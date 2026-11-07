"""Manual online dev smoke test. See USAGE.md; never deploys infrastructure."""

import argparse
import hashlib
import json
import sys
import uuid
from pathlib import Path

ACCOUNT = "710809468662"
REGION = "us-east-2"


def sample(run_id):
    identifier = str(uuid.UUID(run_id))
    if identifier != run_id:
        raise ValueError("Use a canonical lowercase UUID")
    record = {"schema_version": 1, "smoke_id": identifier,
              "shared_goal": "Verify synthetic S3 to PostgreSQL ingestion"}
    return "smoke/e2e/" + identifier + "/sample.json", json.dumps(
        record, sort_keys=True).encode()


def check_identity(session, require_ec2):
    identity = session.client("sts").get_caller_identity()
    if identity["Account"] != ACCOUNT:
        raise ValueError("Wrong AWS account")
    if require_ec2 and session.get_credentials().method != "iam-role":
        raise ValueError("Run on the EC2 host with its instance role, not a profile")


def read_sample(s3, bucket, key, expected):
    response = s3.get_object(Bucket=bucket, Key=key, ExpectedBucketOwner=ACCOUNT)
    with response["Body"] as stream:
        body = stream.read(len(expected) + 1)
    if body != expected:
        raise ValueError("S3 sample content mismatch")
    if response.get("VersionId") in (None, "null"):
        raise ValueError("Sample is not versioned")
    return json.loads(body), response


def verify_database(connection, task_key, uri, goal):
    rows = connection.execute(
        """SELECT r.status, r.artifact_uri, v.outcome_correct, t.shared_goal
           FROM tasks t JOIN runs r USING (task_id)
           JOIN v_run_summary v ON v.run_id = r.run_id
           WHERE t.task_key = %s AND t.dataset_source = 'e2e_smoke'""",
        (task_key,)).fetchall()
    if rows != [("completed", uri, True, goal)]:
        raise ValueError("Database/view verification failed")


def load_database(connection, task_key, uri, record):
    # No upsert: a reused ID must never overwrite an existing test or real data.
    task_id = connection.execute(
        """INSERT INTO tasks
           (task_key,dataset_source,repo_url,base_commit,shared_goal,
            conflict_type,expected_outcome)
           VALUES (%s,'e2e_smoke','https://github.com/UofT-CSC490-F2026/ALIGN',
                   'synthetic',%s,'none','complete_all_goals') RETURNING task_id""",
        (task_key, record["shared_goal"])).fetchone()[0]
    run_id = connection.execute(
        """INSERT INTO runs (task_id,trial_key,arbiter_condition,status,
                            started_at,finished_at,artifact_uri)
           VALUES (%s,%s,'none','completed',now(),now(),%s) RETURNING run_id""",
        (task_id, task_key, uri)).fetchone()[0]
    connection.execute(
        """INSERT INTO evaluations (run_id,evaluator_version,outcome_correct,
                                   shared_goal_satisfied,critical_constraints_passed)
           VALUES (%s,'e2e-smoke-v1',true,true,true)""", (run_id,))
    verify_database(connection, task_key, uri, record["shared_goal"])


def run(args):
    import boto3
    from botocore.config import Config

    key, body = sample(args.run_id)
    session = boto3.Session(region_name=REGION, profile_name=args.profile)
    check_identity(session, args.mode != "verify-backup")
    s3 = session.client("s3", config=Config(
        connect_timeout=10, read_timeout=30, retries={"max_attempts": 2}))
    lake = f"align-dev-lake-{ACCOUNT}"
    bucket = f"align-dev-backup-{ACCOUNT}" if args.mode == "verify-backup" else lake
    if args.mode == "run":
        s3.put_object(Bucket=lake, Key=key, Body=body,
                      ContentType="application/json", IfNoneMatch="*",
                      ExpectedBucketOwner=ACCOUNT)
    record, response = read_sample(s3, bucket, key, body)
    print(json.dumps({"stage": "s3_readback", "status": "passed",
                      "bucket": bucket, "key": key,
                      "version_id": response["VersionId"],
                      "sha256": hashlib.sha256(body).hexdigest()}), flush=True)
    if args.mode == "verify-backup":
        if response.get("ReplicationStatus") != "REPLICA":
            raise ValueError("Backup object is not a replication replica")
        return

    import psycopg

    secret = json.loads(session.client("secretsmanager").get_secret_value(
        SecretId=args.db_secret_arn)["SecretString"])
    # RDS-managed credentials need not contain the host or database name.
    settings = dict(host=args.db_host, port=5432, dbname=args.db_name,
                    user=secret["username"], password=secret["password"],
                    sslmode="verify-full", sslrootcert=args.ca_bundle,
                    connect_timeout=10, application_name="align-e2e-smoke",
                    options="-c statement_timeout=30000 -c lock_timeout=5000")
    task_key = "e2e-smoke:" + args.run_id
    uri = f"s3://{lake}/{key}"
    with psycopg.connect(**settings) as connection:
        if args.mode == "run":
            load_database(connection, task_key, uri, record)
        else:
            verify_database(connection, task_key, uri, record["shared_goal"])
    # A separate connection checks committed data, not just the insert transaction.
    with psycopg.connect(**settings) as connection:
        verify_database(connection, task_key, uri, record["shared_goal"])
    print(json.dumps({"stage": "database_and_view", "status": "passed",
                      "task_key": task_key}), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("run", "verify", "verify-backup"))
    parser.add_argument("--run-id", required=True, help="New UUID for run; reuse for verification")
    parser.add_argument("--profile", help="Operator profile, only for verify-backup")
    parser.add_argument("--db-host")
    parser.add_argument("--db-name")
    parser.add_argument("--db-secret-arn")
    parser.add_argument("--ca-bundle")
    args = parser.parse_args(argv)
    try:
        sample(args.run_id)
    except ValueError:
        parser.error("--run-id must be a canonical lowercase UUID")
    if args.mode != "verify-backup":
        if args.profile or not all((args.db_host, args.db_name, args.db_secret_arn,
                                   args.ca_bundle)):
            parser.error("EC2 modes require DB arguments and CA bundle, and forbid --profile")
        if not Path(args.ca_bundle).is_file():
            parser.error("CA bundle must exist")
        if not args.db_secret_arn.startswith(f"arn:aws:secretsmanager:{REGION}:{ACCOUNT}:secret:"):
            parser.error("Secret must belong to the intended account/region")
    print(json.dumps({"mode": args.mode, "smoke_id": args.run_id}), flush=True)
    try:
        run(args)
    except Exception as error:
        # SDK/DB messages can contain credentials or DSNs. Never print them.
        print(json.dumps({"status": "failed", "error_type": type(error).__name__,
                          "note": "Check the last passed stage; sample may remain"}), file=sys.stderr)
        return 1
    print(json.dumps({"status": "passed", "mode": args.mode}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
