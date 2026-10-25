"""One uniquely labeled fake task and SQL log in an explicitly chosen environment."""
import argparse
import json, sys, uuid
from pathlib import Path

def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--environment', choices=('dev', 'prod'), required=True)
    parser.add_argument('--db-host', required=True, help='Current Terraform db_address output')
    parser.add_argument('--db-secret-arn', required=True, help='Current Terraform db_master_secret_arn output; not a password')
    parser.add_argument('--ca-bundle', type=Path, required=True, help='Downloaded AWS RDS CA bundle')
    parser.add_argument('--apply', action='store_true', help='Required acknowledgement: retain a new fake S3 object and SQL rows')
    args = parser.parse_args(argv)
    if not args.apply:
        parser.error('--apply is required; this command writes retained fake data')
    if not args.ca_bundle.is_file():
        parser.error('--ca-bundle must be an existing file')
    expected_host = f'align-{args.environment}-postgres.'
    if not args.db_host.startswith(expected_host) or not args.db_host.endswith('.us-east-2.rds.amazonaws.com'):
        parser.error('--db-host must identify the selected ALIGN environment in us-east-2')
    if not args.db_secret_arn.startswith('arn:aws:secretsmanager:us-east-2:710809468662:secret:rds!db-'):
        parser.error('--db-secret-arn must be an RDS-managed secret in the deployment account/region')
    return args


def main():
    args = parse_args()
    import boto3
    import psycopg
    from psycopg.types.json import Jsonb
    environment = args.environment
    session = boto3.Session(region_name="us-east-2")
    assert session.client("sts").get_caller_identity()["Account"] == "710809468662"
    assert session.get_credentials().method == "iam-role"
    identifier = str(uuid.uuid4())
    bucket = f"align-{environment}-lake-710809468662"
    key = "demo/task-log/" + identifier + "/task.json"
    task = {"task_key": f"fake-log:{environment}:" + identifier,
            "environment": environment,
            "shared_goal": "Add an audio volume setting",
            "synthetic": True}
    s3 = session.client("s3")
    body = json.dumps(task, sort_keys=True).encode()
    s3.put_object(Bucket=bucket, Key=key, Body=body, ContentType="application/json",
                  IfNoneMatch="*", ExpectedBucketOwner="710809468662")
    response = s3.get_object(Bucket=bucket, Key=key, ExpectedBucketOwner="710809468662")
    assert response["Body"].read() == body
    print(json.dumps({"stage":"s3_verified", "uri":f"s3://{bucket}/{key}"}), flush=True)
    secret = json.loads(session.client("secretsmanager").get_secret_value(
        SecretId=args.db_secret_arn)["SecretString"])
    settings = dict(host=args.db_host,
                    dbname="align", user=secret["username"], password=secret["password"],
                    sslmode="verify-full", sslrootcert=str(args.ca_bundle), connect_timeout=15,
                    options="-c statement_timeout=30000 -c lock_timeout=5000")
    uri = f"s3://{bucket}/{key}"
    message = "Fake log: audio volume setting saved successfully."
    with psycopg.connect(**settings) as c:
        task_id = c.execute("""
            INSERT INTO tasks (task_key,dataset_source,repo_url,base_commit,shared_goal,
                               conflict_type,expected_outcome,metadata)
            VALUES (%s,'fake_log_demo','https://example.invalid/fake-repo','synthetic',%s,
                    'none','complete_all_goals',%s) RETURNING task_id
        """, (task["task_key"], task["shared_goal"], Jsonb({"task_uri":uri,"synthetic":True}))).fetchone()[0]
        run_id = c.execute("""
            INSERT INTO runs (task_id,trial_key,arbiter_condition,status,started_at,finished_at,artifact_uri)
            VALUES (%s,%s,'none','completed',now(),now(),%s) RETURNING run_id
        """, (task_id,identifier,uri)).fetchone()[0]
        event_id = c.execute("""
            INSERT INTO events (run_id,sequence_number,actor_type,actor_id,event_type,payload,artifact_uri)
            VALUES (%s,1,'system','fake-data-generator','task_completed',%s,%s) RETURNING event_id
        """, (run_id,Jsonb({"message":message,"synthetic":True}),uri)).fetchone()[0]
    with psycopg.connect(**settings) as c:
        row = c.execute("""
            SELECT t.task_id,r.run_id,e.event_id,t.shared_goal,r.status,
                   e.payload->>'message',r.artifact_uri
            FROM tasks t JOIN runs r USING(task_id) JOIN events e USING(run_id)
            WHERE t.task_key=%s AND t.dataset_source='fake_log_demo'
        """, (task["task_key"],)).fetchall()
        assert row == [(task_id,run_id,event_id,task["shared_goal"],"completed",message,uri)]
        print(json.dumps({"stage":"committed_sql_verified","task_id":task_id,"run_id":run_id,
                          "event_id":event_id,"message":message,"task_uri":uri}), flush=True)

if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(json.dumps({"status":"failed","error_type":type(error).__name__}),flush=True)
        sys.exit(1)
