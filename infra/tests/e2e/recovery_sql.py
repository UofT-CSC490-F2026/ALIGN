"""Capture/compare the original fake SQL log and S3 task, using read-only SQL."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

from recovery_data import ACCOUNT, REGION, LAKE, require

QUERY = """SELECT t.task_id,t.task_key,t.dataset_source,t.shared_goal,
 r.run_id,r.status,r.artifact_uri,e.event_id,e.event_type,e.payload,
 e.artifact_uri AS event_s3_reference
 FROM tasks t JOIN runs r USING(task_id) JOIN events e USING(run_id)
 WHERE t.task_key=%s AND t.dataset_source='fake_log_demo'"""


def compare(actual, expected):
    require(json.dumps(actual, sort_keys=True) == json.dumps(expected, sort_keys=True), 'Recovered SQL/task data differs from before evidence')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task-key', required=True)
    parser.add_argument('--db-host', required=True)
    parser.add_argument('--db-secret-arn', required=True)
    parser.add_argument('--ca-bundle', required=True, type=Path)
    parser.add_argument('--before', type=Path, help='Omit to capture; provide to compare after recovery')
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    require(not args.output.exists(), 'Evidence output already exists')
    require(args.task_key.startswith('fake-log:prod:'), 'Expected a prod fake-log task key')
    require(args.db_host.startswith('align-prod-postgres.') and args.db_host.endswith('.us-east-2.rds.amazonaws.com'), 'Expected the prod RDS endpoint')
    require(args.db_secret_arn.startswith(f'arn:aws:secretsmanager:{REGION}:{ACCOUNT}:secret:rds!db-'), 'Wrong secret account/region')
    require(args.ca_bundle.is_file(), 'CA bundle missing')
    import boto3
    import psycopg
    session = boto3.Session(region_name=REGION)
    require(session.get_credentials().method == 'iam-role', 'Run on prod EC2 with its instance role')
    require(session.client('sts').get_caller_identity()['Account'] == ACCOUNT, 'Wrong AWS account')
    secret = json.loads(session.client('secretsmanager').get_secret_value(SecretId=args.db_secret_arn)['SecretString'])
    with psycopg.connect(host=args.db_host, dbname='align', user=secret['username'], password=secret['password'], sslmode='verify-full', sslrootcert=str(args.ca_bundle), connect_timeout=15, options='-c default_transaction_read_only=on -c statement_timeout=30000') as conn:
        result = conn.execute(QUERY, (args.task_key,))
        rows = [dict(zip([d.name for d in result.description], row)) for row in result.fetchall()]
    require(len(rows) == 1, 'Expected exactly one original log')
    uri = rows[0]['artifact_uri']
    prefix = f's3://{LAKE}/'
    require(isinstance(uri, str) and uri.startswith(prefix) and uri == rows[0]['event_s3_reference'], 'S3 references mismatch')
    response = session.client('s3').get_object(Bucket=LAKE, Key=uri[len(prefix):], ExpectedBucketOwner=ACCOUNT)
    with response['Body'] as stream:
        body = stream.read()
    task = json.loads(body)
    require(task['task_key'] == args.task_key, 'Task identity mismatch')
    evidence = dict(rows=rows, task=task, sha256=hashlib.sha256(body).hexdigest())
    if args.before:
        compare(evidence, json.loads(args.before.read_text(encoding='utf-8')))
    with args.output.open('x', encoding='utf-8') as output:
        json.dump(evidence, output, indent=2)
    print(json.dumps({'status': 'matched' if args.before else 'captured', 'task_key': args.task_key, 'read_only': True}))


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        print(json.dumps({'status': 'failed', 'error_type': type(error).__name__}), file=sys.stderr)
        sys.exit(1)
