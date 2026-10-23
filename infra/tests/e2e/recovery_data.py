"""Protect/restore prod S3 data. Default restore is read-only; no Terraform execution."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

ACCOUNT = '710809468662'
REGION = 'us-east-2'
LAKE = f'align-prod-lake-{ACCOUNT}'
BACKUP = f'align-prod-backup-{ACCOUNT}'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(response):
    with response['Body'] as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def protect(s3, snapshot):
    """Record every current source version; reject history rather than silently lose it."""
    rows = []
    for page in s3.get_paginator('list_object_versions').paginate(Bucket=LAKE, ExpectedBucketOwner=ACCOUNT):
        require(not page.get('DeleteMarkers'), 'Delete markers need a separately reviewed recovery policy')
        for item in page.get('Versions', []):
            require(item['IsLatest'], 'Historical source versions need a separately reviewed recovery policy')
            source = s3.get_object(Bucket=LAKE, Key=item['Key'], VersionId=item['VersionId'], ExpectedBucketOwner=ACCOUNT)
            backup = s3.get_object(Bucket=BACKUP, Key=item['Key'], ExpectedBucketOwner=ACCOUNT)
            require(source.get('ReplicationStatus') == 'COMPLETED' and backup.get('ReplicationStatus') == 'REPLICA', 'Replication not complete')
            version = backup.get('VersionId')
            require(version not in (None, '', 'null'), 'Backup must be versioned')
            checksum = digest(source)
            require(checksum == digest(backup), 'Backup bytes differ')
            rows.append(dict(key=item['Key'], source_version=item['VersionId'], backup_version=version, sha256=checksum))
    require(bool(rows), 'Source lake is empty')
    return dict(schema_version=1, account=ACCOUNT, region=REGION, lake=LAKE, backup=BACKUP, snapshot=snapshot, objects=rows)


def validate_manifest(manifest):
    require(manifest.get('schema_version') == 1 and manifest.get('account') == ACCOUNT and manifest.get('region') == REGION, 'Manifest account/region/schema mismatch')
    require(manifest.get('lake') == LAKE and manifest.get('backup') == BACKUP, 'Unexpected bucket target')
    rows = manifest.get('objects')
    require(isinstance(rows, list) and bool(rows), 'No objects in manifest')
    keys = set()
    for row in rows:
        key = row.get('key')
        require(isinstance(key, str) and key and key not in keys, 'Missing or duplicate key')
        keys.add(key)
        require(isinstance(row.get('backup_version'), str) and row['backup_version'] not in ('', 'null'), 'Missing backup version')
        checksum = row.get('sha256')
        require(isinstance(checksum, str) and len(checksum) == 64 and all(c in '0123456789abcdef' for c in checksum), 'Invalid checksum')
    return rows


def restore(s3, manifest, apply=False):
    rows = validate_manifest(manifest)
    # Verify ALL selected backup versions before allowing writes.
    for row in rows:
        response = s3.get_object(Bucket=BACKUP, Key=row['key'], VersionId=row['backup_version'], ExpectedBucketOwner=ACCOUNT)
        require(digest(response) == row['sha256'], 'Backup version checksum mismatch')
    if not apply:
        return dict(status='dry-run', verified_backup_objects=len(rows), writes=0)
    # Intended only for a newly recreated, quiescent lake. Never empty it here.
    s3.head_bucket(Bucket=LAKE, ExpectedBucketOwner=ACCOUNT)
    for page in s3.get_paginator('list_object_versions').paginate(Bucket=LAKE, ExpectedBucketOwner=ACCOUNT):
        require(not page.get('Versions') and not page.get('DeleteMarkers'), 'Destination is not empty; inspect partial restores manually')
    for row in rows:
        # S3 server-side version-specific copy retains object metadata. No backup deletion.
        s3.copy_object(Bucket=LAKE, Key=row['key'], CopySource={'Bucket': BACKUP, 'Key': row['key'], 'VersionId': row['backup_version']}, ExpectedBucketOwner=ACCOUNT, ExpectedSourceBucketOwner=ACCOUNT)
        response = s3.get_object(Bucket=LAKE, Key=row['key'], ExpectedBucketOwner=ACCOUNT)
        require(digest(response) == row['sha256'], 'Restored object checksum mismatch')
    return dict(status='restored', objects=len(rows), all_hashes_match=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('protect', 'restore'))
    parser.add_argument('--profile', required=True)
    parser.add_argument('--manifest', required=True, type=Path)
    parser.add_argument('--snapshot', help='Available manual prod snapshot; required for protect')
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    if args.mode == 'protect' and (not args.snapshot or args.apply):
        parser.error('protect requires --snapshot and does not accept --apply')
    if args.mode == 'protect' and args.manifest.exists():
        parser.error('Use a new manifest filename; existing evidence is never overwritten')
    import boto3
    session = boto3.Session(profile_name=args.profile, region_name=REGION)
    require(session.client('sts').get_caller_identity()['Account'] == ACCOUNT, 'Wrong AWS account')
    if args.mode == 'protect':
        snap = session.client('rds').describe_db_snapshots(DBSnapshotIdentifier=args.snapshot)['DBSnapshots'][0]
        require(snap['Status'] == 'available' and snap['SnapshotType'] == 'manual' and snap['DBInstanceIdentifier'] == 'align-prod-postgres', 'Expected an Available manual prod snapshot')
        manifest = protect(session.client('s3'), args.snapshot)
        with args.manifest.open('x', encoding='utf-8') as output:
            json.dump(manifest, output, indent=2)
        print(json.dumps(dict(status='protected', objects=len(manifest['objects']), snapshot=args.snapshot)))
    else:
        manifest = json.loads(args.manifest.read_text(encoding='utf-8'))
        print(json.dumps(restore(session.client('s3'), manifest, args.apply)))


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        print(json.dumps({'status': 'failed', 'error_type': type(error).__name__, 'note': 'No automatic cleanup or retry; inspect any partial restore.'}), file=sys.stderr)
        sys.exit(1)
