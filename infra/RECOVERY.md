# Prod disaster recovery: completed exercise and runbook

Account `710809468662`, region `us-east-2`. This records the approved October 22,
2026 exercise, NOT permission to repeat it. Dev, backup and bootstrap roots are out
of scope. Preserve state, backup buckets, snapshots and their KMS key.

## Completed results

- Terraform deleted 36 prod resources; dev remained intact.
- RDS restored from **a2-snapshot-before-disaster**, confirmed by an AWS event.
- Restore: 1 imported, 35 added, 1 changed, 0 destroyed.
- Five S3 objects recovered from exact backup versions; all SHA-256 hashes matched.
- Original SQL task/run/event IDs 1/1/1 and task contents matched before evidence.
- New fake task `d9274996-d957-49eb-ba9d-b3526b376186` committed IDs 2/2/2 and
  passed fresh-connection readback.
- Final Terraform plan: no changes. Private/encrypted RDS, deletion protection,
  EC2/RDS security groups, S3 versioning and public-access block checked.

This proves the focused synthetic S3/PostgreSQL workflow, not a full benchmark
application. No LLM or application image was required. Screenshots and sanitized
JSON remain separately in the operator checkout under `a2/evidence/`, in
`dr-deletion-20261022`, `before-prod-20261022T235746Z`, and
`dr-restoration-20261022`. They are not committed here.

**Prod is already restored. Always include BOTH prod.tfvars and restore.tfvars in
future prod plans. Removing the snapshot input can propose DB replacement.**
The final-snapshot name in restore.tfvars is for a future approved deletion; it
does not create a snapshot during normal apply.

## Before/protect

The actual preflight verified account identity, backend bucket
`align-tfstate-710809468662-us-east-2`, key `align/prod/terraform.tfstate`,
the original snapshot Available, and its KMS key Enabled. All five source object
versions matched their backup replicas byte-for-byte. No historical source
versions/delete markers existed. ECR was empty and the LLM secret had no value.
A fresh read-only SQL connection retrieved the original log before deletion.

Read-only CLI checks used included:

~~~powershell
$env:AWS_PROFILE = 'align-new'
$env:AWS_REGION = 'us-east-2'
aws sts get-caller-identity
aws rds describe-db-snapshots --db-instance-identifier align-prod-postgres --snapshot-type manual --query 'DBSnapshots[].{Name:DBSnapshotIdentifier,Status:Status}' --output json
aws rds describe-db-instances --db-instance-identifier align-prod-postgres --query 'DBInstances[0].{Status:DBInstanceStatus,DeletionProtection:DeletionProtection}' --output json
~~~

For another exercise, coordinate downtime/freeze writes, capture fresh before
evidence, and check the snapshot recovery point. Later SQL writes are not in that
snapshot. Stop if images, secret values, instance-local files or historical S3
versions now require additional preservation. Keep secrets/raw state/full JSON
plans out of shared logs. Record the empty LLM secret ARN, never its value.

## Terraform deletion commands used

The original operator wrapper executed these commands from `infra/envs/prod`.
Paths below abbreviate the private temporary directory. Every plan was reviewed
before apply; binary plans are not committed.

~~~powershell
terraform plan -input=false -no-color -var-file=prod.tfvars -var-file=teardown.tfvars -out='<private>/align-dr-guards-20261022.tfplan'
terraform apply -input=false -no-color '<private>/align-dr-guards-20261022.tfplan'
# Verified live RDS deletion protection was false, then generated a NEW plan:
terraform plan -input=false -no-color -var-file=prod.tfvars -var-file=teardown.tfvars -out='<private>/align-dr-destroy-20261022.tfplan' -destroy
terraform apply -input=false -no-color '<private>/align-dr-destroy-20261022.tfplan'
~~~

Historical teardown inputs: snapshot=null (the original DB was not yet restored),
deletion_protection=false, lake_force_destroy=true, and final snapshot name
`align-prod-final-dr-20261021-01`. Guard apply changed two resources, destroyed none.
The 36-resource deletion plan excluded backup/state buckets and the manual snapshot.
Terraform created its configured final deletion snapshot as an extra safeguard;
restoration used ONLY the original manual snapshot.

**Do not reuse those old inputs on today's restored database.** For a future
approved deletion, copy `teardown.tfvars.example` to a private operator file,
retain the CURRENT snapshot input, and choose an UNUSED final-snapshot name.
Include prod.tfvars, restore.tfvars, then the reviewed teardown override when
planning. Verify the account/backend/default workspace and every planned resource.
The prior resource count is not authorization. Final snapshots incur storage costs.
Force-destroy removes ALL lake versions; verified backups are mandatory.
Never destroy dev/backup/bootstrap or force-delete extra resources after a partial
failure. Wait for deletion to finish and capture deleted-state screenshots.

## Terraform restoration commands used

The empty LLM-secret metadata was recovered using the Secrets Manager RestoreSecret
API. Its CLI equivalent is shown here, not claimed as the original invocation:

~~~powershell
aws secretsmanager restore-secret --secret-id '<recorded empty-secret ARN>'
~~~

Use only when scheduled for deletion. A CLI import attempt failed on uncreated
network for_each keys; this temporary declarative import worked:

~~~hcl
import {
  to = module.compute.aws_secretsmanager_secret.llm_api_key
  id = "arn:aws:secretsmanager:us-east-2:710809468662:secret:align-prod-llm-api-key-69hLEN"
}
~~~

No secret value was read or committed. Do not import an already tracked resource.
Remove the temporary block after successful apply, as we did.

~~~powershell
terraform plan -input=false -no-color -var-file=prod.tfvars -var-file=restore.tfvars -out='<private>/align-dr-restore-plan-20261022.tfplan'
# Reviewed original snapshot, private/encrypted RDS, managed credentials,
# sizing, deletion protection, 35 creates + secret import/update; no deletions.
terraform apply -input=false -no-color '<private>/align-dr-restore-plan-20261022.tfplan'
terraform output -json
~~~

Only selected non-secret outputs were retained. The restored RDS managed secret
changes; use fresh endpoint/secret/EC2 outputs, never old helper defaults. Wait for
RDS Available and EC2 SSM Online. Do not reseed or recreate the restored schema.

## S3 and SQL operations actually used

The original script read the saved manifest's key, backup version and SHA-256 for
each object, checked the recreated lake was empty, and executed this SDK operation:

~~~python
s3.copy_object(
    Bucket='align-prod-lake-710809468662', Key=item['key'],
    CopySource={'Bucket': 'align-prod-backup-710809468662',
                'Key': item['key'], 'VersionId': item['backup_version']},
    ExpectedBucketOwner='710809468662', ExpectedSourceBucketOwner='710809468662')
restored = s3.get_object(Bucket='align-prod-lake-710809468662', Key=item['key'])
assert hashlib.sha256(restored['Body'].read()).hexdigest() == item['sha256']
~~~

This is a historical excerpt, not a standalone helper. `s3`, `item` and `hashlib`
were supplied by the original script. Copies created new lake versions with the
same bytes, without deleting backups. Keep writers paused during any future copy,
check every hash, and inspect partial failures before retrying. Do not run historical
assert-based checks with Python optimization enabled.

Original task key:
`demo/task-log/9c985e79-81bf-4be3-b32c-2c543e0b1777/task.json`.
SHA-256: `fee1d8ac10eeae5b30fd82b0f386fb651706c92d40d95d9a47fec135d606441a`.

On restored EC2 through SSM, a fresh PostgreSQL connection used TLS verify-full,
the NEW managed secret, and default_transaction_read_only=on. This parameterized
query retrieved the original row:

~~~sql
SELECT t.task_id,t.task_key,t.dataset_source,t.shared_goal,
       r.run_id,r.status,r.artifact_uri,e.event_id,e.event_type,e.payload,
       e.artifact_uri AS event_s3_reference
FROM tasks t JOIN runs r USING(task_id) JOIN events e USING(run_id)
WHERE t.task_key = %s AND t.dataset_source='fake_log_demo';
~~~

The bound parameter was `fake-log:prod:9c985e79-81bf-4be3-b32c-2c543e0b1777`.
The row, S3 reference, JSON contents and hash matched the saved before evidence.
Original IDs remained 1/1/1; message:
`Fake log: audio volume setting saved successfully.`

The focused fake-log test then created ONE new task/log and verified committed
IDs 2/2/2 from a separate connection. PR #17 documents its current explicit DB
arguments; the historical invocation instead injected current outputs into its
older connection defaults. These are synthetic records, not benchmark results.

The final command returned no changes:

~~~powershell
terraform plan -input=false -no-color -var-file=prod.tfvars -var-file=restore.tfvars -out='<private>/align-dr-restore-final-plan-20261022.tfplan'
~~~

Historical local entry points were `prod_dr_execute.py`, `prod_dr_restore.py`,
`collect_before_evidence.py`, and `verify_restored_prod.py`. These are operator
files, not shipped dependencies. This PR contains Terraform recovery configuration,
its mocked tests, and this runbook only; no new Python recovery helpers.
