# Prod disaster recovery

This is a manual, approved downtime exercise for account `710809468662`, region
`us-east-2`. Never execute it just because this PR merges. Dev, backup and bootstrap
roots are out of scope. Preserve Terraform state, backup buckets and snapshots.

Terraform owns deletion/recreation; the small Python scripts only verify/copy data.
They are justified because Terraform does not compare S3 bytes or query application
rows. No alternative infrastructure provisioner or second data pipeline is added.

## Current state and completed exercise

On 2026-10-22 Toronto, prod was deleted and recreated with Terraform. RDS restored
from **`a2-snapshot-before-disaster`**, confirmed by an AWS event. Five S3 files
matched their original SHA-256 checksums. Original SQL task/run/event IDs 1/1/1
and task JSON matched before evidence. A new fake task/log committed IDs 2/2/2 and
passed fresh-connection readback. Final Terraform plan: no changes.

The exercise used operator-local helpers. The reusable helpers in this PR are
cleaned-up versions with local mocked tests; they have NOT been rerun against AWS.
Evidence/screenshots remain outside this PR. This proves the focused synthetic
S3/PostgreSQL path, not CooperBench execution or a running production application.

**The current database is already restored. Always include `restore.tfvars` in
future prod plans. Removing its snapshot input can propose DB replacement.**
That file selects the original recovery snapshot. Its final-snapshot identifier
is only for a future deletion, not a request to create a snapshot on apply.

## 1. Before and protect

Freeze writes/coordinate downtime. Review SQL/S3 evidence and ensure the snapshot
is recent enough; writes after its creation are outside its recovery point.
Use current Terraform outputs for DB endpoint, managed secret ARN and EC2 ID.
Never print a secret value, raw state, or a full JSON plan into shared evidence.

Run the SQL capture on prod EC2 through Session Manager, using the venv/dependencies
and RDS CA bundle described in `tests/e2e/USAGE.md`:

```bash
python recovery_sql.py --task-key '<original fake-log:prod:UUID>' \
  --db-host '<current db_address>' --db-secret-arn '<current db_master_secret_arn>' \
  --ca-bundle global-bundle.pem --output before-sql.json
```

Save this file and the task IDs outside prod. On the operator machine, from repo root:

```powershell
$env:AWS_PROFILE = 'align-new'
$env:AWS_REGION = 'us-east-2'
aws sts get-caller-identity
python infra/tests/e2e/recovery_data.py protect --profile align-new --snapshot a2-snapshot-before-disaster --manifest '<outside-prod>/backup-manifest.json'
```

The helper is read-only against AWS. It validates the manual snapshot and compares
every current lake object to its replica, recording exact backup versions/hashes.
It deliberately refuses historical versions or delete markers: stop and define a
version-history recovery policy instead of discarding them. It works with real task
objects as well as fake ones. It does not prove snapshot SQL contents until restore.
Verify the snapshot KMS key is Enabled and preserve it.

Also record EC2/RDS/security/configuration screenshots and the empty LLM-secret ARN
(metadata only). Confirm ECR is empty and no instance-local artifacts need saving.
Stop if images, secret values or unpreserved data were added since the last exercise.

## 2. Review and delete (explicit approval required)

Use ONLY `infra/envs/prod`. Verify initialized backend is the state bucket
`align-tfstate-710809468662-us-east-2`, key `align/prod/terraform.tfstate`, default
workspace, correct account. Never copy state between accounts or delete it.

Copy `teardown.tfvars.example` to an operator-owned file outside the repo. Keep the
CURRENT `db_snapshot_identifier`, and choose an UNUSED final snapshot name. The
final snapshot is an additional deletion safeguard; obtain approval for its storage
cost. The previous exercise kept one in addition to the original restore source.

```powershell
Set-Location infra/envs/prod
terraform workspace show
terraform plan -var-file=prod.tfvars -var-file=restore.tfvars -var-file='<reviewed teardown file>' -out='<private>/guards.tfplan'
# STOP: review. Only DB deletion protection/final snapshot and lake force_destroy change.
terraform apply '<private>/guards.tfplan'
# Verify live RDS DeletionProtection is false, then generate a NEW plan.
terraform plan -destroy -var-file=prod.tfvars -var-file=restore.tfvars -var-file='<reviewed teardown file>' -out='<private>/destroy.tfplan'
# STOP: review every address/ID before executing. Only prod resources may be deleted.
terraform apply '<private>/destroy.tfplan'
```

The original exercise deleted 36 resources; recheck current scope rather than using
that count as authorization. Backup/state buckets and manual snapshots must NOT
appear. Keep binary plans private. `force_destroy` removes all lake versions; the
backup check is a mandatory prerequisite. Wait for completion/final snapshot and
capture deleted-state screenshots before restoration. Inspect partial failures;
do not force-delete extra resources or automatically retry a destructive operation.

## 3. Recreate from the original snapshot

Check `a2-snapshot-before-disaster` remains Available. Recover the empty LLM-secret
metadata if Secrets Manager shows a scheduled deletion; do not force-delete it:

```powershell
aws secretsmanager restore-secret --secret-id '<recorded empty-secret ARN>'
```

If Terraform no longer tracks it, add a temporary `recovery-import.tf` in prod:

```hcl
import {
  to = module.compute.aws_secretsmanager_secret.llm_api_key
  id = "<recorded empty-secret ARN>"
}
```

Declarative import avoids the CLI-import issue with not-yet-created network keys.
Do not import over an already tracked resource. No secret value is involved.

```powershell
terraform plan -var-file=prod.tfvars -var-file=restore.tfvars -out='<private>/restore.tfplan'
# STOP: confirm original snapshot, private RDS, encryption, sizing, managed credentials,
# deletion protection, and no unintended deletions. Then, with recovery approval:
terraform apply '<private>/restore.tfplan'
```

Remove the temporary import block after success. Wait for RDS Available and new EC2
SSM Online. Use NEW outputs: the managed DB secret changes. Do not reseed or rerun
schema creation over the restored tables. Do not reuse old helper connection defaults.

## 4. Restore files and prove recovery

From repo root on the operator machine:

```powershell
python infra/tests/e2e/recovery_data.py restore --profile align-new --manifest '<saved backup-manifest.json>'
# After the dry-run passes and restoration is approved:
python infra/tests/e2e/recovery_data.py restore --profile align-new --manifest '<saved backup-manifest.json>' --apply
```

The target must be an empty recreated prod lake with ALL writers paused. The helper
copies exact backup versions and verifies hashes; it never deletes backup objects.
This is not an atomic multi-object restore. On partial failure, inspect the destination
and reconcile manually; do not empty it or blindly rerun. No concurrent writes allowed.

Copy saved before-sql.json to restored EC2 and run `recovery_sql.py` with the original
task key, NEW DB outputs, `--before before-sql.json --output after-sql.json` and CA
bundle. This compares the SQL row, S3 reference, JSON and checksum using read-only SQL.
Then use `fake_log_demo.py` from PR #17 to create ONE new task/log and verify a fresh
connection. Preserve its new IDs separately from the original. No LLM call required.

Verify restored security groups (EC2 no inbound, RDS 5432 from EC2 SG only), IAM
permissions, RDS private/encrypted/deletion-protected settings, S3 encryption,
versioning, replication and public-access block. Capture After screenshots, confirm
dev stayed intact, and run a final plan with BOTH prod.tfvars and restore.tfvars.
Record snapshot time, deletion/recovery timestamps and measured verification results.

## Local validation

```powershell
python -m unittest discover -s infra/tests/e2e -v
terraform -chdir=infra/modules/database test
```

These are offline/mocked tests; passing them does not execute another AWS recovery.
