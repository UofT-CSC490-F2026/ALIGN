# Manual connectivity / data-path tests

These are synthetic verification tools, not the CooperBench ingestion pipeline in
`scripts/`. Neither command deploys infrastructure, deletes data, or calls an LLM.

Deployment review warning: this infrastructure PR intentionally excludes the
separate disaster-recovery changes. Do NOT apply this revision to the already
snapshot-restored prod database. Keep using the recovery-enabled deployment
checkout and both prod/restore tfvars until the recovery PR is merged. Removing
snapshot configuration from a restored database can propose replacement.

## Focused fake task and SQL log (dev or prod)

`fake_log_demo.py` writes a uniquely labeled task JSON to the selected lake, reads
it back, inserts one task/run/event in PostgreSQL, commits, and verifies the log
and S3 reference using a fresh connection. It requires EC2 instance-role credentials
in account `710809468662`, region `us-east-2`. Run only on the selected environment's
EC2 host through Session Manager, after schema migrations and with permission to
write retained synthetic data. The DB remains private; do not open port 22 or 5432
to your laptop. It uses the RDS-managed password, not IAM database authentication.

Use CURRENT `db_address` and `db_master_secret_arn` Terraform outputs. The managed
secret changes after snapshot recovery. No passwords or AWS keys belong in the command.
The shared CA/dependency setup below also applies to this test.

```bash
python fake_log_demo.py --environment prod --apply \
  --db-host '<current db_address>' \
  --db-secret-arn '<current db_master_secret_arn>' \
  --ca-bundle global-bundle.pem
```

Use `--environment dev` with dev outputs for development. This command always
creates a NEW UUID. Save its printed S3 URI and task/run/event IDs. It intentionally
retains data labeled `dataset_source = 'fake_log_demo'`; do not include it in real
benchmark statistics. A failure can leave an S3 object without SQL rows; there is
no automatic cleanup. Never remove the referenced S3 object independently of its log.

The focused flow passed live dev/prod checks and a prod recovery exercise before
this PR; the explicit connection arguments are a subsequent portability/safety
change covered locally, not a new live AWS run. Unit tests never prove connectivity.

## Extended dev-only smoke test

This is separate from the offline nightly smoke test. It does not deploy anything,
run agents, call an LLM, or run the team's actual pipeline scripts. It tests a small
synthetic S3 -> Python -> PostgreSQL -> reporting-view path. It is dev-only.

Prerequisites: deploy the backup bucket, dev lake, private RDS and EC2 with its
instance role; apply the database Flyway migrations. Compute must be wired into
dev first. Open a Session Manager shell on that host and copy this directory to it.
Use actual Terraform outputs for the DB arguments; do not copy AWS credentials to EC2.
Install the Python dependencies in a virtual environment. Download the RDS trust
bundle over HTTPS from
https://truststore.pki.rds.amazonaws.com/global/global-bundle.pem.
The RDS connection checks both certificate trust and hostname (`verify-full`).

On EC2 (Linux):

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
curl --fail --show-error https://truststore.pki.rds.amazonaws.com/global/global-bundle.pem -o global-bundle.pem
SMOKE_ID=$(python -c 'import uuid; print(uuid.uuid4())')
echo "$SMOKE_ID"
set -o pipefail
python smoke.py run --run-id "$SMOKE_ID" \
  --db-host '<db_address>' --db-name '<db_name>' \
  --db-secret-arn '<db_master_secret_arn>' --ca-bundle global-bundle.pem \
  | tee smoke-evidence.jsonl
```

Save the UUID and evidence outside the environment being destroyed. `run` creates
one versioned S3 JSON object, one task, one completed run and one evaluation.
It downloads and validates the object before inserting its content, then verifies
committed DB data from a new connection. Reusing an ID for `run` fails rather than
overwriting data. To verify retained/restored data, repeat with `verify` instead of
`run`, the original UUID, and the current DB endpoint/secret. No writes in `verify`.

From your operator machine, after replication has had time to complete:

```powershell
python -m pip install -r infra/tests/e2e/requirements.txt
python infra/tests/e2e/smoke.py verify-backup --profile align-new --run-id '<saved UUID>'
```

This separately checks the backup's bytes, version ID and REPLICA status. A missing
replica is a failure, not a pass; retry later if replication is pending. EC2 does
not need backup access or delete permission.

Samples are deliberately retained for recovery verification. They appear in reports
as dataset_source `e2e_smoke`; exclude these from experiment comparisons. No automatic
cleanup is performed. After the demo, an authorized operator should remove only the
matching evaluation/run/task and the exact `smoke/e2e/<UUID>/sample.json` key's versions
in both buckets. Ordinary S3 deletion only adds a delete marker, and source deletion
does not clean the backup. Never empty a bucket for this test. If a run fails, its S3
object may remain even when its database transaction rolled back; save its UUID too.

Passing does not prove ECR pulls, app/container health, cron execution, arbitrary
internet access, denied IAM operations, or recovery itself. Check those separately
before recording Part 5. No cloud execution is covered by the local unit tests:

```powershell
python -m unittest discover -s infra/tests/e2e -v
```
