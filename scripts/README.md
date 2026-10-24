# CooperBench raw ingestion

Uploads a **pinned, unmodified** snapshot of the official `CodeConflict/cooperbench-dataset` Hugging Face dataset into the existing ALIGN S3 lake. No Terraform, PostgreSQL, agent execution or task normalization is involved.

Requires Python 3.10+, `pip install huggingface_hub boto3`, and (for S3 writes) AWS credentials authorized to access the dev lake. `--revision` is the **40-character Hugging Face dataset commit**, not the CooperBench GitHub commit.

From the repository root (PowerShell):

```powershell
# Copy a full 40-character commit ID from the Hugging Face dataset history.
$revision = "PASTE_FULL_HF_DATASET_COMMIT_SHA_HERE"

# Inspect the pinned dataset without AWS access (downloads from Hugging Face).
python scripts/ingest_cooperbench.py --revision $revision --dry-run

# After verifying the account and that the existing bucket is deployed:
aws sts get-caller-identity
aws s3api head-bucket --bucket align-dev-lake-710809468662
python scripts/ingest_cooperbench.py --revision $revision --bucket align-dev-lake-710809468662

# Verify completion marker and original file paths:
aws s3 ls "s3://align-dev-lake-710809468662/benchmarks/raw/cooperbench/$revision/" --recursive
```

For offline fixture testing, add `--source-dir path/to/dataset --dry-run`; this skips Hugging Face and AWS. Unit tests run without AWS/network:

```powershell
python -m unittest discover -s scripts/tests -p 'test_*.py' -v
```

Objects retain original relative paths at `benchmarks/raw/cooperbench/<revision>/dataset/`. A deterministic `manifest.json` written **last** records the revision, paths, sizes and SHA-256 checksums and serves as the completion marker. Repeat uploads skip matching objects; they fail rather than overwrite a conflicting snapshot. A failed upload before the manifest exists can be resumed safely. Object HEAD metadata is checked after upload; this isn't a substitute for a future full-download byte-for-byte audit.

The raw dataset includes gold implementation patches and hidden evaluation tests. **Do not give coding agents access to this raw prefix**; keep it available to a trusted evaluator only.

## Normalized ALIGN task manifests (PR 2)

`normalize_tasks.py` is an **offline-only** adapter from CooperBench feature pairs or team-authored JSON scenarios to the same database-facing `tasks.jsonl` contract. It does **not** deploy infrastructure, connect to S3, run agents, or write to PostgreSQL. Each record contains one `tasks`-compatible object plus its embedded `goals` for a future `task_goals` loader.

### CooperBench

After PR 1's raw snapshot is available locally (for example, by downloading its S3 prefix), normalize the pinned `flash` subset:

```powershell
$revision = "b612b1a35af722751454813d9e5a7888f065fc9e"
py scripts/normalize_tasks.py cooperbench `
  --dataset-dir ".\cooperbench-snapshot\dataset" `
  --raw-manifest ".\cooperbench-snapshot\manifest.json" `
  --revision $revision `
  --subset flash `
  --output-dir ".\normalized\cooperbench-flash-$revision"
```

`--subset` accepts `flash` (default, 50 pairs), `lite` (100 pairs), or `all` (all unordered feature pairs found in the file tree). `--raw-manifest` is optional for a separately obtained local dataset, but **recommended** when consuming the PR 1 snapshot: it validates provenance, paths, sizes and SHA-256 checksums against the original raw manifest before normalization. Without the manifest, callers are responsible for ensuring that `--revision` truly matches the input directory.

### Team-authored scenarios

Create one JSON file per scenario using the sample contract in `scripts/examples/align_custom_sample.json`. The included example is **not runnable**; it exists to prove both sources produce the same record structure.

```powershell
py scripts/normalize_tasks.py custom `
  --scenarios-dir scripts/examples `
  --revision v1 `
  --output-dir ".\normalized\custom-v1"
```

Both modes create deterministic `tasks.jsonl` and `manifest.json` containing `task_count` and a SHA-256 hash of `tasks.jsonl`. Running the same normalization again is safe; conflicting content is never silently overwritten. A changed source definition needs a new output directory and corresponding source revision/version.

The normalized fields align with the existing `tasks` and `task_goals` tables, but database loading is deliberately deferred to a future PR. Each CooperBench **pair** becomes a single task with two agent-owned goals. Original CooperBench records use `conflict_type: "none"` to mean *no ALIGN-injected conflict classification*, **not** that a coding/merge conflict cannot arise. We retain gold patch and hidden-test references for a **trusted evaluator only**. Do not expose the entire normalized record, gold artifacts, or unrestricted raw S3 prefix to coding agents.

### Offline tests

```powershell
py -m unittest discover -s scripts/tests -p "test_*.py" -v
```

These tests use synthetic local fixtures and never contact AWS, Hugging Face, or PostgreSQL.

## Load normalized tasks into PostgreSQL (PR 3)

`load_tasks.py` checks a PR-2 `tasks.jsonl` against its completion `manifest.json`, validates every record, then optionally inserts into the existing `tasks` and `task_goals` tables. By default, it **only validates files and never connects to a database**.

```powershell
# Generate a small custom snapshot (as in PR 2).
py scripts/normalize_tasks.py custom `
  --scenarios-dir scripts/examples `
  --revision v1 `
  --output-dir ".\normalized\custom-v1"

# Validate the snapshot; no dependencies beyond Python and no DB connection.
py scripts/load_tasks.py --input-dir ".\normalized\custom-v1"
```

To test against your **local development Docker PostgreSQL only** (never a production database), start the existing Compose database and run its Flyway migrations:

```powershell
Push-Location infra/modules/database
docker compose up -d postgres
docker compose run --rm migrate migrate
Pop-Location

# The credentials below are from the repository's LOCAL DEV ONLY compose file.
py -m pip install "psycopg[binary]>=3.2,<4"
$env:ALIGN_DATABASE_URL = "postgresql://align_dev:local_dev_only_change_me@127.0.0.1:5433/align"

py scripts/load_tasks.py --input-dir ".\normalized\custom-v1" --apply
py scripts/load_tasks.py --input-dir ".\normalized\custom-v1" --apply

Remove-Item Env:ALIGN_DATABASE_URL
```

Expected first run: one task inserted; expected repeat: zero new tasks, one unchanged. The example scenario is intentionally **not runnable by the agent harness**; it merely verifies ingestion.

### Safety and replay semantics

- `--apply` and `ALIGN_DATABASE_URL` must **both** be provided before any SQL writes. Database credentials are not accepted as CLI flags or logged.
- The completed manifest's task count, source revision and `tasks.jsonl` SHA-256 are verified **before connecting to PostgreSQL**. Unknown task/goal fields, duplicate identities, and schema mismatches fail closed.
- Inserts are parameterized, and **the entire input batch runs in one transaction**. New tasks and goals roll back together if any record fails. Existing task definitions and goal records are compared field-by-field: identical reloads are skipped; differing content is rejected rather than overwritten.
- No table migrations, deletes, destructive updates, AWS calls, or RDS connectivity are necessary. The actual integration test in CI uses a disposable PostgreSQL 16 service with existing Flyway migrations.
- The database holds evaluator-only test references and scenario metadata. **Do not expose full database task rows directly to coding agents.** The later execution harness must enforce evaluator/agent data separation.

To run offline tests:

```powershell
py -m unittest discover -s scripts/tests -p "test_*.py" -v
```

The PostgreSQL-specific test is skipped locally unless `ALIGN_TEST_DATABASE_URL` is set to a disposable, migrated test database. CI configures this automatically. When recovery work is ongoing, perform the dry-run and unit tests only; defer any `--apply` until the database is ready.


## Run artifacts and S3 upload (PR 4)

`run_artifacts.py` implements the future harness's **version 1 event contract** and a
versioned run bundle. One `events.jsonl` line per event has a canonical `event_uuid`
and `run_key`, strictly contiguous `sequence_number`, UTC `occurred_at`,
`actor_type`, `event_type`, `payload`, optional `trigger_sequence` for
proposal→decision/result associations, and optional S3 `artifact_uri`.
The first event is `run_start` (task key, arbiter condition, configs, seed); a
completed run ends with `run_end` (status and optional cost/goal evidence).
A missing `run_end` is allowed for crash recovery and marked incomplete.

Stage, inspect and stage again with **no AWS access**:

```powershell
py scripts/run_artifacts.py stage --events scripts/examples/example_run.events.jsonl --output-dir normalized/example-run
py scripts/run_artifacts.py verify --run-dir normalized/example-run
```

To upload a reviewed run bundle to an existing S3 lake (requires approved IAM permissions):

```powershell
py scripts/run_artifacts.py upload --run-dir normalized/example-run --bucket YOUR_DEV_LAKE --expected-account YOUR_AWS_ACCOUNT_ID --apply
```

No S3 writes occur without `--apply`. The uploader checks account identity,
refuses conflicting remote objects, reads uploaded bytes back, and writes
`runs/<run_key>/manifest.json` last. To import an existing bundle from S3 use
`download --run-key ... --bucket ... --expected-account ... --output-dir ...`.
The manifest contains SHA-256, byte size, run identity, completion and event count.
S3's immutable objects cannot be appended to; the harness should checkpoint
segments or upload finalized bundles after each run. This first version stores one
**finalized** JSONL object per run, not live streaming checkpoints. It does not
automatically redact secrets: the producer must filter sensitive tool output before
creating events, and large/private data should remain in restricted artifacts.
Never expose raw evaluator materials or privileged instance credentials to agents.

The synthetic fixture is **not an agent run or evaluation ground truth**. Run
`py -m unittest discover -s scripts/tests -p "test_*.py" -v` for offline tests.


## Process run artifacts into PostgreSQL (PR 5)

`process_runs.py` reads a **completed, checksum-verified local run bundle** created
by PR 4 and imports `runs`, `events`, `decision_log` and `action_attempts`
in one PostgreSQL transaction. Sequence numbers map trigger references into
same-run event foreign keys, and a conservative rule checker annotates clearly
unsafe proposals as **provisional**, not gold labels. Absent run_end becomes
a failed run. Raw event UUID is retained in `events.payload.event_uuid`.
No new database migration is required.

```powershell
py scripts/process_runs.py --run-dir normalized/example-run       # no DB connection
# Only with a *disposable/local* migrated PostgreSQL and inserted matching task:
$env:ALIGN_DATABASE_URL = "postgresql://align_dev:local_dev_only_change_me@127.0.0.1:5433/align"
py scripts/process_runs.py --run-dir normalized/example-run --apply
Remove-Item Env:ALIGN_DATABASE_URL
```

A SQL import requires the referenced `task_key` to already exist (load it via
`load_tasks.py`). The source fixture's task is an illustrative custom scenario.
Concurrent importers coordinate with a PostgreSQL transaction advisory lock for
each task and run key, and an identical replay skips rather than duplicating rows;
changed event bytes or changed run metadata fail. The database schema doesn't
yet have a dedicated unique key for `(task_id,trial_key)`; add it by Flyway
if other writers may bypass this importer.

**Limitations:** invalid events are rejected before any DB write, not partially
quarantined. The caller must redact secrets before artifacts are staged; no
automatic key redaction is promised. V1 accepts one finalized bundle rather than
arbitrary streaming segments. An LLM judge and automatic conflict labeling
are evaluator/future work. No AWS or RDS operations occur in CI.
