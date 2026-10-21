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
