"""Copy a pinned CooperBench dataset snapshot to ALIGN's existing S3 data lake."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
from pathlib import Path

SOURCE = "CodeConflict/cooperbench-dataset"
EXCLUDED_DIRS = {".cache", ".git", "__pycache__"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inventory(dataset: Path, revision: str) -> tuple[dict, list[tuple[Path, str]]]:
    """Validate the official task layout and inventory its original files."""
    if not (dataset / "README.md").is_file():
        raise ValueError("Dataset is missing README.md")
    subset_path = dataset / "subsets" / "flash.json"
    if not subset_path.is_file():
        raise ValueError("Dataset is missing subsets/flash.json")

    subset = json.loads(subset_path.read_text(encoding="utf-8"))
    for task in subset["tasks"]:
        task_dir = dataset / task["repo"] / f"task{task['task_id']}"
        if not task_dir.is_dir():
            raise ValueError(f"Missing CooperBench task: {task_dir.relative_to(dataset)}")
        for pair in task["pairs"]:
            if len(pair) != 2:
                raise ValueError(f"Invalid feature pair: {pair}")
            for feature_id in pair:
                for filename in ("feature.md", "feature.patch", "tests.patch"):
                    file = task_dir / f"feature{feature_id}" / filename
                    if not file.is_file():
                        raise ValueError(f"Missing CooperBench file: {file.relative_to(dataset)}")

    entries: list[tuple[Path, str]] = []
    metadata = []
    for path in sorted(dataset.rglob("*")):
        rel = path.relative_to(dataset)
        if any(part in EXCLUDED_DIRS for part in rel.parts):
            continue
        if path.is_symlink():
            raise ValueError(f"Symlinks are not supported: {rel.as_posix()}")
        if not path.is_file():
            continue
        key = f"dataset/{rel.as_posix()}"
        entries.append((path, key))
        metadata.append({"path": key, "size_bytes": path.stat().st_size, "sha256": sha256_file(path)})

    manifest = {
        "schema_version": 1,
        "source": "cooperbench",
        "source_repository": SOURCE,
        "source_revision": revision,
        "file_count": len(metadata),
        "files": metadata,
    }
    return manifest, entries


def manifest_bytes(manifest: dict) -> bytes:
    return (json.dumps(manifest, sort_keys=True, indent=2) + "\n").encode("utf-8")


def optional_object(s3, operation: str, *, bucket: str, key: str):
    """Return None on S3 missing-key errors; propagate permissions/network errors."""
    from botocore.exceptions import ClientError

    try:
        return getattr(s3, operation)(Bucket=bucket, Key=key)
    except ClientError as exc:
        code = str(exc.response.get("Error", {}).get("Code", ""))
        if code in {"404", "NoSuchKey", "NotFound"}:
            return None
        raise


def upload_snapshot(s3, bucket: str, revision: str, manifest: dict, entries: list[tuple[Path, str]]) -> tuple[int, int]:
    """Skip matching objects; reject conflicts; write completion manifest last."""
    prefix = f"benchmarks/raw/cooperbench/{revision}/"
    manifest_key = prefix + "manifest.json"
    expected = manifest_bytes(manifest)
    previous = optional_object(s3, "get_object", bucket=bucket, key=manifest_key)
    if previous is not None and previous["Body"].read() != expected:
        raise ValueError(f"Snapshot manifest differs at s3://{bucket}/{manifest_key}")

    inventory_by_key = {item["path"]: item for item in manifest["files"]}
    uploaded = skipped = 0
    for path, relative_key in entries:
        key = prefix + relative_key
        expected_hash = inventory_by_key[relative_key]["sha256"]
        if sha256_file(path) != expected_hash:
            raise ValueError(f"Source file changed during upload: {relative_key}")
        head = optional_object(s3, "head_object", bucket=bucket, key=key)
        if head is not None:
            if (head.get("ContentLength") != path.stat().st_size or
                    head.get("Metadata", {}).get("sha256") != expected_hash):
                raise ValueError(f"Existing object differs; refusing overwrite: s3://{bucket}/{key}")
            skipped += 1
            continue
        if previous is not None:
            raise ValueError(f"Completed snapshot is missing object: s3://{bucket}/{key}")
        s3.upload_file(str(path), bucket, key, ExtraArgs={"Metadata": {"sha256": expected_hash}})
        confirmed = s3.head_object(Bucket=bucket, Key=key)
        if (confirmed.get("ContentLength") != path.stat().st_size or
                confirmed.get("Metadata", {}).get("sha256") != expected_hash):
            raise RuntimeError(f"Uploaded object verification failed: s3://{bucket}/{key}")
        uploaded += 1

    if previous is None:
        s3.put_object(Bucket=bucket, Key=manifest_key, Body=expected, ContentType="application/json", IfNoneMatch="*")
    return uploaded, skipped


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", required=True, help="Exact 40-character Hugging Face dataset commit SHA")
    parser.add_argument("--bucket", help="Existing ALIGN S3 lake bucket (required unless --dry-run)")
    parser.add_argument("--source-dir", type=Path, help="Local dataset/ directory; skip downloading from Hugging Face")
    parser.add_argument("--dry-run", action="store_true", help="Verify and inventory locally; do not use AWS")
    args = parser.parse_args()

    if not re.fullmatch(r"[0-9a-f]{40}", args.revision):
        parser.error("--revision must be a pinned 40-character lowercase commit SHA")
    if not args.dry_run and not args.bucket:
        parser.error("--bucket is required unless --dry-run")
    if args.source_dir and not args.dry_run:
        parser.error("--source-dir is only supported with --dry-run to preserve provenance")

    with tempfile.TemporaryDirectory(prefix="align-cooperbench-") as workdir:
        if args.source_dir:
            dataset = args.source_dir
        else:
            from huggingface_hub import snapshot_download
            dataset = Path(workdir) / "dataset"
            snapshot_download(repo_id=SOURCE, repo_type="dataset", revision=args.revision,
                              local_dir=str(dataset), token=os.environ.get("HF_TOKEN"))

        manifest, files = inventory(dataset, args.revision)
        print(f"Validated {manifest['file_count']} CooperBench files at revision {args.revision}")
        if args.dry_run:
            print("DRY RUN: No S3 calls made")
            return
        import boto3
        uploaded, skipped = upload_snapshot(boto3.client("s3"), args.bucket, args.revision, manifest, files)
        print(f"Complete: {uploaded} uploaded, {skipped} already matching")
        print(f"s3://{args.bucket}/benchmarks/raw/cooperbench/{args.revision}/manifest.json")


if __name__ == "__main__":
    main()
