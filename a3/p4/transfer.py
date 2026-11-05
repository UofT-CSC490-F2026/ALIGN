"""Verified, small evidence exports; weights and training data stay on Modal.

The default local export is deliberately a partial copy of the remote bundle.
Its full remote index and a separate selection manifest make that scope explicit.
An explicit full export remains available when local model weights are needed.
"""
from __future__ import annotations

import json
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import subprocess
import tempfile
import time

from .artifacts import BUNDLE_MANIFEST, atomic_json, sha256_file, verify_bundle
from .config import VOLUME_NAME, PROFILE
from .data import verify_prepared

P4 = Path(__file__).resolve().parent
LOCAL_MANIFEST = "LOCAL_EXPORT_MANIFEST.json"
MAX_EVIDENCE_BYTES = 10 * 1024 * 1024
_EVIDENCE_SUFFIXES = {".json", ".jsonl", ".csv", ".md", ".txt", ".log", ".png", ".svg", ".pdf", ".yaml", ".yml", ".toml"}
_BULKY_SUFFIXES = {".pt", ".bin", ".parquet", ".wandb", ".safetensors", ".ckpt", ".zip", ".gz"}
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


def _validate_id(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,99}", value):
        raise ValueError("Unsafe experiment/preparation ID")


def _relative(name):
    if not isinstance(name, str) or not name or "\\" in name:
        raise ValueError("Unsafe artifact path")
    path = PurePosixPath(name)
    if path.is_absolute() or any(part in ("", ".", "..") for part in name.split("/")):
        raise ValueError(f"Unsafe artifact path: {name}")
    return path


def _get(remote, destination):
    subprocess.run([str(P4 / ".venv/bin/modal"), "volume", "get", "--profile", PROFILE,
                    VOLUME_NAME, remote, str(destination)], check=True)


def _records(manifest):
    if manifest.get("schema_version") != 1 or not isinstance(manifest.get("files"), dict):
        raise ValueError("Invalid remote artifact manifest")
    for name, record in manifest["files"].items():
        _relative(name)
        if name in (LOCAL_MANIFEST, BUNDLE_MANIFEST, "manifest.json"):
            raise ValueError(f"Remote index contains a reserved manifest path: {name}")
        if (not isinstance(record, dict) or not isinstance(record.get("sha256"), str)
                or not _SHA256.fullmatch(record["sha256"])
                or type(record.get("size_bytes")) is not int or record["size_bytes"] < 0):
            raise ValueError(f"Invalid remote artifact record: {name}")
    return manifest["files"]


def _selection(records, scope):
    selected, skipped = {}, {}
    for name, record in sorted(records.items()):
        path = PurePosixPath(name)
        reason = None
        if scope == "prepared_full":
            pass
        elif record["size_bytes"] > MAX_EVIDENCE_BYTES:
            reason = "larger_than_10_MiB"
        elif path.suffix.lower() in _BULKY_SUFFIXES:
            reason = "remote_model_or_training_blob"
        elif scope == "prepared_metadata":
            if not (name in ("evaluation.jsonl", "token_bytes.npy") or path.parts[0] == "tokenizer"):
                reason = "training_stream_or_document_index_stays_remote"
        elif scope == "evidence_only":
            if path.suffix.lower() not in _EVIDENCE_SUFFIXES:
                reason = "not_small_report_or_log_evidence"
        else:
            raise ValueError(f"Unknown export scope: {scope}")
        if reason:
            skipped[name] = {**record, "reason": reason}
        else:
            selected[name] = record
    return selected, skipped


def _verify_file(path, record):
    if (path.is_symlink() or not path.is_file() or path.stat().st_size != record["size_bytes"]
            or sha256_file(path) != record["sha256"]):
        raise ValueError(f"Exported artifact missing or checksum mismatch: {path.name}")


def _partial_manifest(directory, source_name, source_manifest, scope):
    selected, skipped = _selection(_records(source_manifest), scope)
    manifest = {
        "schema_version": 1,
        "scope": scope,
        "remote_manifest_file": source_name,
        "remote_manifest_sha256": sha256_file(directory / source_name),
        "selection_policy": {"version": 1, "maximum_evidence_file_bytes": MAX_EVIDENCE_BYTES},
        "selected_files": sorted(selected),
        "files": selected,
        "skipped_files": skipped,
        "storage_note": "Skipped artifacts remain in the persistent Modal Volume; this is not a full local model backup.",
    }
    atomic_json(directory / LOCAL_MANIFEST, manifest)
    return manifest


def _verify_partial(root, expected_source, expected_scopes):
    root = Path(root)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("Export root must be a real directory")
    local_path = root / LOCAL_MANIFEST
    if local_path.is_symlink():
        raise ValueError("Export manifest must not be a symlink")
    local = json.loads(local_path.read_text())
    if (local.get("schema_version") != 1 or local.get("scope") not in expected_scopes
            or local.get("remote_manifest_file") != expected_source
            or local.get("selection_policy") != {"version": 1, "maximum_evidence_file_bytes": MAX_EVIDENCE_BYTES}):
        raise ValueError("Invalid local export manifest")
    source_path = root / expected_source
    if source_path.is_symlink() or not source_path.is_file() or sha256_file(source_path) != local.get("remote_manifest_sha256"):
        raise ValueError("Remote index missing or checksum mismatch")
    source = json.loads(source_path.read_text())
    selected, skipped = _selection(_records(source), local["scope"])
    if (local.get("files") != selected or local.get("selected_files") != sorted(selected)
            or local.get("skipped_files") != skipped):
        raise ValueError("Local export selection differs from the preserved remote index")
    allowed = set(selected) | {LOCAL_MANIFEST, expected_source}
    actual = set()
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError("Symlinks are not allowed in local exports")
        if path.is_file():
            if not stat.S_ISREG(path.stat().st_mode):
                raise ValueError("Non-regular export artifact")
            if path.name == ".DS_Store":
                continue
            actual.add(path.relative_to(root).as_posix())
    if actual != allowed:
        raise ValueError(f"Local export audit failed: missing={sorted(allowed - actual)}, unexpected={sorted(actual - allowed)}")
    for name, record in selected.items():
        _verify_file(root / name, record)
    return local, source


def verify_export(root):
    """Audit a light run export, or an earlier complete downloaded bundle."""
    root = Path(root)
    if (root / LOCAL_MANIFEST).exists():
        return _verify_partial(root, BUNDLE_MANIFEST, {"evidence_only"})[0]
    return verify_bundle(root)


def verify_prepared_export(root):
    """Audit local preparation metadata; return the full remote preparation index."""
    root = Path(root)
    if (root / LOCAL_MANIFEST).exists():
        return _verify_partial(root, "manifest.json", {"prepared_metadata", "prepared_full"})[1]
    # Migrate an earlier export whose redundant large local files were removed
    # only after their persistent remote copies were independently verified.
    source = json.loads((root / "manifest.json").read_text())
    records = _records(source)
    selected, skipped = _selection(records, "prepared_metadata")
    for name, record in selected.items():
        _verify_file(root / name, record)
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError("Symlinks are not allowed in local preparation")
        if not path.is_file() or path.name == ".DS_Store":
            continue
        name = path.relative_to(root).as_posix()
        if name == "manifest.json":
            continue
        if name not in records:
            raise ValueError(f"Unexpected preparation file: {name}")
        _verify_file(path, records[name])
    return source


def _publish(incoming, destination, verifier):
    """Keep old evidence until the new copy verifies; retain one old light copy."""
    verifier(incoming)
    previous = None
    if destination.exists():
        verifier(destination)
        previous = destination.with_name(f"{destination.name}.previous-{time.time_ns()}")
        destination.rename(previous)
    try:
        incoming.rename(destination)
    except BaseException:
        if previous is not None and not destination.exists():
            previous.rename(destination)
        raise
    # Only prune verified light snapshots created by this exporter. Preserve
    # legacy/full snapshots containing weights for explicit user review.
    light = []
    for candidate in sorted(destination.parent.glob(f"{destination.name}.previous-*"), reverse=True):
        marker = candidate / LOCAL_MANIFEST
        if not candidate.is_dir() or candidate.is_symlink() or not marker.is_file() or marker.is_symlink():
            continue
        recorded = json.loads(marker.read_text())
        if recorded.get("scope") not in ("evidence_only", "prepared_metadata"):
            continue
        verifier(candidate)
        light.append(candidate)
    for candidate in light[1:]:
        shutil.rmtree(candidate)


def export_run(experiment_id, include_checkpoints=False):
    """Default: download small, audited evidence and retain a complete remote index."""
    _validate_id(experiment_id)
    outputs = P4 / "outputs"
    outputs.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".download-", dir=outputs))
    try:
        if include_checkpoints:
            # The recursive CLI retains the remote folder name beneath an
            # already-existing directory, so audit that nested child.
            _get(f"/runs/{experiment_id}", staging)
            incoming = staging / experiment_id
            manifest = verify_bundle(incoming)
            scope, skipped = "full_bundle", {}
        else:
            incoming = staging / experiment_id
            incoming.mkdir()
            _get(f"/runs/{experiment_id}/{BUNDLE_MANIFEST}", incoming / BUNDLE_MANIFEST)
            source = json.loads((incoming / BUNDLE_MANIFEST).read_text())
            selected, skipped = _selection(_records(source), "evidence_only")
            for name, record in selected.items():
                target = incoming / name
                target.parent.mkdir(parents=True, exist_ok=True)
                _get(f"/runs/{experiment_id}/{name}", target)
                _verify_file(target, record)
            manifest = _partial_manifest(incoming, BUNDLE_MANIFEST, source, "evidence_only")
            verify_export(incoming)
            scope = "evidence_only"
        destination = outputs / experiment_id
        _publish(incoming, destination, verify_export)
        return {"directory": str(destination), "files_verified": len(manifest["files"]),
                "scope": scope, "skipped_files": len(skipped),
                "skipped_bytes": sum(row["size_bytes"] for row in skipped.values())}
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def export_prepared(preparation_id, include_training_data=False):
    """Download tokenizer/evaluation metadata by default, not the training stream."""
    _validate_id(preparation_id)
    parent = P4 / "local/prepared"
    parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".download-", dir=parent))
    try:
        incoming = staging / preparation_id
        incoming.mkdir()
        _get(f"/prepared/{preparation_id}/manifest.json", incoming / "manifest.json")
        source = json.loads((incoming / "manifest.json").read_text())
        destination = parent / preparation_id
        if destination.exists():
            old_source = verify_prepared_export(destination)
            if old_source != source:
                raise ValueError("Existing local preparation has a different identity")
        scope = "prepared_full" if include_training_data else "prepared_metadata"
        selected, skipped = _selection(_records(source), scope)
        for name, record in selected.items():
            target = incoming / name
            target.parent.mkdir(parents=True, exist_ok=True)
            old_file = destination / name
            if old_file.is_file() and not old_file.is_symlink():
                _verify_file(old_file, record)
                shutil.copy2(old_file, target)
            else:
                _get(f"/prepared/{preparation_id}/{name}", target)
            _verify_file(target, record)
        local = _partial_manifest(incoming, "manifest.json", source, scope)
        verify_prepared_export(incoming)
        _publish(incoming, destination, verify_prepared_export)
        return {"directory": str(destination), "files_verified": len(local["files"]),
                "scope": scope, "skipped_files": len(skipped),
                "skipped_bytes": sum(row["size_bytes"] for row in skipped.values())}
    finally:
        shutil.rmtree(staging, ignore_errors=True)
