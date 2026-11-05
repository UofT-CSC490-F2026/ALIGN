"""Durable experiment artifacts; checkpoint deserialization is for our own files only.

Checkpoint writers must be single-writer. A successful ``save`` means its explicit
storage commit completed. JSONL and final bundle manifests also need a caller's
Volume commit when used on Modal; fsync alone does not commit a Modal Volume.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import stat
import tempfile
from typing import Any, Callable


BUNDLE_MANIFEST = "ARTIFACT_MANIFEST.json"
_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,159}\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sync_directory(path: Path) -> None:
    """Persist directory entries on filesystems that support directory fsync."""
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _json_bytes(payload: Any) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def atomic_json(path: str | Path, payload: Any) -> None:
    """Atomically replace one JSON file, rejecting non-finite numeric values."""
    data = _json_bytes(payload)  # Validate before creating anything.
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise ValueError(f"Refusing a symlink JSON target: {path}")
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
        _sync_directory(path.parent)
    finally:
        temporary_path.unlink(missing_ok=True)


def append_jsonl(path: str | Path, row: Any) -> None:
    """Append and fsync a complete metrics row; one writer must own this file."""
    data = (json.dumps(row, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise ValueError(f"Refusing a symlink JSONL target: {path}")
    with path.open("ab") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    _sync_directory(path.parent)


def _root_path(root: str | Path, *, create: bool = False) -> Path:
    path = Path(root)
    if path.is_symlink():
        raise ValueError(f"Artifact root must not be a symlink: {path}")
    if create:
        path.mkdir(parents=True, exist_ok=True)
    if not path.is_dir():
        raise ValueError(f"Artifact root is not a directory: {path}")
    return path.resolve()


def _inside(root: Path, relative: str) -> Path:
    """Reject traversal and every symlink beneath the declared artifact root."""
    name = Path(relative)
    if name.is_absolute() or not name.parts or any(part in (".", "..") for part in name.parts):
        raise ValueError(f"Unsafe artifact path: {relative}")
    target = root / name
    current = root
    for part in name.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError(f"Symlink artifacts are not allowed: {relative}")
    if not target.resolve().is_relative_to(root):
        raise ValueError(f"Artifact path escapes root: {relative}")
    return target


class CheckpointStore:
    """Immutable named checkpoints with hashes and an explicit persistence hook."""

    def __init__(self, root: str | Path, commit: Callable[[], None] = lambda: None):
        self.root = _root_path(root, create=True)
        self.commit = commit
        directory = _inside(self.root, "checkpoints")
        directory.mkdir(exist_ok=True)

    def _paths(self, name: str) -> tuple[Path, Path]:
        if not isinstance(name, str) or not _NAME.fullmatch(name):
            raise ValueError("Checkpoint name must be a simple filename without directory separators")
        return (_inside(self.root, f"checkpoints/{name}.pt"),
                _inside(self.root, f"checkpoints/{name}.json"))

    def save(self, name: str, state: Any, metadata: dict[str, Any]) -> dict[str, Any]:
        """Save, validate and commit; raise rather than return an uncommitted result.

        Existing names, including incomplete leftover payloads, cannot be reused.
        If a commit fails, local completion files are removed. The storage service
        may have persisted data before an acknowledgement failure; callers must
        inspect storage before choosing a new recovery attempt.
        """
        checkpoint, manifest_path = self._paths(name)
        if checkpoint.exists() or manifest_path.exists():
            raise FileExistsError(f"Checkpoint names are immutable: {name}")
        if not isinstance(metadata, dict):
            raise TypeError("Checkpoint metadata must be a JSON object")
        _json_bytes(metadata)
        import torch

        descriptor, temporary = tempfile.mkstemp(prefix=f".{name}.", suffix=".tmp", dir=checkpoint.parent)
        temporary_path = Path(temporary)
        published_checkpoint = False
        published_manifest = False
        try:
            with os.fdopen(descriptor, "wb") as stream:
                torch.save(state, stream)
                stream.flush()
                os.fsync(stream.fileno())
            # Validate serialization before publishing the payload. Only load
            # trusted files created by this experiment, never arbitrary uploads.
            validated = torch.load(temporary_path, map_location="cpu", weights_only=False)
            del validated
            manifest = {
                "schema_version": 1,
                "name": name,
                "file": checkpoint.relative_to(self.root).as_posix(),
                "sha256": sha256_file(temporary_path),
                "size_bytes": temporary_path.stat().st_size,
                "metadata": metadata,
            }
            # Use same-directory rename, rather than hard links which some
            # mounted/distributed filesystems do not support. One writer owns
            # this run directory; the extra check prevents ordinary name reuse.
            if checkpoint.exists() or manifest_path.exists():
                raise FileExistsError(f"Checkpoint names are immutable: {name}")
            os.replace(temporary_path, checkpoint)
            published_checkpoint = True
            _sync_directory(checkpoint.parent)
            atomic_json(manifest_path, manifest)
            published_manifest = True
            self.commit()
            return manifest
        except BaseException:
            if published_manifest:
                manifest_path.unlink(missing_ok=True)
            if published_checkpoint:
                checkpoint.unlink(missing_ok=True)
            _sync_directory(checkpoint.parent)
            raise
        finally:
            temporary_path.unlink(missing_ok=True)

    def load(self, name: str, map_location: Any = "cpu") -> tuple[Any, dict[str, Any]]:
        """Verify the completed manifest and bytes before loading trusted state."""
        checkpoint, manifest_path = self._paths(name)
        if not manifest_path.is_file():
            raise FileNotFoundError(f"No completed checkpoint manifest: {name}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (manifest.get("schema_version") != 1 or manifest.get("name") != name
                or manifest.get("file") != checkpoint.relative_to(self.root).as_posix()
                or not isinstance(manifest.get("sha256"), str)
                or not _SHA256.fullmatch(manifest["sha256"])
                or type(manifest.get("size_bytes")) is not int
                or manifest["size_bytes"] < 0
                or not isinstance(manifest.get("metadata"), dict)):
            raise ValueError(f"Invalid checkpoint manifest: {name}")
        if (not checkpoint.is_file() or checkpoint.stat().st_size != manifest["size_bytes"]
                or sha256_file(checkpoint) != manifest["sha256"]):
            raise ValueError(f"Checkpoint missing or checksum mismatch: {name}")
        import torch
        return torch.load(checkpoint, map_location=map_location, weights_only=False), manifest


def _bundle_files(root: Path) -> dict[str, dict[str, Any]]:
    files = {}
    for directory, directories, names in os.walk(root, followlinks=False):
        parent = Path(directory)
        for name in directories:
            if (parent / name).is_symlink():
                raise ValueError(f"Symlink directory in artifact bundle: {parent / name}")
        for name in names:
            path = parent / name
            relative = path.relative_to(root).as_posix()
            _inside(root, relative)
            if relative == BUNDLE_MANIFEST or name.endswith(".tmp"):
                continue
            if not stat.S_ISREG(path.stat().st_mode):
                raise ValueError(f"Non-file entry in artifact bundle: {relative}")
            files[relative] = {"sha256": sha256_file(path), "size_bytes": path.stat().st_size}
    return dict(sorted(files.items()))


def write_bundle_manifest(root: str | Path) -> dict[str, Any]:
    """Snapshot a finished run directory. Caller commits/exports it afterward."""
    directory = _root_path(root)
    manifest = {"schema_version": 1, "files": _bundle_files(directory)}
    atomic_json(directory / BUNDLE_MANIFEST, manifest)
    return manifest


def verify_bundle(root: str | Path) -> dict[str, Any]:
    """Require the complete exported file set and every recorded byte hash."""
    directory = _root_path(root)
    manifest_path = _inside(directory, BUNDLE_MANIFEST)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1 or not isinstance(manifest.get("files"), dict):
        raise ValueError("Invalid artifact bundle manifest")
    for name in manifest["files"]:
        _inside(directory, name)
        if name == BUNDLE_MANIFEST or name.endswith(".tmp"):
            raise ValueError(f"Invalid recorded bundle entry: {name}")
    actual = _bundle_files(directory)
    expected = manifest["files"]
    missing = sorted(set(expected) - set(actual))
    unexpected = sorted(set(actual) - set(expected))
    changed = sorted(name for name in set(actual) & set(expected) if actual[name] != expected[name])
    if missing or unexpected or changed:
        raise ValueError(f"Bundle audit failed: missing={missing}, unexpected={unexpected}, changed={changed}")
    return manifest
