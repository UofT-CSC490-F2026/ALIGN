"""Persistence contract tests; no cloud or network resources are used."""
import importlib.util
import json
import math
from pathlib import Path
import pickle
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

from p4.artifacts import (
    BUNDLE_MANIFEST, CheckpointStore, append_jsonl, atomic_json,
    sha256_file, verify_bundle, write_bundle_manifest,
)


def fake_torch():
    def save(value, stream):
        pickle.dump(value, stream)

    def load(path, *, map_location, weights_only):
        assert weights_only is False
        with Path(path).open("rb") as stream:
            return pickle.load(stream)

    return types.SimpleNamespace(save=save, load=load)


class ArtifactTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def test_jsonl_preserves_rows_and_rejects_nonfinite_metrics(self):
        path = self.root / "metrics.jsonl"
        append_jsonl(path, {"step": 1, "loss": 2.5})
        append_jsonl(path, {"step": 2, "loss": 2.0})
        with self.assertRaises(ValueError):
            append_jsonl(path, {"loss": math.nan})
        self.assertEqual([json.loads(line)["step"] for line in path.read_text().splitlines()], [1, 2])

    def test_bad_json_does_not_replace_existing_evidence(self):
        path = self.root / "config.json"
        atomic_json(path, {"seed": 42})
        with self.assertRaises(ValueError):
            atomic_json(path, {"seed": float("inf")})
        self.assertEqual(json.loads(path.read_text()), {"seed": 42})
        self.assertFalse(list(self.root.glob("*.tmp")))

    def test_immutable_checkpoint_roundtrip_and_commit_order(self):
        observed = []

        def commit():
            observed.append((self.root / "checkpoints/stage1.json").is_file())

        with patch.dict(sys.modules, {"torch": fake_torch()}):
            store = CheckpointStore(self.root, commit)
            manifest = store.save("stage1", {"step": 5, "optimizer": {"momentum": 0.9}}, {"context": 512})
            state, restored = store.load("stage1")
            self.assertEqual(state["optimizer"], {"momentum": 0.9})
            self.assertEqual(restored, manifest)
            self.assertEqual(observed, [True])
            self.assertEqual(manifest["sha256"], sha256_file(self.root / manifest["file"]))
            with self.assertRaises(FileExistsError):
                store.save("stage1", {"step": 6}, {})

    def test_failed_storage_commit_cannot_be_loaded_as_complete(self):
        def fail():
            raise OSError("storage unavailable")

        with patch.dict(sys.modules, {"torch": fake_torch()}):
            store = CheckpointStore(self.root, fail)
            with self.assertRaisesRegex(OSError, "storage unavailable"):
                store.save("stage1", {"step": 5}, {})
            self.assertFalse(list((self.root / "checkpoints").iterdir()))
            with self.assertRaises(FileNotFoundError):
                store.load("stage1")

    def test_corrupt_checkpoint_is_rejected_before_deserialization(self):
        with patch.dict(sys.modules, {"torch": fake_torch()}):
            store = CheckpointStore(self.root)
            manifest = store.save("stage1", {"step": 5}, {})
            (self.root / manifest["file"]).write_bytes(b"broken")
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                store.load("stage1")

    def test_manifest_cannot_redirect_checkpoint_outside_root(self):
        with patch.dict(sys.modules, {"torch": fake_torch()}):
            store = CheckpointStore(self.root)
            manifest = store.save("stage1", {"step": 5}, {})
            manifest["file"] = "../outside.pt"
            atomic_json(self.root / "checkpoints/stage1.json", manifest)
            with self.assertRaisesRegex(ValueError, "Invalid checkpoint manifest"):
                store.load("stage1")

    def test_checkpoint_names_cannot_traverse_or_follow_symlinks(self):
        store = CheckpointStore(self.root)
        for name in ["../escape", "/tmp/escape", "a/b", ".", "..", ""]:
            with self.subTest(name=name), self.assertRaises(ValueError):
                store.save(name, {}, {})
        (self.root / "checkpoints/linked.pt").symlink_to(self.root / "elsewhere")
        with self.assertRaises(ValueError):
            store.load("linked")

    def test_bundle_detects_changed_missing_and_unexpected_files(self):
        atomic_json(self.root / "config.json", {"seed": 42})
        append_jsonl(self.root / "metrics.jsonl", {"step": 1})
        (self.root / ".partial.tmp").write_text("unfinished")
        manifest = write_bundle_manifest(self.root)
        self.assertEqual(set(manifest["files"]), {"config.json", "metrics.jsonl"})
        self.assertEqual(verify_bundle(self.root), manifest)
        (self.root / "config.json").write_text("changed")
        (self.root / "metrics.jsonl").unlink()
        (self.root / "unrecorded.txt").write_text("extra")
        with self.assertRaisesRegex(ValueError, "missing=.*metrics.jsonl.*unexpected=.*unrecorded.txt.*changed=.*config.json"):
            verify_bundle(self.root)

    def test_bundle_rejects_symlink_and_traversal_manifest_entries(self):
        (self.root / "linked").symlink_to(self.root / "missing")
        with self.assertRaisesRegex(ValueError, "Symlink"):
            write_bundle_manifest(self.root)
        (self.root / "linked").unlink()
        write_bundle_manifest(self.root)
        atomic_json(self.root / BUNDLE_MANIFEST, {"schema_version": 1, "files": {"../escape": {}}})
        with self.assertRaises(ValueError):
            verify_bundle(self.root)

    @unittest.skipUnless(importlib.util.find_spec("torch") is not None, "torch not installed")
    def test_real_torch_checkpoint_roundtrip(self):
        import torch
        store = CheckpointStore(self.root)
        value = {"model": {"weight": torch.arange(4)}, "rng": torch.get_rng_state(), "step": 2}
        store.save("real", value, {"context": 512})
        restored, _ = store.load("real")
        torch.testing.assert_close(value["model"]["weight"], restored["model"]["weight"])
        torch.testing.assert_close(value["rng"], restored["rng"])


if __name__ == "__main__":
    unittest.main()
