"""Light exports keep verifiable evidence without downloading model/data blobs."""
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from p4 import transfer
from p4.artifacts import atomic_json, sha256_file, write_bundle_manifest


class LightExportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.p4 = self.root / "p4"
        self.p4.mkdir()
        self.remote = self.root / "remote"
        self.remote.mkdir()
        self.requests = []
        self.location = patch.object(transfer, "P4", self.p4)
        self.location.start()
        self.addCleanup(self.location.stop)

    def get(self, prefix):
        def download(remote, destination):
            self.assertTrue(remote.startswith(prefix))
            self.requests.append(remote)
            source = self.remote / remote[len(prefix):]
            shutil.copy2(source, destination)
        return download

    def run_bundle(self):
        atomic_json(self.remote / "config.json", {"seed": 42})
        atomic_json(self.remote / "report/summary.json", {"status": "complete"})
        atomic_json(self.remote / "checkpoints/checkpoint1.json", {"sha256": "a" * 64})
        (self.remote / "checkpoints/checkpoint1.pt").write_bytes(b"large model placeholder")
        write_bundle_manifest(self.remote)

    def test_default_run_skips_weights_and_audits_selected_hashes(self):
        self.run_bundle()
        with patch.object(transfer, "_get", side_effect=self.get("/runs/run-a/")):
            result = transfer.export_run("run-a")
        directory = Path(result["directory"])
        manifest = transfer.verify_export(directory)
        self.assertEqual(result["scope"], "evidence_only")
        self.assertEqual(result["skipped_files"], 1)
        self.assertIn("checkpoints/checkpoint1.pt", manifest["skipped_files"])
        self.assertFalse((directory / "checkpoints/checkpoint1.pt").exists())
        self.assertFalse(any(name.endswith(".pt") for name in self.requests))
        (directory / ".DS_Store").write_bytes(b"Finder metadata")
        transfer.verify_export(directory)
        (directory / "config.json").write_text("tampered")
        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
            transfer.verify_export(directory)

    def test_selection_tampering_cannot_hide_a_missing_metric(self):
        self.run_bundle()
        with patch.object(transfer, "_get", side_effect=self.get("/runs/run-a/")):
            result = transfer.export_run("run-a")
        directory = Path(result["directory"])
        marker = directory / transfer.LOCAL_MANIFEST
        value = json.loads(marker.read_text())
        del value["files"]["config.json"]
        marker.write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError, "selection differs"):
            transfer.verify_export(directory)

    def prepared_bundle(self):
        (self.remote / "tokenizer").mkdir()
        for name in ("evaluation.jsonl", "token_bytes.npy", "tokenizer/tokenizer.pkl",
                     "train.bin", "training-documents.jsonl"):
            (self.remote / name).write_bytes(name.encode())
        files = {str(p.relative_to(self.remote)): {"sha256": sha256_file(p), "size_bytes": p.stat().st_size}
                 for p in self.remote.rglob("*") if p.is_file()}
        atomic_json(self.remote / "manifest.json", {"schema_version": 1, "token_count": 40000000, "files": files})

    def test_prepared_metadata_reexport_omits_stream_and_limits_old_light_copies(self):
        self.prepared_bundle()
        with patch.object(transfer, "_get", side_effect=self.get("/prepared/prep-a/")):
            for _ in range(3):
                result = transfer.export_prepared("prep-a")
        directory = Path(result["directory"])
        self.assertEqual(transfer.verify_prepared_export(directory)["token_count"], 40000000)
        self.assertFalse((directory / "train.bin").exists())
        self.assertFalse((directory / "training-documents.jsonl").exists())
        self.assertFalse(any(name.endswith("/train.bin") for name in self.requests))
        self.assertEqual(len(list(directory.parent.glob("prep-a.previous-*"))), 1)

    def test_trimmed_legacy_preparation_can_be_migrated_without_redownloading_stream(self):
        self.prepared_bundle()
        directory = self.p4 / "local/prepared/prep-a"
        directory.mkdir(parents=True)
        for name in ("manifest.json", "evaluation.jsonl", "token_bytes.npy", "tokenizer/tokenizer.pkl"):
            target = directory / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(self.remote / name, target)
        transfer.verify_prepared_export(directory)
        with patch.object(transfer, "_get", side_effect=self.get("/prepared/prep-a/")):
            transfer.export_prepared("prep-a")
        self.assertTrue((directory / transfer.LOCAL_MANIFEST).is_file())
        self.assertFalse((directory / "train.bin").exists())


if __name__ == "__main__":
    unittest.main()
