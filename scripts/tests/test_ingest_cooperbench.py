"""Offline tests for the CooperBench raw snapshot uploader."""

import io
import runpy
import shutil
import sys
from unittest.mock import Mock, patch
import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from botocore.exceptions import ClientError

MODULE = Path(__file__).resolve().parents[1] / "ingest_cooperbench.py"
spec = importlib.util.spec_from_file_location("ingest_cooperbench", MODULE)
ingest = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ingest)
REVISION = "a" * 40


class FakeS3:
    def __init__(self):
        self.objects = {}
        self.uploads = 0

    def get_object(self, *, Bucket, Key):
        obj = self.objects.get((Bucket, Key))
        if obj is None:
            raise ClientError({"Error": {"Code": "NoSuchKey", "Message": "missing"}}, "GetObject")
        from io import BytesIO
        return {"Body": BytesIO(obj[0])}

    def head_object(self, *, Bucket, Key):
        obj = self.objects.get((Bucket, Key))
        if obj is None:
            raise ClientError({"Error": {"Code": "404", "Message": "missing"}}, "HeadObject")
        return {"ContentLength": len(obj[0]), "Metadata": obj[1]}

    def upload_file(self, filename, bucket, key, ExtraArgs):
        self.uploads += 1
        self.objects[(bucket, key)] = (Path(filename).read_bytes(), ExtraArgs["Metadata"])

    def put_object(self, *, Bucket, Key, Body, ContentType, IfNoneMatch):
        assert IfNoneMatch == "*"
        assert (Bucket, Key) not in self.objects
        self.objects[(Bucket, Key)] = (Body, {})


class IngestTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dataset = Path(self.tmp.name)
        self.write("README.md", "CooperBench fixture\n")
        self.write("subsets/flash.json", json.dumps({"tasks": [{"repo": "example_task", "task_id": 1, "pairs": [[1, 2]]}]}))
        for feature_id in (1, 2):
            for filename in ("feature.md", "feature.patch", "tests.patch"):
                self.write(f"example_task/task1/feature{feature_id}/{filename}", f"{feature_id} {filename}\n")
        self.write("example_task/task1/Dockerfile", "FROM scratch\n")
        self.write(".cache/huggingface/ignored", "should not be uploaded")
        self.s3 = FakeS3()

    def write(self, rel, content):
        path = self.dataset / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    def test_inventory_requires_dataset_metadata(self):
        for name in ("README.md", "subsets/flash.json"):
            with self.subTest(name=name):
                path = self.dataset / name
                data = path.read_bytes()
                path.unlink()
                with self.assertRaisesRegex(ValueError, name):
                    ingest.inventory(self.dataset, REVISION)
                path.write_bytes(data)

    def test_subset_rejects_missing_task_and_incomplete_pair(self):
        for task_id, pair, error in [(2, [1, 2], "Missing CooperBench task"), (1, [1], "Invalid feature pair")]:
            with self.subTest(task_id=task_id):
                self.write("subsets/flash.json", json.dumps({"tasks": [
                    {"repo": "example_task", "task_id": task_id, "pairs": [pair]}]}))
                with self.assertRaisesRegex(ValueError, error):
                    ingest.inventory(self.dataset, REVISION)

    def test_symlink_is_not_snapshotted(self):
        (self.dataset / "alias").symlink_to(self.dataset / "README.md")
        with self.assertRaisesRegex(ValueError, "Symlinks"):
            ingest.inventory(self.dataset, REVISION)

    def test_incomplete_readback_never_publishes_manifest(self):
        manifest, entries = ingest.inventory(self.dataset, REVISION)
        with patch.object(self.s3, "upload_file"), self.assertRaisesRegex(ClientError, "404"):
            ingest.upload_snapshot(self.s3, "lake", REVISION, manifest, entries)
        with patch.object(self.s3, "head_object", side_effect=[
                ClientError({"Error": {"Code": "404"}}, "HeadObject"), {"ContentLength": 0}]), \
                self.assertRaisesRegex(RuntimeError, "verification failed"):
            ingest.upload_snapshot(self.s3, "lake", REVISION, manifest, entries)
        self.assertFalse(any(key.endswith("manifest.json") for _, key in self.s3.objects))

    def test_completed_snapshot_with_missing_file_is_rejected(self):
        manifest, entries = ingest.inventory(self.dataset, REVISION)
        ingest.upload_snapshot(self.s3, "lake", REVISION, manifest, entries)
        del self.s3.objects["lake", f"benchmarks/raw/cooperbench/{REVISION}/{entries[0][1]}"]
        with self.assertRaisesRegex(ValueError, "missing object"):
            ingest.upload_snapshot(self.s3, "lake", REVISION, manifest, entries)

    def test_access_denied_is_not_treated_as_missing(self):
        with patch.object(self.s3, "get_object", side_effect=ClientError({"Error": {"Code": "AccessDenied"}}, "GetObject")), \
                self.assertRaisesRegex(ClientError, "AccessDenied"):
            ingest.optional_object(self.s3, "get_object", bucket="lake", key="key")

    def test_cli_validates_revision_bucket_and_local_source(self):
        for args in (["--revision", "main"], ["--revision", REVISION],
                     ["--revision", REVISION, "--bucket", "lake", "--source-dir", str(self.dataset)]):
            with self.subTest(args=args), patch.object(sys, "argv", [str(MODULE), *args]), \
                    patch("sys.stderr", new_callable=io.StringIO), self.assertRaises(SystemExit) as error:
                ingest.main()
            self.assertEqual(error.exception.code, 2)

    def test_cli_dry_run_and_pinned_download(self):
        args = [str(MODULE), "--revision", REVISION]
        with patch.object(sys, "argv", args + ["--dry-run", "--source-dir", str(self.dataset)]), \
                patch("sys.stdout", new_callable=io.StringIO) as output:
            runpy.run_path(str(MODULE), run_name="__main__")
        self.assertIn("DRY RUN", output.getvalue())
        hub = Mock()
        hub.snapshot_download.side_effect = lambda **kwargs: shutil.copytree(self.dataset, kwargs["local_dir"])
        with patch.dict(sys.modules, {"huggingface_hub": hub}), patch("boto3.client", return_value=self.s3), \
                patch.object(sys, "argv", args + ["--bucket", "lake"]), patch("sys.stdout", new_callable=io.StringIO):
            ingest.main()
        self.assertEqual(hub.snapshot_download.call_args.kwargs["revision"], REVISION)
        self.assertIn(("lake", f"benchmarks/raw/cooperbench/{REVISION}/manifest.json"), self.s3.objects)

    def test_inventory_preserves_relative_paths_and_checksums(self):
        manifest, entries = ingest.inventory(self.dataset, REVISION)
        self.assertEqual(manifest["source_revision"], REVISION)
        self.assertEqual(manifest["file_count"], len(entries))
        self.assertTrue(any(f["path"] == "dataset/example_task/task1/feature1/feature.md" for f in manifest["files"]))
        self.assertFalse(any(".cache" in f["path"] for f in manifest["files"]))
        self.assertEqual(manifest["files"][0]["sha256"], hashlib.sha256(entries[0][0].read_bytes()).hexdigest())

    def test_upload_retries_are_idempotent(self):
        manifest, entries = ingest.inventory(self.dataset, REVISION)
        uploaded, skipped = ingest.upload_snapshot(self.s3, "test-lake", REVISION, manifest, entries)
        self.assertEqual((uploaded, skipped), (len(entries), 0))
        uploaded, skipped = ingest.upload_snapshot(self.s3, "test-lake", REVISION, manifest, entries)
        self.assertEqual((uploaded, skipped), (0, len(entries)))
        self.assertEqual(self.s3.uploads, len(entries))
        self.assertIn(("test-lake", f"benchmarks/raw/cooperbench/{REVISION}/manifest.json"), self.s3.objects)

    def test_existing_object_conflict_fails_without_overwrite(self):
        manifest, entries = ingest.inventory(self.dataset, REVISION)
        key = f"benchmarks/raw/cooperbench/{REVISION}/{entries[0][1]}"
        self.s3.objects[("test-lake", key)] = (b"different", {"sha256": "invalid"})
        with self.assertRaisesRegex(ValueError, "refusing overwrite"):
            ingest.upload_snapshot(self.s3, "test-lake", REVISION, manifest, entries)
        self.assertEqual(self.s3.uploads, 0)

    def test_incomplete_upload_can_resume(self):
        manifest, entries = ingest.inventory(self.dataset, REVISION)
        key = f"benchmarks/raw/cooperbench/{REVISION}/{entries[0][1]}"
        data = entries[0][0].read_bytes()
        self.s3.objects[("test-lake", key)] = (data, {"sha256": hashlib.sha256(data).hexdigest()})
        uploaded, skipped = ingest.upload_snapshot(self.s3, "test-lake", REVISION, manifest, entries)
        self.assertEqual((uploaded, skipped), (len(entries) - 1, 1))

    def test_source_file_mutation_during_upload_is_rejected(self):
        manifest, entries = ingest.inventory(self.dataset, REVISION)
        self.write("example_task/task1/feature1/feature.md", "modified after inventory")
        with self.assertRaisesRegex(ValueError, "Source file changed"):
            ingest.upload_snapshot(self.s3, "test-lake", REVISION, manifest, entries)

    def test_completed_revision_cannot_change(self):
        manifest, entries = ingest.inventory(self.dataset, REVISION)
        ingest.upload_snapshot(self.s3, "test-lake", REVISION, manifest, entries)
        self.write("example_task/task1/feature1/feature.md", "changed")
        changed, new_entries = ingest.inventory(self.dataset, REVISION)
        with self.assertRaisesRegex(ValueError, "manifest differs"):
            ingest.upload_snapshot(self.s3, "test-lake", REVISION, changed, new_entries)

    def test_missing_feature_rejected(self):
        (self.dataset / "example_task/task1/feature2/tests.patch").unlink()
        with self.assertRaisesRegex(ValueError, "Missing CooperBench file"):
            ingest.inventory(self.dataset, REVISION)


if __name__ == "__main__":
    unittest.main()
