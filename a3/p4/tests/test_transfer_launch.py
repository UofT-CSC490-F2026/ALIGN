"""Offline evidence-transfer and preflight tests; never import a cloud app."""
import builtins
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from p4 import bootstrap, run as launcher, transfer
from p4.artifacts import atomic_json, verify_bundle, write_bundle_manifest
from p4.config import UPSTREAM_COMMIT


class TransferTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.p4 = self.root / "p4"
        self.p4.mkdir()
        self.location_patch = patch.object(transfer, "P4", self.p4)
        self.location_patch.start()
        self.addCleanup(self.location_patch.stop)

    def bundle(self, name, value):
        directory = self.root / name
        directory.mkdir()
        atomic_json(directory / "config.json", {"marker": value})
        atomic_json(directory / "report/summary.json", {"status": "complete", "marker": value})
        write_bundle_manifest(directory)
        return directory

    def download(self, source, *, corrupt=False):
        """Simulate the installed SDK's nested-directory download contract."""
        def get(remote, destination):
            self.assertEqual(remote, "/runs/run-a")
            destination = Path(destination)
            # Nonexistent destinations are interpreted differently by Modal's
            # actual CLI, so asserting this is the important regression guard.
            self.assertTrue(destination.is_dir())
            self.assertEqual(destination.parent, self.p4 / "outputs")
            self.assertTrue(destination.name.startswith(".download-"))
            shutil.copytree(source, destination / "run-a")
            if corrupt:
                (destination / "run-a/config.json").write_text("damaged download")
        return get

    def test_exports_nested_verified_child_from_preexisting_staging(self):
        source = self.bundle("remote", "new")
        with patch.object(transfer, "_get", side_effect=self.download(source)) as get:
            result = transfer.export_run("run-a", include_checkpoints=True)
        self.assertEqual(get.call_count, 1)
        destination = self.p4 / "outputs/run-a"
        self.assertEqual(Path(result["directory"]), destination)
        self.assertEqual(result["files_verified"], 2)
        self.assertEqual(verify_bundle(destination), verify_bundle(source))
        self.assertEqual(json.loads((destination / "config.json").read_text())["marker"], "new")
        self.assertEqual(list((self.p4 / "outputs").glob(".download-*")), [])

    def test_corrupt_download_preserves_previous_verified_export(self):
        old = self.bundle("old", "old")
        source = self.bundle("remote", "new")
        destination = self.p4 / "outputs/run-a"
        destination.parent.mkdir()
        shutil.copytree(old, destination)
        expected = verify_bundle(destination)
        with patch.object(transfer, "_get", side_effect=self.download(source, corrupt=True)):
            with self.assertRaisesRegex(ValueError, "Bundle audit failed"):
                transfer.export_run("run-a", include_checkpoints=True)
        self.assertEqual(verify_bundle(destination), expected)
        self.assertEqual(json.loads((destination / "config.json").read_text())["marker"], "old")
        self.assertEqual(list(destination.parent.glob("run-a.previous-*")), [])

    def test_failed_cli_download_preserves_previous_verified_export(self):
        old = self.bundle("old", "old")
        destination = self.p4 / "outputs/run-a"
        destination.parent.mkdir()
        shutil.copytree(old, destination)
        expected = verify_bundle(destination)
        with patch.object(transfer, "_get", side_effect=subprocess.CalledProcessError(1, ["modal", "volume", "get"])):
            with self.assertRaises(subprocess.CalledProcessError):
                transfer.export_run("run-a", include_checkpoints=True)
        self.assertEqual(verify_bundle(destination), expected)
        self.assertEqual(list(destination.parent.glob("run-a.previous-*")), [])

    def test_successful_replacement_keeps_verified_previous_export(self):
        old = self.bundle("old", "old")
        source = self.bundle("remote", "new")
        destination = self.p4 / "outputs/run-a"
        destination.parent.mkdir()
        shutil.copytree(old, destination)
        old_manifest = verify_bundle(destination)
        with patch.object(transfer, "_get", side_effect=self.download(source)):
            transfer.export_run("run-a", include_checkpoints=True)
        self.assertEqual(verify_bundle(destination), verify_bundle(source))
        previous = list(destination.parent.glob("run-a.previous-*"))
        self.assertEqual(len(previous), 1)
        self.assertEqual(verify_bundle(previous[0]), old_manifest)


class LaunchGateTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.p4 = Path(self.temporary.name)
        self.imports = []
        actual_import = builtins.__import__

        def reject_cloud(name, *args, **kwargs):
            if name == "modal" or name.startswith("modal.") or name == "p4.modal_app":
                self.imports.append(name)
                raise AssertionError("A cloud import happened before the preflight gate")
            return actual_import(name, *args, **kwargs)

        self.reject_cloud = reject_cloud

    def test_missing_rehearsal_blocks_training_before_cloud_import_or_reservation(self):
        with (patch.object(launcher, "P4", self.p4),
              patch.object(launcher, "activate_local", return_value={"unchanged": True}),
              patch.object(launcher, "verify_prepared_export", return_value={"token_count": 40_000_000}),
              patch.object(launcher, "reserve") as reserve,
              patch.object(sys, "argv", ["run.py", "short", "--experiment-id", "fresh-run"]),
              patch("builtins.__import__", side_effect=self.reject_cloud)):
            with self.assertRaisesRegex(RuntimeError, "passed.*rehearsal"):
                launcher.main()
        self.assertEqual(self.imports, [])
        reserve.assert_not_called()
        self.assertFalse((self.p4 / "local/fresh-run-config.json").exists())

    def test_invalid_namespace_blocks_before_cloud_import(self):
        with (patch.object(launcher, "P4", self.p4),
              patch.object(launcher, "activate_local", return_value={"unchanged": True}),
              patch.object(launcher, "verify_prepared_export") as prepared,
              patch.object(launcher, "reserve") as reserve,
              patch.object(sys, "argv", ["run.py", "short", "--experiment-id", "../escape"]),
              patch("builtins.__import__", side_effect=self.reject_cloud)):
            with self.assertRaisesRegex(ValueError, "simple unique"):
                launcher.main()
        self.assertEqual(self.imports, [])
        prepared.assert_not_called()
        reserve.assert_not_called()

    def test_stale_rehearsal_source_identity_blocks_before_cloud_import(self):
        atomic_json(self.p4 / "local/rehearsal-passed.json",
                    {"status": "passed", "source_identity": {"training.py": "old"}})
        with (patch.object(launcher, "P4", self.p4),
              patch.object(launcher, "activate_local", return_value={"unchanged": True}),
              patch.object(launcher, "verify_prepared_export", return_value={"token_count": 40_000_000}),
              patch.object(launcher, "source_identity", return_value={"training.py": "changed"}),
              patch.object(launcher, "reserve") as reserve,
              patch.object(sys, "argv", ["run.py", "long", "--experiment-id", "fresh-run"]),
              patch("builtins.__import__", side_effect=self.reject_cloud)):
            with self.assertRaisesRegex(RuntimeError, "implementation changed"):
                launcher.main()
        self.assertEqual(self.imports, [])
        reserve.assert_not_called()

    def test_dirty_upstream_is_rejected_without_mutating_real_checkout(self):
        with (patch.object(bootstrap.subprocess, "check_output",
                           side_effect=[UPSTREAM_COMMIT + "\n", " M nanochat/gpt.py\n"]) as git,
              patch.object(bootstrap, "verify_jinhao_sources") as boundary):
            with self.assertRaisesRegex(RuntimeError, "modified tracked files"):
                bootstrap.activate_local()
        self.assertEqual(git.call_count, 2)
        self.assertIn("status", git.call_args_list[1].args[0])
        boundary.assert_not_called()


if __name__ == "__main__":
    unittest.main()
