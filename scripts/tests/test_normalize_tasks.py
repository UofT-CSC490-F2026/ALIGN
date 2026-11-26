"""No-AWS tests for the ALIGN dataset-neutral task normalizer."""

import runpy
import sys
from unittest.mock import patch
import importlib.util
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

MODULE = Path(__file__).resolve().parents[1] / "normalize_tasks.py"
spec = importlib.util.spec_from_file_location("normalize_tasks", MODULE)
normalize = importlib.util.module_from_spec(spec)
spec.loader.exec_module(normalize)

REVISION = "b612b1a35af722751454813d9e5a7888f065fc9e"
BASE = "d791537317947448a3960662d00acf160eff2602"


class NormalizeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.dataset = self.root / "dataset"
        task = "pallets_click_task/task2068"
        self.write(self.dataset / "subsets/flash.json", json.dumps({"name": "flash", "tasks": [
            {"repo": "pallets_click_task", "task_id": 2068, "pairs": [[1, 2]]},
        ]}))
        self.write(self.dataset / task / "setup.sh", f'BASE_COMMIT="{BASE}"\n')
        for filename in ("Dockerfile", "runner.sh", "run_tests.sh", "combined.patch"):
            self.write(self.dataset / task / filename, f"{filename}\n")
        for fid in (1, 2, 3):
            for name in ("feature.md", "feature.patch", "tests.patch"):
                self.write(self.dataset / task / f"feature{fid}" / name, f"Feature {fid}: {name}\n")
        self.custom = self.root / "custom"
        self.custom.mkdir()
        self.write(self.custom / "example.json", json.dumps(self.custom_case()))

    def write(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value, encoding="utf-8")

    @staticmethod
    def custom_case():
        return {
            "scenario_id": "shared-file-conflict", "version": 1,
            "repo_url": "https://github.com/UofT-CSC490-F2026/ALIGN", "base_commit": BASE,
            "shared_goal": "Coordinate changes to a shared configuration file",
            "conflict_type": "execution_conflict", "expected_outcome": "escalate",
            "constraints": {"protected_paths": ["tests/"]},
            "setup_spec": {"execution_mode": "fixture"},
            "metadata": {"synthetic_example": True},
            "goals": [{"goal_key": "safe_edit", "owner_type": "agent", "owner_agent_id": "agent1",
                       "description": "Propose a safe edit without touching tests",
                       "success_criteria": {"type": "manual_review"}, "is_critical": True}],
        }

    def test_cli_normalizes_custom_directory(self):
        out = self.root / "out"
        args = [str(MODULE), "custom", "--scenarios-dir", str(self.custom), "--revision", "v1", "--output-dir", str(out)]
        with patch.object(sys, "argv", args):
            runpy.run_path(str(MODULE), run_name="__main__")
        self.assertEqual(json.loads((out / "tasks.jsonl").read_text()), normalize.normalized_custom(self.custom, "v1")[0])

    def test_cooperbench_cli_defaults_to_flash_without_manifest(self):
        out = self.root / "flash-output"
        args = [str(MODULE), "cooperbench", "--dataset-dir", str(self.dataset),
                "--revision", REVISION, "--output-dir", str(out)]
        with patch.object(sys, "argv", args):
            normalize.main()
        expected = normalize.normalized_cooperbench(self.dataset, REVISION, "flash")
        records = [json.loads(line) for line in (out / "tasks.jsonl").read_text().splitlines()]
        self.assertEqual(records, expected)

    def test_failed_flush_leaves_no_partial_output(self):
        out = self.root / "out"
        records = normalize.normalized_custom(self.custom, "v1")
        with patch.object(normalize.os, "fsync", side_effect=OSError("disk full")), self.assertRaisesRegex(OSError, "disk full"):
            normalize.write_output(records, out, "align-custom", "v1", "all")
        self.assertEqual(list(out.iterdir()), [])

    def test_cooperbench_pair_has_expected_database_fields(self):
        records = normalize.normalized_cooperbench(self.dataset, REVISION, "flash")
        self.assertEqual(len(records), 1)
        task = records[0]
        self.assertEqual(task["repo_url"], "https://github.com/pallets/click")
        self.assertEqual(task["base_commit"], BASE)
        self.assertEqual(task["dataset_source"], "cooperbench")
        self.assertEqual(task["conflict_type"], "none")
        self.assertEqual(task["metadata"]["feature_ids"], [1, 2])
        self.assertEqual([g["owner_agent_id"] for g in task["goals"]], ["agent1", "agent2"])
        self.assertEqual([g["goal_key"] for g in task["goals"]], ["feature1", "feature2"])
        self.assertTrue(task["goals"][0]["description"].startswith("Feature 1"))
        self.assertIn("feature1/tests.patch", task["goals"][0]["success_criteria"]["evaluator_test_patch"])

    def test_all_selects_all_unordered_feature_pairs(self):
        (self.dataset / "pallets_click_task" / "docs").mkdir()
        records = normalize.normalized_cooperbench(self.dataset, REVISION, "all")
        self.assertEqual(len(records), 3)
        self.assertEqual([r["metadata"]["feature_ids"] for r in records], [[1, 2], [1, 3], [2, 3]])

    def test_unknown_repository_is_rejected(self):
        p = self.dataset / "subsets/flash.json"
        self.write(p, json.dumps({"name": "flash", "tasks": [{"repo": "unknown_task", "task_id": 1, "pairs": [[1, 2]]}]}))
        with self.assertRaisesRegex(ValueError, "Unknown CooperBench repository"):
            normalize.normalized_cooperbench(self.dataset, REVISION, "flash")

    def test_missing_required_feature_files_fails(self):
        (self.dataset / "pallets_click_task/task2068/feature1/tests.patch").unlink()
        with self.assertRaisesRegex(ValueError, "Missing feature file"):
            normalize.normalized_cooperbench(self.dataset, REVISION, "flash")

    def test_missing_base_commit_fails(self):
        self.write(self.dataset / "pallets_click_task/task2068/setup.sh", "#!/bin/bash\n")
        with self.assertRaisesRegex(ValueError, "BASE_COMMIT"):
            normalize.normalized_cooperbench(self.dataset, REVISION, "flash")

    def test_duplicate_pair_is_rejected(self):
        self.write(self.dataset / "subsets/flash.json", json.dumps({"name": "flash", "tasks": [
            {"repo": "pallets_click_task", "task_id": 2068, "pairs": [[1, 2], [2, 1]]},
        ]}))
        with self.assertRaisesRegex(ValueError, "Duplicate feature pairs"):
            normalize.normalized_cooperbench(self.dataset, REVISION, "flash")

    def test_custom_case_uses_same_schema(self):
        r = normalize.normalized_custom(self.custom, "v1")[0]
        normalize.validate_task(r)
        self.assertEqual(r["dataset_source"], "align-custom")
        self.assertEqual(r["conflict_type"], "execution_conflict")
        self.assertEqual(r["expected_outcome"], "escalate")
        self.assertEqual(r["metadata"]["source_revision"], "v1")
        self.assertEqual(r["task_key"], "align-custom:shared-file-conflict:v1:v1")

    def test_invalid_custom_goal_rejected(self):
        bad = self.custom_case()
        bad["goals"][0]["owner_agent_id"] = None
        self.write(self.custom / "example.json", json.dumps(bad))
        with self.assertRaisesRegex(ValueError, "owner_agent_id"):
            normalize.normalized_custom(self.custom, "v1")

    def test_multiple_custom_cases_preserve_identity(self):
        item = self.custom_case()
        item["scenario_id"] = "shared-file-conflict-2"
        self.write(self.custom / "example2.json", json.dumps(item))
        records = normalize.normalized_custom(self.custom, "v1")
        self.assertEqual(len(records), 2)
        self.assertEqual(len({r["task_key"] for r in records}), 2)

    def test_immutable_output_and_sha(self):
        records = normalize.normalized_cooperbench(self.dataset, REVISION, "flash")
        out = self.root / "output"
        self.assertTrue(normalize.write_output(records, out, "cooperbench", REVISION, "flash"))
        self.assertFalse(normalize.write_output(records, out, "cooperbench", REVISION, "flash"))
        manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["task_count"], 1)
        self.assertEqual(manifest["tasks_sha256"], normalize.sha256_bytes((out / "tasks.jsonl").read_bytes()))
        self.assertEqual(json.loads((out / "tasks.jsonl").read_text(encoding="utf-8")), records[0])

    def test_existing_output_never_silently_overwritten(self):
        records = normalize.normalized_custom(self.custom, "v1")
        out = self.root / "normalized"
        normalize.write_output(records, out, "align-custom", "v1", "all")
        records[0]["shared_goal"] = "Different goal"
        with self.assertRaisesRegex(ValueError, "manifest differs|modified"):
            normalize.write_output(records, out, "align-custom", "v1", "all")

    def test_partially_written_output_can_resume_if_matching(self):
        records = normalize.normalized_custom(self.custom, "v1")
        out = self.root / "partial"
        normalize.write_output(records, out, "align-custom", "v1", "all")
        (out / "manifest.json").unlink()
        self.assertTrue(normalize.write_output(records, out, "align-custom", "v1", "all"))
        self.assertTrue((out / "manifest.json").exists())

    def test_raw_snapshot_provenance_and_checksums(self):
        files = []
        for path in sorted(self.dataset.rglob("*")):
            if path.is_file():
                contents = path.read_bytes()
                files.append({"path": f"dataset/{path.relative_to(self.dataset).as_posix()}",
                              "size_bytes": len(contents), "sha256": hashlib.sha256(contents).hexdigest()})
        raw_manifest = self.root / "raw_manifest.json"
        self.write(raw_manifest, json.dumps({"source": "cooperbench",
                                              "source_repository": "CodeConflict/cooperbench-dataset",
                                              "source_revision": REVISION, "file_count": len(files), "files": files}))
        out = self.root / "verified"
        with patch.object(sys, "argv", [str(MODULE), "cooperbench", "--dataset-dir", str(self.dataset),
                                       "--revision", REVISION, "--raw-manifest", str(raw_manifest), "--output-dir", str(out)]):
            normalize.main()
        self.assertEqual(json.loads((out / "manifest.json").read_text())["task_count"], 1)
        self.write(self.dataset / "pallets_click_task/task2068/feature2/tests.patch", "modified")
        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
            normalize.verify_raw_manifest(self.dataset, REVISION, raw_manifest)
        with self.assertRaisesRegex(ValueError, "provenance"):
            normalize.verify_raw_manifest(self.dataset, "c" * 40, raw_manifest)

    def test_revision_affects_identity(self):
        a = normalize.normalized_cooperbench(self.dataset, REVISION, "flash")[0]
        b = normalize.normalized_cooperbench(self.dataset, "c" * 40, "flash")[0]
        self.assertNotEqual(a["task_key"], b["task_key"])
        self.assertNotEqual(a["external_task_id"], b["external_task_id"])


if __name__ == "__main__":
    unittest.main()
