"""CPU rehearsal tests for ancestry, exact cursor resumption and saved evidence."""
import copy
import hashlib
import importlib.util
import json
import os
import shutil
import tempfile
import unittest
from unittest.mock import Mock, patch
from pathlib import Path

from p4.training import learning_rate_multiplier


HAS_TORCH = importlib.util.find_spec("torch") is not None


class ScheduleTests(unittest.TestCase):
    def test_schedule_is_global_across_context_boundary(self):
        config = {"short_steps": 500, "long_steps": 500, "warmup_steps": 20, "warmdown_start": 700}
        self.assertEqual(learning_rate_multiplier(0, config), .05)
        self.assertEqual(learning_rate_multiplier(499, config), 1)
        self.assertEqual(learning_rate_multiplier(500, config), 1)
        self.assertAlmostEqual(learning_rate_multiplier(999, config), .05)
        with self.assertRaises(ValueError):
            learning_rate_multiplier(1000, config)


@unittest.skipUnless(HAS_TORCH, "requires isolated CPU torch installation")
class TrainingRehearsalTests(unittest.TestCase):
    def setUp(self):
        import numpy as np
        import torch
        from p4.artifacts import atomic_json, sha256_file

        torch.set_num_threads(1)
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.prepared = self.root / "prepared"
        self.prepared.mkdir()
        np.asarray([i % 53 + 1 for i in range(1024)], dtype="<u4").tofile(self.prepared / "train.bin")
        np.save(self.prepared / "token_bytes.npy", np.ones(320, dtype=np.int32))
        examples = [{"id": "heldout-1", "tokens": [i % 47 + 1 for i in range(17)]},
                    {"id": "heldout-2", "tokens": [(i + 3) % 47 + 1 for i in range(17)]}]
        (self.prepared / "evaluation.jsonl").write_text("".join(json.dumps(row) + "\n" for row in examples))
        files = {path.name: {"sha256": sha256_file(path), "size_bytes": path.stat().st_size}
                 for path in self.prepared.iterdir()}
        atomic_json(self.prepared / "manifest.json", {"schema_version": 1, "vocab_size": 320,
                    "token_count": 1024, "files": files})
        self.config = {
            "seed": 42, "variant": "fox", "depth": 2, "width": 128, "vocab_size": 320,
            "short_context": 8, "long_context": 16, "short_steps": 2, "long_steps": 2,
            "tokens_per_step": 16, "short_microbatch": 2, "long_microbatch": 1,
            "checkpoint_every": 1, "log_every": 1, "eval_every": 2,
            "eval_target_tokens": 4, "eval_examples": 2, "warmup_steps": 1,
            "warmdown_start": 3, "max_seconds_per_stage": 120,
            "wandb_mode": "disabled", "budget_usd": 50,
        }

    def tearDown(self):
        self.temporary.cleanup()

    def test_two_stage_continuation_and_partial_resume_preserve_trajectory(self):
        import torch
        from p4.artifacts import CheckpointStore
        from p4.training import model_state_sha256, train_stage

        run = self.root / "chain"
        commits = []
        short = train_stage(self.config, self.prepared, run, "short", commit=lambda: commits.append(True), device="cpu")
        self.assertEqual(short["status"], "complete")
        checkpoint1_bytes = (run / "checkpoints/checkpoint1.pt").read_bytes()
        checkpoint1_hash = hashlib.sha256(checkpoint1_bytes).hexdigest()
        long = train_stage(self.config, self.prepared, run, "long", device="cpu")
        self.assertEqual(long["status"], "complete")
        self.assertEqual(long["parent_checkpoint_sha256"], checkpoint1_hash)
        self.assertEqual((run / "checkpoints/checkpoint1.pt").read_bytes(), checkpoint1_bytes)
        proof = json.loads((run / "continuation.json").read_text())
        self.assertTrue(proof["weights_verified_before_update"])
        self.assertTrue(proof["optimizer_state_loaded"])
        self.assertEqual(proof["global_step_before_update"], 2)
        self.assertEqual(proof["tokens_before_update"], 32)
        self.assertEqual(proof["data_stream_before_update"]["cursor"], 32)
        final, _ = CheckpointStore(run).load("checkpoint2")
        self.assertEqual(final["data_stream"]["cursor"], 64)
        self.assertEqual(final["cumulative_tokens"], 64)
        self.assertEqual(final["optimizer_kind"], "adamw_cpu_correctness_test_only")
        self.assertGreater(len(commits), 2)
        for name in ("checkpoint1", "checkpoint2"):
            rows = [json.loads(line) for line in (run / f"evaluation-{name}.jsonl").read_text().splitlines()]
            self.assertEqual(len(rows), 4)
            self.assertEqual({row["context_length"] for row in rows}, {8, 16})

        # Reconstruct a run interrupted immediately after the durable step-3
        # checkpoint. Resuming must reach exactly the same weights and cursor.
        recovered = self.root / "recovered"
        shutil.copytree(run, recovered)
        for name in ("checkpoints/checkpoint2.pt", "checkpoints/checkpoint2.json",
                     "evaluation-checkpoint2.jsonl", "long-result.json"):
            (recovered / name).unlink()
        resumed = train_stage(self.config, self.prepared, recovered, "long", device="cpu", resume="long-step000003")
        self.assertEqual(resumed["steps_this_attempt"], 1)
        restored, _ = CheckpointStore(recovered).load("checkpoint2")
        self.assertEqual(restored["data_stream"], final["data_stream"])
        self.assertEqual(model_state_sha256(restored["model"]), model_state_sha256(final["model"]))
        for key in final["model"]:
            torch.testing.assert_close(restored["model"][key], final["model"][key], rtol=0, atol=0)

    def test_source_data_and_result_reuse_are_rejected(self):
        from p4.training import train_stage

        run = self.root / "chain"
        train_stage(self.config, self.prepared, run, "short", device="cpu")
        with self.assertRaises(FileExistsError):
            train_stage(self.config, self.prepared, run, "short", device="cpu")
        changed = copy.deepcopy(self.config)
        changed["seed"] += 1
        with self.assertRaisesRegex(ValueError, "config differs"):
            train_stage(changed, self.prepared, run, "long", device="cpu")
        with (self.prepared / "train.bin").open("ab") as handle:
            handle.write(b"\x00\x00\x00\x00")
        with self.assertRaisesRegex(ValueError, "integrity failure"):
            train_stage(self.config, self.prepared, run, "long", device="cpu")

    def test_recovery_retention_preserves_final_checkpoint_and_audit(self):
        from p4.training import train_stage

        config = dict(self.config, short_steps=3, long_steps=1, keep_last_recovery_checkpoints=1)
        run = self.root / "retention"
        train_stage(config, self.prepared, run, "short", device="cpu")
        self.assertTrue((run / "checkpoints/checkpoint1.pt").is_file())
        self.assertTrue((run / "checkpoints/short-step000002.pt").is_file())
        self.assertFalse((run / "checkpoints/short-step000001.pt").exists())
        rows = [json.loads(line) for line in (run / "checkpoint-history.jsonl").read_text().splitlines()]
        self.assertEqual(rows[0]["checkpoint"], "short-step000001")
        self.assertEqual(rows[0]["global_step"], 1)
        self.assertEqual(len(rows[0]["checkpoint_sha256"]), 64)
        self.assertEqual(len(rows[0]["manifest_sha256"]), 64)

    def test_rng_round_trip_includes_python_numpy_and_torch(self):
        import numpy as np
        import random
        import torch
        from p4.training import capture_rng_state, restore_rng_state

        state = capture_rng_state()
        first = (random.random(), float(np.random.random()), torch.rand(2))
        restore_rng_state(state)
        second = (random.random(), float(np.random.random()), torch.rand(2))
        self.assertEqual(first[:2], second[:2])
        torch.testing.assert_close(first[2], second[2], rtol=0, atol=0)

    def test_final_wandb_evaluation_uses_a_new_internal_history_row(self):
        from p4.training import _WandbMirror

        mirror = _WandbMirror.__new__(_WandbMirror)
        mirror.run, mirror.root, mirror.failed = Mock(), self.root, False
        mirror.log({"tokens": 32, "train/loss": 2.0}, 2)
        mirror.log({"tokens": 32, "val/bpb": 1.25}, 2)
        calls = mirror.run.log.call_args_list
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0].kwargs, {})
        self.assertEqual(calls[1].kwargs, {})
        self.assertEqual(calls[0].args[0]["global_step"], 2)
        self.assertEqual(calls[1].args[0]["val/bpb"], 1.25)

    @unittest.skipUnless(importlib.util.find_spec("wandb") is not None and os.environ.get("P4_RUN_WANDB_INTEGRATION") == "1",
                         "opt-in offline SDK integration requires local socket permission")
    def test_offline_wandb_mirror_does_not_create_export_blocking_symlinks(self):
        from p4.training import _WandbMirror

        run = self.root / "offline"
        run.mkdir()
        config = dict(self.config, wandb_mode="offline")
        with patch.dict(os.environ, {"WANDB_CACHE_DIR": str(run / "cache"),
                                     "WANDB_CONFIG_DIR": str(run / "config"),
                                     "WANDB_DATA_DIR": str(run / "data")}):
            mirror = _WandbMirror(config, run, "short", "test-attempt")
            mirror.log({"tokens": 16, "train/loss": 2.0}, 1)
            mirror.log({"tokens": 16, "val/bpb": 1.25}, 1)
            mirror.finish()
            import wandb
            wandb.teardown(exit_code=0)
        self.assertTrue(any(run.rglob("*.wandb")))
        self.assertFalse(any(path.is_symlink() for path in run.rglob("*")))


if __name__ == "__main__":
    unittest.main()
