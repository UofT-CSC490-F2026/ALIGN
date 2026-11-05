import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from p4.artifacts import atomic_json, sha256_file, CheckpointStore
from p4.standard_eval import engine
from p4.standard_eval.control import train_control, assert_common_initialization


class ToyLossModel(torch.nn.Module):
    def get_device(self):
        return torch.device("cpu")

    def forward(self, x, y, loss_reduction="none"):
        return torch.where(y >= 0, x.float() / 10 + 1, 0).reshape(-1)


class StandardTests(unittest.TestCase):
    def test_upstream_metric_masks_special_and_ignored_targets(self):
        arrays = {"inputs": np.array([[[1, 2, 3, 4]], [[5, 6, 7, 8]]]),
                  "targets": np.array([[[0, 1, 2, -1]], [[1, 0, 2, 1]]])}
        metric, rows = engine.evaluate_frozen(ToyLossModel(), arrays, torch.tensor([0, 1, 2]), "cpu")
        expected_loss = 1.2 + 1.3 + 1.5 + 1.7 + 1.8
        self.assertAlmostEqual(metric, expected_loss / (np.log(2) * 7), places=6)
        self.assertEqual(sum(r["scored_non_special_tokens"] for r in rows), 5)
        self.assertEqual(sum(r["target_bytes"] for r in rows), 7)

    def test_frozen_loader_copies_mutable_upstream_buffers(self):
        from nanochat.tokenizer import RustBPETokenizer
        import nanochat.dataloader as dataloader
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            prepared, out = root / "prepared", root / "out"
            (prepared / "source").mkdir(parents=True)
            (prepared / "tokenizer").mkdir()
            (prepared / "source/shard_06542.parquet").write_bytes(b"test")
            (prepared / "tokenizer/tokenizer.pkl").write_bytes(b"test")
            np.save(prepared / "token_bytes.npy", np.ones(32))
            atomic_json(prepared / "manifest.json", {"dataset_revision": "frozen-test"})
            def reusable(tokenizer, batch, length, split, device):
                x = torch.zeros((batch, length), dtype=torch.int64)
                y = x.clone()
                for i in range(100):
                    x.fill_(i)
                    y.fill_(i + 1)
                    yield x, y
            with patch.object(engine, "TOKEN_BUDGET", 32768), patch.object(RustBPETokenizer, "from_directory", return_value=None), patch.object(dataloader, "tokenizing_distributed_data_loader_bos_bestfit", reusable):
                manifest = engine.freeze_validation(prepared, out)
                with np.load(out / "validation-512.npz") as a:
                    self.assertEqual(int(a["inputs"][0, 0, 0]), 0)
                    self.assertEqual(int(a["inputs"][1, 0, 0]), 1)
                    self.assertEqual(int(a["inputs"][-1, 0, 0]), 7)
                self.assertEqual(engine.freeze_validation(prepared, out), manifest)
                (out / "validation-512.npz").write_bytes(b"corrupted")
                with self.assertRaises(ValueError):
                    engine.freeze_validation(prepared, out)

    def test_baseline_continuation_matches_token_cursor_and_parent(self):
        from p4.config import DEFAULT
        config = copy.deepcopy(DEFAULT)
        config.update(variant="baseline", depth=2, width=64, vocab_size=32,
                      short_context=8, long_context=16, short_steps=2, long_steps=2,
                      short_microbatch=1, long_microbatch=1, tokens_per_step=32,
                      checkpoint_every=1, log_every=1, warmdown_start=3, warmup_steps=1,
                      wandb_mode="disabled", max_seconds_per_stage=60)
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            prepared, run = root / "prepared", root / "run"
            prepared.mkdir()
            np.tile(np.arange(32, dtype="<u4"), 20).tofile(prepared / "train.bin")
            np.save(prepared / "token_bytes.npy", np.ones(32, dtype=np.int64))
            (prepared / "evaluation.jsonl").write_text('{}\n')
            files = {name: {"sha256": sha256_file(prepared / name), "size_bytes": (prepared / name).stat().st_size}
                     for name in ("train.bin", "token_bytes.npy", "evaluation.jsonl")}
            atomic_json(prepared / "manifest.json", {"vocab_size": 32, "files": files})
            common = assert_common_initialization(config)
            self.assertTrue(common["common_initial_tensors_identical"])
            short = train_control(config, prepared, run, "short", device="cpu")
            long = train_control(config, prepared, run, "long", device="cpu")
            self.assertEqual(long["parent_checkpoint_sha256"], short["checkpoint"]["sha256"])
            state, _ = CheckpointStore(run).load("checkpoint2")
            self.assertEqual(state["data_stream"]["cursor"], 128)
            self.assertEqual(state["global_step"], 4)
            with self.assertRaises(FileExistsError):
                train_control(config, prepared, run, "long", device="cpu")


if __name__ == "__main__":
    unittest.main()
