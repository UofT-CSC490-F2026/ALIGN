import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from p4.budget import reserve
from p4.config import load_config, rehearsal_config, validate
from p4.data import TokenStream, verify_prepared
from p4.artifacts import sha256_file


class DataConfigTests(unittest.TestCase):
    def test_resume_keeps_exact_next_tokens_when_context_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "train.bin"
            np.arange(101, dtype="<u4").tofile(path)
            stream = TokenStream(path)
            x, y = stream.next_batch(2, 8)
            self.assertEqual(x.flatten().tolist(), list(range(16)))
            self.assertEqual(y.flatten().tolist(), list(range(1, 17)))
            state = stream.state_dict()
            resumed = TokenStream(path)
            resumed.load_state_dict(state)
            x, y = resumed.next_batch(1, 16)
            self.assertEqual(x.flatten().tolist(), list(range(16, 32)))
            self.assertEqual(y.flatten().tolist(), list(range(17, 33)))
            self.assertEqual(resumed.state_dict()["cursor"], 32)

    def test_wrapping_is_explicit_and_cursor_preserves_epoch_progress(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "train.bin"
            np.arange(5, dtype="<u4").tofile(path)
            stream = TokenStream(path, cursor=4)
            self.assertEqual(stream.next_batch(1, 4)[0].tolist(), [[4, 0, 1, 2]])
            self.assertEqual(stream.state_dict()["cursor"], 8)
            with self.assertRaises(ValueError):
                stream.load_state_dict({"cursor": 1, "token_count": 6})

    def test_prepared_tampering_and_budget_overrun_block_launch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "train.bin"
            path.write_bytes(b"data")
            (root / "manifest.json").write_text(json.dumps({"files": {"train.bin": {"sha256": sha256_file(path), "size_bytes": 4}}}))
            verify_prepared(root)
            path.write_bytes(b"oops")
            with self.assertRaises(ValueError):
                verify_prepared(root)
            config = load_config()
            config["budget_usd"] = .01
            with self.assertRaises(RuntimeError):
                reserve(root / "budget.jsonl", "gpu", config, seconds=1800, gpu=True)
            self.assertFalse((root / "budget.jsonl").exists())

    def test_bounded_configs(self):
        config = rehearsal_config()
        self.assertEqual(config["long_steps"], 2)
        config["budget_usd"] = 101
        with self.assertRaises(ValueError):
            validate(config)
