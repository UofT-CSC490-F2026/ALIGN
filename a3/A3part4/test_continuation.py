"""Check that both contexts score identical targets without future-token leakage."""
from pathlib import Path
import json
import tempfile
import unittest
from continuation_eval import window, completed_rows


class ContinuationTests(unittest.TestCase):
    def test_identical_targets_and_shift(self):
        tokens = list(range(2048))
        for context in (512, 2048):
            x, y = window(tokens, 9999, context)
            self.assertEqual(len(x), context)
            self.assertEqual(len(y), context)
            self.assertEqual(x[0], 9999)
            self.assertEqual(y[-128:], tokens[-128:])
            self.assertEqual(y[:-128], [-1]*(context-128))
            self.assertEqual(x[-128:], tokens[-129:-1])
            self.assertNotIn(tokens[-1], x)

    def test_reject_short_document(self):
        with self.assertRaises(ValueError):
            window([1]*2047, 0, 512)

    def test_duplicate_scores_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'scores.json'
            row = dict(checkpoint_step=500, context=512, index=0, text_sha256='a')
            path.write_text(json.dumps([row, row]))
            with self.assertRaises(ValueError):
                completed_rows(path, [{'text_sha256': 'a'}])


if __name__ == '__main__':
    unittest.main()
