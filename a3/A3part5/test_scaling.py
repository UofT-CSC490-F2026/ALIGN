"""Offline checks for study isolation, accounting and extrapolation."""
import json
import math
from pathlib import Path
import tempfile
import unittest
import sys
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scaling import train_command, predict, freeze_prediction
import scaling


class ScalingTests(unittest.TestCase):
    def test_training_command_is_bounded_and_native(self):
        c = train_command(6, 43, 250)
        self.assertIn('A3part5.scaling_train', c)
        self.assertIn('--target-param-data-ratio=8', c)
        self.assertIn('--max-seq-len=2048', c)
        self.assertIn('--resume-from-step=250', c)
        self.assertIn('--core-metric-every=-1', c)
        self.assertNotIn('--num-iterations', ' '.join(c))
        with self.assertRaises(ValueError):
            train_command(24, 42)
        with self.assertRaises(ValueError):
            train_command(12, 42)
        with self.assertRaises(ValueError):
            train_command(6, 44)

    def test_prediction_arithmetic(self):
        self.assertAlmostEqual(predict({'intercept': math.log(2), 'slope': -.5}, 100), .2)

    def test_nano_freeze_uses_only_three_small_sizes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            hashes = {'A': 'a', 'B': 'b'}
            (root / 'evaluation_manifest.json').write_text(json.dumps({k: {'sha256': v} for k, v in hashes.items()}))
            configs = {str(d): dict(steps=d, tokens=d*100, scaling_parameters=d*1000) for d in (4, 6, 8, 24)}
            (root / 'protocol.json').write_text(json.dumps({'configurations': configs}))
            for d in (4, 6, 8):
                for s in (42, 43):
                    p = root / f'd{d}-seed{s}'
                    p.mkdir()
                    (p / 'result.json').write_text(json.dumps(dict(depth=d, seed=s, step=d, tokens=d*100,
                        scaling_parameters=d*1000, eval_hashes=hashes, bpb={'A': 1., 'B': 2.})))
            with patch.object(scaling, 'STUDY', root), patch.object(scaling, 'power_fit', return_value={'intercept': 0., 'slope': -.1}) as fit:
                first = freeze_prediction()
                self.assertEqual(len(fit.call_args_list[0].args[0]), 6)
                self.assertEqual(first['target_depth'], 24)
                self.assertEqual(first, freeze_prediction())
                row_path = root / 'd4-seed42/result.json'
                row = json.loads(row_path.read_text())
                row['bpb']['A'] = 3.
                row_path.write_text(json.dumps(row))
                with self.assertRaises(ValueError):
                    freeze_prediction()


if __name__ == '__main__':
    unittest.main()
