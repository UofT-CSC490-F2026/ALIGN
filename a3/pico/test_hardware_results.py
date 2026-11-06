"""Offline checks of historical benchmark evidence; no cloud access."""
import json
import unittest
from pathlib import Path

FOLDER = Path(__file__).resolve().parents[1] / 'hardware_results'


class HardwareEvidenceTests(unittest.TestCase):
    def test_complete_grid_and_measurements(self):
        seen = set()
        for hardware in ('A10', 'A100-80GB', 'H100'):
            worker = json.loads((FOLDER / f'{hardware}.json').read_text())
            for row in worker['results']:
                key = (hardware, row['variant'], row['seed'])
                self.assertNotIn(key, seen)
                seen.add(key)
                self.assertEqual(row['status'], 'complete')
                self.assertEqual(row['completed_steps'], 500)
                self.assertEqual(row['tokens_seen'], 16384000)
                times = row['step_seconds']
                self.assertEqual(len(times), 500)
                self.assertTrue(all(t > 0 for t in times))
                self.assertAlmostEqual(sum(times), row['train_seconds'], places=5)
                rate = 495 * 32768 / sum(times[5:])
                self.assertAlmostEqual(rate, row['steady_tokens_per_second'], places=5)
                expected_mfu = 100 * row['model_flops_per_token'] * rate / row['dense_bf16_peak_flops']
                self.assertAlmostEqual(expected_mfu, row['estimated_mfu_percent'], places=5)
                self.assertAlmostEqual(row['elapsed_seconds'] * row['gpu_dollars_per_second'],
                                       row['seed_elapsed_gpu_cost_estimate'], places=6)
        self.assertEqual(seen, {(h, v, s) for h in ('A10', 'A100-80GB', 'H100')
                               for v in ('baseline', 'fox') for s in (42, 43)})

    def test_reported_mfu_uses_actual_device(self):
        rows = json.loads((FOLDER / 'reported_results.json').read_text())
        self.assertEqual(len(rows), 12)
        for row in rows:
            peak = 70e12 if 'a10g' in row['gpu'].lower() else {'A100-80GB': 312e12, 'H100': 989e12}[row['hardware']]
            self.assertEqual(row['dense_bf16_peak_flops'], peak)
            self.assertAlmostEqual(row['estimated_mfu_percent'],
                                   100 * row['model_flops_per_token'] * row['steady_tokens_per_second'] / peak)


if __name__ == '__main__':
    unittest.main()
