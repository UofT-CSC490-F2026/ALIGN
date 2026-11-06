"""Offline consistency checks; never allocate GPUs or contact services."""
import json
import math
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1] / 'speedrun_results'


class EvidenceTests(unittest.TestCase):
    def test_final_and_prediction(self):
        final = json.loads((ROOT / 'final.json').read_text())
        forecast = json.loads((ROOT / 'predict.json').read_text())
        self.assertEqual(final['status'], 'complete')
        self.assertEqual(final['steps'], 5571)
        self.assertEqual(final['tokens'], final['steps'] * final['global_batch'])
        self.assertEqual(final['tokens'], 5841616896)
        self.assertEqual(final['fixed_eval_sha256'], forecast['fixed_eval_sha256'])
        self.assertEqual(final['model_source_sha256'], forecast['model_source_sha256'])
        predicted = math.exp(forecast['intercept']) * final['scaling_parameters'] ** forecast['slope']
        self.assertAlmostEqual(predicted, final['predicted_bpb'])
        self.assertAlmostEqual(final['calibration_bpb'] - predicted, final['prediction_error'])

    def test_calibration_and_preflight(self):
        forecast = json.loads((ROOT / 'predict.json').read_text())
        self.assertEqual({r['depth'] for r in forecast['calibration']}, {4, 6, 8})
        for row in forecast['calibration']:
            self.assertEqual(row['status'], 'complete')
            self.assertEqual(row['fixed_eval_sha256'], forecast['fixed_eval_sha256'])
        check = json.loads((ROOT / 'kernel-check.json').read_text())
        self.assertEqual(check['status'], 'passed')
        self.assertEqual(check['model_source_sha256'], forecast['model_source_sha256'])
        smoke = json.loads((ROOT / 'h100-check.json').read_text())
        self.assertEqual(smoke['status'], 'complete')
        self.assertIn('H100', smoke['gpu_name'])


if __name__ == '__main__':
    unittest.main()
