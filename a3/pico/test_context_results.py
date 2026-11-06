"""Offline checks of saved Part 4 evidence; no cloud calls or model imports."""
import json
import math
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1] / 'baseline_context_results'


class ContextEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.data = json.loads((ROOT / 'comparison.json').read_text())

    def test_continuation_grid(self):
        rows = self.data['results']
        self.assertEqual(len(rows), 4)
        self.assertEqual({(r['seed'], r['training_context']) for r in rows},
                         {(s, c) for s in (42, 43) for c in (512, 2048)})
        for row in rows:
            self.assertEqual(row['status'], 'complete')
            self.assertEqual(row['variant'], 'baseline')
            self.assertTrue(row['native_baseline'])
            self.assertEqual(row['completed_steps'], 1024)
            self.assertEqual(row['tokens_seen'], 33554432)
            self.assertEqual(row['seed_training_tokens_before'], 16384000)
            self.assertEqual(row['source_sha256'], self.data['source_sha256'][str(row['seed'])])

    def test_targets_and_byte_weighted_bpb(self):
        cells = self.data['originals'] + self.data['cross_eval']
        self.assertEqual(len(cells), 12)
        identity = None
        for cell in cells:
            result = cell['suffix_eval']
            docs = result['documents']
            self.assertEqual(len(docs), 48)
            self.assertEqual(result['target_tokens'], 12288)
            current = [(r['document'], r['bytes']) for r in docs]
            if identity is None:
                identity = current
            self.assertEqual(identity, current)
            measured = sum(r['nats'] for r in docs) / (math.log(2) * sum(r['bytes'] for r in docs))
            self.assertAlmostEqual(measured, result['bpb'])

    def test_evaluation_artifact_matches(self):
        rows = self.data['results']
        self.assertEqual(len({r['tokenizer_sha256'] for r in rows}), 1)
        self.assertTrue(all(r['eval_suite_sha256'] == self.data['eval_suite_sha256'] for r in rows))


if __name__ == '__main__':
    unittest.main()
