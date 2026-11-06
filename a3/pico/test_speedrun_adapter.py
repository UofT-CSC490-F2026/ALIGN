"""Read-only regression checks: preserve upstream training/optimizer semantics."""
import ast
import os
import unittest
from pathlib import Path
from pico.speedrun_train import source_adapter


class AdapterTests(unittest.TestCase):
    def test_training_loop_unchanged(self):
        root = Path(os.environ.get('NANOCHAT_SOURCE', Path(__file__).resolve().parents[2] / 'nanochat'))
        source=(root/'scripts/base_train.py').read_text(encoding='utf-8')
        original=ast.parse(source)
        revised=ast.parse(source_adapter(source))
        loops=lambda tree:[ast.dump(n) for n in tree.body if isinstance(n,ast.While)]
        self.assertEqual(loops(original),loops(revised))
        functions=lambda tree:[ast.dump(n) for n in tree.body if isinstance(n,ast.FunctionDef)]
        self.assertEqual(functions(original),functions(revised))

    def test_drift_rejected(self):
        with self.assertRaises(RuntimeError):
            source_adapter('print("unexpected source")')


if __name__=='__main__':
    unittest.main()
