"""Alignment, denominator and window-policy regressions for Part 4 evaluation."""
from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from p4.evaluation import (aggregate_results, evaluate_checkpoint,
                           evaluation_window_policy, make_scoring_window)
from p4.report import (COST_ESTIMATE_SCOPE, _metric_series, _metric_table,
                       _validation_series, create_report)


class ScoringWindowTests(unittest.TestCase):
    def test_windows_score_the_same_suffix_with_correct_next_token_alignment(self):
        tokens = list(range(10))
        short = make_scoring_window(tokens, 4, 2)
        long = make_scoring_window(tokens, 8, 2)
        self.assertEqual(short["input_ids"], [5, 6, 7, 8])
        self.assertEqual(short["target_ids"], [6, 7, 8, 9])
        self.assertEqual(short["target_suffix_ids"], [8, 9])
        self.assertEqual(short["target_suffix_ids"], long["target_suffix_ids"])
        self.assertEqual(short["target_ids"][short["score_start"]:], [8, 9])

    def test_rejects_short_or_invalid_examples_and_oversized_suffix(self):
        for tokens, context, suffix in (([1, 2], 2, 1), ([1, -1, 2], 2, 1),
                                        ([1, True, 2], 2, 1), ([1, 2, 3], 2, 3)):
            with self.subTest(tokens=tokens, context=context, suffix=suffix):
                with self.assertRaises(ValueError):
                    make_scoring_window(tokens, context, suffix)

    def test_window_and_config_restored_even_after_failure(self):
        class Model:
            config = SimpleNamespace(sequence_len=4, window_pattern="SSSL")
            window_sizes = [(1, 0), (4, 0)]

            def _compute_window_sizes(self, config):
                return [(config.sequence_len // 4, 0), (config.sequence_len, 0)]

        model = Model()
        old_config, old_windows = model.config, model.window_sizes
        with self.assertRaisesRegex(RuntimeError, "deliberate"):
            with evaluation_window_policy(model, 8) as policy:
                self.assertEqual(model.config.sequence_len, 8)
                self.assertEqual(policy["window_sizes"], [[2, 0], [8, 0]])
                raise RuntimeError("deliberate")
        self.assertIs(model.config, old_config)
        self.assertIs(model.window_sizes, old_windows)

    def test_aggregation_weights_by_total_bytes(self):
        rows = [
            dict(checkpoint="checkpoint1", context_length=4, example_id="a", nll_nats=2., target_bytes=1),
            dict(checkpoint="checkpoint1", context_length=4, example_id="b", nll_nats=4., target_bytes=4),
        ]
        result = aggregate_results(rows)[0]
        self.assertAlmostEqual(result["bpb"], 6 / (math.log(2) * 5))
        self.assertNotAlmostEqual(result["bpb"], (2 + 1) / (2 * math.log(2)))
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            aggregate_results(rows + rows[:1])

    def test_target_identity_changes_are_rejected(self):
        base = dict(checkpoint="checkpoint1", context_length=4, example_id="a", nll_nats=2., target_bytes=1, target_sha256="one")
        other = dict(base, context_length=8, target_sha256="two")
        with self.assertRaisesRegex(ValueError, "Target changed"):
            aggregate_results([base, other])


@unittest.skipUnless(importlib.util.find_spec("torch"), "CPU torch is not installed")
class TensorEvaluationTests(unittest.TestCase):
    def _model(self):
        import torch

        class NextTokenModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.config = SimpleNamespace(sequence_len=4, window_pattern="SSSL")
                self.window_sizes = [(1, 0), (4, 0)]
                self.variant = "fox"
                self.calls = []

            def _compute_window_sizes(self, config):
                return [(config.sequence_len // 4, 0), (config.sequence_len, 0)]

            def forward(self, x):
                self.calls.append((x.shape[1], list(self.window_sizes)))
                logits = torch.zeros((*x.shape, 16), device=x.device)
                logits.scatter_(-1, ((x + 1) % 16).unsqueeze(-1), 2.)
                return logits

        return NextTokenModel()

    def test_evaluation_masks_special_tokens_and_restores_training_mode(self):
        model = self._model()
        original_config, original_windows = model.config, model.window_sizes
        # Targets end in [0, 1]; ID 0 is special and contributes no loss/bytes.
        tokens = list(range(9, 16)) + [0, 1]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "evaluation.jsonl"
            rows = evaluate_checkpoint(model, [{"id": "doc", "tokens": tokens}], [0] + [2] * 15,
                                       path, "checkpoint1", contexts=(4, 8), target_tokens=2)
            self.assertEqual(len(rows), 2)
            expected_loss = math.log(math.exp(2) + 15) - 2
            for row in rows:
                self.assertAlmostEqual(row["nll_nats"], expected_loss, places=6)
                self.assertEqual(row["target_bytes"], 2)
                self.assertEqual(row["scored_tokens"], 1)
                self.assertEqual(row["target_token_ids"], [0, 1])
                self.assertIsNone(row["target_nll_nats"][0])
            self.assertEqual(rows[0]["target_sha256"], rows[1]["target_sha256"])
            self.assertFalse(rows[0]["out_of_training_range"])
            self.assertTrue(rows[1]["out_of_training_range"])
            self.assertEqual(model.calls, [(4, [(1, 0), (4, 0)]), (8, [(2, 0), (8, 0)])])
            self.assertIs(model.config, original_config)
            self.assertIs(model.window_sizes, original_windows)
            self.assertTrue(model.training)
            self.assertEqual(len(path.read_text().splitlines()), 2)
            with self.assertRaises(FileExistsError):
                evaluate_checkpoint(model, [{"id": "doc", "tokens": tokens}], [0] + [2] * 15,
                                    path, "checkpoint1", contexts=(4, 8), target_tokens=2)

    def test_invalid_request_does_not_create_partial_output(self):
        model = self._model()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "evaluation.jsonl"
            with self.assertRaisesRegex(ValueError, "No scored bytes"):
                evaluate_checkpoint(model, [{"id": "doc", "tokens": [1] * 9}], [0] * 16,
                                    path, "checkpoint1", contexts=(4, 8), target_tokens=2)
            self.assertFalse(path.exists())


class ReportTests(unittest.TestCase):
    def _rows(self):
        return [dict(event="evaluation_example", checkpoint=checkpoint, context_length=context,
                     example_id=example, target_sha256=example, nll_nats=nll, target_bytes=2, scored_tokens=1)
                for checkpoint in ("checkpoint1", "checkpoint2")
                for context in (4, 8)
                for example, nll in (("a", 1.), ("b", 2.))]

    def test_complete_four_cells_and_periodic_validation_are_separate(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            rows = self._rows()
            rows += [dict(rows[0], checkpoint="validation-step-100")]
            (root / "evaluations.jsonl").write_text("\n".join(json.dumps(row) for row in rows))
            (root / "metrics.jsonl").write_text(json.dumps({"event": "train", "tokens": 100, "train/loss": 2.5, "val/bpb": 1.4}))
            with patch("p4.report._draw_figures", return_value=([], "test_disabled")):
                report = create_report(root, expected_contexts=(4, 8))
            self.assertEqual(report["status"], "complete")
            self.assertEqual(len(report["evaluation"]), 4)
            self.assertEqual(report["training_metric_rows"], 1)
            self.assertTrue((root / "report/evaluation_examples.csv").is_file())
            self.assertEqual(len((root / "report/evaluation_summary.csv").read_text().splitlines()), 5)

    def test_missing_cell_is_not_fabricated(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            rows = [r for r in self._rows() if not (r["checkpoint"] == "checkpoint2" and r["context_length"] == 8)]
            (root / "evaluations.jsonl").write_text("\n".join(json.dumps(row) for row in rows))
            with patch("p4.report._draw_figures", return_value=([], "test_disabled")):
                report = create_report(root, expected_contexts=(4, 8))
            self.assertEqual(report["status"], "incomplete")
            self.assertEqual(report["missing_cells"], [{"checkpoint": "checkpoint2", "context_length": 8}])
            self.assertIsNone(report["context_benefit"][1]["long_context_benefit_bpb"])

    def test_validation_series_use_explicit_contexts_instead_of_generic_bpb(self):
        metrics = _metric_table([
            {"stage": "short", "attempt": "a", "tokens": 10, "context": 512,
             "val/bpb": 99., "val/bpb_context_512": 1.5, "val/bpb_context_2048": 1.4},
            {"stage": "long", "attempt": "b", "tokens": 20, "context": 2048,
             "val/bpb": 88., "val/bpb_context_512": 1.3, "val/bpb_context_2048": 1.2},
        ])
        series = _validation_series(metrics)
        self.assertEqual(len(series), 4)
        self.assertEqual(sorted(value for points in series.values() for _, value in points), [1.2, 1.3, 1.4, 1.5])
        self.assertTrue(all("input window 512" in label or "input window 2048" in label for label in series))

    def test_legacy_validation_fallback_discloses_stage_and_context(self):
        metrics = _metric_table([
            {"stage": "short", "attempt": "a", "tokens": 10, "context": 512, "val/bpb": 1.5},
            {"stage": "long", "tokens": 20, "val/bpb": 1.3},
        ])
        labels = list(_validation_series(metrics))
        self.assertTrue(any("short" in label and "input window 512" in label for label in labels))
        self.assertTrue(any("long" in label and "evaluation window not recorded" in label for label in labels))

    def test_recovered_attempts_remain_separate_and_keep_log_order(self):
        rows = [
            {"stage": "short", "attempt": "failed", "tokens": 20, "train/loss": 2., "estimated_cost_usd": .01},
            {"stage": "short", "attempt": "failed", "tokens": 30, "train/loss": 1.9, "estimated_cost_usd": .02},
            {"stage": "short", "attempt": "resumed", "tokens": 21, "train/loss": 1.95, "estimated_cost_usd": .03},
            {"stage": "short", "tokens": 21, "val/bpb_context_512": 1.4},
            {"stage": "short", "attempt": "resumed", "tokens": 22, "train/loss": 1.93, "estimated_cost_usd": .04},
        ]
        metrics = _metric_table(rows)
        for field in ("train_loss", "estimated_cost_usd"):
            series = _metric_series(metrics, field)
            self.assertEqual(len(series), 2)
            self.assertEqual(next(points for label, points in series.items() if "failed" in label)[0][0], 20)
            self.assertEqual([p[0] for label, points in series.items() if "resumed" in label for p in points], [21, 22])
        self.assertEqual(metrics[3]["attempt"], "resumed")
        self.assertTrue(metrics[3]["attempt_inferred"])
        self.assertEqual(json.loads(metrics[0]["raw_metric_json"]), rows[0])

    def test_report_exports_context_fields_and_explicit_cost_scope(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = {"stage": "short", "attempt": "a", "tokens": 10, "context": 512,
                      "val/bpb": 1.5, "val/bpb_context_512": 1.5, "val/bpb_context_2048": 1.4,
                      "estimated_cost_usd": .01, "hardware/unrecognized_future_metric": 7}
            raw = json.dumps(source) + "\n"
            (root / "metrics.jsonl").write_text(raw)
            with patch("p4.report._draw_figures", return_value=([], "test_disabled")):
                summary = create_report(root)
            self.assertEqual((root / "metrics.jsonl").read_text(), raw)
            csv = (root / "report/training_metrics.csv").read_text()
            self.assertIn("val_bpb_context_512", csv)
            self.assertIn("val_bpb_context_2048", csv)
            self.assertIn("unrecognized_future_metric", csv)
            self.assertEqual(summary["cost_estimate_scope"], COST_ESTIMATE_SCOPE)
            for exclusion in ("startup", "CPU data preparation", "rehearsal", "storage", "final W&B finalization"):
                self.assertIn(exclusion, summary["cost_estimate_scope"])


if __name__ == "__main__":
    unittest.main()
