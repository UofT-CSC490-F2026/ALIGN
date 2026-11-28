"""Offline arbiter evaluation: mock arbiters, hand-computed metrics, label validation."""
import contextlib
import io
import json
from pathlib import Path
import runpy
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest import mock
import warnings

from evaluation import offline


def observation(sequence, command="echo hi"):
    return {"schema_version": 1, "task_key": "task-a", "shared_goal": "goal",
            "agent_goals": {"agent1": "one", "agent2": "two"}, "constraints": {},
            "actor_id": "agent1", "proposal_sequence": sequence,
            "proposed_action": {"action": {"tool": "bash", "command": command}, "files_touched": []},
            "recent_events": [], "repository_state": {}}


def record(sequence, expected, primary=None, source="runner_event", command="echo hi", split=None):
    row = {"checkpoint_event_uuid": f"uuid-{sequence}", "task_key": "task-a",
           "expected_actions": expected, "primary_action": primary or expected[0],
           "label_source": "human", "review_status": "reviewed",
           "prompt": [{"role": "system", "content": "prompt"},
                      {"role": "user", "content": json.dumps(observation(sequence, command))}]}
    if source:
        row["observation_source"] = source
    if split:
        row["split"] = split
    return row


def write(folder, rows):
    path = Path(folder) / "records.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


class Table:
    """Answers by proposal_sequence: an action name, a ready-made dict, or an exception."""
    def __init__(self, table):
        self.table = table

    def decide(self, obs):
        value = self.table[obs["proposal_sequence"]]
        if isinstance(value, Exception):
            raise value
        if isinstance(value, dict):
            return value
        return {"action": value, "reason": "fixture", "affected_agents": [obs["actor_id"]], "evidence": {}}


def six_records():
    return [record(1, ["block"]), record(2, ["block"]), record(3, ["continue"]),
            record(4, ["query", "mediate"], "query", "public_reconstruction"),
            record(5, ["continue"], source="public_reconstruction"),
            record(6, ["mediate"], source="public_reconstruction")]


SIX_ANSWERS = {1: "block", 2: "continue", 3: "block", 4: "mediate", 5: "continue",
               6: RuntimeError("endpoint down")}


class LoadRecordsTests(unittest.TestCase):
    def load(self, rows):
        with TemporaryDirectory() as folder:
            return offline.load_records(write(folder, rows))

    def test_blank_lines_are_ignored(self):
        with TemporaryDirectory() as folder:
            path = write(folder, [record(1, ["block"])])
            path.write_text("\n" + path.read_text(encoding="utf-8") + "\n  \n", encoding="utf-8")
            self.assertEqual(len(offline.load_records(path)), 1)

    def test_valid_records_gain_a_parsed_observation(self):
        loaded = self.load([record(1, ["block", "query"])])
        self.assertEqual(loaded[0]["observation"], observation(1))
        self.assertEqual(loaded[0]["expected_actions"], ["block", "query"])

    def test_invalid_records_are_rejected(self):
        bad_prompt = record(1, ["block"])
        bad_prompt["prompt"][-1]["content"] = "[1, 2]"
        provisional = record(1, ["block"])
        provisional["review_status"] = "provisional"
        teacher = record(1, ["block"])
        teacher["label_source"] = "teacher"
        cases = {
            "empty file": [],
            "duplicate checkpoint": [record(1, ["block"]), record(1, ["continue"])],
            "empty expected": [record(1, [], "block")],
            "unknown action": [record(1, ["explode"])],
            "expected not a list": [record(1, "block")],
            "primary missing": [record(1, ["block"], "query")],
            "not reviewed": [provisional],
            "not human": [teacher],
            "prompt is not an observation": [bad_prompt],
        }
        for name, rows in cases.items():
            with self.subTest(name), self.assertRaises(ValueError):
                self.load(rows)


class JudgeTests(unittest.TestCase):
    def test_truth_table(self):
        cases = [
            (["continue"], None, (False, False, False)),
            (["continue"], "continue", (True, False, False)),
            (["block"], "continue", (False, True, False)),
            (["continue", "query"], "continue", (True, False, False)),
            (["continue"], "block", (False, False, True)),
            (["continue", "query"], "block", (False, False, False)),
            (["continue", "query"], "query", (True, False, False)),
            (["query", "mediate"], "mediate", (True, False, False)),
        ]
        for expected, action, flags in cases:
            with self.subTest(expected=expected, action=action):
                result = offline.judge(expected, action)
                self.assertEqual((result["accepted"], result["missed"], result["unnecessary"]), flags)


class RunArbiterTests(unittest.TestCase):
    def results(self, answers, rows):
        with TemporaryDirectory() as folder:
            records = offline.load_records(write(folder, rows))
        return offline.run_arbiter(Table(answers), records)

    def test_failures_are_recorded_not_raised(self):
        malformed = {"action": "block"}
        unknown_agent = {"action": "block", "reason": "x", "affected_agents": ["agent9"], "evidence": {}}
        results = self.results(
            {1: "block", 2: RuntimeError("down"), 3: malformed, 4: unknown_agent},
            [record(1, ["block"]), record(2, ["block"]), record(3, ["block"]), record(4, ["block"])])
        self.assertEqual([row["action"] for row in results], ["block", None, None, None])
        self.assertEqual([row["error"] for row in results], [None, "RuntimeError", "ValueError", "ValueError"])
        self.assertEqual([row["accepted"] for row in results], [True, False, False, False])

    def test_row_carries_identity_and_defaults_source(self):
        results = self.results({7: "continue"}, [record(7, ["continue"], source=None)])
        row = results[0]
        self.assertEqual((row["checkpoint_event_uuid"], row["task_key"], row["source"]),
                         ("uuid-7", "task-a", "unknown"))
        self.assertIsInstance(row["latency_ms"], int)

    def test_arbiter_cannot_change_the_stored_observation(self):
        class Tamper:
            def decide(self, obs):
                obs["agent_goals"]["agent1"] = "changed"
                return {"action": "continue", "reason": "x", "affected_agents": [], "evidence": {}}
        with TemporaryDirectory() as folder:
            records = offline.load_records(write(folder, [record(1, ["continue"])]))
        offline.run_arbiter(Tamper(), records)
        self.assertEqual(records[0]["observation"]["agent_goals"]["agent1"], "one")


class SummarizeTests(unittest.TestCase):
    def summary(self):
        with TemporaryDirectory() as folder:
            records = offline.load_records(write(folder, six_records()))
        return offline.summarize(offline.run_arbiter(Table(SIX_ANSWERS), records))

    def test_overall_counts(self):
        overall = self.summary()["overall"]
        self.assertEqual((overall["records"], overall["valid"], overall["accepted"]), (6, 5, 3))
        self.assertAlmostEqual(overall["valid_rate"], 5 / 6)
        self.assertAlmostEqual(overall["accepted_rate"], 0.5)
        self.assertEqual((overall["missed"], overall["unnecessary"]), (1, 1))

    def test_error_types_are_counted(self):
        summary = self.summary()
        self.assertEqual(summary["overall"]["errors"], {"RuntimeError": 1})
        self.assertEqual(summary["by_source"]["runner_event"]["errors"], {})
        self.assertEqual(summary["by_source"]["public_reconstruction"]["errors"], {"RuntimeError": 1})

    def test_per_action_precision_and_recall(self):
        per_action = self.summary()["overall"]["per_action"]
        self.assertEqual(set(per_action), {"block", "continue", "query", "mediate"})
        self.assertEqual(per_action["block"], {"expected": 2, "chosen": 2, "recall": 0.5, "precision": 0.5})
        self.assertEqual(per_action["continue"], {"expected": 2, "chosen": 2, "recall": 0.5, "precision": 0.5})
        self.assertEqual(per_action["query"], {"expected": 1, "chosen": 0, "recall": 0.0, "precision": None})
        self.assertEqual(per_action["mediate"], {"expected": 2, "chosen": 1, "recall": 0.5, "precision": 1.0})

    def test_confusion_matrix_uses_primary_action_and_marks_invalid(self):
        self.assertEqual(self.summary()["overall"]["confusion"], {
            "block": {"block": 1, "continue": 1},
            "continue": {"block": 1, "continue": 1},
            "mediate": {"invalid": 1},
            "query": {"mediate": 1}})

    def test_sources_are_reported_separately(self):
        by_source = self.summary()["by_source"]
        self.assertEqual(list(by_source), ["public_reconstruction", "runner_event"])
        runner, public = by_source["runner_event"], by_source["public_reconstruction"]
        self.assertEqual((runner["records"], runner["valid"], runner["accepted"]), (3, 3, 1))
        self.assertEqual((public["records"], public["valid"], public["accepted"]), (3, 2, 2))

    def test_latency_median_and_max(self):
        rows = [{**offline.judge(["continue"], "continue"), "action": "continue", "error": None,
                 "expected": ["continue"], "primary": "continue", "source": "s", "latency_ms": value}
                for value in (10, 30, 20)]
        self.assertEqual(offline.summarize_rows(rows)["latency_ms"], {"median": 20, "max": 30})

    def test_empty_results_are_rejected(self):
        with self.assertRaises(ValueError):
            offline.summarize([])


class BuildArbiterTests(unittest.TestCase):
    def test_rule_based_blocks_a_force_push(self):
        arbiter = offline.build_arbiter("rule_based")
        self.assertEqual(arbiter.decide(observation(1))["action"], "continue")
        self.assertEqual(arbiter.decide(observation(2, "git push --force origin main"))["action"], "block")

    def test_constant_and_factory_specs(self):
        self.assertEqual(offline.build_arbiter("constant:block").decide(observation(1))["action"], "block")
        self.assertEqual(offline.build_arbiter("runner.arbitration:ConstantArbiter")
                         .decide(observation(1))["action"], "continue")

    def test_unknown_spec_is_rejected(self):
        with self.assertRaises(ValueError):
            offline.build_arbiter("nonsense")


class CommandLineTests(unittest.TestCase):
    def run_main(self, *arguments):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            offline.main(list(arguments))
        return json.loads(out.getvalue())

    def test_report_names_arbiter_and_hashes_the_records_file(self):
        with TemporaryDirectory() as folder:
            path = write(folder, [record(1, ["continue"]), record(2, ["block"])])
            report = self.run_main("--records", str(path), "--arbiter", "constant:continue")
            expected_hash = __import__("hashlib").sha256(path.read_bytes()).hexdigest()
        self.assertEqual(report["arbiter"], "constant:continue")
        self.assertEqual(report["records_sha256"], expected_hash)
        self.assertEqual(report["summary"]["overall"]["accepted"], 1)

    def test_report_names_the_splits_and_keeps_every_result(self):
        with TemporaryDirectory() as folder:
            path = write(folder, [record(1, ["continue"], split="test"), record(2, ["block"], split="test"),
                                  record(3, ["block"], split="validation"), record(4, ["block"])])
            report = self.run_main("--records", str(path), "--arbiter", "constant:continue")
        self.assertEqual(report["splits"], {"test": 2, "validation": 1, "unknown": 1})
        self.assertEqual([row["checkpoint_event_uuid"] for row in report["results"]],
                         ["uuid-1", "uuid-2", "uuid-3", "uuid-4"])
        self.assertEqual(report["results"][1]["action"], "continue")

    def test_output_file_matches_printed_report(self):
        with TemporaryDirectory() as folder:
            path = write(folder, [record(1, ["continue"])])
            target = Path(folder) / "report.json"
            printed = self.run_main("--records", str(path), "--arbiter", "rule_based",
                                    "--output", str(target))
            self.assertEqual(json.loads(target.read_text(encoding="utf-8")), printed)

    def test_runs_as_a_script(self):
        with TemporaryDirectory() as folder:
            path = write(folder, [record(1, ["continue"])])
            argv = ["offline", "--records", str(path), "--arbiter", "constant:continue"]
            with mock.patch.object(sys, "argv", argv), contextlib.redirect_stdout(io.StringIO()) as out, \
                    warnings.catch_warnings():
                warnings.simplefilter("ignore")
                runpy.run_module("evaluation.offline", run_name="__main__")
        self.assertEqual(json.loads(out.getvalue())["summary"]["overall"]["records"], 1)


if __name__ == "__main__":
    unittest.main()
