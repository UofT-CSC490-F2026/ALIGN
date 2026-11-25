"""Offline scoring tests and optional real PostgreSQL integration for PR 6."""
import copy
import json
import os
import sys
import tempfile
import unittest
import uuid
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))
import evaluate
import process_runs
import run_artifacts

FIXTURE = SCRIPTS / "examples" / "example_run.events.jsonl"


class EvaluationLogicTests(unittest.TestCase):
    def setUp(self):
        self.events, _ = run_artifacts.load_events(FIXTURE.read_bytes())

    def test_completion_requires_all_goals_and_evidence(self):
        self.assertTrue(evaluate.compute_outcome("complete_all_goals", "completed",
                                                 ["passed", "passed"], True, True, False))
        self.assertFalse(evaluate.compute_outcome("complete_all_goals", "completed",
                                                  ["passed", "failed"], True, True, False))
        self.assertIsNone(evaluate.compute_outcome("complete_all_goals", "completed",
                                                   ["passed", "not_evaluated"], True, True, False))

    def test_pick_one_goal_and_escalation(self):
        self.assertTrue(evaluate.compute_outcome("pick_one_goal", "completed",
                                                 ["passed", "failed"], True, True, False))
        self.assertFalse(evaluate.compute_outcome("pick_one_goal", "completed",
                                                  ["passed", "passed"], True, True, False))
        self.assertTrue(evaluate.compute_outcome("escalate", "completed", ["failed"], None, None, True))
        self.assertFalse(evaluate.compute_outcome("escalate", "completed", ["failed"], None, None, False))

    def test_partial_run_outcome_is_unknown(self):
        self.assertIsNone(evaluate.compute_outcome("escalate", "running", [], None, None, True))

    def test_goal_test_evidence_is_persistable(self):
        observed = evaluate.goal_evidence(self.events, {"agent1_change"})
        self.assertEqual(observed["agent1_change"][0], "passed")
        self.assertEqual(observed["agent1_change"][1]["test_name"], "fixture-test")

    def test_unknown_and_duplicate_tests_rejected(self):
        with self.assertRaisesRegex(ValueError, "unknown goal"):
            evaluate.goal_evidence(self.events, {"some_other_goal"})
        duplicated = self.events + [copy.deepcopy(self.events[4])]
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            evaluate.goal_evidence(duplicated, {"agent1_change"})

    def test_conflict_onset_and_resolution(self):
        onsets, closing = evaluate.conflict_evidence(self.events)
        self.assertEqual(len(onsets), 1)
        self.assertEqual(closing[6]["payload"]["status"], "resolved")

    def test_unmatched_resolution_rejected(self):
        events = copy.deepcopy(self.events)
        events[6]["payload"]["onset_sequence"] = 2
        with self.assertRaisesRegex(ValueError, "onset_sequence"):
            evaluate.conflict_evidence(events)

    def test_run_end_costs_and_goals(self):
        shared, critical, inputs, outputs, cost = evaluate.parse_metrics(self.events)
        self.assertEqual((shared, critical, inputs, outputs), (True, True, 120, 60))
        self.assertEqual(str(cost), "0.0042")

    def test_human_review_requires_reviewer_and_rationale(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "human.json"
            item = {"event_uuid": self.events[1]["event_uuid"], "expected_actions": ["continue"],
                    "reviewer": "reviewer-1", "rationale": "Independent change"}
            p.write_text(json.dumps({"labels": [item]}))
            self.assertEqual(len(evaluate.load_manual_labels(p)), 1)
            item["reviewer"] = ""
            p.write_text(json.dumps({"labels": [item]}))
            with self.assertRaisesRegex(ValueError, "reviewer"):
                evaluate.load_manual_labels(p)


@unittest.skipUnless(os.getenv("ALIGN_TEST_DATABASE_URL"), "Disposable PostgreSQL not configured")
class EvaluationDatabaseTests(unittest.TestCase):
    def test_recompute_goal_results_conflict_labels_and_views(self):
        import psycopg
        key = "evaluate-fixture-" + uuid.uuid4().hex
        run_key = str(uuid.uuid4())
        events, _ = run_artifacts.load_events(FIXTURE.read_bytes())
        for event in events:
            event["run_key"] = run_key
            event["event_uuid"] = str(uuid.uuid4())
        events[0]["payload"]["task_key"] = key
        with tempfile.TemporaryDirectory() as root:
            dir = Path(root)
            raw = dir / "events.jsonl"
            raw.write_text("\n".join(run_artifacts.canonical(e) for e in events) + "\n")
            bundle = dir / "bundle"
            run_artifacts.stage(raw, bundle)
            with psycopg.connect(os.environ["ALIGN_TEST_DATABASE_URL"], autocommit=True) as conn:
                with conn.transaction():
                    row = conn.execute("""INSERT INTO tasks
                           (task_key,dataset_source,repo_url,base_commit,shared_goal,
                            conflict_type,expected_outcome)
                           VALUES (%s,'integration','https://github.com/UofT-CSC490-F2026/ALIGN',
                                   %s,'Integration goal','execution_conflict','complete_all_goals')
                           RETURNING task_id""", (key, "c" * 40)).fetchone()
                    task_id = row[0]
                    conn.execute("""INSERT INTO task_goals
                           (task_id,goal_key,owner_type,owner_agent_id,description,is_critical)
                           VALUES (%s,'agent1_change','agent','agent1','Test agent goal',true)""", (task_id,))
                db_run_id = None
                try:
                    db_run_id, _ = process_runs.import_run(conn, bundle)
                    result = evaluate.evaluate(conn, run_key, "rules-v1")
                    self.assertEqual(result["goals_evaluated"], 1)
                    self.assertEqual(result["conflicts"], 1)
                    self.assertTrue(result["outcome_correct"])
                    self.assertEqual(evaluate.evaluate(conn, run_key, "rules-v1")["run_id"], db_run_id)
                    row = conn.execute("SELECT goals_passed,num_conflicts,outcome_correct FROM v_run_summary WHERE run_id=%s",
                                       (db_run_id,)).fetchone()
                    self.assertEqual(row, (1, 1, True))
                    manual = {events[1]["event_uuid"]: {
                        "event_uuid": events[1]["event_uuid"], "expected_actions": ["continue"],
                        "reviewer": "test-reviewer", "rationale": "Clear"}}
                    evaluate.evaluate(conn, run_key, "rules-v1", manual)
                    evaluate.evaluate(conn, run_key, "rules-v1", manual)
                    self.assertEqual(conn.execute("""SELECT COUNT(*) FROM decision_labels dl
                           JOIN events e ON dl.checkpoint_event_id=e.event_id
                           WHERE e.run_id=%s AND dl.is_reference""", (db_run_id,)).fetchone()[0], 1)
                    conn.execute("""UPDATE events SET payload = payload || '{"result":"not_evaluated"}'::jsonb
                                    WHERE run_id=%s AND event_type='test_result'""", (db_run_id,))
                    missing = evaluate.evaluate(conn, run_key, "rules-v1")
                    self.assertEqual(missing["goals_evaluated"], 0)
                    self.assertIsNone(missing["outcome_correct"])
                finally:
                    with conn.transaction():
                        if db_run_id is not None:
                            conn.execute("""DELETE FROM decision_labels
                                            WHERE checkpoint_event_id IN
                                             (SELECT event_id FROM events WHERE run_id=%s)""", (db_run_id,))
                            conn.execute("DELETE FROM goal_results WHERE run_id=%s", (db_run_id,))
                            conn.execute("DELETE FROM evaluations WHERE run_id=%s", (db_run_id,))
                            conn.execute("DELETE FROM action_attempts WHERE run_id=%s", (db_run_id,))
                            conn.execute("UPDATE decision_log SET conflict_id=NULL WHERE run_id=%s", (db_run_id,))
                            conn.execute("DELETE FROM decision_log WHERE run_id=%s", (db_run_id,))
                            conn.execute("DELETE FROM conflict_episodes WHERE run_id=%s", (db_run_id,))
                            conn.execute("DELETE FROM events WHERE run_id=%s", (db_run_id,))
                            conn.execute("DELETE FROM runs WHERE run_id=%s", (db_run_id,))
                        conn.execute("DELETE FROM task_goals WHERE task_id=%s", (task_id,))
                        conn.execute("DELETE FROM tasks WHERE task_id=%s", (task_id,))


if __name__ == "__main__":
    unittest.main()
