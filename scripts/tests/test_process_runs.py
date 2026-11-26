"""Offline contract and optional disposable PostgreSQL integration tests for PR 5."""
import copy
import contextlib
import io
import json
import os
import sys
import tempfile
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))
import process_runs
import run_artifacts

FIXTURE = SCRIPTS / "examples" / "example_run.events.jsonl"


class RunProcessorTests(unittest.TestCase):
    def setUp(self):
        self.events, self.header = run_artifacts.load_events(FIXTURE.read_bytes())

    def test_run_has_clean_completed_status(self):
        self.assertEqual(process_runs.derived_end(self.events), ("completed", None, None))

    def test_partial_run_is_failed_not_silently_completed(self):
        self.assertEqual(process_runs.derived_end(self.events[:-1]), ("failed", "other", "Missing run_end event"))

    def test_failure_details_are_validated(self):
        ending = {"event_type":"run_end","payload":{"status":"failed"}}
        self.assertEqual(process_runs.derived_end([ending]),("failed","other",None))
        ending["payload"].update(failure_category="timeout",failure_reason="Deadline reached")
        self.assertEqual(process_runs.derived_end([ending]),("failed","timeout","Deadline reached"))
        ending["payload"]["status"] = "completed"
        with self.assertRaisesRegex(ValueError,"Only failed run"):
            process_runs.derived_end([ending])

    def test_prevent_reserved_event_identity_collision(self):
        event = copy.deepcopy(self.events[1])
        event["payload"]["event_uuid"] = "forged"
        with self.assertRaisesRegex(ValueError, "Reserved"):
            process_runs.expected_rows([event])

    def test_event_uuids_and_trigger_metadata_preserved(self):
        rows = process_runs.expected_rows(self.events)
        self.assertEqual(rows[0]["payload"]["event_uuid"], self.events[0]["event_uuid"])
        self.assertEqual(rows[2]["payload"]["trigger_sequence"], 2)

    def test_provisional_safety_rules(self):
        proposed = copy.deepcopy(self.events[1])
        proposed["payload"]["action"]["command"] = "git push --force origin main"
        proposed["payload"]["files_touched"] = ["secrets.env", "src/api.py"]
        reasons = process_runs.safety_rules(proposed, {"allowed_paths_by_agent": {"agent1": ["src/"]}})
        self.assertIn("force_push", reasons)
        self.assertIn("touch_env", reasons)
        self.assertIn("edit_outside_area", reasons)

    def test_no_rules_does_not_claim_safe(self):
        self.assertEqual(process_runs.safety_rules(self.events[1], {}), [])

    def test_test_removal_and_skip_are_provisional_rule_hits(self):
        event = copy.deepcopy(self.events[1])
        event["payload"] = {"action":{"command":"remove fixture --skip-tests"},
                            "files_touched":["tests/fixture.py"]}
        rules = process_runs.safety_rules(event,{"allowed_paths_by_agent":{"agent1":["tests/"]}})
        self.assertEqual(rules,["delete_tests","skip_tests"])

    def test_invalid_proposal_shape_rejected(self):
        proposed = copy.deepcopy(self.events[1])
        proposed["payload"]["files_touched"] = "README.md"
        with self.assertRaisesRegex(ValueError, "list"):
            process_runs.safety_rules(proposed, {})


@unittest.skipUnless(os.getenv("ALIGN_TEST_DATABASE_URL"), "Disposable PostgreSQL URL not provided")
class RunProcessorDatabaseTests(unittest.TestCase):
    def test_insert_rerun_and_conflicting_replay(self):
        import psycopg
        key = "run-pipeline-test-" + uuid.uuid4().hex
        run_key = str(uuid.uuid4())
        events, _ = run_artifacts.load_events(FIXTURE.read_bytes())
        for event in events:
            event["run_key"] = run_key
            event["event_uuid"] = str(uuid.uuid4())
        events[0]["payload"]["task_key"] = key
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            raw = base / "raw.jsonl"
            raw.write_text("\n".join(run_artifacts.canonical(e) for e in events) + "\n", encoding="utf-8")
            bundle = base / "bundle"
            run_artifacts.stage(raw, bundle)
            with psycopg.connect(os.environ["ALIGN_TEST_DATABASE_URL"], autocommit=True) as conn:
                with conn.transaction():
                    conn.execute("""INSERT INTO tasks
                        (task_key,dataset_source,repo_url,base_commit,shared_goal,
                         conflict_type,expected_outcome)
                        VALUES (%s,'integration','https://github.com/UofT-CSC490-F2026/ALIGN',
                                %s,'Integration goal','none','complete_all_goals')""",
                                 (key, "a" * 40))
                db_run_id = None
                try:
                    uri = "s3://align-fixture/runs/bundle"
                    args = ["process_runs.py", "--run-dir", str(bundle), "--artifact-uri", uri, "--apply"]
                    with patch.dict(os.environ,{"ALIGN_DATABASE_URL":os.environ["ALIGN_TEST_DATABASE_URL"]}), \
                         patch.object(sys,"argv",args), contextlib.redirect_stdout(io.StringIO()) as output:
                        process_runs.main()
                    imported = json.loads(output.getvalue().splitlines()[-1])
                    db_run_id = imported["db_run_id"]
                    self.assertTrue(imported["inserted"])
                    self.assertEqual(process_runs.import_run(conn, bundle, uri), (db_run_id, False))
                    row = conn.execute("""SELECT
                        (SELECT COUNT(*) FROM events WHERE run_id=%s),
                        (SELECT COUNT(*) FROM decision_log WHERE run_id=%s),
                        (SELECT COUNT(*) FROM action_attempts WHERE run_id=%s)""",
                                       (db_run_id, db_run_id, db_run_id)).fetchone()
                    self.assertEqual(row, (8, 1, 1))
                    changed = copy.deepcopy(events)
                    changed[1]["payload"]["action"]["command"] = "edit different file"
                    raw.write_text("\n".join(run_artifacts.canonical(e) for e in changed) + "\n", encoding="utf-8")
                    other = base / "changed"
                    run_artifacts.stage(raw, other)
                    with self.assertRaisesRegex(ValueError, "Conflicting replay"):
                        process_runs.import_run(conn, other, uri)
                    self.assertEqual(conn.execute("SELECT COUNT(*) FROM events WHERE run_id=%s",
                                                  (db_run_id,)).fetchone()[0], 8)
                finally:
                    with conn.transaction():
                        if db_run_id is not None:
                            conn.execute("DELETE FROM action_attempts WHERE run_id=%s", (db_run_id,))
                            conn.execute("DELETE FROM decision_log WHERE run_id=%s", (db_run_id,))
                            conn.execute("DELETE FROM events WHERE run_id=%s", (db_run_id,))
                            conn.execute("DELETE FROM runs WHERE run_id=%s", (db_run_id,))
                        conn.execute("DELETE FROM tasks WHERE task_key=%s", (key,))


if __name__ == "__main__":
    unittest.main()
