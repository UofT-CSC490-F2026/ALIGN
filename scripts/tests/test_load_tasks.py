"""PR-3 loader contract tests; live DB test runs only when explicitly configured."""

import io
import runpy
from unittest.mock import MagicMock, patch
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
import normalize_tasks
import load_tasks

BASE = "829219e06ebe74c8872b59dd78dfefef37cdca00"


def scenario(source_revision="v1", task_id="example", version=1):
    return {
        "schema_version": 1, "task_key": f"align-custom:{task_id}:v{version}:{source_revision}",
        "dataset_source": "align-custom", "external_task_id": f"{task_id}@{source_revision}",
        "version": version, "repo_url": "https://github.com/UofT-CSC490-F2026/ALIGN",
        "base_commit": BASE, "shared_goal": "Coordinate changes", "conflict_type": "execution_conflict",
        "expected_outcome": "escalate", "constraints": {"protected_paths": ["tests/"]},
        "setup_spec": {"execution_mode": "fixture_only"},
        "metadata": {"source_revision": source_revision, "scenario_type": "custom"},
        "goals": [{"goal_key": "agent1_change", "owner_type": "agent",
                   "owner_agent_id": "agent1", "description": "Propose a safe edit",
                   "success_criteria": {"type": "manual_review"}, "is_critical": True}],
    }


class FakeTransaction:
    def __init__(self, conn):
        self.conn = conn

    def __enter__(self):
        self.before = copy.deepcopy(self.conn.tasks)
        self.before_goals = copy.deepcopy(self.conn.goals)
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc_type:
            self.conn.tasks = self.before
            self.conn.goals = self.before_goals
        return False


class FakeCursor:
    def __init__(self, conn):
        self.conn = conn
        self.last = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def execute(self, sql, params):
        if sql == load_tasks.TASK_INSERT:
            values = dict(zip(load_tasks.TASK_COLUMNS, params))
            for field in ("constraints", "setup_spec", "metadata"):
                values[field] = json.loads(values[field])
            key = values["task_key"]
            if key in self.conn.tasks:
                self.last = [None]
            else:
                ident = len(self.conn.tasks) + 1
                self.conn.tasks[key] = (ident, values)
                self.last = [(ident,)]
        elif sql == load_tasks.GOAL_INSERT:
            ident = params[0]
            values = dict(zip(load_tasks.GOAL_COLUMNS, params[1:]))
            values["success_criteria"] = json.loads(values["success_criteria"])
            self.conn.goals.setdefault(ident, []).append(values)
        elif sql == load_tasks.TASK_SELECT:
            entry = self.conn.tasks.get(params[0])
            self.last = [(entry[0], *(entry[1][k] for k in load_tasks.TASK_COLUMNS))] if entry else [None]
        elif sql == load_tasks.GOAL_SELECT:
            self.last = [tuple(row[k] for k in load_tasks.GOAL_COLUMNS)
                         for row in sorted(self.conn.goals.get(params[0], []), key=lambda x: x["goal_key"])]
        else:
            raise AssertionError(f"Unexpected SQL: {sql}")

    def fetchone(self):
        return self.last.pop(0)

    def fetchall(self):
        return list(self.last)


class FakeConnection:
    def __init__(self):
        self.tasks = {}
        self.goals = {}

    def transaction(self):
        return FakeTransaction(self)

    def cursor(self):
        return FakeCursor(self)


class LoaderTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)
        self.task = scenario()
        normalize_tasks.write_output([self.task], self.path, "align-custom", "v1", "all")

    def test_disappearing_conflicting_row_aborts_import(self):
        conn = MagicMock()
        conn.cursor.return_value.__enter__.return_value.fetchone.return_value = None
        with self.assertRaisesRegex(RuntimeError, "Task vanished"):
            load_tasks.import_batch(conn, [self.task])

    def test_cli_dry_run_and_apply(self):
        args = [load_tasks.__file__, "--input-dir", str(self.path)]
        db = FakeConnection()
        with patch("psycopg.connect") as connect:
            connect.return_value.__enter__.return_value = db
            with patch.object(sys, "argv", args), patch("sys.stdout", new_callable=io.StringIO) as output:
                runpy.run_path(load_tasks.__file__, run_name="__main__")
            self.assertIn("DRY RUN", output.getvalue())
            connect.assert_not_called()
            with patch.object(sys, "argv", args + ["--apply"]):
                with patch.dict(os.environ, {"ALIGN_DATABASE_URL": ""}), patch("sys.stderr", new_callable=io.StringIO), self.assertRaises(SystemExit):
                    load_tasks.main()
                connect.assert_not_called()
                with patch.dict(os.environ, {"ALIGN_DATABASE_URL": "fixture"}), patch("sys.stdout", new_callable=io.StringIO) as output:
                    load_tasks.main()
            self.assertIn("Committed 1 new tasks", output.getvalue())
            connect.assert_called_once_with("fixture", autocommit=True, connect_timeout=10)

    def test_manifest_checksum_and_parse(self):
        self.assertEqual(load_tasks.read_batch(self.path), [self.task])

    def test_corrupted_jsonl_is_rejected(self):
        (self.path / "tasks.jsonl").write_text("{}", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
            load_tasks.read_batch(self.path)

    def test_missing_manifest_rejected(self):
        (self.path / "manifest.json").unlink()
        with self.assertRaisesRegex(ValueError, "Expected both"):
            load_tasks.read_batch(self.path)

    def test_mismatched_source_revision_rejected(self):
        task = scenario("v2")
        other = self.path / "mismatch"
        normalize_tasks.write_output([task], other, "align-custom", "v1", "all")
        with self.assertRaisesRegex(ValueError, "Source revision mismatch"):
            load_tasks.read_batch(other)

    def test_duplicate_task_keys_rejected_before_sql(self):
        records = [self.task, self.task]
        other = self.path / "dups"
        other.mkdir()
        rows = "\n".join(normalize_tasks.canonical_json(x) for x in records) + "\n"
        raw = rows.encode("utf-8")
        (other / "tasks.jsonl").write_bytes(raw)
        import hashlib
        (other / "manifest.json").write_text(json.dumps({"schema_version": 1, "dataset_source": "align-custom", "source_revision": "v1", "task_count": 2, "tasks_sha256": hashlib.sha256(raw).hexdigest()}), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Duplicate task_key"):
            load_tasks.read_batch(other)

    def test_unknown_fields_rejected_instead_of_ignored(self):
        task = copy.deepcopy(self.task)
        task["unexpected"] = "not stored in SQL"
        other = self.path / "extra"
        normalize_tasks.write_output([task], other, "align-custom", "v1", "all")
        with self.assertRaisesRegex(ValueError, "Unexpected or missing task fields"):
            load_tasks.read_batch(other)

    def test_new_and_repeat_import_are_idempotent(self):
        conn = FakeConnection()
        self.assertEqual(load_tasks.import_batch(conn, [self.task]), (1, 0))
        self.assertEqual(load_tasks.import_batch(conn, [self.task]), (0, 1))
        self.assertEqual(len(conn.tasks), 1)
        self.assertEqual(len(conn.goals[1]), 1)

    def test_task_mutation_fails_and_rolls_back_prior_insert(self):
        conn = FakeConnection()
        load_tasks.import_batch(conn, [self.task])
        modified = copy.deepcopy(self.task)
        modified["shared_goal"] = "Modified content with unchanged key"
        brand_new = scenario(task_id="a-new-task")
        with self.assertRaisesRegex(ValueError, "Existing task differs"):
            load_tasks.import_batch(conn, [brand_new, modified])
        self.assertEqual(len(conn.tasks), 1)
        self.assertEqual(len(conn.goals), 1)

    def test_goal_mutation_fails_and_rolls_back(self):
        conn = FakeConnection()
        load_tasks.import_batch(conn, [self.task])
        modified = copy.deepcopy(self.task)
        modified["goals"][0]["description"] = "Changed goal"
        with self.assertRaisesRegex(ValueError, "Existing goals differ"):
            load_tasks.import_batch(conn, [modified])
        self.assertEqual(len(conn.goals[1]), 1)
        self.assertEqual(conn.goals[1][0]["description"], "Propose a safe edit")


@unittest.skipUnless(os.environ.get("ALIGN_TEST_DATABASE_URL"),
                     "Set ALIGN_TEST_DATABASE_URL for isolated PostgreSQL integration test")
class PostgresLoaderIntegration(unittest.TestCase):
    def test_insert_rerun_conflict_and_atomic_rollback(self):
        import psycopg
        task = scenario(task_id=f"integration-{uuid.uuid4().hex}")
        another = scenario(task_id=f"integration-{uuid.uuid4().hex}")
        with psycopg.connect(os.environ["ALIGN_TEST_DATABASE_URL"], autocommit=True) as conn:
            try:
                self.assertEqual(load_tasks.import_batch(conn, [task]), (1, 0))
                self.assertEqual(load_tasks.import_batch(conn, [task]), (0, 1))
                with conn.cursor() as cur:
                    cur.execute("SELECT count(*) FROM task_goals g JOIN tasks t USING (task_id) WHERE t.task_key = %s", (task["task_key"],))
                    self.assertEqual(cur.fetchone()[0], 1)
                modified = copy.deepcopy(task)
                modified["goals"][0]["description"] = "Not the same goal"
                with self.assertRaisesRegex(ValueError, "Existing goals differ"):
                    load_tasks.import_batch(conn, [another, modified])
                with conn.cursor() as cur:
                    cur.execute("SELECT count(*) FROM tasks WHERE task_key = %s", (another["task_key"],))
                    self.assertEqual(cur.fetchone()[0], 0)
            finally:
                with conn.transaction():
                    with conn.cursor() as cur:
                        for key in (task["task_key"], another["task_key"]):
                            cur.execute("DELETE FROM task_goals WHERE task_id IN (SELECT task_id FROM tasks WHERE task_key = %s)", (key,))
                            cur.execute("DELETE FROM tasks WHERE task_key = %s", (key,))


if __name__ == "__main__":
    unittest.main()
