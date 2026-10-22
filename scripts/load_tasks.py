"""Validate a normalized ALIGN tasks.jsonl snapshot and optionally import into PostgreSQL.

No AWS access. Writes require --apply and ALIGN_DATABASE_URL. Each batch is atomic.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from normalize_tasks import validate_task

TASK_COLUMNS = (
    "task_key", "dataset_source", "external_task_id", "version", "repo_url",
    "base_commit", "shared_goal", "conflict_type", "expected_outcome",
    "constraints", "setup_spec", "metadata",
)
GOAL_COLUMNS = (
    "goal_key", "owner_type", "owner_agent_id", "description",
    "success_criteria", "is_critical",
)
JSON_COLUMNS = {"constraints", "setup_spec", "metadata", "success_criteria"}

TASK_INSERT = """
INSERT INTO tasks (
    task_key, dataset_source, external_task_id, version, repo_url, base_commit,
    shared_goal, conflict_type, expected_outcome, constraints, setup_spec, metadata
)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s::jsonb, %s::jsonb)
ON CONFLICT (task_key) DO NOTHING
RETURNING task_id
"""
TASK_SELECT = """
SELECT task_id, task_key, dataset_source, external_task_id, version, repo_url,
       base_commit, shared_goal, conflict_type, expected_outcome,
       constraints, setup_spec, metadata
FROM tasks WHERE task_key = %s FOR UPDATE
"""
GOAL_INSERT = """
INSERT INTO task_goals (
    task_id, goal_key, owner_type, owner_agent_id, description, success_criteria, is_critical
)
VALUES (%s, %s, %s, %s, %s, %s::jsonb, %s)
"""
GOAL_SELECT = """
SELECT goal_key, owner_type, owner_agent_id, description, success_criteria, is_critical
FROM task_goals WHERE task_id = %s ORDER BY goal_key
"""


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def read_batch(input_dir: Path) -> list[dict]:
    """Verify completion manifest and strict v1 schema before any database writes."""
    manifest_path = input_dir / "manifest.json"
    tasks_path = input_dir / "tasks.jsonl"
    _check(manifest_path.is_file() and tasks_path.is_file(),
           "Expected both manifest.json and tasks.jsonl in input directory")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _check(manifest.get("schema_version") == 1, "Unsupported manifest schema_version")
    _check(isinstance(manifest.get("dataset_source"), str) and bool(manifest["dataset_source"]),
           "Invalid manifest dataset_source")
    _check(isinstance(manifest.get("source_revision"), str) and bool(manifest["source_revision"]),
           "Invalid manifest source_revision")
    raw = tasks_path.read_bytes()
    _check(hashlib.sha256(raw).hexdigest() == manifest.get("tasks_sha256"),
           "tasks.jsonl checksum mismatch")
    rows = []
    seen_keys: set[str] = set()
    seen_external: set[tuple[str, str, int]] = set()
    for lineno, line in enumerate(raw.decode("utf-8").splitlines(), 1):
        _check(bool(line.strip()), f"Blank tasks.jsonl line {lineno}")
        row = json.loads(line)
        _check(isinstance(row, dict), f"Task at line {lineno} is not an object")
        validate_task(row)
        _check(set(row) == set(TASK_COLUMNS) | {"schema_version", "goals"},
               f"Unexpected or missing task fields at line {lineno}")
        _check(row["dataset_source"] == manifest["dataset_source"],
               f"Dataset source mismatch at line {lineno}")
        _check(row["metadata"].get("source_revision") == manifest["source_revision"],
               f"Source revision mismatch at line {lineno}")
        _check(row["task_key"] not in seen_keys, f"Duplicate task_key: {row['task_key']}")
        seen_keys.add(row["task_key"])
        external = (row["dataset_source"], row["external_task_id"], row["version"])
        _check(external not in seen_external, f"Duplicate external task identity: {external}")
        seen_external.add(external)
        for goal in row["goals"]:
            _check(set(goal) == set(GOAL_COLUMNS),
                   f"Unexpected or missing goal fields in {row['task_key']}")
        rows.append(row)
    _check(bool(rows) and type(manifest.get("task_count")) is int and
           manifest["task_count"] == len(rows), "Manifest task_count mismatch")
    return rows


def _values(record: dict, columns: tuple[str, ...]) -> tuple:
    return tuple(json.dumps(record[column], sort_keys=True, ensure_ascii=False)
                 if column in JSON_COLUMNS else record[column] for column in columns)


def import_batch(conn, records: list[dict]) -> tuple[int, int]:
    """Atomically insert new tasks and goals; reject any changed existing task."""
    inserted = unchanged = 0
    with conn.transaction():
        with conn.cursor() as cursor:
            for task in records:
                cursor.execute(TASK_INSERT, _values(task, TASK_COLUMNS))
                result = cursor.fetchone()
                if result is not None:
                    task_id = result[0]
                    for goal in task["goals"]:
                        cursor.execute(GOAL_INSERT, (task_id, *_values(goal, GOAL_COLUMNS)))
                    inserted += 1
                    continue

                cursor.execute(TASK_SELECT, (task["task_key"],))
                found = cursor.fetchone()
                if found is None:
                    raise RuntimeError(f"Task vanished during concurrent import: {task['task_key']}")
                task_id = found[0]
                actual = dict(zip(TASK_COLUMNS, found[1:]))
                for field in TASK_COLUMNS:
                    if actual[field] != task[field]:
                        raise ValueError(f"Existing task differs: {task['task_key']} ({field})")

                cursor.execute(GOAL_SELECT, (task_id,))
                existing_goals = [dict(zip(GOAL_COLUMNS, row)) for row in cursor.fetchall()]
                expected_goals = sorted(task["goals"], key=lambda goal: goal["goal_key"])
                if existing_goals != expected_goals:
                    raise ValueError(f"Existing goals differ: {task['task_key']}")
                unchanged += 1
    return inserted, unchanged


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True,
                        help="Directory containing normalized tasks.jsonl and manifest.json")
    parser.add_argument("--apply", action="store_true",
                        help="Actually write to PostgreSQL; default only validates locally")
    args = parser.parse_args()
    records = read_batch(args.input_dir)
    print(f"Validated {len(records)} normalized tasks")
    if not args.apply:
        print("DRY RUN: no database connection or writes")
        return
    dsn = os.environ.get("ALIGN_DATABASE_URL")
    if not dsn:
        parser.error("Set ALIGN_DATABASE_URL before --apply")
    import psycopg
    with psycopg.connect(dsn, autocommit=True, connect_timeout=10) as conn:
        created, existing = import_batch(conn, records)
    print(f"Committed {created} new tasks; {existing} unchanged")


if __name__ == "__main__":
    main()
