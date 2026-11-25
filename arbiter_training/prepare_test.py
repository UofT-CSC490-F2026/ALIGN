"""Prepare an independently labeled held-out test file for offline scoring.

Only tasks already marked test in the split manifest are accepted. This script
does not produce training or validation records.
"""

import argparse
import json
from pathlib import Path

from .prepare import build_records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--events-dir", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--history-limit", type=int, default=8)
    args = parser.parse_args()
    records = build_records(args.tasks, args.events_dir, args.labels,
                            args.history_limit, include_test=True)
    if not records:
        raise ValueError("No labeled held-out checkpoints found")
    if any(row["split"] != "test" for row in records):
        raise ValueError("Test preparation received a non-test task")
    if any(row["label_source"] != "human" or row["review_status"] != "reviewed"
           for row in records):
        raise ValueError("Held-out reference labels must be independently reviewed")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as stream:
        for row in records:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    print(json.dumps({"test_records": len(records),
                      "test_tasks": len({row["task_key"] for row in records}),
                      "output": str(args.output)}))


if __name__ == "__main__":
    main()
