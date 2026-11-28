"""Overlay completed human reviews on provisional teacher labels."""

import argparse
import json
from pathlib import Path

from .contract import ACTIONS
from .prepare import read_jsonl


def merge(teacher_rows, review_rows):
    labels = {row["event_uuid"]: dict(row) for row in teacher_rows}
    if len(labels) != len(teacher_rows):
        raise ValueError("Duplicate teacher event UUID")
    reviewed = set()
    changed = 0
    for row in review_rows:
        event_uuid = row["event_uuid"]
        if event_uuid not in labels or event_uuid in reviewed:
            raise ValueError(f"Unknown or duplicate review event UUID: {event_uuid}")
        actions = row.get("expected_actions")
        primary = row.get("primary_action")
        rationale = row.get("rationale")
        reviewer = row.get("reviewer")
        if not actions and not primary and not rationale and not reviewer:
            continue
        if (not isinstance(actions, list) or not actions or len(set(actions)) != len(actions)
                or not set(actions) <= ACTIONS or primary not in actions
                or not isinstance(rationale, str) or not rationale.strip()
                or not isinstance(reviewer, str) or not reviewer.strip()):
            raise ValueError(f"Incomplete human review for {event_uuid}")
        old = labels[event_uuid]
        changed += primary != old.get("primary_action", old["expected_actions"][0])
        labels[event_uuid] = {
            "event_uuid": event_uuid,
            "expected_actions": actions,
            "primary_action": primary,
            "reviewer": reviewer,
            "rationale": rationale.strip(),
            "label_source": "human",
            "review_status": "reviewed",
        }
        reviewed.add(event_uuid)
    if not reviewed:
        raise ValueError("No completed human reviews in packet")
    return list(labels.values()), {"reviewed": len(reviewed),
                                   "changed_primary_actions": changed,
                                   "remaining_provisional": len(labels) - len(reviewed)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-labels", type=Path, required=True)
    parser.add_argument("--reviewed-packet", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    teacher_rows = list(read_jsonl(args.teacher_labels))
    review_rows = list(read_jsonl(args.reviewed_packet))
    merged, summary = merge(teacher_rows, review_rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as stream:
        for row in merged:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
