"""Make provenance-tracked decision-level SFT records from A2 run bundles."""

import argparse
import hashlib
import json
from pathlib import Path

from .contract import (ACTIONS, SCHEMA_VERSION, decision_for_observation,
                       messages_for_observation, observation_from_checkpoint)


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if line.strip():
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc
                if not isinstance(value, dict):
                    raise ValueError(f"Expected an object at {path}:{line_number}")
                yield value


def task_context(task):
    # Metadata, setup specifications, hidden tests, and gold patches are excluded.
    return {
        "task_key": task["task_key"],
        "repo_url": task.get("repo_url"),
        "base_commit": task.get("base_commit"),
        "shared_goal": task["shared_goal"],
        "conflict_type": task.get("conflict_type"),
        "constraints": task.get("constraints", {}),
        "goals": [
            {"goal_key": g.get("goal_key"), "owner_agent_id": g.get("owner_agent_id"),
             "description": g["description"]}
            for g in task.get("goals", [])
        ],
    }


def validated_labels(path):
    labels = {}
    for item in read_jsonl(path):
        event_uuid = item.get("event_uuid")
        actions = item.get("expected_actions")
        reviewer = item.get("reviewer")
        rationale = item.get("rationale")
        source = item.get("label_source", "human")
        review_status = item.get("review_status", "reviewed" if source == "human" else "provisional")
        if not isinstance(event_uuid, str) or not isinstance(actions, list) or not actions:
            raise ValueError("Each label needs event_uuid and nonempty expected_actions")
        if len(set(actions)) != len(actions) or not set(actions) <= ACTIONS:
            raise ValueError(f"Invalid actions for {event_uuid}")
        if source not in {"human", "teacher_gpt"} or review_status not in {"reviewed", "provisional"}:
            raise ValueError(f"Invalid label provenance for {event_uuid}")
        if source == "human" and (not isinstance(reviewer, str) or not reviewer.strip()):
            raise ValueError(f"Human label {event_uuid} needs a reviewer")
        if not isinstance(rationale, str) or not rationale.strip():
            raise ValueError(f"Label {event_uuid} needs a rationale")
        primary = item.get("primary_action", actions[0])
        if primary not in actions:
            raise ValueError(f"Primary action must be acceptable for {event_uuid}")
        if event_uuid in labels:
            raise ValueError(f"Duplicate label for {event_uuid}")
        labels[event_uuid] = (actions, primary, reviewer, rationale, source, review_status)
    return labels


def build_records(tasks_path, events_dir, labels_path, history_limit=20,
                  excluded_tasks=frozenset(), include_test=False):
    tasks = {}
    task_splits = {}
    for task in read_jsonl(tasks_path):
        key = task["task_key"]
        if key in tasks:
            raise ValueError(f"Duplicate task_key {key}")
        tasks[key] = task_context(task)
        task_splits[key] = task.get("metadata", {}).get("split")
    labels = validated_labels(labels_path)
    used = set()
    withheld = set()
    records = []
    files = sorted(Path(events_dir).rglob("events.jsonl"))
    if not files:
        raise ValueError("No events.jsonl bundles found")
    for path in files:
        events = list(read_jsonl(path))
        if not events or events[0].get("event_type") != "run_start":
            raise ValueError(f"Missing run_start in {path}")
        task_key = events[0].get("payload", {}).get("task_key")
        if task_key not in tasks:
            raise ValueError(f"Unknown task_key {task_key} in {path}")
        split = task_splits[task_key]
        if task_key in excluded_tasks or (split == "test" and not include_test):
            withheld.update(e.get("event_uuid") for e in events
                            if e.get("event_type") == "proposed_action")
            continue
        run_key = events[0].get("run_key")
        history = []
        for index, event in enumerate(events, 1):
            if event.get("run_key") != run_key or event.get("sequence_number") != index:
                raise ValueError(f"Invalid run identity or event order in {path}")
            if event.get("event_type") == "proposed_action" and event.get("event_uuid") in labels:
                event_uuid = event["event_uuid"]
                if event_uuid in used:
                    raise ValueError(f"Repeated labeled event_uuid {event_uuid}")
                actions, primary, reviewer, rationale, source, review_status = labels[event_uuid]
                checkpoint = {
                    "schema_version": SCHEMA_VERSION,
                    "task": tasks[task_key],
                    "run_key": run_key,
                    "checkpoint_event_uuid": event_uuid,
                    "sequence_number": index,
                    "agent_id": event.get("actor_id"),
                    "proposal": event.get("payload", {}),
                    "prior_events": history[-history_limit:],
                }
                observation = observation_from_checkpoint(checkpoint)
                messages = messages_for_observation(observation)
                records.append({
                    "schema_version": SCHEMA_VERSION,
                    "task_key": task_key,
                    "run_key": run_key,
                    "checkpoint_event_uuid": event_uuid,
                    "source_bundle": str(path),
                    "expected_actions": actions,
                    "primary_action": primary,
                    "reviewer": reviewer,
                    "label_source": source,
                    "review_status": review_status,
                    "split": split,
                    "prompt": messages,
                    "completion": [{"role": "assistant", "content": json.dumps(
                        decision_for_observation(primary, rationale, observation),
                        ensure_ascii=False)}],
                })
                used.add(event_uuid)
            history.append({
                "sequence_number": index,
                "actor_type": event.get("actor_type"),
                "actor_id": event.get("actor_id"),
                "event_type": event.get("event_type"),
                "payload": event.get("payload", {}),
            })
    missing = set(labels) - used - withheld
    if missing:
        raise ValueError(f"Labels without matching proposals: {len(missing)}")
    return records


def split_records(records, validation_fraction):
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between zero and one")
    train, validation = [], []
    for record in records:
        # Split entire tasks together; the evaluation owner's held-out tasks stay outside input.
        digest = hashlib.sha256(record["task_key"].encode()).digest()
        fraction = int.from_bytes(digest[:8], "big") / 2**64
        if record.get("split") == "validation":
            validation.append(record)
        elif record.get("split") == "train":
            train.append(record)
        else:
            (validation if fraction < validation_fraction else train).append(record)
    return train, validation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, required=True, help="Normalized A2 tasks.jsonl")
    parser.add_argument("--events-dir", type=Path, required=True, help="Directory of events.jsonl bundles")
    parser.add_argument("--labels", type=Path, required=True, help="Reviewed A2 human-label JSONL")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--history-limit", type=int, default=20)
    parser.add_argument("--exclude-task-keys", type=Path,
                        help="One held-out task key per line; none are exported")
    args = parser.parse_args()
    if args.history_limit < 0:
        parser.error("history-limit must be nonnegative")
    excluded = set()
    if args.exclude_task_keys:
        excluded = {line.strip() for line in args.exclude_task_keys.read_text(encoding="utf-8").splitlines()
                    if line.strip() and not line.lstrip().startswith("#")}
    records = build_records(args.tasks, args.events_dir, args.labels, args.history_limit, excluded)
    if not records:
        parser.error("No labeled checkpoint examples were found")
    train, validation = split_records(records, args.validation_fraction)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in (("train", train), ("validation", validation)):
        with (args.output_dir / f"{name}.jsonl").open("w", encoding="utf-8", newline="\n") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    manifest = {"schema_version": SCHEMA_VERSION, "train": len(train),
                "validation": len(validation), "tasks": len({r['task_key'] for r in records}),
                "label_policy": "provenance preserved; first acceptable action unless primary_action supplied"}
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest))


if __name__ == "__main__":
    main()
