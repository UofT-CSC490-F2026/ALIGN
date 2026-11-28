"""Select varied, prefix-only checkpoints for teacher labeling and audit."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

from .prepare import (event_bundles, observation_for_proposal, recorded_observations,
                      read_jsonl, task_context, task_splits_for)


def category(command):
    command = command.lower()
    if "send_message" in command or "coop-send" in command:
        return "communication"
    if any(token in command for token in ("git merge", "git rebase", "git cherry-pick", "git push", "gh pr create")):
        return "integration"
    if any(token in command for token in ("git reset --hard", "git clean -", "rm -rf ")):
        return "destructive"
    if any(token in command for token in ("apply_patch", "cat >", "sed -i", "python3 - <<", "tee ")):
        return "possible_edit"
    if any(token in command for token in ("pytest", "npm test", "go test", "cargo test")):
        return "test"
    if any(token in command for token in ("git ", "gh ")):
        return "git"
    return "inspection"


def checkpoints(tasks_path, events_dir, history_limit, split_manifest=None):
    tasks = {t["task_key"]: t for t in read_jsonl(tasks_path)}
    splits = task_splits_for(tasks, split_manifest)
    for path, events in event_bundles(events_dir):
        task_key = events[0]["payload"]["task_key"]
        if task_key not in tasks:
            raise ValueError(f"Unknown task_key {task_key} in {path}")
        task = tasks[task_key]
        observations = recorded_observations(events, path)
        history = []
        for index, event in enumerate(events, 1):
            if event["event_type"] == "proposed_action":
                proposal = event["payload"]
                command = proposal["action"].get("command", "")
                if proposal["action"].get("tool") == "message":
                    command = "send_message " + proposal["action"].get("content", "")
                prior_events = history[-history_limit:] if history_limit else []
                observation = observation_for_proposal(
                    events, index, task_context(task), prior_events, observations, path)
                yield {"event_uuid": event["event_uuid"], "task_key": task_key,
                       "split": splits[task_key], "category": category(command),
                       "source_bundle": str(path),
                       "observation_source": "runner_event" if observations else "public_reconstruction",
                       "observation": observation,
                       "checkpoint": {
                           "schema_version": 1, "task": task_context(task),
                           "run_key": event["run_key"],
                           "checkpoint_event_uuid": event["event_uuid"],
                           "sequence_number": event["sequence_number"],
                           "agent_id": event["actor_id"],
                           "proposal": event["payload"],
                           "prior_events": prior_events,
                       }}
            history.append({"sequence_number": event["sequence_number"],
                            "actor_type": event["actor_type"], "actor_id": event["actor_id"],
                            "event_type": event["event_type"], "payload": event["payload"]})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--events-dir", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path,
                        help="CooperBench split JSON; required when tasks lack metadata.split")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit-per-split", type=int, default=12)
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--validation-limit", type=int)
    parser.add_argument("--test-limit", type=int, default=0)
    parser.add_argument("--history-limit", type=int, default=8)
    args = parser.parse_args()
    groups = defaultdict(lambda: defaultdict(list))
    for item in checkpoints(args.tasks, args.events_dir, args.history_limit,
                            args.split_manifest):
        groups[item["split"]][item["category"]].append(item)
    selected = []
    for split in ("train", "validation", "test"):
        limit = ({"train": args.train_limit,
                  "validation": args.validation_limit,
                  "test": args.test_limit})[split]
        if limit is None:
            limit = args.limit_per_split
        buckets = groups[split]
        task_counts = defaultdict(int)
        while len([x for x in selected if x["split"] == split]) < limit:
            added = False
            for kind in ("destructive", "integration", "possible_edit", "communication", "git", "test", "inspection"):
                if buckets[kind]:
                    chosen = min(range(len(buckets[kind])),
                                 key=lambda i: (task_counts[buckets[kind][i]["task_key"]], i))
                    item = buckets[kind].pop(chosen)
                    selected.append(item)
                    task_counts[item["task_key"]] += 1
                    added = True
                    if len([x for x in selected if x["split"] == split]) >= limit:
                        break
            if not added:
                break
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as stream:
        for item in selected:
            stream.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(json.dumps({"selected": len(selected), "categories": {
        split: {kind: sum(x["split"] == split and x["category"] == kind for x in selected)
                for kind in groups[split]} for split in ("train", "validation", "test")}}, indent=2))


if __name__ == "__main__":
    main()
