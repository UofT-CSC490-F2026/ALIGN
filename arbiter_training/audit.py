"""Audit prepared arbiter examples for leakage, provenance, and class balance."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from .contract import parse_decision
from .prepare import read_jsonl


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    rows = {split: list(read_jsonl(args.prepared_dir / f"{split}.jsonl"))
            for split in ("train", "validation")}
    task_keys = {split: {r["task_key"] for r in values}
                 for split, values in rows.items()}
    def repository(row):
        return (row["task_key"].split(":")[1] if ":" in row["task_key"]
                else row["task_key"])

    repositories = {split: {repository(row) for row in values}
                    for split, values in rows.items()}
    if task_keys["train"] & task_keys["validation"]:
        raise ValueError("Task leakage across train and validation")
    if repositories["train"] & repositories["validation"]:
        raise ValueError("Repository leakage across train and validation")
    seen_events = set()
    prompt_hashes = set()
    review_candidates = []
    for split, values in rows.items():
        for row in values:
            if row.get("split") != split:
                raise ValueError("Record split disagrees with output file")
            event_uuid = row["checkpoint_event_uuid"]
            if event_uuid in seen_events:
                raise ValueError("Duplicate checkpoint across prepared data")
            seen_events.add(event_uuid)
            if row["label_source"] == "teacher_gpt" and row["review_status"] != "provisional":
                raise ValueError("Teacher label was incorrectly marked reviewed")
            if row["primary_action"] not in row["expected_actions"]:
                raise ValueError("Primary decision not in acceptable decisions")
            prompt = row["prompt"]
            if len(prompt) != 2 or prompt[1]["role"] != "user":
                raise ValueError("Unexpected prompt format")
            observation = json.loads(prompt[1]["content"])
            required = {"schema_version", "task_key", "shared_goal", "agent_goals",
                        "constraints", "actor_id", "proposal_sequence",
                        "proposed_action", "recent_events", "repository_state"}
            if set(observation) != required or observation["task_key"] != row["task_key"]:
                raise ValueError("Prompt does not match Person 1 observation schema")
            if any(e["sequence_number"] >= observation["proposal_sequence"]
                   for e in observation["recent_events"]):
                raise ValueError("Future event leaked into a prompt")
            if observation["repository_state"] and row.get("scenario_source") == "public_trace":
                raise ValueError("Public trace has no reliable repository snapshot")
            decision = parse_decision(row["completion"][0]["content"],
                                      observation=observation, enforceable=True)
            if decision["action"] != row["primary_action"]:
                raise ValueError("Completion action differs from label")
            if decision["evidence"].get("proposal_sequence") != observation["proposal_sequence"]:
                raise ValueError("Completion cites a different proposal")
            command = str(observation["proposed_action"]["action"].get(
                "command", "")).lower()
            if row["primary_action"] != "continue" or any(
                token in command for token in
                ("git reset --hard", "git checkout --", "git merge", "gh pr create")
            ):
                review_candidates.append({"event_uuid": event_uuid, "split": split,
                                          "action": row["primary_action"],
                                          "scenario_source": row.get("scenario_source", "public_trace")})
            digest = hashlib.sha256(prompt[1]["content"].encode("utf-8")).hexdigest()
            if digest in prompt_hashes:
                raise ValueError("Duplicate prompt in prepared data")
            prompt_hashes.add(digest)
    report = {"examples": {s: len(values) for s, values in rows.items()},
              "repositories": {s: sorted(names) for s, names in repositories.items()},
              "actions": {s: dict(Counter(r["primary_action"] for r in values))
                          for s, values in rows.items()},
              "label_sources": {s: dict(Counter(r["label_source"] for r in values))
                                for s, values in rows.items()},
              "max_prompt_characters": {s: max((len(r["prompt"][1]["content"])
                                              for r in values), default=0)
                                        for s, values in rows.items()},
              "warnings": []}
    report["review_candidates"] = review_candidates
    if sum(r["primary_action"] != "continue" for r in rows["train"]) < 5:
        report["warnings"].append("Too few train interventions for a meaningful fine-tune")
    if all(r["review_status"] == "provisional" for values in rows.values() for r in values):
        report["warnings"].append("All labels are provisional; independent review remains")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
