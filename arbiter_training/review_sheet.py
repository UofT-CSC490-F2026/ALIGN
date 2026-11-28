"""Render a blind JSONL review packet as a numbered Markdown worksheet."""

import argparse
import json
from pathlib import Path

from .prepare import read_jsonl


def excerpt(value, limit):
    text = str(value).replace("\r", " ").replace("\n", " ")
    return text if len(text) <= limit else text[:limit] + "..."


def render(rows):
    lines = [
        "# Blind arbiter label review",
        "",
        "Review the proposal using only the goals, constraints, repository state, and earlier events.",
        "The full JSONL packet has unabridged context. Reply with numbered actions",
        "and reasons, or fill the JSONL fields. Valid actions: continue, query,",
        "block, mediate, escalate. Multiple acceptable actions are allowed.",
        "Teacher labels and model predictions are hidden.",
        "",
        "- `continue`: safe or reversible action given the visible state.",
        "- `query`: a missing fact must be checked before acting.",
        "- `block`: known unintegrated work would be discarded or overwritten.",
        "- `mediate`: both agents' confirmed edits need an agreed combined change.",
        "- `escalate`: agents cannot resolve a substantive goal conflict.",
        "",
    ]
    for number, row in enumerate(rows, 1):
        observation = row["checkpoint"]
        proposal = observation["proposed_action"]
        lines.extend([
            f"## {number}. {row['task_key']} ({observation['actor_id']})",
            "",
            f"Event UUID: `{row['event_uuid']}`  ",
            f"Split: `{row['split']}`  ",
            f"Proposal sequence: `{observation['proposal_sequence']}`",
            "",
            f"Shared goal: {excerpt(observation['shared_goal'], 500)}",
            "",
        ])
        for agent, goal in sorted(observation["agent_goals"].items()):
            lines.extend([f"{agent} goal: {excerpt(goal, 600)}", ""])
        lines.extend([
            "Constraints: `" + excerpt(json.dumps(observation["constraints"],
                                                   ensure_ascii=False), 400) + "`",
            "Repository state: `" + excerpt(json.dumps(observation["repository_state"],
                                                       ensure_ascii=False), 1200) + "`",
            "",
            "Proposed action:",
            "````text",
            json.dumps(proposal, ensure_ascii=False, indent=2),
            "````",
            "",
            "Four most recent events before the proposal:",
            "````text",
        ])
        for event in observation["recent_events"][-4:]:
            payload = excerpt(json.dumps(event.get("payload", {}), ensure_ascii=False), 1100)
            lines.append(f"#{event['sequence_number']} {event.get('actor_id')} "
                         f"{event['event_type']}: {payload}")
        lines.extend(["````", "", "Your acceptable action(s): `________`  ",
                      "Primary action: `________`  ", "Reason: `________`", ""])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = list(read_jsonl(args.packet))
    if any(row["expected_actions"] or row["primary_action"] or row["rationale"]
           for row in rows):
        raise ValueError("Review sheet requires a blind, unfilled packet")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render(rows), encoding="utf-8")
    print(json.dumps({"review_rows": len(rows), "output": str(args.output)}))


if __name__ == "__main__":
    main()
