"""Print a compact, prefix-only view of teacher labeling checkpoints."""

import argparse
import json
from pathlib import Path


def short(value, limit):
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    text = " ".join(text.split())
    return text[:limit] + ("..." if len(text) > limit else "")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("queue", type=Path)
    parser.add_argument("ids", nargs="+")
    parser.add_argument("--history", type=int, default=8)
    args = parser.parse_args()
    for line in args.queue.read_text(encoding="utf-8").splitlines():
        item = json.loads(line)
        if not any(item["event_uuid"].startswith(prefix) for prefix in args.ids):
            continue
        checkpoint = item["checkpoint"]
        print(f"\n{item['event_uuid']} {item['split']} {item['task_key']} "
              f"seq={checkpoint['sequence_number']} {checkpoint['agent_id']}")
        for goal in checkpoint["task"]["goals"]:
            print(f"GOAL {goal['owner_agent_id']}: {short(goal['description'], 700)}")
        for event in checkpoint["prior_events"][-args.history:]:
            payload = event["payload"]
            content = (payload.get("action", {}).get("command")
                       if event["event_type"] == "proposed_action"
                       else payload.get("output_excerpt", payload))
            print(f"PRIOR {event['sequence_number']} {event['actor_id']} "
                  f"{event['event_type']}: {short(content, 300)}")
        print(f"PROPOSAL: {short(checkpoint['proposal']['action']['command'], 2200)}")


if __name__ == "__main__":
    main()
