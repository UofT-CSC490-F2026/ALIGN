"""Export a blind annotation packet from prepared arbiter examples.

The packet deliberately omits teacher decisions, model predictions, and future
events. A reviewer can fill expected_actions, rationale, and reviewer, then use
the resulting label JSONL with prepare.py.
"""

import argparse
import json
from pathlib import Path

from .prepare import read_jsonl


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit", type=Path,
                        help="Optional audit.json; export only flagged decisions")
    parser.add_argument("--include-synthetic", action="store_true",
                        help="Include counterfactuals for review; these cannot be fed to prepare.py")
    args = parser.parse_args()
    flagged = None
    if args.audit:
        report = json.loads(args.audit.read_text(encoding="utf-8"))
        flagged = {item["event_uuid"] for item in report["review_candidates"]}
    rows = []
    for split in ("train", "validation"):
        for record in read_jsonl(args.prepared_dir / f"{split}.jsonl"):
            event_uuid = record["checkpoint_event_uuid"]
            if flagged is not None and event_uuid not in flagged:
                continue
            if record.get("scenario_source") == "synthetic_counterfactual" and not args.include_synthetic:
                continue
            rows.append({
                "event_uuid": event_uuid,
                "task_key": record["task_key"],
                "split": split,
                "scenario_source": record.get("scenario_source", "public_trace"),
                "checkpoint": json.loads(record["prompt"][1]["content"]),
                "expected_actions": [],
                "primary_action": None,
                "rationale": "",
                "reviewer": "",
            })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({"review_rows": len(rows),
                      "output": str(args.output)}))


if __name__ == "__main__":
    main()
