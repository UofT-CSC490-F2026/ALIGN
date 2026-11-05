"""Generate the Part 4 FoX-only standard-validation table and summary."""

import csv
import hashlib
import json
from pathlib import Path

P4 = Path(__file__).resolve().parents[2]
ROOT = P4 / "results" / "standard-validation-seed42-v1"


def main() -> None:
    verification = json.loads((ROOT / "fresh-worker-verification.json").read_text())
    if verification["status"] != "passed":
        raise ValueError("Fresh-worker checkpoint and evidence verification did not pass")
    result = json.loads((ROOT / "fox-standard.json").read_text())
    cells = {(r["checkpoint"], r["evaluation_context"]): r for r in result["results"]}
    if len(cells) != 4:
        raise ValueError("Expected four FoX checkpoint/context cells")
    rows = []
    for context in (512, 2048):
        first = cells[("checkpoint1", context)]
        second = cells[("checkpoint2", context)]
        if (first["validation_manifest_sha256"] != second["validation_manifest_sha256"]
                or first["target_bytes"] != second["target_bytes"]
                or first["nominal_tokens"] != second["nominal_tokens"]):
            raise ValueError(f"Validation data differ across checkpoints at {context}")
        rows.append({
            "evaluation_context": context,
            "checkpoint1_bpb": first["bpb"],
            "checkpoint2_bpb": second["bpb"],
            "checkpoint2_minus_checkpoint1_bpb": second["bpb"] - first["bpb"],
            "nominal_target_positions": first["nominal_tokens"],
            "scored_non_special_tokens": first["scored_non_special_tokens"],
            "target_bytes": first["target_bytes"],
        })
    with (ROOT / "fox_standard_validation.csv").open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "purpose": "Part 4 FoX C1-to-C2 standard validation, not an architecture comparison",
        "seed": 42,
        "rows": rows,
        "source_result_sha256": hashlib.sha256((ROOT / "fox-standard.json").read_bytes()).hexdigest(),
        "limitations": [
            "Checkpoint 2 has 16.384M more training tokens and a later learning-rate schedule.",
            "The 512 and 2048 best-fit streams have different target tokens; compare checkpoints only within a fixed evaluation context.",
            "The procedure matches Part 2, but the original Part 2 tokenizer and validation inputs are unavailable.",
            "One FoX seed was trained for Part 4.",
        ],
    }
    (ROOT / "fox_report_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
