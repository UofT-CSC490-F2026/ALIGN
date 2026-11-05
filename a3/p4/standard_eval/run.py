"""Explicit, one-phase-at-a-time launcher. Never starts jobs on import."""
import argparse
import json
import os
from pathlib import Path
import sys

P4 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(P4.parent))
os.environ["MODAL_PROFILE"] = "a3-p4"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

from p4.bootstrap import activate_local
from p4.artifacts import atomic_json, sha256_file
from p4.config import load_config
from p4.budget import reserve
from p4.standard_eval.engine import BASELINE_ID, EVALUATION_ID


def export_evidence():
    import modal
    output = P4 / "results" / EVALUATION_ID
    output.mkdir(parents=True, exist_ok=True)
    volume = modal.Volume.from_name("a3-p4-context")
    selected = []
    for directory, local_prefix in ((f"/standard_eval/{EVALUATION_ID}", ""), (f"/runs/{BASELINE_ID}", "baseline-control")):
        for entry in volume.iterdir(directory, recursive=True):
            path = Path(entry.path)
            if path.suffix not in (".json", ".jsonl", ".csv", ".png", ".md") or "wandb" in path.parts:
                continue
            # iterators may return absolute or volume-relative paths.
            rel = path.relative_to(directory.lstrip("/")) if not path.is_absolute() else path.relative_to(directory)
            target = output / local_prefix / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("wb") as handle:
                for chunk in volume.read_file(entry.path):
                    handle.write(chunk)
            selected.append({"remote": entry.path, "local": str(target.relative_to(output)),
                             "sha256": sha256_file(target), "size_bytes": target.stat().st_size})
    atomic_json(output / "LOCAL_EXPORT_MANIFEST.json", {"scope": "Small supplemental evidence only; weights, raw shards and frozen token arrays stay on Modal.", "files": selected})
    return {"directory": str(output), "files": len(selected)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("check", "fox-eval", "baseline-short", "baseline-long", "baseline-eval", "verify", "export"))
    args = parser.parse_args()
    boundary = activate_local()
    config = load_config()
    if args.phase == "check":
        print(json.dumps({"sources": boundary, "standard_token_budget": 262144,
                          "control": "same data/tokenizer/seed/schedule/batches as original FoX; baseline architecture only"}, indent=2))
        return
    if args.phase == "export":
        print(json.dumps(export_evidence(), indent=2))
        return
    if args.phase.startswith("baseline-") and args.phase != "baseline-eval":
        config.update(variant="baseline", experiment_id=BASELINE_ID, wandb_project="a3-p4-standard-validation")
    else:
        config.update(experiment_id=EVALUATION_ID, wandb_project="a3-p4-standard-validation")
    reserve(P4 / "local/budget-reservations.jsonl", "supplemental-" + args.phase, config,
            seconds=1800 if args.phase in ("baseline-short", "baseline-long") else 300 if args.phase == "verify" else 900,
            gpu=args.phase != "verify")
    from modal import enable_output
    from p4.standard_eval.modal_app import app, baseline_training, standard_evaluation, verify
    with enable_output(), app.run():
        if args.phase in ("baseline-short", "baseline-long"):
            result = baseline_training.remote(config, args.phase.split("-")[1])
        elif args.phase == "verify":
            result = verify.remote()
        else:
            result = standard_evaluation.remote(config, args.phase.split("-")[0])
    destination = P4 / "local/standard-validation"
    destination.mkdir(exist_ok=True)
    atomic_json(destination / f"{args.phase}.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
