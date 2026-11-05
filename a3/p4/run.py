"""Local preflight and explicit Modal stage launcher.

Guards run BEFORE importing or initializing the cloud app. Export/audit paths
never construct a cloud app or allocate a GPU.
"""
import argparse
import json
import os
from pathlib import Path
import re
import sys

P4 = Path(__file__).resolve().parent
sys.path.insert(0, str(P4.parent))
os.environ["MODAL_PROFILE"] = "a3-p4"
os.environ.setdefault("MPLCONFIGDIR", str(P4 / ".cache/matplotlib"))
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

from p4.artifacts import atomic_json, sha256_file
from p4.bootstrap import activate_local, verify_jinhao_sources
from p4.budget import reserve
from p4.config import load_config, rehearsal_config
from p4.transfer import export_prepared, export_run, verify_export, verify_prepared_export


def safe_id(value):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,99}", value):
        raise ValueError("Use a simple unique experiment/preparation ID")
    return value


def source_identity():
    files = [*P4.glob("*.py"), P4.parent / "pico/variants.py",
             P4 / "vendor/nanochat/uv.lock"]
    return {str(path.relative_to(P4.parent)): sha256_file(path) for path in files}


def invoke_training(function, config, preparation_id, experiment_id, stage, resume=None):
    try:
        return function.remote(config, preparation_id, experiment_id, stage, resume)
    except BaseException:
        try:
            exported = export_run(experiment_id)
            print("Failure evidence exported to " + exported["directory"], file=sys.stderr)
        except Exception:
            print("Automatic failure export unavailable; inspect the persistent Modal Volume.", file=sys.stderr)
        raise


def publish_small_report(experiment_id, exported, result):
    """Copy small research deliverables into a Git-friendly results folder."""
    import shutil
    source = Path(exported["directory"])
    verify_export(source)
    destination = P4 / "results" / safe_id(experiment_id)
    destination.mkdir(parents=True, exist_ok=True)
    names = ["config.json", "hardware.json", "provenance.json", "continuation.json",
             "short-result.json", "long-result.json", "ARTIFACT_MANIFEST.json"]
    paths = [source / name for name in names] + list((source / "report").rglob("*"))
    for path in paths:
        if not path.is_file() or path.is_symlink() or path.stat().st_size > 2 * 1024 * 1024:
            continue
        if path.suffix.lower() not in (".json", ".csv", ".md", ".png", ".svg"):
            continue
        target = destination / path.relative_to(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    atomic_json(destination / "collected-status.json", result)
    (destination / "ARTIFACT_LOCATIONS.md").write_text(
        "# Artifact locations\n\n"
        "These small summaries and figures can be committed to GitHub.\n\n"
        f"Full checkpoints, logs and per-example evidence are in Modal Volume `a3-p4-context`, `/runs/{experiment_id}`.\n"
        "`ARTIFACT_MANIFEST.json` records hashes and sizes of the full remote files; it is an index, not the model itself.\n\n"
        f"Optional full download: `a3/p4/.venv/bin/python a3/p4/run.py export --experiment-id {experiment_id} --full-export`.\n"
        "Downloaded weights/data remain ignored by Git.\n")
    return str(destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("check", "probe", "prepare", "rehearsal", "short", "long", "export", "export-prepared", "audit"))
    parser.add_argument("--experiment-id", default="")
    parser.add_argument("--preparation-id", default="climbmix-v1")
    parser.add_argument("--config", default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--full-export", action="store_true", help="Also download large model weights or the training stream; default keeps these on Modal")
    args = parser.parse_args()
    boundary = activate_local()
    config = load_config(args.config)
    safe_id(args.preparation_id)
    local = P4 / "local"
    local.mkdir(exist_ok=True)
    if args.stage == "check":
        print(json.dumps({"source_boundary": boundary, "config": config,
                          "account_authentication_not_tested": True}, indent=2))
        return
    if args.stage == "export-prepared":
        print(json.dumps(export_prepared(args.preparation_id, include_training_data=args.full_export), indent=2))
        return
    if args.stage in ("export", "audit"):
        safe_id(args.experiment_id)
        if args.stage == "export":
            result = export_run(args.experiment_id, include_checkpoints=args.full_export)
        else:
            result = verify_export(P4 / "outputs" / args.experiment_id)
        print(json.dumps(result, indent=2))
        return
    ledger = local / "budget-reservations.jsonl"
    if args.stage in ("prepare", "probe"):
        reserve(ledger, args.stage, config, seconds=1800 if args.stage == "prepare" else 120)
    else:
        safe_id(args.experiment_id)
        prepared = verify_prepared_export(local / "prepared" / args.preparation_id)
        if prepared["token_count"] < (config["short_steps"] + config["long_steps"]) * config["tokens_per_step"] + 1:
            raise ValueError("Frozen preparation is too small for the main training horizon")
        if args.stage == "rehearsal":
            config = rehearsal_config()
            for label in ("short", "long"):
                reserve(ledger, "rehearsal-" + label, config, seconds=1800, gpu=True)
                reserve(ledger, "rehearsal-verify-" + label, config, seconds=300)
        else:
            gate_path = local / "rehearsal-passed.json"
            if not gate_path.exists():
                raise RuntimeError("Full training requires a passed save/reload/evaluate/export rehearsal")
            gate = json.loads(gate_path.read_text())
            if gate.get("status") != "passed" or gate.get("source_identity") != source_identity():
                raise RuntimeError("Rehearsal is missing or the implementation changed afterward")
            verify_export(Path(gate["final_export"]["directory"]))
            reserve(ledger, f"{args.experiment_id}-{args.stage}", config, seconds=1800, gpu=True)
            reserve(ledger, f"{args.experiment_id}-verify", config, seconds=300)
        config["experiment_id"] = args.experiment_id
        config_file = local / f"{args.experiment_id}-config.json"
        if config_file.exists() and json.loads(config_file.read_text()) != config:
            raise ValueError("Experiment config changed; use a new run ID")
        atomic_json(config_file, config)

    # Only now may Modal build its images, resolve the W&B secret and create
    # functions. No model training is launched merely by importing this file.
    if args.stage in ("prepare", "probe"):
        # CPU preparation must not attempt to register GPU functions, which
        # require an account payment method even when credits are available.
        os.environ["A3_P4_CPU_PREP_ONLY"] = "1"
    else:
        os.environ.pop("A3_P4_CPU_PREP_ONLY", None)
    from modal import enable_output
    from p4.modal_app import app, prepare, training, verify_saved, probe_tracking
    with enable_output(), app.run():
        if args.stage == "probe":
            result = probe_tracking.remote(config)
            atomic_json(local / "tracking-probe.json", result)
        elif args.stage == "prepare":
            result = prepare.remote(config, args.preparation_id)
            atomic_json(local / f"prepared-{args.preparation_id}.json", result)
            result["local_export"] = export_prepared(args.preparation_id, include_training_data=args.full_export)
        elif args.stage == "rehearsal":
            short = invoke_training(training, config, args.preparation_id, args.experiment_id, "short")
            if short["status"] != "complete":
                export_run(args.experiment_id)
                raise RuntimeError("Rehearsal short stage did not complete")
            before = verify_saved.remote(args.experiment_id, "checkpoint1")
            first = export_run(args.experiment_id, include_checkpoints=args.full_export)
            long = invoke_training(training, config, args.preparation_id, args.experiment_id, "long")
            if long["status"] != "complete":
                export_run(args.experiment_id)
                raise RuntimeError("Rehearsal continuation did not complete")
            after = verify_saved.remote(args.experiment_id, "checkpoint2")
            final = export_run(args.experiment_id, include_checkpoints=args.full_export)
            summary = json.loads((Path(final["directory"]) / "report/summary.json").read_text())
            if summary.get("status") != "complete" or long["parent_checkpoint_sha256"] != before["checkpoint_sha256"]:
                raise RuntimeError("Rehearsal evaluation or ancestry audit failed")
            if not short.get("wandb_url") or not long.get("wandb_url") or short.get("wandb_replay_required") or long.get("wandb_replay_required"):
                raise RuntimeError("Rehearsal must have working online W&B logging")
            result = {"status": "passed", "experiment_id": args.experiment_id,
                      "source_identity": source_identity(), "short": short, "long": long,
                      "before": before, "after": after, "first_export": first, "final_export": final}
            atomic_json(local / "rehearsal-passed.json", result)
        else:
            result = invoke_training(training, config, args.preparation_id, args.experiment_id, args.stage, args.resume)
            if result["status"] == "complete":
                checkpoint = "checkpoint1" if args.stage == "short" else "checkpoint2"
                result["fresh_worker_verification"] = verify_saved.remote(args.experiment_id, checkpoint)
            result["local_export"] = export_run(args.experiment_id, include_checkpoints=args.full_export)
            result["modal_app_id"] = app.app_id
            atomic_json(local / f"{args.experiment_id}-{args.stage}-collected.json", result)
            if result.get("status") == "complete":
                result["github_ready_report"] = publish_small_report(args.experiment_id, result["local_export"], result)
    verify_jinhao_sources()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    import fcntl
    lock_dir = P4 / "local"
    lock_dir.mkdir(exist_ok=True)
    with (lock_dir / "launch.lock").open("a") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("A Part 4 launcher is already running; avoid concurrent jobs.")
        main()
