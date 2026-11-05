"""Separate bounded app; original Part Four source and outputs stay immutable."""
from pathlib import Path
import json
import modal

from p4.modal_app import image as original_image, volume, activate
from p4.artifacts import atomic_json, sha256_file, write_bundle_manifest
from .engine import FOX_ID, BASELINE_ID, EVALUATION_ID

app = modal.App("a3-p4-standard-validation")
image = original_image
for source in sorted(Path(__file__).parent.glob("*.py")):
    image = image.add_local_file(source, f"/app/a3/p4/standard_eval/{source.name}")
resources = dict(image=image, volumes={"/vol": volume}, cpu=(4, 4), memory=(16384, 16384),
                 retries=0, scaledown_window=2, max_containers=1, single_use_containers=True,
                 include_source=False)
secret = modal.Secret.from_name("a3-p4-wandb", required_keys=["WANDB_API_KEY"])


@app.function(**resources, gpu="H100", secrets=[secret], timeout=1800)
def baseline_training(config, stage):
    activate()
    from .control import train_control
    root = Path("/vol/runs") / BASELINE_ID
    try:
        return train_control(config, "/vol/prepared/climbmix-v1", root, stage, volume.commit)
    finally:
        if root.exists():
            write_bundle_manifest(root)
            volume.commit()


@app.function(**resources, gpu="H100", secrets=[secret], timeout=900)
def standard_evaluation(config, family):
    activate()
    from .engine import evaluate_family
    root = Path("/vol/standard_eval") / EVALUATION_ID
    try:
        return evaluate_family(config, "/vol/prepared/climbmix-v1", "/vol/runs", root, family, volume.commit)
    finally:
        if root.exists():
            write_bundle_manifest(root)
            volume.commit()


@app.function(**resources, secrets=[secret], timeout=300)
def verify():
    activate()
    from p4.artifacts import CheckpointStore, verify_bundle
    from p4.training import model_state_sha256
    root = Path("/vol/standard_eval") / EVALUATION_ID
    verify_bundle(root)
    validation_hash = sha256_file(root / "validation-manifest.json")
    results = []
    checkpoints = []
    parents = {}
    for family, run_id in (("fox", FOX_ID), ("baseline", BASELINE_ID)):
        record = json.loads((root / f"{family}-standard.json").read_text())
        if record["validation_manifest_sha256"] != validation_hash:
            raise ValueError("Families were evaluated on different frozen batches")
        results.extend(record["results"])
        store = CheckpointStore(Path("/vol/runs") / run_id)
        for name in ("checkpoint1", "checkpoint2"):
            state, manifest = store.load(name)
            if model_state_sha256(state["model"]) != state["model_state_sha256"]:
                raise ValueError("Fresh-worker checkpoint verification failed")
            if name == "checkpoint1":
                parents[family] = manifest["sha256"]
            elif state["parent_checkpoint_sha256"] != parents[family]:
                raise ValueError("Continuation parent hash does not match checkpoint 1")
            if family == "baseline":
                fox_state, _ = CheckpointStore(Path("/vol/runs") / FOX_ID).load(name)
                if state["cumulative_tokens"] != fox_state["cumulative_tokens"] or state["data_stream"] != fox_state["data_stream"]:
                    raise ValueError("Baseline and FoX training token positions are not matched")
                del fox_state
            checkpoints.append({"family": family, "checkpoint": name, "sha256": manifest["sha256"],
                                "tokens": state["cumulative_tokens"], "stream": state["data_stream"]})
            del state
    if len(results) != 8:
        raise ValueError("The standard validation matrix is incomplete")
    for checkpoint in ("checkpoint1", "checkpoint2"):
        for length in (512, 2048):
            pair = [r for r in results if r["checkpoint"] == checkpoint and r["evaluation_context"] == length]
            if len(pair) != 2 or pair[0]["target_bytes"] != pair[1]["target_bytes"] or pair[0]["scored_non_special_tokens"] != pair[1]["scored_non_special_tokens"]:
                raise ValueError("The architecture comparison scored different targets/bytes")
    import wandb
    from urllib.parse import urlparse
    tracking = []
    tracking_records = [json.loads((root / f"{family}-standard.json").read_text()) for family in ("fox", "baseline")]
    tracking_records.extend(json.loads((Path("/vol/runs") / BASELINE_ID / f"{stage}-result.json").read_text()) for stage in ("short", "long"))
    for record in tracking_records:
        url = record["wandb_url"]
        parts = urlparse(url).path.strip("/").split("/")
        if len(parts) != 4 or parts[2] != "runs":
            raise ValueError("Unexpected W&B run URL")
        history = list(wandb.Api(timeout=30).run("/".join((parts[0], parts[1], parts[3]))).scan_history())
        expected = "train/loss" if "stage" in record else "standard/bpb_512"
        if not any(expected in row for row in history):
            raise ValueError("Saved W&B history is missing " + expected)
        tracking.append({"url": url, "history_rows": len(history), "verified_metric": expected})
    from .engine import supplemental_sources
    scientific = supplemental_sources()
    training_sources = json.loads((Path("/vol/runs") / BASELINE_ID / "provenance.json").read_text())["supplemental_sources"]
    if any(scientific[k] != training_sources[k] for k in ("engine.py", "control.py")):
        raise ValueError("Scientific evaluation or control source changed after training")
    result = {"status": "passed", "fresh_cpu_worker": True, "validation_manifest_sha256": validation_hash,
              "same_frozen_inputs_per_context": True, "training_tokens_and_cursors_matched": True,
              "cells": 8, "checkpoints": checkpoints, "results": results, "wandb_history": tracking,
              "scientific_sources_unchanged": True, "verifier_source_sha256": scientific["modal_app.py"]}
    atomic_json(root / "fresh-worker-verification.json", result)
    write_bundle_manifest(root)
    volume.commit()
    return result
