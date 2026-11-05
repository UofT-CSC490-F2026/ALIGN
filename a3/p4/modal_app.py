"""Explicit Modal stages; running this file never starts an automatic sweep.

Launch with the plain-Python preflight wrapper, not modal run:
  a3/p4/.venv/bin/python a3/p4/run.py prepare
  ... rehearsal --experiment-id rehearsal-YYYYMMDD-01
  ... short --experiment-id fox-seed42-v1
  ... long --experiment-id fox-seed42-v1

The main training stages require a verified, locally downloaded rehearsal.
"""
from pathlib import Path
import json
import os
import re
import sys
import time

import modal

P4 = Path(__file__).resolve().parent
A3 = P4.parent
if str(A3) not in sys.path:
    sys.path.insert(0, str(A3))

from p4.config import APP_NAME, VOLUME_NAME, SECRET_NAME, load_config, rehearsal_config

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME)
CPU_PREPARATION_ONLY = os.environ.get("A3_P4_CPU_PREP_ONLY") == "1"
image = (modal.Image.debian_slim(python_version="3.12")
         .apt_install("git", "build-essential")
         .pip_install("uv==0.13.7")
         .add_local_file(P4 / "vendor/nanochat/pyproject.toml", "/opt/nanochat/pyproject.toml", copy=True)
         .add_local_file(P4 / "vendor/nanochat/uv.lock", "/opt/nanochat/uv.lock", copy=True)
         .workdir("/opt/nanochat")
         .run_commands("uv sync --frozen --extra gpu --no-install-project",
                       "uv pip install --python .venv/bin/python matplotlib==3.10.8 numpy==1.26.4 contourpy==1.3.3")
         .env({"PATH": "/opt/nanochat/.venv/bin:/usr/local/bin:/usr/bin:/bin",
               "PYTHONPATH": "/opt/nanochat:/app/a3", "OMP_NUM_THREADS": "1",
               "HF_HOME": "/vol/cache/huggingface",
               "NANOCHAT_BASE_DIR": "/vol/cache/nanochat",
               "PYTORCH_ALLOC_CONF": "expandable_segments:True",
               "MPLCONFIGDIR": "/tmp/matplotlib", "PYTHONDONTWRITEBYTECODE": "1"})
         .add_local_dir(P4 / "vendor/nanochat/nanochat", "/opt/nanochat/nanochat",
                        ignore=lambda path: path.suffix != ".py")
         .add_local_dir(A3 / "pico", "/app/a3/pico", ignore=lambda path: path.suffix != ".py"))
# Enumerate the Part 4 source files explicitly. Runtime logs, downloaded data,
# credentials, checkpoints and the local virtual environment are never mounts.
for source_file in sorted(P4.glob("*.py")):
    image = image.add_local_file(source_file, f"/app/a3/p4/{source_file.name}")


def activate():
    import site
    site.addsitedir("/opt/nanochat/.venv/lib/python3.12/site-packages")


def safe_id(value):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,99}", value):
        raise ValueError("Use a simple unique experiment/preparation ID")
    return value


@app.function(image=image, volumes={"/vol": volume}, cpu=(4, 4), memory=(16384, 16384),
              timeout=1800, retries=0, scaledown_window=2, max_containers=1,
              single_use_containers=True, include_source=False)
def prepare(config, preparation_id):
    activate()
    from p4.data import prepare_data
    from p4.artifacts import atomic_json
    started = time.monotonic()
    root = Path("/vol/prepared") / safe_id(preparation_id)
    result = prepare_data(root, config, commit=volume.commit)
    atomic_json(root / "preparation-timing.json", {"elapsed_seconds": time.monotonic() - started,
                "requested_cpu_cores": 4, "requested_memory_mb": 16384,
                "scope": "worker entrypoint; excludes image build and startup"})
    volume.commit()
    return {"preparation_id": preparation_id, "manifest": result,
            "cpu_preparation": {"elapsed_seconds": time.monotonic() - started,
                                "requested_cpu_cores": 4, "requested_memory_mb": 16384,
                                "scope": "worker entrypoint; excludes image build and startup"}}


@app.function(image=image, volumes={"/vol": volume}, gpu=None if CPU_PREPARATION_ONLY else "H100",
              cpu=(4, 4), memory=(16384, 16384),
              secrets=[modal.Secret.from_name(SECRET_NAME, required_keys=["WANDB_API_KEY"])],
              timeout=1800, retries=0, scaledown_window=2, max_containers=1,
              single_use_containers=True, include_source=False)
def training(config, preparation_id, experiment_id, stage, resume=None):
    activate()
    from p4.training import train_stage
    from p4.artifacts import atomic_json, write_bundle_manifest
    root = Path("/vol/runs") / safe_id(experiment_id)
    prepared = Path("/vol/prepared") / safe_id(preparation_id)
    # Each stage gets a new worker; metadata/cursor and Volume state are loaded
    # afresh rather than relying on a previous process's memory or mount cache.
    try:
        result = train_stage(config, prepared, root, stage, commit=volume.commit,
                             device="cuda", resume=resume)
    except BaseException:
        # Package any already-durable checkpoints/metrics even when training or
        # tracking fails, so the local launcher can export failure evidence.
        if root.exists():
            try:
                write_bundle_manifest(root)
                volume.commit()
            except Exception:
                pass  # Preserve the original error; no success is reported.
        raise
    if result["status"] == "complete":
        from p4.report import create_report
        summary = create_report(root)
        if stage == "long" and summary.get("status") != "complete":
            raise RuntimeError("Long-stage evaluation evidence is incomplete")
    write_bundle_manifest(root)
    volume.commit()
    return result


@app.function(image=image, volumes={"/vol": volume}, cpu=(1, 1), memory=(4096, 4096),
              secrets=[modal.Secret.from_name(SECRET_NAME, required_keys=["WANDB_API_KEY"])],
              timeout=300, retries=0, scaledown_window=2, max_containers=1,
              single_use_containers=True, include_source=False)
def verify_saved(experiment_id, checkpoint):
    activate()
    from p4.artifacts import CheckpointStore, verify_bundle
    root = Path("/vol/runs") / safe_id(experiment_id)
    manifest = verify_bundle(root)
    state, record = CheckpointStore(root).load(checkpoint)
    # Check the actual online history, not just a locally recorded run URL.
    # This catches successful training whose asynchronous chart upload failed.
    import wandb
    stage_result = json.loads((root / f"{state['stage']}-result.json").read_text())
    url = stage_result.get("wandb_url")
    if not url:
        raise RuntimeError("Saved stage has no W&B run URL")
    from urllib.parse import urlparse
    parts = urlparse(url).path.strip("/").split("/")
    if len(parts) != 4 or parts[2] != "runs":
        raise RuntimeError("Unexpected W&B run URL format")
    run_path = "/".join((parts[0], parts[1], parts[3]))
    tracked = None
    for attempt in range(3):
        history = list(wandb.Api(timeout=30).run(run_path).scan_history())
        loss_present = any("train/loss" in row for row in history)
        evaluation_present = any("val/bpb" in row for row in history)
        if loss_present and evaluation_present:
            tracked = {"run_url": url, "history_rows": len(history),
                       "training_loss_verified": True, "final_evaluation_verified": True}
            break
        if attempt < 2:
            time.sleep(3)
    if tracked is None:
        raise RuntimeError("W&B history is missing required training/evaluation metrics")
    return {"experiment_id": experiment_id, "checkpoint": checkpoint,
            "checkpoint_sha256": record["sha256"],
            "global_step": state["global_step"], "tokens": state["cumulative_tokens"],
            "data_stream": state["data_stream"], "files_verified": len(manifest["files"]),
            "verified_in_fresh_cpu_worker": True, "wandb_history": tracked}


@app.function(image=image, volumes={"/vol": volume}, cpu=(1, 1), memory=(1024, 1024),
              secrets=[modal.Secret.from_name(SECRET_NAME, required_keys=["WANDB_API_KEY"])],
              timeout=120, retries=0, scaledown_window=2, max_containers=1,
              single_use_containers=True, include_source=False)
def probe_tracking(config):
    """Prove account/chart access using CPU only, before spending GPU time."""
    activate()
    import uuid
    from p4.training import _WandbMirror
    from p4.artifacts import atomic_json, write_bundle_manifest
    import wandb
    attempt = uuid.uuid4().hex[:12]
    root = Path("/vol/setup") / attempt
    root.mkdir(parents=True)
    mirror = _WandbMirror(config, root, "setup", attempt)
    mirror.log({"tokens": 0, "setup/connection_check": 1.0,
                "purpose": "Connection test only; no model training"}, 0)
    url = mirror.url
    mirror.finish()
    from urllib.parse import urlparse
    parts = urlparse(url).path.strip("/").split("/")
    run_path = "/".join((parts[0], parts[1], parts[3]))
    history = list(wandb.Api(timeout=30).run(run_path).scan_history())
    if mirror.failed or not any(row.get("setup/connection_check") == 1.0 for row in history):
        raise RuntimeError("W&B connection test was not visible in online history")
    result = {"status": "passed", "wandb_url": url,
              "project_url": url.rsplit("/runs/", 1)[0],
              "history_rows_verified": len(history), "gpu_used": False,
              "purpose": "Connection test only; no model training"}
    atomic_json(root / "result.json", result)
    write_bundle_manifest(root)
    volume.commit()
    return result


