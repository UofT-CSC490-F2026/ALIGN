"""Detached Part 4 preparation, training, and native evaluation."""
import hashlib
import json
from pathlib import Path
import re
import subprocess

import modal

HERE = Path(__file__).resolve().parent
REVISION = "92d63d4e8bb4df75c3b71618f31ddde2378b2bcd"
app = modal.App("a3-part4-baseline")
volume = modal.Volume.from_name("a3-part4-baseline", create_if_missing=True)
image = (modal.Image.debian_slim(python_version="3.12")
         .apt_install("git", "build-essential")
         .pip_install("uv")
         .run_commands("git clone https://github.com/karpathy/nanochat.git /root/nanochat",
                       f"git -C /root/nanochat checkout {REVISION}")
         .workdir("/root/nanochat")
         .run_commands("uv sync --python 3.12 --frozen --extra gpu --no-install-project")
         .env({"PATH": "/root/nanochat/.venv/bin:/usr/local/bin:/usr/bin:/bin",
               "PYTHONPATH": "/root/nanochat:/root/a3", "OMP_NUM_THREADS": "1",
               "NANOCHAT_BASE_DIR": "/vol/cache", "HF_HOME": "/vol/hf",
               "PYTORCH_ALLOC_CONF": "expandable_segments:True"})
         .add_local_dir(str(HERE.parent / "pico"), "/root/a3/pico",
                        ignore=["__pycache__", "*.pyc"]))
for filename in ("__init__.py", "train.py", "evaluate.py"):
    image = image.add_local_file(str(HERE / filename), f"/root/a3/A3part4/{filename}")


def activate():
    import site
    site.addsitedir("/root/nanochat/.venv/lib/python3.12/site-packages")


@app.function(image=image, volumes={"/vol": volume}, gpu="H100", cpu=4,
              memory=16384, timeout=3600, retries=0, max_containers=1,
              scaledown_window=2, include_source=True,
              secrets=[modal.Secret.from_name("a3-p4-wandb")])
def experiment(run_id: str):
    activate()
    volume.reload()
    from A3part4.train import train
    from A3part4.evaluate import evaluate
    output = Path("/vol/runs") / run_id
    training = train(output, volume.commit)
    evaluation = evaluate(output, volume.commit)
    return {"training": training, "evaluation": evaluation}


@app.function(image=image, volumes={"/vol": volume}, cpu=4, memory=16384,
              timeout=5400, retries=0, max_containers=1, scaledown_window=2,
              include_source=True, secrets=[modal.Secret.from_name("a3-p4-wandb")])
def pipeline(run_id: str):
    activate()
    import wandb
    from A3part4.train import check_baseline, write_json
    from nanochat.common import download_file_with_lock
    from scripts.base_eval import EVAL_BUNDLE_URL, place_eval_bundle
    output = Path("/vol/runs") / run_id
    output.mkdir(parents=True, exist_ok=False)
    try:
        write_json(output / "status.json", {"status": "preparing"})
        volume.commit()
        # Fail before allocating a GPU if account access or model compatibility fails.
        wandb.Api(timeout=30).viewer
        checks = check_baseline()
        print(json.dumps(checks), flush=True)
        subprocess.run(["python", "-m", "nanochat.dataset", "-n", "4"], check=True)
        if not Path("/vol/cache/tokenizer/tokenizer.pkl").exists():
            subprocess.run(["python", "-m", "scripts.tok_train", "--max-chars=100000000",
                            "--vocab-size=32768"], check=True)
        if not Path("/vol/cache/eval_bundle").exists():
            download_file_with_lock(EVAL_BUNDLE_URL, "eval_bundle.zip", postprocess_fn=place_eval_bundle)
        artifacts = []
        for path in sorted(Path("/vol/cache").rglob("*")):
            if path.is_file() and (path.suffix == ".parquet" or path.parent.name == "tokenizer"):
                digest = hashlib.sha256()
                with path.open("rb") as handle:
                    for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                        digest.update(chunk)
                artifacts.append({"path": str(path), "bytes": path.stat().st_size,
                                  "sha256": digest.hexdigest()})
        write_json(output / "preparation.json", {"upstream_commit": REVISION,
                   "checks": checks, "dataset_training_shards": 4,
                   "tokenizer_max_chars": 100000000, "vocab_size": 32768,
                   "artifacts": artifacts})
        write_json(output / "status.json", {"status": "prepared_waiting_for_gpu"})
        volume.commit()
        result = experiment.remote(run_id)
        volume.reload()
        write_json(output / "result.json", result)
        write_json(output / "status.json", {"status": "complete"})
        volume.commit()
        return result
    except Exception as error:
        volume.reload()
        write_json(output / "status.json", {"status": "failed", "error": str(error)})
        volume.commit()
        raise


@app.local_entrypoint()
def main(run_id: str = "baseline-context-seed42-v1"):
    if not re.fullmatch(r"[a-zA-Z0-9_-]+", run_id):
        raise ValueError("Use letters, digits, dashes, or underscores for the run ID")
    print(f"Outputs: volume a3-part4-baseline /runs/{run_id}", flush=True)
    print(json.dumps(pipeline.remote(run_id), indent=2))
