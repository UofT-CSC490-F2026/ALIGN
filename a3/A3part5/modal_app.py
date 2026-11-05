"""One detached pipeline: CPU preparation, small reference, 8-H100 training, BPB."""
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time

import modal

HERE = Path(__file__).resolve().parent
REVISION = "92d63d4e8bb4df75c3b71618f31ddde2378b2bcd"
app = modal.App("a3-part5-fox")
volume = modal.Volume.from_name("a3-part5-fox", create_if_missing=True)
secret = modal.Secret.from_name("a3-p4-wandb")
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
               "PYTORCH_ALLOC_CONF": "expandable_segments:True",
               "TORCHINDUCTOR_CACHE_DIR": "/vol/compile", "TRITON_CACHE_DIR": "/vol/triton"})
         .add_local_dir(str(HERE.parent / "pico"), "/root/a3/pico", ignore=["__pycache__", "*.pyc"]))
for filename in ("__init__.py", "fox.py", "run_train.py", "evaluate.py",
                 "evaluation_questions.json", "scaling_prediction.json"):
    image = image.add_local_file(str(HERE / filename), f"/root/a3/A3part5/{filename}")


def activate():
    import site
    site.addsitedir("/root/nanochat/.venv/lib/python3.12/site-packages")


def record(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def execute(command, output, env=None, timeout=None):
    """Persist native stdout and commit after complete checkpoint notifications."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    record(output / "command.json", command)
    print("COMMAND " + " ".join(command), flush=True)
    last_commit = time.monotonic()
    with (output / "console.log").open("a", buffering=1) as log:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, env=env)
        try:
            for line in process.stdout:
                log.write(line)
                print(line, end="", flush=True)
                if re.search(r"(?:loss:|Validation bpb:)\s*(?:nan|inf)", line, re.I):
                    raise FloatingPointError("Nonfinite native training/evaluation metric")
                if "P5_CHECKPOINT_READY" in line or time.monotonic() - last_commit > 60:
                    volume.commit()
                    last_commit = time.monotonic()
            code = process.wait(timeout=timeout)
            if code:
                raise subprocess.CalledProcessError(code, command)
        finally:
            if process.poll() is None:
                process.terminate()
            volume.commit()


def train_command(run_name, smoke=False):
    command = ["torchrun", "--standalone", "--nproc_per_node=8", "-m", "A3part5.run_train",
               "--depth=24", "--max-seq-len=2048", "--target-param-data-ratio=8",
               "--device-batch-size=2", "--fp8", "--core-metric-every=-1", "--sample-every=-1",
               "--eval-every=500", "--eval-tokens=262144", "--save-every=500",
               "--model-tag=fox-d24", f"--run={run_name}"]
    if smoke:
        command += ["--num-iterations=12"]
    return command


def train_environment(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for name in ("tokenizer", "base_data_climbmix"):
        target = output / name
        if not target.exists():
            target.symlink_to(Path("/vol/cache") / name, target_is_directory=True)
    return dict(os.environ, NANOCHAT_BASE_DIR=str(output), P5_OUTPUT=str(output))


@app.function(image=image, gpu="H100", cpu=4, memory=32768, timeout=1800,
              volumes={"/vol": volume}, secrets=[secret], retries=0,
              max_containers=1, scaledown_window=2)
def pico_reference(run_id):
    volume.reload()
    output = Path("/vol/runs") / run_id / "pico"
    execute(["python", "-m", "A3part5.evaluate", "pico", str(output)], output)


@app.function(image=image, gpu="H100:8", cpu=16, memory=131072, timeout=1200,
              volumes={"/vol": volume}, secrets=[secret], retries=0,
              max_containers=1, scaledown_window=2)
def rehearsal(run_id, attempt="smoke"):
    volume.reload()
    output = Path("/vol/runs") / run_id / attempt
    execute(train_command(run_id + "-smoke", smoke=True), output, train_environment(output))
    # Exclude compilation/startup; use the last three native step timers.
    times = re.findall(r"dt: ([0-9.]+)ms", (output / "console.log").read_text())
    if len(times) < 12:
        raise RuntimeError("Distributed rehearsal did not complete twelve updates")
    seconds = sum(float(t) for t in times[-3:]) / 3000
    model = json.loads((output.parent / "configuration.json").read_text())
    estimate = seconds * model["planned_updates"] * 1.3
    report = {"steady_step_seconds": seconds, "training_seconds_with_30pct_allowance": estimate,
              "training_compute_estimate_usd": estimate * (8*.001097 + 16*.0000131 + 128*.00000222)}
    record(output / "rehearsal.json", report)
    volume.commit()
    if estimate > 9 * 3600:
        raise RuntimeError("Measured run exceeds the bounded training allocation; main training was not launched")
    return report


@app.function(image=image, gpu="H100:8", cpu=16, memory=131072, timeout=36000,
              volumes={"/vol": volume}, secrets=[secret], retries=0,
              max_containers=1, scaledown_window=2)
def training(run_id):
    volume.reload()
    output = Path("/vol/runs") / run_id
    record(output / "status.json", {"status": "training", "run_id": run_id})
    volume.commit()
    execute(train_command(run_id), output, train_environment(output))
    record(output / "status.json", {"status": "trained"})
    volume.commit()


@app.function(image=image, gpu="H100", cpu=4, memory=32768, timeout=1800,
              volumes={"/vol": volume}, secrets=[secret], retries=0,
              max_containers=1, scaledown_window=2)
def evaluation(run_id):
    volume.reload()
    output = Path("/vol/runs") / run_id
    record(output / "status.json", {"status": "evaluating_bpb_and_questions"})
    volume.commit()
    execute(["python", "-m", "A3part5.evaluate", "nano", str(output)],
            output / "evaluation-log", train_environment(output))
    return json.loads((output / "evaluation.json").read_text())


@app.function(image=image, cpu=8, memory=65536, timeout=43200,
              volumes={"/vol": volume}, secrets=[secret], retries=0,
              max_containers=1, scaledown_window=2)
def pipeline(run_id):
    activate()
    import torch
    import wandb
    from nanochat.gpt import GPTConfig
    from A3part5.fox import install, verify, FoXGPT
    output = Path("/vol/runs") / run_id
    output.mkdir(parents=True, exist_ok=False)
    try:
        record(output / "status.json", {"status": "preparing"})
        volume.commit()
        wandb.Api(timeout=30).viewer
        install()
        checks = verify()
        with torch.device("meta"):
            model = FoXGPT(GPTConfig(n_layer=24, n_embd=1536, n_head=12, n_kv_head=12))
            ref = FoXGPT(GPTConfig(n_layer=12, n_embd=768, n_head=6, n_kv_head=6))
        counts = model.num_scaling_params()
        ref_counts = ref.num_scaling_params()
        nscale = counts["transformer_matrices"] + counts["lm_head"]
        nref = ref_counts["transformer_matrices"] + ref_counts["lm_head"]
        batch = 2 ** round(__import__('math').log2(2**19 * (nscale/nref)**.383))
        config = {"upstream_commit": REVISION, "architecture": "Part 2 FoX adaptation",
                  "depth": 24, "width": 1536, "heads": 12, "context": 2048,
                  "counts": counts, "seed": 42, "training_shards": 170,
                  "target_param_data_ratio": 8, "total_batch_size": batch,
                  "planned_updates": 8*nscale//batch, "planned_tokens": (8*nscale//batch)*batch,
                  "device_batch_size": 2, "gradient_checkpointing": True,
                  "precision": "native FP8 linear conversion, BF16 attention/embeddings; FP32 attention scores",
                  "budget_cap_usd": 500, "checks": checks,
                  "gpu": "H100:8", "core_disabled": True, "sft": False}
        record(output / "configuration.json", config)
        print(json.dumps(config), flush=True)
        execute(["python", "-m", "nanochat.dataset", "-n", "8"], output / "prepare")
        if not Path("/vol/cache/tokenizer/tokenizer.pkl").exists():
            execute(["python", "-m", "scripts.tok_train"], output / "prepare")
        execute(["python", "-m", "scripts.tok_eval"], output / "prepare")
        execute(["python", "-m", "nanochat.dataset", "-n", "170"], output / "prepare")
        hashes = {}
        for path in sorted(Path("/vol/cache/tokenizer").glob("*")):
            if path.is_file(): hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        record(output / "tokenizer_hashes.json", hashes)
        record(output / "status.json", {"status": "pico_reference"})
        volume.commit()
        pico_reference.remote(run_id)
        volume.reload()
        pico = json.loads((output / "pico/comparison.json").read_text())
        prediction = json.loads(Path("/root/a3/A3part5/scaling_prediction.json").read_text())
        prediction.update(pico_bpb_512=pico["bpb"][0]["bpb"],
                          nano_transformer_params=counts["transformer_matrices"],
                          pico_transformer_params=pico["counts"]["transformer_matrices"],
                          recorded_before_main_training=True)
        prediction["predicted_bpb_512"] = prediction["pico_bpb_512"] * (
            prediction["nano_transformer_params"] / prediction["pico_transformer_params"])**(-prediction["exponent"])
        record(output / "scaling_prediction.json", prediction)
        record(output / "status.json", {"status": "ready_for_main_training"})
        volume.commit()
        training.remote(run_id)
        result = evaluation.remote(run_id)
        volume.reload()
        record(output / "status.json", {"status": "complete", "question_grading": "manual review pending"})
        volume.commit()
        return result
    except Exception as error:
        volume.reload()
        record(output / "status.json", {"status": "failed", "error": str(error)})
        volume.commit()
        raise


@app.function(image=image, cpu=1, memory=2048, timeout=41400,
              volumes={"/vol": volume}, secrets=[secret], retries=0,
              max_containers=1, scaledown_window=2)
def continue_prepared(run_id):
    """Explicit recovery after cancellation; reuse completed preparation only."""
    volume.reload()
    output = Path("/vol/runs") / run_id
    for name in ("configuration.json", "scaling_prediction.json", "pico/comparison.json", "pico/answers.json"):
        if not (output / name).is_file():
            raise RuntimeError(f"Missing prerequisite: {name}")
    if (output / "console.log").exists():
        raise RuntimeError("Main training has already started; inspect its checkpoints before recovery")
    try:
        record(output / "status.json", {"status": "ready_for_main_training", "recovery": True})
        volume.commit()
        training.remote(run_id)
        result = evaluation.remote(run_id)
        volume.reload()
        record(output / "status.json", {"status": "complete", "question_grading": "manual review pending"})
        volume.commit()
        return result
    except Exception as error:
        volume.reload()
        record(output / "status.json", {"status": "failed", "error": str(error)})
        volume.commit()
        raise


@app.local_entrypoint()
def main(run_id: str = "fox-d24-seed42-v1", resume_prepared: bool = False):
    if not re.fullmatch(r"[A-Za-z0-9_-]+", run_id):
        raise ValueError("Invalid run ID")
    call = (continue_prepared if resume_prepared else pipeline).spawn(run_id)
    print(f"Submitted remote pipeline: {call.object_id}", flush=True)
