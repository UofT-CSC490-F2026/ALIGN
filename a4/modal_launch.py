"""Run nanochat SFT or midtraining on the A3 FoX checkpoint in the Modal volume a4-base.

    modal run a4/modal_launch.py --script sft --mode dry          # 1 GPU, 5 steps, nothing saved to the model
    modal run a4/modal_launch.py --script sft --mode full --run a4-sft-default   # 8 GPUs, default settings

The volume is mounted as nanochat's base directory, so the scripts read
base_checkpoints/fox-d24 and tokenizer/ from it and write chatsft_checkpoints/ or
mid_checkpoints/ back into it. Every job has a timeout.
"""
import shlex
import subprocess
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
A3 = HERE.parent / "a3"
REVISION = "92d63d4e8bb4df75c3b71618f31ddde2378b2bcd"
MODEL_TAG, MODEL_STEP = "fox-d24", "5571"
VENV = "/root/nanochat/.venv/bin"

app = modal.App("a4-launch")
volume = modal.Volume.from_name("a4-base")
secrets = [modal.Secret.from_name("wandb-secret")]
image = (modal.Image.debian_slim(python_version="3.12")
         .apt_install("git", "build-essential")
         .pip_install("uv")
         .run_commands("git clone https://github.com/karpathy/nanochat.git /root/nanochat",
                       f"git -C /root/nanochat checkout {REVISION}")
         .workdir("/root/nanochat")
         .run_commands("uv sync --python 3.12 --frozen --extra gpu --no-install-project")
         .env({"PYTHONPATH": "/root/nanochat:/root/a3:/root/a3/A3part5:/root/a4",
               "NANOCHAT_BASE_DIR": "/vol", "HF_HOME": "/vol/hf", "OMP_NUM_THREADS": "1"})
         .add_local_dir(str(A3 / "pico"), "/root/a3/pico", ignore=["__pycache__", "*.pyc"])
         .add_local_file(str(A3 / "A3part5" / "fox.py"), "/root/a3/A3part5/fox.py")
         .add_local_file(str(HERE / "fox_cache.py"), "/root/a4/fox_cache.py")
         .add_local_file(str(HERE / "run_script.py"), "/root/a4/run_script.py"))

MODULES = {"sft": "scripts.chat_sft", "mid": "scripts.mid_train"}
# A few steps of tiny batches: checks that the model, data and evaluation run, not that it learns.
DRY_ARGS = {
    "sft": ["--load-optimizer", "0", "--num-iterations", "5", "--total-batch-size", "16384",
            "--eval-every", "-1", "--eval-tokens", "65536",
            "--chatcore-every", "5", "--chatcore-max-sample", "4", "--chatcore-max-cat", "8"],
    "mid": ["--dry-run", "--device-batch-size", "2", "--num-iterations", "5", "--total-batch-size", "16384",
            "--eval-every", "-1", "--eval-tokens", "65536"],
}
# mid_train.py does not inherit the pretraining batch size, so give it A3's per-GPU batch of 2.
FULL_ARGS = {"sft": [], "mid": ["--device-batch-size", "2"]}


def build_command(script, mode, run, extra):
    """The shell command for one job, as a list of arguments."""
    module = MODULES[script]
    arguments = ["--model-tag", MODEL_TAG, "--model-step", MODEL_STEP, "--run", run]
    if mode == "dry":
        return [f"{VENV}/python", "/root/a4/run_script.py", module, *arguments, *DRY_ARGS[script], *extra]
    return [f"{VENV}/torchrun", "--standalone", "--nproc_per_node=8", "/root/a4/run_script.py",
            module, *arguments, *FULL_ARGS[script], *extra]


def execute(command):
    if "scripts.mid_train" in command and not Path("/vol/identity_conversations.jsonl").is_file():
        raise FileNotFoundError("mid_train needs /vol/identity_conversations.jsonl in the a4-base volume")
    print("COMMAND", shlex.join(command), flush=True)
    try:
        subprocess.run(command, check=True)
    finally:
        volume.commit()  # keep checkpoints even if the job failed late


@app.function(image=image, gpu="H100", timeout=1800, volumes={"/vol": volume}, secrets=secrets, retries=0)
def one_gpu(command):
    execute(command)


@app.function(image=image, gpu="H100:8", timeout=3 * 3600, volumes={"/vol": volume}, secrets=secrets, retries=0)
def eight_gpus(command):
    execute(command)


@app.local_entrypoint()
def main(script: str = "sft", mode: str = "dry", run: str = "dummy", extra: str = ""):
    if script not in MODULES or mode not in ("dry", "full"):
        raise SystemExit("--script must be sft or mid, and --mode must be dry or full")
    command = build_command(script, mode, run, shlex.split(extra))
    (one_gpu if mode == "dry" else eight_gpus).remote(command)
