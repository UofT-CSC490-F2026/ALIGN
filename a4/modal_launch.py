"""Run nanochat SFT or midtraining on the A3 FoX checkpoint in the Modal volume a4-base.

    modal run a4/modal_launch.py --script sft --mode dry     # 1 GPU, one step, 30 minute limit
    modal run a4/modal_launch.py --script sft --mode smoke   # 8 GPUs, a few steps, 25 minute limit
    modal run a4/modal_launch.py --script sft --mode full --run a4-sft-default   # 8 GPUs, default settings
    modal run a4/modal_launch.py --script sft --mode full --run a4-sft-math --extra-data extra_math.jsonl
    modal run a4/modal_launch.py --script rl --mode full --run a4-rl --start a4-sft-default   # RL from a saved SFT model
    modal run --detach a4/modal_launch.py ... --spawn   # start and return: the job no longer depends on this terminal;
                                                        # follow it with: modal app logs <app id>
    modal run a4/modal_launch.py --script answers --mode full --run x --extra "--models sft:a4-sft-default"

The volume is mounted as nanochat's base directory, so the scripts read
base_checkpoints/fox-d24 and tokenizer/ from it and write chatsft_checkpoints/ or
mid_checkpoints/ back into it. The scripts name their output folder after the base model's tag, so a
full run moves it to a folder named after --run (RL then loads it with --model-tag <run>), and dry and
smoke runs delete it. A job refuses to start if it would overwrite an existing folder. Every job has a timeout.
"""
import re
import shlex
import shutil
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
         .add_local_file(str(HERE / "run_script.py"), "/root/a4/run_script.py")
         .add_local_file(str(HERE / "eval_models.py"), "/root/a4/eval_models.py")
         .add_local_file(str(HERE / "dump_answers.py"), "/root/a4/dump_answers.py")
         .add_local_file(str(HERE / "diagnose_stops.py"), "/root/a4/diagnose_stops.py")
         # midtraining was deleted from nanochat before the pinned commit; see a4/nanochat/README.md
         .add_local_file(str(HERE / "nanochat/scripts/mid_train.py"), "/root/nanochat/scripts/mid_train.py")
         # chat_rl.py with --rollouts-dir added (saves every sampled answer); see a4/nanochat/README.md
         .add_local_file(str(HERE / "nanochat/scripts/chat_rl.py"), "/root/nanochat/scripts/chat_rl.py")
         .add_local_file(str(HERE / "nanochat/tasks/customjson.py"), "/root/nanochat/tasks/customjson.py")
         .add_local_file(str(HERE / "nanochat/tasks/spellingbee.py"), "/root/nanochat/tasks/spellingbee.py"))

MODULES = {"sft": "scripts.chat_sft", "mid": "scripts.mid_train", "eval": "eval_models",
           "answers": "dump_answers", "diagnose": "diagnose_stops", "rl": "scripts.chat_rl"}
OUTPUT_ROOT = {"sft": "/vol/chatsft_checkpoints", "mid": "/vol/mid_checkpoints", "eval": "/vol/eval_results",  # eval and answers write no checkpoints
               "answers": "/vol/answers", "diagnose": "/vol/answers", "rl": "/vol/chatrl_checkpoints"}
# A few steps of tiny batches: checks that the model, data and evaluation run, not that it learns.
DRY_ARGS = {
    "sft": ["--load-optimizer", "0", "--num-iterations", "5", "--total-batch-size", "16384",
            "--eval-every", "-1", "--eval-tokens", "65536",
            "--chatcore-every", "5", "--chatcore-max-sample", "4", "--chatcore-max-cat", "8"],
    "mid": ["--dry-run", "--device-batch-size", "2", "--num-iterations", "5", "--total-batch-size", "16384",
            "--eval-every", "-1", "--eval-tokens", "65536"],
    "eval": ["--max-problems", "16"],
    "answers": ["--max-problems", "16"],
    "diagnose": ["--max-problems", "40"],
    "rl": ["--eval-examples", "8", "--save-every", "100000"],
}
# mid_train.py does not inherit the pretraining batch size, so give it A3's per-GPU batch of 2.
FULL_ARGS = {"sft": [], "mid": ["--device-batch-size", "2"], "eval": [], "answers": [], "diagnose": [], "rl": []}
# A few steps at the real settings on 8 GPUs (warm-start on). --num-iterations counts mini-batches and one
# step is 32 of them here, so 100 is about 3 steps. Evaluation is shortened; ChatCORE already ran in dry mode.
SMOKE_ARGS = {
    "sft": ["--num-iterations", "100", "--eval-every", "-1", "--eval-tokens", "1048576", "--chatcore-every", "-1"],
    "mid": ["--dry-run", "--num-iterations", "100", "--eval-every", "-1", "--eval-tokens", "1048576"],
    "eval": ["--max-problems", "64"],
    "answers": ["--max-problems", "64"],
    "rl": ["--eval-examples", "8", "--save-every", "100000"],
}


def build_command(script, mode, run, extra, extra_data="", start=""):
    """The shell command for one job, as a list of arguments. extra_data names a CustomJSON file in the volume
    that run_script.py adds to the training mix."""
    module = MODULES[script]
    prefix = ["env", f"EXTRA_DATA=/vol/{extra_data}"] if extra_data else []
    arguments = ["--model-tag", MODEL_TAG, "--model-step", MODEL_STEP, "--run", run]
    if script == "rl":  # RL starts from a saved SFT folder, whose last step is used
        arguments = ["--model-tag", start, "--run", run, "--rollouts-dir", f"/vol/rollouts/{run}"]
    if mode == "dry":
        return [*prefix, f"{VENV}/python", "/root/a4/run_script.py", module, *arguments, *DRY_ARGS[script], *extra]
    shortened = SMOKE_ARGS[script] if mode == "smoke" else []
    return [*prefix, f"{VENV}/torchrun", "--standalone", "--nproc_per_node=8", "/root/a4/run_script.py",
            module, *arguments, *FULL_ARGS[script], *shortened, *extra]


def execute(command, output_root, run, keep, tag):
    """Run one job, then move its output from <output_root>/<tag> to <output_root>/<run>, or delete it. The scripts
    name their output folder after the model tag they were given."""
    if "scripts.mid_train" in command and not Path("/vol/identity_conversations.jsonl").is_file():
        raise FileNotFoundError("mid_train needs /vol/identity_conversations.jsonl in the a4-base volume")
    source, target = Path(output_root) / tag, Path(output_root) / run
    if source.exists() or (keep and target.exists()):
        raise FileExistsError(f"{source} or {target} already exists; move or delete it first")
    for argument in command:
        if argument.startswith("EXTRA_DATA=") and not Path(argument.split("=", 1)[1]).is_file():
            raise FileNotFoundError(f"{argument} is not in the a4-base volume")
    print("COMMAND", shlex.join(command), flush=True)
    try:
        subprocess.run(command, check=True)
    finally:
        if source.exists():
            shutil.move(source, target) if keep else shutil.rmtree(source)
        volume.commit()  # keep what was written even if the job failed late


@app.function(image=image, gpu="H100", timeout=1800, volumes={"/vol": volume}, secrets=secrets, retries=0)
def one_gpu(command, output_root, run, keep, tag):
    execute(command, output_root, run, keep, tag)


@app.function(image=image, gpu="H100:8", timeout=25 * 60, volumes={"/vol": volume}, secrets=secrets, retries=0)
def eight_gpus_smoke(command, output_root, run, keep, tag):
    execute(command, output_root, run, keep, tag)


@app.function(image=image, gpu="H100:8", timeout=4 * 3600, volumes={"/vol": volume}, secrets=secrets, retries=0)
def eight_gpus(command, output_root, run, keep, tag):
    execute(command, output_root, run, keep, tag)


@app.local_entrypoint()
def main(script: str = "sft", mode: str = "dry", run: str = "dummy", extra: str = "", extra_data: str = "",
         start: str = "", spawn: bool = False):
    functions = {"dry": one_gpu, "smoke": eight_gpus_smoke, "full": eight_gpus}
    if script not in MODULES or mode not in functions:
        raise SystemExit("--script must be sft, mid, eval, answers or rl, and --mode must be dry, smoke or full")
    if mode == "full" and (run in ("dummy", MODEL_TAG) or not re.fullmatch(r"[A-Za-z0-9_-]+", run)):
        raise SystemExit(f"--run needs its own name (letters, digits, - and _), not dummy or {MODEL_TAG}")
    if script == "rl" and not start:
        raise SystemExit("--script rl needs --start <name of a folder in chatsft_checkpoints>")
    command = build_command(script, mode, run, shlex.split(extra), extra_data, start)
    call = functions[mode].spawn if spawn else functions[mode].remote
    call(command, OUTPUT_ROOT[script], run, mode == "full", start if script == "rl" else MODEL_TAG)
