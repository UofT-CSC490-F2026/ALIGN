"""Run nanochat training (BILLABLE: never use this module for a local preview).

RL dry/smoke each cap optimizer updates to 1, retain isolated outputs,
and refuse existing experiment folders. Use preview_launch.py for local/free commands.

Run nanochat SFT or midtraining on the A3 FoX checkpoint in the Modal volume a4-base.

    modal run a4/modal_launch.py --script sft --mode dry     # 1 GPU, one step, 30 minute limit
    modal run a4/modal_launch.py --script sft --mode smoke   # 8 GPUs, 10 minute limit
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
A run folder holds a file named COMPLETE if the job finished and INCOMPLETE if it failed or was stopped, so a partial
run is never mistaken for a finished one.
"""
import json
import re
import shlex
import shutil
import subprocess
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
A3 = HERE.parent / "a3"
REVISION = "92d63d4e8bb4df75c3b71618f31ddde2378b2bcd"

# "H100!" asks for exactly H100s. Plain "H100" lets Modal upgrade a job to an H200, and two of our SFT runs got one.
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
         .add_local_file(str(HERE / "rewards.py"), "/root/a4/rewards.py")
         .add_local_file(str(HERE / "arithmetic_reward.py"), "/root/a4/arithmetic_reward.py")
         .add_local_file(str(HERE / "reward_integration.py"), "/root/a4/reward_integration.py")
         .add_local_file(str(HERE / "part5_distillation.py"), "/root/a4/part5_distillation.py")
         .add_local_file(str(HERE / "launcher_config.py"), "/root/a4/launcher_config.py")
         .add_local_file(str(HERE / "run_script.py"), "/root/a4/run_script.py")
         .add_local_file(str(HERE / "eval_models.py"), "/root/a4/eval_models.py")
         .add_local_file(str(HERE / "dump_answers.py"), "/root/a4/dump_answers.py")
         .add_local_file(str(HERE / "diagnose_stops.py"), "/root/a4/diagnose_stops.py")
         .add_local_file(str(HERE / "diagnose_mc.py"), "/root/a4/diagnose_mc.py")
         # midtraining was deleted from nanochat before the pinned commit; see a4/nanochat/README.md
         .add_local_file(str(HERE / "nanochat/scripts/mid_train.py"), "/root/nanochat/scripts/mid_train.py")
         # chat_rl.py with --rollouts-dir added (saves every sampled answer); see a4/nanochat/README.md
         .add_local_file(str(HERE / "nanochat/scripts/chat_rl.py"), "/root/nanochat/scripts/chat_rl.py")
         .add_local_file(str(HERE / "nanochat/scripts/chat_distill.py"), "/root/nanochat/scripts/chat_distill.py")
         .add_local_file(str(HERE / "nanochat/tasks/customjson.py"), "/root/nanochat/tasks/customjson.py")
         .add_local_file(str(HERE / "nanochat/tasks/spellingbee.py"), "/root/nanochat/tasks/spellingbee.py"))

from launcher_config import (MODEL_TAG, MODEL_STEP, VENV, MODULES, OUTPUT_ROOT, DRY_ARGS, FULL_ARGS, SMOKE_ARGS,
                             SINGLE_PROCESS, build_command, latest_checkpoint, merge_move, mark_run, plan_recovery)


def execute_isolated_rl(command, output_root, run):
    """Isolated training with explicit resume and guarded Modal preemption recovery.

    The original launch command is retained for restart comparison. Existing
    completed runs, changed configurations, and incomplete checkpoints fail
    closed. Running two concurrent jobs with the same name is not supported.
    """
    target = Path(output_root) / run
    original_command = list(command)
    resume_source = ""
    resume_step = None
    if "--resume-from" in command:
        resume_source = command[command.index("--resume-from") + 1]
        resume_step = int(command[command.index("--resume-step") + 1])
    command, resuming = plan_recovery(
        output_root, "/vol/rollouts", run, original_command,
        resume_source, resume_step
    )
    if not target.exists():
        target.mkdir(parents=True, exist_ok=False)
        mark_run(target, run, False)
        (target / "launch_command.json").write_text(json.dumps(original_command))
        volume.commit()
    if resuming:
        print(f"RESUMING {run} from {command[-3]} step {command[-1]}", flush=True)
        print("NOTE: optimizer state is not restored", flush=True)
    completed = False
    try:
        process = subprocess.Popen(command)
        while True:
            try:
                code = process.wait(timeout=120)
                break
            except subprocess.TimeoutExpired:
                volume.commit()
        if code:
            raise subprocess.CalledProcessError(code, command)
        completed = True
    finally:
        mark_run(target, run, completed)
        volume.commit()


def execute(command, output_root, run, keep, tag):
    """Run one job, then move its output from <output_root>/<tag> to <output_root>/<run>, or delete it. The scripts
    name their output folder after the model tag they were given."""
    if "scripts.chat_rl" in command or "scripts.chat_distill" in command:
        return execute_isolated_rl(command, output_root, run)
    if "scripts.mid_train" in command and not Path("/vol/identity_conversations.jsonl").is_file():
        raise FileNotFoundError("mid_train needs /vol/identity_conversations.jsonl in the a4-base volume")
    source, target = Path(output_root) / tag, Path(output_root) / run
    if (target / "COMPLETE").exists():
        raise FileExistsError(f"{target} is already a finished run; use a new --run name")
    if source.exists() or (keep and target.exists()):
        raise FileExistsError(f"{source} or {target} already exists; move or delete it first")
    for argument in command:
        if argument.startswith("EXTRA_DATA=") and not Path(argument.split("=", 1)[1]).is_file():
            raise FileNotFoundError(f"{argument} is not in the a4-base volume")
    print("COMMAND", shlex.join(command), flush=True)
    completed = False
    try:
        # Commit the volume every two minutes while the job runs. Modal can preempt a GPU job at any time, and
        # anything not committed by then (checkpoints, saved answers) is lost.
        process = subprocess.Popen(command)
        while True:
            try:
                code = process.wait(timeout=120)
                break
            except subprocess.TimeoutExpired:
                volume.commit()
        if code:
            raise subprocess.CalledProcessError(code, command)
        completed = True
    finally:
        if source.exists():
            if keep:
                merge_move(source, target)
                # A failed, stopped or preempted job leaves partial checkpoints; say so, so it is never mistaken for a
                # finished run. A resumed job that finishes replaces the earlier INCOMPLETE marker.
                mark_run(target, run, completed)
            else:
                shutil.rmtree(source)
        volume.commit()  # keep what was written even if the job failed late


@app.function(image=image, gpu="H100!", timeout=1800, volumes={"/vol": volume}, secrets=secrets, retries=0)
def one_gpu(command, output_root, run, keep, tag):
    execute(command, output_root, run, keep, tag)


@app.function(image=image, gpu="H100!:8", timeout=10 * 60, volumes={"/vol": volume}, secrets=secrets, retries=0)
def eight_gpus_smoke(command, output_root, run, keep, tag):
    execute(command, output_root, run, keep, tag)


@app.function(image=image, gpu="H100!:8", timeout=4 * 3600, volumes={"/vol": volume}, secrets=secrets, retries=0)
def eight_gpus(command, output_root, run, keep, tag):
    execute(command, output_root, run, keep, tag)


@app.local_entrypoint()
def main(script: str = "sft", mode: str = "dry", run: str = "dummy", extra: str = "", extra_data: str = "",
         start: str = "", spawn: bool = False, mix_seed: int = -1,
         reward_mode: str = "baseline", completion_weight: float = .15, arithmetic_weight: float = .20,
         teacher_tag: str = "", resume_from: str = "", resume_step: int = -1):
    functions = {"dry": one_gpu, "smoke": eight_gpus_smoke, "full": eight_gpus}
    if script not in MODULES or mode not in functions:
        raise SystemExit("--script must be sft, mid, eval, answers, rl or distill; mode dry, smoke or full")
    if mode == "full" and (run in ("dummy", MODEL_TAG) or not re.fullmatch(r"[A-Za-z0-9_-]+", run)):
        raise SystemExit(f"--run needs its own name (letters, digits, - and _), not dummy or {MODEL_TAG}")
    if script in ("rl", "distill") and not start:
        raise SystemExit("RL/distill needs --start <SFT checkpoint folder>")
    if script == "distill" and not teacher_tag:
        raise SystemExit("--script distill needs --teacher-tag <completed RL folder>")
    command = build_command(script, mode, run, shlex.split(extra), extra_data, start, mix_seed,
                            reward_mode, completion_weight, arithmetic_weight, teacher_tag,
                            resume_from, None if resume_step == -1 else resume_step)
    function = one_gpu if script in SINGLE_PROCESS else functions[mode]  # one GPU is enough for these scripts
    call = function.spawn if spawn else function.remote
    call(command, OUTPUT_ROOT[script], run, mode == "full", start if script in ("rl", "distill") else MODEL_TAG)
