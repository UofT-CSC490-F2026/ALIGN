"""Local-only launcher configuration and filesystem helpers. No Modal import."""
import json
import re
import shutil
from pathlib import Path
MODEL_TAG, MODEL_STEP = "fox-d24", "5571"
VENV = "/root/nanochat/.venv/bin"

MODULES = {"sft": "scripts.chat_sft", "mid": "scripts.mid_train", "eval": "eval_models",
           "answers": "dump_answers", "diagnose": "diagnose_stops", "mcdiag": "diagnose_mc", "rl": "scripts.chat_rl"}
OUTPUT_ROOT = {"sft": "/vol/chatsft_checkpoints", "mid": "/vol/mid_checkpoints", "eval": "/vol/eval_results",  # eval and answers write no checkpoints
               "answers": "/vol/answers", "diagnose": "/vol/answers", "mcdiag": "/vol/answers", "rl": "/vol/chatrl_checkpoints"}
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
    "mcdiag": ["--stride", "5"],
    "rl": ["--max-steps", "1", "--eval-examples", "8", "--save-every", "100000"],
}
# mid_train.py does not inherit the pretraining batch size, so give it A3's per-GPU batch of 2.
FULL_ARGS = {"sft": [], "mid": ["--device-batch-size", "2"], "eval": [], "answers": [], "diagnose": [], "mcdiag": [], "rl": []}
# A few steps at the real settings on 8 GPUs (warm-start on). --num-iterations counts mini-batches and one
# step is 32 of them here, so 100 is about 3 steps. Evaluation is shortened; ChatCORE already ran in dry mode.
SMOKE_ARGS = {
    "sft": ["--num-iterations", "100", "--eval-every", "-1", "--eval-tokens", "1048576", "--chatcore-every", "-1"],
    "mid": ["--dry-run", "--num-iterations", "100", "--eval-every", "-1", "--eval-tokens", "1048576"],
    "eval": ["--max-problems", "64"],
    "answers": ["--max-problems", "64"],
    "rl": ["--max-steps", "1", "--eval-examples", "8", "--save-every", "100000"],
}


# Scripts that are not written for several processes. Under torchrun every process would repeat the same work and print the
# same output, so they always run as one process on one GPU.
SINGLE_PROCESS = {"mcdiag"}


def build_command(script, mode, run, extra, extra_data="", start="", mix_seed=-1, reward_mode="baseline", completion_weight=.15, arithmetic_weight=.20):
    """The shell command for one job, as a list of arguments. extra_data names a CustomJSON file in the volume
    that run_script.py adds to the training mix."""
    if mode not in ('dry', 'smoke', 'full'):
        raise ValueError('invalid mode')
    if script == 'rl':
        from argparse import Namespace
        from reward_integration import validate_reward_arguments, MODES
        if reward_mode not in MODES:
            raise ValueError('invalid reward mode')
        if not start or run in ('dummy', start) or not re.fullmatch(r'[A-Za-z0-9_-]+', run):
            raise ValueError('RL needs a unique run name and a starting SFT checkpoint')
        validate_reward_arguments(Namespace(reward_mode=reward_mode, completion_weight=completion_weight,
                                            arithmetic_weight=arithmetic_weight, max_steps=None, output_tag=run))
        protected = {'--model-tag', '--output-tag', '--run', '--rollouts-dir', '--reward-mode',
                     '--completion-weight', '--arithmetic-weight', '--resume-from', '--resume-step'}
        if mode != 'full':
            protected.add('--max-steps')
        if any(arg.split('=')[0] in protected for arg in extra):
            raise ValueError('extra arguments cannot override experiment identity, rewards, or smoke cap')
    module = MODULES[script]
    variables = ([f"EXTRA_DATA=/vol/{extra_data}"] if extra_data else []) + ([f"MIX_SEED={mix_seed}"] if mix_seed >= 0 else [])
    prefix = ["env", *variables] if variables else []
    arguments = ["--model-tag", MODEL_TAG, "--model-step", MODEL_STEP, "--run", run]
    if script == "rl":  # RL starts from a saved SFT folder, whose last step is used
        arguments = ["--model-tag", start, "--run", run, "--rollouts-dir", f"/vol/rollouts/{run}",
                     "--output-tag", run, "--reward-mode", reward_mode,
                     "--completion-weight", str(completion_weight), "--arithmetic-weight", str(arithmetic_weight)]
    if mode == "dry":
        return [*prefix, f"{VENV}/python", "/root/a4/run_script.py", module, *arguments, *DRY_ARGS[script], *extra]
    if script in SINGLE_PROCESS:
        return [*prefix, f"{VENV}/python", "/root/a4/run_script.py", module, *arguments, *FULL_ARGS[script], *extra]
    shortened = SMOKE_ARGS[script] if mode == "smoke" else []
    return [*prefix, f"{VENV}/torchrun", "--standalone", "--nproc_per_node=8", "/root/a4/run_script.py",
            module, *arguments, *FULL_ARGS[script], *shortened, *extra]


def latest_checkpoint(output_root, names):
    """The newest fully saved step among the folders called `names` under output_root, as (folder name, step), or None.
    nanochat writes model_<step>.pt first and meta_<step>.json after it, so a step with no readable meta file was cut
    off mid-save and is skipped."""
    found = []
    for name in names:
        for path in (Path(output_root) / name).glob("model_*.pt"):
            meta = path.with_name(path.name.replace("model_", "meta_").replace(".pt", ".json"))
            try:
                json.loads(meta.read_text())
            except (OSError, ValueError):
                continue
            found.append((int(path.stem.split("_")[1]), name))
    return (max(found)[1], max(found)[0]) if found else None


def merge_move(source, target):
    """Move the files in source into target, creating it if needed, and remove source."""
    target.mkdir(parents=True, exist_ok=True)
    collisions = [path.name for path in source.iterdir() if (target / path.name).exists()]
    if collisions:
        raise FileExistsError(f'refusing to overwrite existing files: {collisions}')
    for path in source.iterdir():
        shutil.move(path, target / path.name)
    source.rmdir()


def mark_run(target, run, completed):
    """Leave a file named COMPLETE or INCOMPLETE in a run folder, replacing the other one."""
    marker, stale = ("COMPLETE", "INCOMPLETE") if completed else ("INCOMPLETE", "COMPLETE")
    (target / stale).unlink(missing_ok=True)
    (target / marker).write_text(f"{run}\n")
