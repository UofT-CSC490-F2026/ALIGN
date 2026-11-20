"""Evaluate several saved models on the same tasks in one job.

    torchrun --standalone --nproc_per_node=8 a4/run_script.py eval_models --models base:fox-d24,sft:a4-sft-default

Each model is source:tag, with source base, sft, mid or rl (the folder under the nanochat base directory). The scores are
printed and saved to <base dir>/eval_results/<--run>.json.
"""
import argparse
import json
import os
import sys

import torch

from nanochat.checkpoint_manager import load_model_from_dir
from nanochat.common import autodetect_device_type, compute_cleanup, compute_init, get_base_dir, print0
from nanochat.engine import Engine
from scripts.chat_eval import run_chat_eval

FOLDERS = {"base": "base_checkpoints", "sft": "chatsft_checkpoints", "mid": "mid_checkpoints", "rl": "chatrl_checkpoints"}
CHANCE = {"ARC-Easy": 0.25, "ARC-Challenge": 0.25, "MMLU": 0.25, "GSM8K": 0.0}

parser = argparse.ArgumentParser()
parser.add_argument("--models", required=True)
parser.add_argument("--tasks", default="GSM8K|ARC-Easy|ARC-Challenge|MMLU")
parser.add_argument("--max-problems", type=int, default=None)
parser.add_argument("--run", default="eval")
args, _ = parser.parse_known_args()  # run_script.py's launcher also passes the training options

result_path = os.path.join(get_base_dir(), "eval_results", f"{args.run}.json")
if args.run != "dummy" and os.path.exists(result_path):
    raise FileExistsError(f"{result_path} already exists; use a new --run name")
_, rank, _, _, device = compute_init(autodetect_device_type())
results = {}
for entry in args.models.split(","):
    source, tag = entry.split(":")
    model, tokenizer, _ = load_model_from_dir(os.path.join(get_base_dir(), FOLDERS[source]), device, "eval", model_tag=tag)
    engine = Engine(model, tokenizer)
    scores = {task: run_chat_eval(task, model, tokenizer, engine, max_problems=args.max_problems)
              for task in args.tasks.split("|")}
    if all(task in scores for task in CHANCE):  # ChatCORE without HumanEval
        scores["ChatCORE_no_humaneval"] = sum((scores[t] - c) / (1 - c) for t, c in CHANCE.items()) / len(CHANCE)
    results[entry] = scores
    print0(f"RESULT {entry} {json.dumps(scores)}")
    del model, engine
    torch.cuda.empty_cache()
if rank == 0:
    os.makedirs(os.path.join(get_base_dir(), "eval_results"), exist_ok=True)
    with open(result_path, "w") as f:
        json.dump(results, f, indent=2)
compute_cleanup()
