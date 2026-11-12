"""Find out why some answers stop early: does the model itself predict the end token there, or is it a cache artifact?

    python a4/run_script.py diagnose_stops --models rl:a4-rl --answers-file <run>_rl_a4-rl.jsonl --max-problems 40

Takes the first problems whose saved answer (from dump_answers.py) has no '####', regenerates each with the cached
Engine, then checks the token right after the answer with a plain full forward pass (no cache).
"""
import argparse
import json
import os
from collections import Counter

import torch

from nanochat.checkpoint_manager import load_model_from_dir
from nanochat.common import autodetect_device_type, compute_init, get_base_dir
from nanochat.engine import Engine
from tasks.gsm8k import GSM8K

FOLDERS = {"sft": "chatsft_checkpoints", "mid": "mid_checkpoints", "rl": "chatrl_checkpoints"}
parser = argparse.ArgumentParser()
parser.add_argument("--models", required=True)
parser.add_argument("--max-problems", type=int, default=40)
parser.add_argument("--answers-file", default=None, help="file in <base dir>/answers written by dump_answers.py")
args, _ = parser.parse_known_args()

device = compute_init(autodetect_device_type())[-1]
task = GSM8K(subset="main", split="test")
source, tag = args.models.split(",")[0].split(":")
saved = [json.loads(line) for line in open(os.path.join(get_base_dir(), "answers", args.answers_file or f"{source}_{tag}.jsonl"))]
stopped = [row for row in saved if "####" not in row["answer"]][:args.max_problems]
model, tokenizer, _ = load_model_from_dir(os.path.join(get_base_dir(), FOLDERS[source]), device, "eval", model_tag=tag)
engine = Engine(model, tokenizer)

names = {tokenizer.encode_special(s): s for s in ["<|assistant_end|>", "<|bos|>", "<|python_start|>", "<|python_end|>"]}
agree, same_text, outcomes = 0, 0, Counter()
for row in stopped:
    prompt = tokenizer.render_for_completion(task[row["index"]])
    sequences, _ = engine.generate_batch(prompt, num_samples=1, max_tokens=512, temperature=0.0)
    ids = sequences[0]
    same_text += tokenizer.decode(ids[len(prompt):]) == row["answer"]
    with torch.no_grad():
        logits = model(torch.tensor([ids], device=device))[0, -1].float()  # full forward, no cache
    probs = logits.softmax(-1)
    top = int(probs.argmax())
    outcomes[names.get(top, "other token")] += 1
    print(f"idx {row['index']:4d} generated {len(ids) - len(prompt):3d} tokens | full forward predicts "
          f"{names.get(top, repr(tokenizer.decode([top])))} p={probs[top]:.2f} | last words: {row['answer'][-40:]!r}")
print(f"SUMMARY problems={len(stopped)} regenerated_text_identical={same_text} next_token_by_full_forward={dict(outcomes)}")
