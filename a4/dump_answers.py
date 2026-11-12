"""Save every answer a model writes to the GSM8K test problems, for error analysis.

    torchrun --standalone --nproc_per_node=8 a4/run_script.py dump_answers --models sft:a4-sft-default2

Each model is source:tag, with source base, sft, mid or rl. Decoding is greedy. The answers are written to
<base dir>/answers/<run>_<source>_<tag>.jsonl (the run name keeps later runs from overwriting earlier ones), one line per problem: index, question, reference answer, the model's answer and
whether it was right.
"""
import argparse
import json
import os

import torch.distributed as dist

from nanochat.checkpoint_manager import load_model_from_dir
from nanochat.common import autodetect_device_type, compute_cleanup, compute_init, get_base_dir, print0
from nanochat.engine import Engine
from tasks.gsm8k import GSM8K

FOLDERS = {"base": "base_checkpoints", "sft": "chatsft_checkpoints", "mid": "mid_checkpoints", "rl": "chatrl_checkpoints"}

parser = argparse.ArgumentParser()
parser.add_argument("--models", required=True)
parser.add_argument("--max-problems", type=int, default=None)
parser.add_argument("--max-new-tokens", type=int, default=512)
parser.add_argument("--run", default="dummy")
args, _ = parser.parse_known_args()  # run_script.py's launcher also passes the training options

ddp, rank, _, world_size, device = compute_init(autodetect_device_type())
task = GSM8K(subset="main", split="test")
count = len(task) if args.max_problems is None else min(len(task), args.max_problems)
out_dir = os.path.join(get_base_dir(), "answers")
os.makedirs(out_dir, exist_ok=True)

for entry in args.models.split(","):
    source, tag = entry.split(":")
    stem = f"{args.run}_{source}_{tag}"
    if args.run != "dummy" and os.path.exists(os.path.join(out_dir, f"{stem}.jsonl")):
        raise FileExistsError(f"{stem}.jsonl already exists in {out_dir}; use a new --run name")
    model, tokenizer, _ = load_model_from_dir(os.path.join(get_base_dir(), FOLDERS[source]), device, "eval", model_tag=tag)
    engine = Engine(model, tokenizer)
    shard = os.path.join(out_dir, f"{stem}.rank{rank}.jsonl")
    with open(shard, "w") as f:
        for i in range(rank, count, world_size):
            conversation = task[i]
            prompt = tokenizer.render_for_completion(conversation)
            sequences, _ = engine.generate_batch(prompt, num_samples=1, max_tokens=args.max_new_tokens, temperature=0.0)
            answer = tokenizer.decode(sequences[0][len(prompt):])
            f.write(json.dumps({
                "index": i,
                "question": conversation["messages"][0]["content"],
                "reference": conversation["messages"][-1]["content"][-1]["text"],
                "answer": answer,
                "correct": task.evaluate(conversation, answer),
            }) + "\n")
    if ddp:
        dist.barrier()
    if rank == 0:  # merge the per-GPU shards into one file in problem order
        rows = []
        for r in range(world_size):
            shard_path = os.path.join(out_dir, f"{stem}.rank{r}.jsonl")
            with open(shard_path) as f:
                rows += [json.loads(line) for line in f]
            os.remove(shard_path)
        rows.sort(key=lambda row: row["index"])
        with open(os.path.join(out_dir, f"{stem}.jsonl"), "w") as f:
            f.writelines(json.dumps(row) + "\n" for row in rows)
        print0(f"ANSWERS {entry} {sum(r['correct'] for r in rows)}/{len(rows)}")
    del model, engine
compute_cleanup()
