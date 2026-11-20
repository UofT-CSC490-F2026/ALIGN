"""Build the control file for the extra-data comparison.

    python a4/prepare_control_data.py extra_math.jsonl control_chat.jsonl

The extra-data runs train for more steps than the baselines, so part of any gain could come from the extra training and
not from the math. This builds a file with the same number of tokens as extra_math.jsonl, made of chat conversations
that are already in nanochat's SFT mix (SmolTalk train, repeated), so it adds training steps and no new information.
Needs nanochat on PYTHONPATH and NANOCHAT_BASE_DIR set to a folder with the tokenizer.
"""
import json
import random
import sys

from nanochat.tokenizer import get_tokenizer
from tasks.smoltalk import SmolTalk


def main(math_path, output_path, seed=0):
    tokenizer = get_tokenizer()
    target = sum(len(tokenizer.render_conversation({"messages": json.loads(line)})[0]) for line in open(math_path))
    task = SmolTalk(split="train")
    order = list(range(len(task)))
    random.Random(seed).shuffle(order)
    total = rows = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for i in order:
            messages = [m for m in task[i]["messages"] if m["role"] != "system"]  # CustomJSON has no system role
            if len(messages) % 2:
                messages = messages[:-1]
            if len(messages) < 2 or not all(isinstance(m["content"], str) for m in messages):
                continue
            total += len(tokenizer.render_conversation({"messages": messages})[0])
            f.write(json.dumps(messages) + "\n")
            rows += 1
            if total >= target:
                break
    print(f"{rows} rows, {total:,} tokens (target {target:,}) -> {output_path}")


if __name__ == "__main__":
    main(*sys.argv[1:3])
