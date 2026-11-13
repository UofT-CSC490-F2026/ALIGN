"""Find out what a model does on multiple-choice questions.

    python a4/run_script.py diagnose_mc --models sft:a4-sft-default,sft:a4-sft-math --stride 5

It repeats nanochat's MMLU scoring (the model's scores for the answer letters at the first answer position, highest
wins) on every <stride>-th test question and reports, for each model: how often each letter is chosen, how often each
letter is right, accuracy, the average score each letter gets, and the token the model would write first if it could
write anything (so we can see whether it even starts an answer with a letter).
"""
import argparse
import json
import os
from collections import Counter

import torch

from nanochat.checkpoint_manager import load_model_from_dir
from nanochat.common import autodetect_device_type, compute_init, get_base_dir
from tasks.mmlu import MMLU

FOLDERS = {"base": "base_checkpoints", "sft": "chatsft_checkpoints", "mid": "mid_checkpoints"}
parser = argparse.ArgumentParser()
parser.add_argument("--models", required=True)
parser.add_argument("--stride", type=int, default=5)
args, _ = parser.parse_known_args()

device = compute_init(autodetect_device_type())[-1]
task = MMLU(subset="all", split="test")
indices = list(range(0, len(task), args.stride))
for entry in args.models.split(","):
    source, tag = entry.split(":")
    model, tokenizer, _ = load_model_from_dir(os.path.join(get_base_dir(), FOLDERS[source]), device, "eval", model_tag=tag)
    bos = tokenizer.get_bos_token_id()
    letter_ids = {letter: tokenizer.encode(letter)[0] for letter in "ABCD"}
    chosen, gold, first_token = Counter(), Counter(), Counter()
    score_sum = {letter: 0.0 for letter in "ABCD"}
    correct = total = 0
    for start in range(0, len(indices), 16):
        conversations = [task[i] for i in indices[start:start + 16]]
        prompts = [tokenizer.render_for_completion(c) for c in conversations]
        width = max(len(p) for p in prompts)
        batch = torch.tensor([p + [bos] * (width - len(p)) for p in prompts], dtype=torch.long, device=device)
        with torch.no_grad():
            logits = model(batch)
        for row, (conversation, prompt) in enumerate(zip(conversations, prompts)):
            position = len(prompt) - 1
            scores = logits[row, position, [letter_ids[l] for l in conversation["letters"]]].float()
            predicted = conversation["letters"][int(scores.argmax())]
            chosen[predicted] += 1
            gold[conversation["messages"][-1]["content"]] += 1
            for letter, score in zip(conversation["letters"], scores.tolist()):
                score_sum[letter] += score
            first_token[tokenizer.decode([int(logits[row, position].argmax())])] += 1
            correct += int(task.evaluate(conversation, predicted))
            total += 1
    summary = {"gpu": torch.cuda.get_device_name(), "questions": total, "accuracy": round(correct / total, 4),
               "letter_chosen": {l: chosen[l] for l in "ABCD"}, "letter_correct": {l: gold[l] for l in "ABCD"},
               "mean_score_per_letter": {l: round(score_sum[l] / total, 2) for l in "ABCD"},
               "first_token_if_unrestricted": first_token.most_common(5)}
    print(f"MC {entry} {json.dumps(summary)}", flush=True)
    del model
    torch.cuda.empty_cache()
