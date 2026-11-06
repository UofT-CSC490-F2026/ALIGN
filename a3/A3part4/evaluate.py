"""Resumable saved-checkpoint comparisons using native nanochat BPB."""
import json
import math
from pathlib import Path

import wandb
from nanochat.common import compute_init
from nanochat.checkpoint_manager import load_checkpoint
from nanochat.dataloader import tokenizing_distributed_data_loader_bos_bestfit
from nanochat.loss_eval import evaluate_bpb
from nanochat.tokenizer import get_tokenizer, get_token_bytes
from pico.variants import build
from A3part4.train import PROJECT, VALIDATION_TOKENS, write_json


def completed_results(output):
    """Accept only complete cells matching this experiment's fixed protocol."""
    path = Path(output) / "bpb.json"
    rows = json.loads(path.read_text()) if path.exists() else []
    seen = set()
    for row in rows:
        key = (row["checkpoint_step"], row["evaluation_context"])
        if (key in seen or key[0] not in (500, 1000) or key[1] not in (512, 2048)
                or row["training_context"] != (512 if key[0] == 500 else 2048)
                or row["nominal_validation_tokens"] != VALIDATION_TOKENS
                or not math.isfinite(row["bpb"]) or row["bpb"] < 0):
            raise ValueError("Saved BPB results do not match the Part 4 protocol")
        seen.add(key)
    return rows


def evaluate(output, commit):
    output = Path(output)
    results = completed_results(output)
    completed = {(r["checkpoint_step"], r["evaluation_context"]) for r in results}
    if len(completed) == 4:
        return {"bpb": results}
    compute_init("cuda")
    tokenizer = get_tokenizer()
    token_bytes = get_token_bytes(device="cuda")
    run = wandb.init(project=PROJECT, name=output.name + "-evaluation",
                     group=output.name, dir=str(output), mode="online",
                     config={"bpb_tokens": VALIDATION_TOKENS, "batch_size": 8,
                             "contexts": [512, 2048]})
    try:
        for step in (500, 1000):
            if all((step, context) in completed for context in (512, 2048)):
                continue
            weights, _, metadata = load_checkpoint(str(output / "checkpoints"), step,
                                                    device="cuda", load_optimizer=False)
            for context in (512, 2048):
                if (step, context) in completed:
                    continue
                write_json(output / "status.json", {"status": "evaluating_bpb",
                           "checkpoint_step": step, "evaluation_context": context,
                           "wandb_url": run.url})
                commit()
                model = build("baseline", seed=42, seq_len=context,
                              vocab_size=tokenizer.get_vocab_size())
                model.load_state_dict(weights, strict=True)
                model.eval()
                loader = tokenizing_distributed_data_loader_bos_bestfit(tokenizer, 8, context, "val")
                bpb = evaluate_bpb(model, loader, VALIDATION_TOKENS // (8 * context), token_bytes)
                row = dict(checkpoint_step=step, training_context=metadata["model_config"]["sequence_len"],
                           evaluation_context=context, nominal_validation_tokens=VALIDATION_TOKENS,
                           bpb=bpb)
                results.append(row)
                write_json(output / "bpb.json", results)
                commit()
                run.log(row)
                run.summary[f"checkpoint_{step}/bpb_{context}"] = bpb
                print(json.dumps(row, allow_nan=False), flush=True)
                commit()
                del model, loader
            del weights
        return {"bpb": results, "wandb_url": run.url}
    finally:
        try:
            run.finish()
        finally:
            commit()
