"""Saved-checkpoint comparisons using nanochat BPB and CORE functions."""
import json
from pathlib import Path

import torch
import wandb
from nanochat.common import compute_init
from nanochat.checkpoint_manager import load_checkpoint
from nanochat.dataloader import tokenizing_distributed_data_loader_bos_bestfit
from nanochat.loss_eval import evaluate_bpb
from nanochat.tokenizer import get_tokenizer, get_token_bytes
from pico.variants import build
from scripts.base_eval import evaluate_core
from A3part4.train import PROJECT, VALIDATION_TOKENS, write_json


def evaluate(output, commit):
    output = Path(output)
    compute_init("cuda")
    tokenizer = get_tokenizer()
    token_bytes = get_token_bytes(device="cuda")
    run = wandb.init(project=PROJECT, name=output.name + "-evaluation",
                     group=output.name, dir=str(output), mode="online",
                     config={"bpb_tokens": VALIDATION_TOKENS, "core_context": 2048,
                             "core_max_per_task": -1})
    results = []
    try:
        for step in (500, 1000):
            weights, _, metadata = load_checkpoint(str(output / "checkpoints"), step,
                                                    device="cuda", load_optimizer=False)
            for context in (512, 2048):
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
                run.log(row)
                run.summary[f"checkpoint_{step}/bpb_{context}"] = bpb
                print(json.dumps(row, allow_nan=False), flush=True)
                commit()
                if context == 2048:
                    write_json(output / "status.json", {"status": "evaluating_core",
                                                         "checkpoint_step": step, "wandb_url": run.url})
                    commit()
                    core = evaluate_core(model, tokenizer, torch.device("cuda"), max_per_task=-1)
                    core.update(checkpoint_step=step, evaluation_context=context, max_per_task=-1)
                    write_json(output / f"core-{step}.json", core)
                    run.summary[f"checkpoint_{step}/core"] = core["core_metric"]
                    run.log({f"checkpoint_{step}/core": core["core_metric"]})
                    commit()
                del model, loader
            del weights
        return {"bpb": results, "wandb_url": run.url}
    finally:
        try:
            run.finish()
        finally:
            commit()
