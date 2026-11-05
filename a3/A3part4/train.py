"""Part 2 baseline loop, extended with a second context and native checkpoints."""
import json
import time
from dataclasses import asdict
from pathlib import Path

import torch
import wandb
from nanochat.common import compute_init
from nanochat.checkpoint_manager import save_checkpoint, load_checkpoint
from nanochat.dataloader import (
    tokenizing_distributed_data_loader_bos_bestfit,
    tokenizing_distributed_data_loader_with_state_bos_bestfit,
)
from nanochat.loss_eval import evaluate_bpb
from nanochat.tokenizer import get_tokenizer, get_token_bytes
from pico.variants import build

UPSTREAM = "92d63d4e8bb4df75c3b71618f31ddde2378b2bcd"
PARAMETERS = 73_531_646
STEPS = 500
TOKENS_PER_STEP = 32_768
VALIDATION_TOKENS = 262_144
PROJECT = "a3-part4-baseline"
OPTIMIZER = dict(unembedding_lr=.008, embedding_lr=.3,
                 matrix_lr=.02, scalar_lr=.5, weight_decay=.28)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def check_baseline():
    """Check Part 2 baseline compatibility with nanochat's native GPT loader."""
    from nanochat.gpt import GPT
    model = build("baseline", seed=42, device="cpu")
    assert sum(p.numel() for p in model.parameters()) == PARAMETERS
    native = GPT(model.config).to("cpu")
    native.init_weights()
    native.load_state_dict(model.state_dict(), strict=True)
    model.eval()
    native.eval()
    inputs = torch.arange(16).reshape(1, 16)
    with torch.no_grad():
        torch.testing.assert_close(model(inputs), native(inputs), rtol=1e-5, atol=1e-5)
    return {"parameters": PARAMETERS, "native_gpt_agreement": True}


def train(output, commit):
    output = Path(output)
    if (output / "training.json").exists():
        raise FileExistsError("This run already started; choose a new run ID for another run")
    compute_init("cuda")
    tokenizer = get_tokenizer()
    assert tokenizer.get_vocab_size() == 32768
    token_bytes = get_token_bytes(device="cuda")
    config = dict(variant="baseline", seed=42, upstream_commit=UPSTREAM,
                  depth=6, width=384, heads=6, kv_heads=6, vocab_size=32768,
                  window_pattern="SSSL", parameters=PARAMETERS,
                  contexts=[512, 2048], steps_per_stage=STEPS,
                  tokens_per_step=TOKENS_PER_STEP, optimizer=OPTIMIZER,
                  lr_schedule="Part 2 schedule repeated independently in each stage",
                  validation_tokens=VALIDATION_TOKENS,
                  gpu=torch.cuda.get_device_name(), torch_version=str(torch.__version__),
                  precision="bf16", compiled_model=False)
    run = wandb.init(project=PROJECT, name=output.name + "-training",
                     group=output.name, config=config, dir=str(output), mode="online")
    run.define_metric("tokens")
    run.define_metric("*", step_metric="tokens")
    write_json(output / "training.json", dict(config, wandb_url=run.url))
    print(f"Training dashboard: {run.url}", flush=True)
    commit()
    results = []
    checkpoints = output / "checkpoints"
    model, optimizer, data_state = None, None, None
    global_step = 0
    try:
        for stage, context, batch in ((1, 512, 8), (2, 2048, 2)):
            write_json(output / "status.json", {"status": "training", "stage": stage,
                                                 "context": context, "wandb_url": run.url})
            commit()
            # This is the same baseline build and optimizer setup used in Part 2.
            model = build("baseline", seed=42, seq_len=context,
                          vocab_size=tokenizer.get_vocab_size())
            optimizer = model.setup_optimizer(**OPTIMIZER)
            assert sum(p.numel() for p in model.parameters()) == PARAMETERS
            if stage == 2:
                weights, optimizer_state, parent = load_checkpoint(
                    str(checkpoints), STEPS, device="cuda", load_optimizer=True)
                model.load_state_dict(weights, strict=True)
                optimizer.load_state_dict(optimizer_state)
                data_state = parent["dataloader_state"]
                torch.set_rng_state(torch.tensor(parent["torch_rng"], dtype=torch.uint8))
                torch.cuda.set_rng_state(torch.tensor(parent["cuda_rng"], dtype=torch.uint8))
                del weights, optimizer_state
            loader = tokenizing_distributed_data_loader_with_state_bos_bestfit(
                tokenizer, batch, context, "train", resume_state_dict=data_state)
            accumulation = TOKENS_PER_STEP // (batch * context)
            timings = []
            started = time.monotonic()
            torch.cuda.reset_peak_memory_stats()
            for step in range(STEPS):
                # Exact Part 2 formula, with the stage-local step for both stages.
                multiplier = min(1., (step + 1) / 20)
                if step >= int(STEPS * .35):
                    multiplier *= 1 - .95 * (step - int(STEPS * .35)) / max(1, STEPS - int(STEPS * .35))
                for group in optimizer.param_groups:
                    group["lr"] = group["initial_lr"] * multiplier
                torch.cuda.synchronize()
                step_start = time.monotonic()
                model.train()
                optimizer.zero_grad(set_to_none=True)
                loss_sum = torch.zeros((), device="cuda")
                for _ in range(accumulation):
                    x, y, data_state = next(loader)
                    loss = model(x, y)
                    (loss / accumulation).backward()
                    loss_sum += loss.detach() / accumulation
                if not torch.isfinite(loss_sum):
                    raise FloatingPointError("Nonfinite training loss")
                optimizer.step()
                torch.cuda.synchronize()
                timings.append(time.monotonic() - step_start)
                global_step = (stage - 1) * STEPS + step + 1
                metrics = dict(stage=stage, context=context, step=global_step,
                               tokens=global_step * TOKENS_PER_STEP,
                               **{"train/loss": loss_sum.item(),
                                  "train/lr_multiplier": multiplier,
                                  "train/tokens_per_second": TOKENS_PER_STEP / timings[-1],
                                  "hardware/peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30})
                # Save the trained model before final validation can fail.
                if step == STEPS - 1:
                    metadata = dict(step=global_step, model_config=asdict(model.config),
                                    stage=stage, training_tokens=global_step * TOKENS_PER_STEP,
                                    dataloader_state=data_state,
                                    torch_rng=torch.get_rng_state().tolist(),
                                    cuda_rng=torch.cuda.get_rng_state().tolist(),
                                    parent_step=STEPS if stage == 2 else None, complete=True)
                    save_checkpoint(str(checkpoints), global_step, model.state_dict(),
                                    optimizer.state_dict(), metadata)
                    commit()
                if (step + 1) % 100 == 0:
                    model.eval()
                    validation = tokenizing_distributed_data_loader_bos_bestfit(
                        tokenizer, 8, context, "val")
                    metrics["val/bpb"] = evaluate_bpb(
                        model, validation, VALIDATION_TOKENS // (8 * context), token_bytes)
                if (step + 1) % 10 == 0:
                    with (output / "training.jsonl").open("a") as handle:
                        handle.write(json.dumps(metrics, allow_nan=False) + "\n")
                    run.log(metrics, step=global_step)
                    print(json.dumps(metrics), flush=True)
                    commit()
            result = dict(stage=stage, context=context, step=global_step,
                          tokens=global_step * TOKENS_PER_STEP, val_bpb=metrics["val/bpb"],
                          train_seconds=sum(timings), elapsed_seconds=time.monotonic() - started,
                          steady_tokens_per_second=TOKENS_PER_STEP / (sum(timings[5:]) / len(timings[5:])),
                          peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
                          checkpoint=f"checkpoints/model_{global_step:06d}.pt")
            write_json(output / f"stage{stage}.json", result)
            results.append(result)
            run.summary[f"stage{stage}/bpb"] = result["val_bpb"]
            commit()
            del loader, model, optimizer
            model, optimizer = None, None
        write_json(output / "training-results.json", results)
        return results
    finally:
        try:
            run.finish()
        finally:
            commit()
