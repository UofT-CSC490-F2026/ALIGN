"""Fixed-token, two-seed screening, with persisted JSON and W&B metrics."""
import json
import math
import os
import time
import hashlib
from pathlib import Path
from dataclasses import asdict
import torch
import wandb
from nanochat.common import compute_init
from nanochat.tokenizer import get_tokenizer, get_token_bytes
from nanochat.dataloader import tokenizing_distributed_data_loader_bos_bestfit
from nanochat.loss_eval import evaluate_bpb
from pico.variants import build


def train(variant, seed, output, steps=500, seconds=1000):
    started = time.monotonic()
    compute_init('cuda')
    tokenizer = get_tokenizer()
    model = build(variant, seed=seed, vocab_size=tokenizer.get_vocab_size())
    optimizer = model.setup_optimizer(unembedding_lr=.008, embedding_lr=.3,
                                     matrix_lr=.02, scalar_lr=.5, weight_decay=.28)
    # Eager graph for all variants: no torch.compile startup differences.
    # The upstream optimizer still compiles its internal update kernels.
    batch, seq, accumulation = 8, 512, 8
    tokens_per_step = batch * seq * accumulation
    config = dict(variant=variant, seed=seed, steps=steps, tokens_per_step=tokens_per_step,
                  model=asdict(model.config), precision='bf16', compiled_model=False,
                  parameter_matching=False, purpose='short fixed-token screening',
                  gpu=torch.cuda.get_device_name(), torch_version=str(torch.__version__),
                  upstream_commit='92d63d4e8bb4df75c3b71618f31ddde2378b2bcd',
                  variant_source_sha256=hashlib.sha256(Path(__file__).with_name('variants.py').read_bytes()).hexdigest())
    run = wandb.init(project='a3-picochat-pilot',
                     name=f'{variant}-seed{seed}',
                     group='pilot-v1', config=config)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    loader = tokenizing_distributed_data_loader_bos_bestfit(tokenizer, batch, seq, 'train')
    token_bytes = get_token_bytes(device='cuda')
    timings, losses, history = [], [], []
    timed_out = False
    train_seconds = 0.0
    torch.cuda.reset_peak_memory_stats()
    try:
        for step in range(steps):
            if time.monotonic() - started > seconds:
                timed_out = True
                break
            # Upstream-style warmup and linear warmdown, fixed across variants.
            multiplier = min(1., (step + 1) / 20)
            if step >= int(steps * .35):
                multiplier *= 1 - .95 * (step - int(steps * .35)) / max(1, steps - int(steps * .35))
            for group in optimizer.param_groups:
                group['lr'] = group['initial_lr'] * multiplier
            torch.cuda.synchronize()
            start_step = time.monotonic()
            model.train()
            optimizer.zero_grad(set_to_none=True)
            loss_sum = torch.zeros((), device='cuda')
            for _ in range(accumulation):
                x, y = next(loader)
                loss = model(x, y)
                (loss / accumulation).backward()
                loss_sum += loss.detach() / accumulation
            if not torch.isfinite(loss_sum):
                raise FloatingPointError('Nonfinite training loss')
            optimizer.step()
            torch.cuda.synchronize()
            elapsed = time.monotonic() - start_step
            train_seconds += elapsed
            timings.append(elapsed)
            losses.append(loss_sum.item())
            metrics = {'step': step + 1, 'tokens': (step + 1) * tokens_per_step,
                       'train/loss': losses[-1], 'train/tokens_per_second': tokens_per_step / elapsed}
            if (step + 1) % 100 == 0 or step == steps - 1:
                model.eval()
                validation = tokenizing_distributed_data_loader_bos_bestfit(tokenizer, batch, seq, 'val')
                metrics['val/bpb'] = evaluate_bpb(model, validation, 64, token_bytes)
                history.append(dict(metrics))
                print(json.dumps(metrics), flush=True)
            if (step + 1) % 10 == 0:
                run.log(metrics, step=step + 1)
            if (step + 1) % 50 == 0 and 'val/bpb' not in metrics:
                print(json.dumps(metrics), flush=True)
        model.eval()
        validation = tokenizing_distributed_data_loader_bos_bestfit(tokenizer, batch, seq, 'val')
        bpb = evaluate_bpb(model, validation, 64, token_bytes)
        result = dict(config, status='timeout_partial' if timed_out else 'complete',
                      completed_steps=len(losses), tokens_seen=len(losses) * tokens_per_step,
                      val_bpb=bpb, train_loss=losses[-1] if losses else None,
                      parameters=sum(p.numel() for p in model.parameters()),
                      train_seconds=train_seconds, elapsed_seconds=time.monotonic() - started,
                      steady_tokens_per_second=tokens_per_step / (sum(timings[5:]) / len(timings[5:])) if len(timings)>5 else None,
                      peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
                      wandb_url=run.url, history=history)
        torch.save({'model': model.state_dict(), 'config': config}, output / 'model.pt')
        (output / 'result.json').write_text(json.dumps(result, indent=2))
        run.summary.update({k: v for k, v in result.items() if k not in ('history', 'model')})
        return result
    finally:
        run.finish()
