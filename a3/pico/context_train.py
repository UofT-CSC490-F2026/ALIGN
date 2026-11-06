"""Budgeted Parts 4-5 training; explicit checkpoint and evaluation provenance."""
import hashlib
import json
import math
import time
from dataclasses import asdict
from pathlib import Path
import torch
import wandb
from nanochat.common import compute_init
from nanochat.tokenizer import get_tokenizer, get_token_bytes
from nanochat.dataloader import tokenizing_distributed_data_loader_with_state_bos_bestfit
from pico.variants import build
from pico.context_eval import suite, bpb, suffix_eval

BASE = Path('/vol/results/part4-baseline-v1-eval')
RATE = .001097 + 4*.0000131 + 16*.00000222


def save_json(path, data):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, indent=2), encoding='utf-8')


def load_model(path, seq=512, native_baseline=False):
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    config = checkpoint['config']
    shape = config['model']
    if native_baseline:
        from nanochat.gpt import GPT, GPTConfig
        if config['variant'] != 'baseline':
            raise ValueError('Native baseline requires a baseline checkpoint')
        torch.manual_seed(config['seed'])
        model = GPT(GPTConfig(**{**shape, 'sequence_len':seq})).to('cuda')
        model.init_weights()
    else:
        model = build(config['variant'], seed=config['seed'], depth=shape['n_layer'],
                      width=shape['n_embd'], seq_len=seq, vocab_size=shape['vocab_size'])
    model.load_state_dict(checkpoint['model'], strict=True)
    return model, config


def train(name, depth=6, width=384, seq=512, steps=3072, seed=42,
          source=None, seconds=3000, commit=lambda: None, eval_every=512,
          constant_lr=False, native_baseline=False,
          output_base=None, run_group='part4-baseline-v1'):
    """32768 tokens/update. Continuations restart optimizer; recorded explicitly."""
    dest = Path(output_base or BASE) / name
    if (dest / 'result.json').exists():
        raise RuntimeError(f'Result already exists: {name}; refusing accidental rerun')
    started = time.monotonic()
    compute_init('cuda')
    # Upstream optimizer kernels specialize by parameter shape. Modal may reuse
    # one worker for multiple model sizes; do not accumulate its old shape cache.
    torch._dynamo.reset()
    torch._dynamo.config.recompile_limit = 32
    if source is None or not native_baseline:
        raise ValueError('Part 4 requires an original baseline checkpoint and native GPT')
    tokenizer, token_bytes = get_tokenizer(), get_token_bytes(device='cuda')
    data = suite(tokenizer, BASE / 'eval_suite.pt')
    if source:
        model, parent = load_model(source, seq, native_baseline=native_baseline)
        depth, width = model.config.n_layer, model.config.n_embd
    config = dict(name=name, variant=parent['variant'], seed=seed, model=asdict(model.config),
                  native_baseline=native_baseline,
                  started_at_unix=time.time(),
                  upstream_commit='92d63d4e8bb4df75c3b71618f31ddde2378b2bcd',
                  source_checkpoint=str(source) if source else None,
                  source_sha256=hashlib.sha256(Path(source).read_bytes()).hexdigest() if source else None,
                  variant_source_sha256=hashlib.sha256(Path(__file__).with_name('variants.py').read_bytes()).hexdigest(),
                  training_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  evaluation_source_sha256=hashlib.sha256(Path(__file__).with_name('context_eval.py').read_bytes()).hexdigest(),
                  optimizer_restarted=bool(source), data_order_restarted=True,
                  training_context=seq, steps=steps, tokens_per_step=32768,
                  seed_training_tokens_before=parent.get('tokens_seen', 16384000) if source else 0,
                  training_shards=4,
                  tokenizer_sha256=hashlib.sha256(Path('/vol/cache/tokenizer/tokenizer.pkl').read_bytes()).hexdigest(),
                  eval_suite_sha256=hashlib.sha256((BASE/'eval_suite.pt').read_bytes()).hexdigest(),
                  gpu=torch.cuda.get_device_name(), torch_version=str(torch.__version__),
                  parameters=sum(p.numel() for p in model.parameters()),
                  lr_schedule='20-step warmup then constant' if constant_lr else '20-step warmup then cosine to 10%',
                  checkpoint_contains_optimizer=True)
    optimizer = model.setup_optimizer(unembedding_lr=.008, embedding_lr=.3,
                                      matrix_lr=.02, scalar_lr=.5, weight_decay=.28)
    # Conservative restart rate for short-to-long continuation; same for 512 control.
    if source:
        for group in optimizer.param_groups:
            group['initial_lr'] *= .2
    run = wandb.init(project='a3-picochat-pilot', group=run_group, name=name,
                     job_type='context',
                     config=config, mode='online')
    run.define_metric('tokens')
    run.define_metric('train/*', step_metric='tokens')
    run.define_metric('val/*', step_metric='tokens')
    batch = 8 if seq == 512 else 2
    accumulation = 32768 // (batch*seq)
    loader = tokenizing_distributed_data_loader_with_state_bos_bestfit(tokenizer, batch, seq, 'train')
    history, timings = [], []
    completed = 0
    data_state = None
    succeeded = False
    dest.mkdir(parents=True, exist_ok=True)
    torch.cuda.reset_peak_memory_stats()
    try:
        for step in range(steps):
            if time.monotonic()-started > seconds:
                break
            mult = min(1., (step+1)/20)
            if not constant_lr:
                mult *= .1 + .9*.5*(1+math.cos(math.pi*step/max(1,steps-1)))
            for group in optimizer.param_groups:
                group['lr'] = group['initial_lr']*mult
            model.train()
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            tick = time.monotonic()
            loss_sum = torch.zeros((), device='cuda')
            for _ in range(accumulation):
                x, y, data_state = next(loader)
                loss = model(x, y)
                (loss/accumulation).backward()
                loss_sum += loss.detach()/accumulation
            if not torch.isfinite(loss_sum):
                raise FloatingPointError('Nonfinite training loss')
            optimizer.step()
            torch.cuda.synchronize()
            timings.append(time.monotonic()-tick)
            completed = step+1
            metrics = dict(tokens=completed*32768, step=completed,
                           **{'train/loss':loss_sum.item(),
                              'train/tokens_per_second':32768/timings[-1],
                              'train/estimated_runtime_cost_usd':(time.monotonic()-started)*RATE})
            if completed % eval_every == 0 or completed == steps:
                # Monitoring uses fixed packed 512-token windows, separate from suffix evaluation.
                metrics['val/calibration_bpb'] = bpb(model, data, token_bytes)
                history.append(dict(metrics))
                print(json.dumps(metrics), flush=True)
                save_json(dest/'progress.json', dict(config, history=history, wandb_url=run.url))
            if completed % 10 == 0 or 'val/calibration_bpb' in metrics:
                run.log(metrics, step=completed)
            if completed % 2048 == 0:
                torch.save({'model':model.state_dict(), 'optimizer':optimizer.state_dict(),
                            'config':dict(config, tokens_seen=completed*32768),
                            'rng_state':torch.get_rng_state(), 'cuda_rng_state':torch.cuda.get_rng_state()}, dest/'latest.pt')
                commit()
        cal, heldout = bpb(model,data,token_bytes), bpb(model,data,token_bytes,'test')
        result = dict(config, status='complete' if completed == steps else 'timeout_partial',
                      completed_steps=completed, tokens_seen=completed*32768,
                      final_loader_state=data_state,
                      calibration_bpb=cal, test_bpb=heldout, history=history,
                      steady_tokens_per_second=32768/(sum(timings[5:])/len(timings[5:])) if len(timings)>5 else None,
                      peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                      wandb_url=run.url)
        if name.startswith('context'):
            result['suffix_eval'] = suffix_eval(model,data,token_bytes,seq)
        result['elapsed_seconds'] = time.monotonic()-started
        result['estimated_runtime_cost_usd'] = result['elapsed_seconds']*RATE
        torch.save({'model':model.state_dict(), 'optimizer':optimizer.state_dict(),
                    'config':dict(config,tokens_seen=completed*32768)},dest/'model.pt')
        save_json(dest/'result.json',result)
        run.summary.update({k:v for k,v in result.items() if k not in ('history','questions','model','suffix_eval')})
        commit()
        succeeded = True
        return result
    finally:
        run.finish(exit_code=0 if succeeded else 1)
