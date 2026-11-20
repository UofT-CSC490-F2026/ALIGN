"""Time FoX generation with and without the KV cache on the real A3 checkpoint.

    modal run a4/modal_fox_timing.py

Reads the checkpoint from the Modal volume a4-base (base_checkpoints/fox-d24, step 5571,
plus tokenizer/). Runs on one GPU for a few minutes and prints a JSON summary.
"""
import json
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
A3 = HERE.parent / "a3"
REVISION = "92d63d4e8bb4df75c3b71618f31ddde2378b2bcd"
app = modal.App("a4-fox-timing")
volume = modal.Volume.from_name("a4-base")
image = (modal.Image.debian_slim(python_version="3.12")
         .apt_install("git", "build-essential")
         .pip_install("uv")
         .run_commands("git clone https://github.com/karpathy/nanochat.git /root/nanochat",
                       f"git -C /root/nanochat checkout {REVISION}")
         .workdir("/root/nanochat")
         .run_commands("uv sync --python 3.12 --frozen --extra gpu --no-install-project")
         .env({"PYTHONPATH": "/root/nanochat:/root/a3:/root/a3/A3part5:/root/a4",
               "NANOCHAT_BASE_DIR": "/vol", "OMP_NUM_THREADS": "1"})
         .add_local_dir(str(A3 / "pico"), "/root/a3/pico", ignore=["__pycache__", "*.pyc"])
         .add_local_file(str(A3 / "A3part5" / "fox.py"), "/root/a3/A3part5/fox.py")
         .add_local_file(str(HERE / "fox_cache.py"), "/root/a4/fox_cache.py"))

MODEL_TAG, STEP = "fox-d24", 5571
QUESTIONS = [
    "Maria has 24 apples. She gives 7 to her neighbor and then buys 15 more. How many apples does she have now?",
    "A train travels 60 miles per hour for 2.5 hours, then 40 miles per hour for 3 hours. How far does it travel in total?",
]
CONFIGS = [(1, 64), (8, 64), (8, 256)]  # (samples per prompt, new tokens); RL uses 8 per pass and up to 256


@app.function(image=image, gpu="H100", timeout=1800, volumes={"/vol": volume})
def timing():
    import site
    import time
    site.addsitedir("/root/nanochat/.venv/lib/python3.12/site-packages")
    import torch
    import fox_cache
    fox_cache.install()
    import nanochat.checkpoint_manager as checkpoints
    from nanochat.common import COMPUTE_DTYPE
    from nanochat.engine import Engine

    device = torch.device("cuda")
    model, tokenizer, _ = checkpoints.load_model("base", device, phase="eval", model_tag=MODEL_TAG, step=STEP)
    model.eval()
    cfg = model.config
    prompts = [tokenizer.render_for_completion({"messages": [{"role": "user", "content": q},
                                                             {"role": "assistant", "content": ""}]})
               for q in QUESTIONS]

    def new_cache(batch, seq_len):
        return fox_cache.FoXKVCache(batch_size=batch, num_heads=cfg.n_kv_head, seq_len=seq_len,
                                    head_dim=cfg.n_embd // cfg.n_head, num_layers=cfg.n_layer,
                                    device=device, dtype=COMPUTE_DTYPE)

    @torch.inference_mode()
    def uncached(prompt, n_new, batch):
        ids = torch.tensor([prompt] * batch, device=device)
        for _ in range(n_new):
            ids = torch.cat([ids, model(ids)[:, -1].argmax(-1, keepdim=True)], 1)
        return ids[0, len(prompt):].tolist()

    @torch.inference_mode()
    def cached(prompt, n_new, batch):
        prefill = new_cache(1, len(prompt))
        logits = model(torch.tensor([prompt], device=device), kv_cache=prefill)[:, -1].expand(batch, -1)
        cache = new_cache(batch, len(prompt) + n_new)
        cache.prefill(prefill)
        out = []
        for _ in range(n_new):
            token = logits.argmax(-1, keepdim=True)
            out.append(token)
            logits = model(token, kv_cache=cache)[:, -1]
        return torch.cat(out, 1)[0].tolist()

    def timed(fn, *args):
        torch.cuda.synchronize()
        start = time.perf_counter()
        result = fn(*args)
        torch.cuda.synchronize()
        return result, time.perf_counter() - start

    @torch.inference_mode()
    def teacher_forced(prompt, continuation):
        """Same tokens through both paths: how far apart are the logits?"""
        full = model(torch.tensor([prompt + continuation], device=device))[0, len(prompt) - 1:-1].float()
        cache = new_cache(1, len(prompt) + len(continuation))
        steps = [model(torch.tensor([prompt], device=device), kv_cache=cache)[0, -1:]]
        steps += [model(torch.tensor([[t]], device=device), kv_cache=cache)[0] for t in continuation[:-1]]
        part = torch.cat(steps, 0).float()
        return {"max_abs_logit_diff": (part - full).abs().max().item(),
                "argmax_agreement": (part.argmax(-1) == full.argmax(-1)).float().mean().item()}

    uncached(prompts[0], 4, 1), cached(prompts[0], 4, 1)  # warm-up
    rows = []
    for index, prompt in enumerate(prompts):
        for batch, n_new in CONFIGS:
            slow, t_slow = timed(uncached, prompt, n_new, batch)
            fast, t_fast = timed(cached, prompt, n_new, batch)
            match = next((i for i, (a, b) in enumerate(zip(slow, fast)) if a != b), len(slow))
            rows.append({"prompt": index, "prompt_tokens": len(prompt), "samples": batch, "new_tokens": n_new,
                         "uncached_s": round(t_slow, 3), "cached_s": round(t_fast, 3),
                         "speedup": round(t_slow / t_fast, 1), "identical_tokens": match, "of": len(slow)})
            print(json.dumps(rows[-1]), flush=True)

    check = teacher_forced(prompts[0], uncached(prompts[0], 64, 1))
    rows_engine, _ = Engine(model, tokenizer).generate_batch(prompts[0], num_samples=8, max_tokens=64, temperature=0)
    return {"gpu": torch.cuda.get_device_name(0), "dtype": str(COMPUTE_DTYPE), "timing": rows,
            "teacher_forced_64_tokens": check,
            "engine_generate_batch_ok": len(rows_engine) == 8 and all(len(r) > len(prompts[0]) for r in rows_engine)}


@app.local_entrypoint()
def main():
    print(json.dumps(timing.remote(), indent=2))
