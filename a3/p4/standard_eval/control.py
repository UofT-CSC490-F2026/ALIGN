"""Baseline control: reuse the frozen Part Four data, LR rule and optimizer.

Only the architecture intervention differs: baseline retains RoPE and omits
FoX forget gates. Original FoX training sources/checkpoints are not changed.
"""
from pathlib import Path
from dataclasses import asdict
import gc
import hashlib
import json
import random
import time
import uuid

from p4.artifacts import atomic_json, append_jsonl, sha256_file, CheckpointStore
from p4.data import TokenStream
from p4.training import (learning_rate_multiplier, model_state_sha256, capture_rng_state,
                         restore_rng_state, _prepared_manifest, _hardware, _estimated_cost,
                         _source_hashes, _WandbMirror, _telemetry)
from .engine import supplemental_sources


def assert_common_initialization(config, device="cpu"):
    """Prove the intervention does not change initial shared learned tensors."""
    import torch
    from pico.variants import build
    args = dict(seed=config["seed"], device=device, depth=config["depth"], width=config["width"],
                seq_len=config["short_context"], vocab_size=config["vocab_size"])
    baseline, fox = build("baseline", **args), build("fox", **args)
    a, b = baseline.state_dict(), fox.state_dict()
    if not set(a).issubset(b):
        raise ValueError("Unexpected baseline-only learned tensors")
    if not all(torch.equal(v, b[k]) for k, v in a.items()):
        raise ValueError("Baseline and FoX shared initial tensors differ")
    extra = sorted(set(b) - set(a))
    if not extra or any(not k.endswith("attn.forget.weight") for k in extra):
        raise ValueError("The architecture difference includes unexpected tensors")
    return {"common_initial_tensors_identical": True, "common_initial_state_sha256": model_state_sha256(a),
            "baseline_parameters": sum(p.numel() for p in baseline.parameters()),
            "fox_parameters": sum(p.numel() for p in fox.parameters()), "fox_only_learned_tensors": extra}


def train_control(config, prepared, root, stage, commit=lambda: None, device="cuda"):
    import numpy as np
    import torch
    from pico.variants import build

    if config["variant"] != "baseline" or stage not in ("short", "long"):
        raise ValueError("This separate trainer only runs the baseline control")
    prepared, root = Path(prepared), Path(root)
    root.mkdir(parents=True, exist_ok=True)
    result_path = root / f"{stage}-result.json"
    if result_path.exists():
        raise FileExistsError("This control stage has already completed")
    manifest = _prepared_manifest(prepared, config)
    identity = {"config": config, "prepared_manifest_sha256": sha256_file(prepared / "manifest.json"),
                "original_training_sources": _source_hashes(), "supplemental_sources": supplemental_sources()}
    config_hash = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    if (root / "provenance.json").exists():
        if json.loads((root / "provenance.json").read_text()) != identity:
            raise ValueError("Baseline source/data/config changed between stages")
    else:
        atomic_json(root / "provenance.json", identity)
        atomic_json(root / "config.json", config)
        atomic_json(root / "initialization-comparison.json", assert_common_initialization(config))
    commit()
    started = time.monotonic()
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    cuda = device.startswith("cuda")
    if cuda:
        torch.cuda.manual_seed_all(config["seed"])
        torch.set_float32_matmul_precision("high")
    context, batch = config[f"{stage}_context"], config[f"{stage}_microbatch"]
    accumulation = config["tokens_per_step"] // (context * batch)
    if accumulation * context * batch != config["tokens_per_step"]:
        raise ValueError("Training batch is not token matched")
    model = build("baseline", seed=config["seed"], device=device, depth=config["depth"],
                  width=config["width"], seq_len=context, vocab_size=config["vocab_size"])
    if cuda:
        optimizer = model.setup_optimizer(**config["optimizer"])
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.01)
        for group in optimizer.param_groups:
            group["initial_lr"] = group["lr"]
    stream, store = TokenStream(prepared / "train.bin"), CheckpointStore(root, commit=commit)
    step, parent = 0, None
    if stage == "long":
        state, parent_record = store.load("checkpoint1")
        if state["config_sha256"] != config_hash or state["global_step"] != config["short_steps"]:
            raise ValueError("Baseline continuation has the wrong parent")
        model.load_state_dict(state["model"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        stream.load_state_dict(state["data_stream"])
        restore_rng_state(state["rng"])
        step, parent = state["global_step"], parent_record["sha256"]
        if model_state_sha256(model.state_dict()) != state["model_state_sha256"]:
            raise ValueError("Baseline parent learned weights changed")
        atomic_json(root / "continuation.json", {"checkpoint1_sha256": parent, "optimizer_state_loaded": True,
                    "weights_verified_before_update": True, "data_stream_before_update": stream.state_dict(),
                    "global_step_before_update": step, "tokens_before_update": state["cumulative_tokens"]})
        del state
        gc.collect()
    end = config["short_steps"] + (config["long_steps"] if stage == "long" else 0)
    start_step, train_seconds = step, 0.0
    mirror = _WandbMirror(config, root, stage, uuid.uuid4().hex[:12])
    atomic_json(root / f"{stage}-hardware.json", _hardware(config, device))

    def save(name):
        weights = model.state_dict()
        state = {"schema_version": 1, "stage": stage, "model": weights, "model_config": asdict(model.config),
                 "variant": "baseline", "optimizer": optimizer.state_dict(), "global_step": step,
                 "cumulative_tokens": step * config["tokens_per_step"], "data_stream": stream.state_dict(),
                 "rng": capture_rng_state(), "model_state_sha256": model_state_sha256(weights),
                 "config_sha256": config_hash, "parent_checkpoint_sha256": parent}
        saved = store.save(name, state, {"global_step": step, "stage": stage, "context": context,
                                 "tokens": state["cumulative_tokens"], "model_state_sha256": state["model_state_sha256"]})
        append_jsonl(root / "metrics.jsonl", {"event": "checkpoint_saved", "checkpoint": name,
                      "step": step, "sha256": saved["sha256"], "tokens": state["cumulative_tokens"]})
        recoveries = sorted((root / "checkpoints").glob("recovery-*.json"), reverse=True)
        for old in recoveries[2:]:
            record = json.loads(old.read_text())
            append_jsonl(root / "checkpoint-history.jsonl", {"pruned": old.stem, "sha256": record["sha256"]})
            commit()
            (root / record["file"]).unlink()
            old.unlink()
        commit()
        return saved

    try:
        while step < end:
            if time.monotonic() - started >= config["max_seconds_per_stage"]:
                partial = save(f"recovery-{step:06d}")
                raise TimeoutError(f"Baseline stopped at {step}; checkpoint {partial['name']} was persisted")
            multiplier = learning_rate_multiplier(step, config)
            for group in optimizer.param_groups:
                group["lr"] = group["initial_lr"] * multiplier
            if cuda:
                torch.cuda.synchronize()
            before = time.monotonic()
            model.train()
            optimizer.zero_grad(set_to_none=True)
            total = torch.zeros((), device=device)
            for _ in range(accumulation):
                x, y = stream.next_batch(batch, context, device)
                loss = model(x, y)
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError("Nonfinite baseline loss")
                (loss / accumulation).backward()
                total += loss.detach() / accumulation
            optimizer.step()
            if cuda:
                torch.cuda.synchronize()
            elapsed = time.monotonic() - before
            train_seconds += elapsed
            step += 1
            row = {"event": "train_step", "family": "baseline", "stage": stage, "step": step,
                   "tokens": step * config["tokens_per_step"], "train/loss": float(total),
                   "train/lr_multiplier": multiplier, "train/step_seconds": elapsed,
                   "train/tokens_per_second": config["tokens_per_step"] / elapsed,
                   "stream_cursor": stream.cursor}
            append_jsonl(root / "metrics.jsonl", row)
            if step % config["log_every"] == 0 or step == end:
                row.update(_telemetry(device))
                mirror.log(row, step)
                print(json.dumps(row), flush=True)
                commit()
            if step % config["checkpoint_every"] == 0 and step != end:
                save(f"recovery-{step:06d}")
        checkpoint = save("checkpoint1" if stage == "short" else "checkpoint2")
        result = {"status": "complete", "family": "baseline", "stage": stage, "global_step": step,
                  "tokens": step * config["tokens_per_step"], "steps_this_attempt": step - start_step,
                  "parameters": sum(p.numel() for p in model.parameters()), "context": context,
                  "checkpoint": checkpoint, "parent_checkpoint_sha256": parent, "train_seconds": train_seconds,
                  "elapsed_seconds": time.monotonic() - started, "wandb_url": mirror.url,
                  "estimated_worker_cost_usd": _estimated_cost(time.monotonic() - started, config, device),
                  "cost_scope": "Worker timer only, excluding startup/build/storage/final tracking finalization.",
                  "validation_during_training": "No validation-based selection; evaluate both fixed final checkpoints separately."}
        atomic_json(result_path, result)
        commit()
        return result
    finally:
        mirror.finish()
        commit()
