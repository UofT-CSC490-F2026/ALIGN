"""Explicit, bounded experiment configuration (no cloud calls on import)."""
from copy import deepcopy
import json
from pathlib import Path

UPSTREAM_COMMIT = "92d63d4e8bb4df75c3b71618f31ddde2378b2bcd"
APP_NAME = "a3-p4-context-extension"
VOLUME_NAME = "a3-p4-context"
SECRET_NAME = "a3-p4-wandb"
PROFILE = "a3-p4"
MAX_BUDGET_USD = 100.0

DEFAULT = {
    "schema_version": 1, "variant": "fox", "seed": 42,
    "upstream_commit": UPSTREAM_COMMIT,
    "depth": 6, "width": 384, "vocab_size": 32768,
    "short_context": 512, "long_context": 2048,
    "short_steps": 500, "long_steps": 500,
    "tokens_per_step": 32768, "short_microbatch": 8, "long_microbatch": 2,
    "checkpoint_every": 50, "keep_last_recovery_checkpoints": 2, "log_every": 10, "eval_every": 100,
    "eval_target_tokens": 128, "eval_examples": 64,
    "warmup_steps": 20, "warmdown_start": 700,
    "max_seconds_per_stage": 1500,
    "optimizer": {"unembedding_lr": .008, "embedding_lr": .3,
                  "matrix_lr": .02, "scalar_lr": .5, "weight_decay": .28},
    "wandb_project": "a3-p4-context-extension", "wandb_mode": "online",
    "gpu": "H100", "cpu": 4, "memory_mb": 16384,
    "budget_usd": MAX_BUDGET_USD,
    "rates": {"gpu_per_second": .001097, "cpu_core_per_second": .0000131,
              "memory_gib_per_second": .00000222,
              "verified_date": "2026-11-04", "source": "https://modal.com/pricing"},
    "train_shards": 4, "train_token_limit": 40000000,
    "tokenizer_chars": 100000000,
    "dataset_revision": "915333b4f8b8684f39aeaafea600fea6f43fb703",
}


def validate(config):
    if config.get("variant") != "fox":
        raise ValueError("Part 4 keeps the FoX architecture fixed.")
    for key in ("depth", "width", "vocab_size", "short_context", "long_context",
                "short_steps", "long_steps", "tokens_per_step", "checkpoint_every",
                "log_every", "eval_every", "eval_target_tokens", "eval_examples",
                "short_microbatch", "long_microbatch", "max_seconds_per_stage",
                "train_shards", "train_token_limit", "tokenizer_chars", "keep_last_recovery_checkpoints"):
        if isinstance(config.get(key), bool) or not isinstance(config.get(key), int) or config[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    if config["width"] % 64 or config["depth"] < 2:
        raise ValueError("Model requires depth >= 2 and width divisible by 64")
    if not 0 < config["eval_target_tokens"] < config["short_context"] < config["long_context"]:
        raise ValueError("Require targets < short context < long context")
    for stage in ("short", "long"):
        if config["tokens_per_step"] % (config[f"{stage}_context"] * config[f"{stage}_microbatch"]):
            raise ValueError("tokens_per_step must fit each microbatch exactly")
    if not 0 < config.get("budget_usd", 0) <= MAX_BUDGET_USD:
        raise ValueError("Part 4 gross usage must remain within the US$100 ceiling")
    if config.get("upstream_commit") != UPSTREAM_COMMIT:
        raise ValueError("Unexpected upstream nanochat revision")
    if not 1 <= config["train_shards"] <= 6542:
        raise ValueError("Training shards must exclude the held-out shard 6542")
    required = (config["short_steps"] + config["long_steps"]) * config["tokens_per_step"] + 1
    if config["train_token_limit"] < required:
        raise ValueError("Training token subset is too small for a no-wrap chain")
    if config.get("dataset_revision"):
        import re
        if not re.fullmatch(r"[0-9a-f]{40}", config["dataset_revision"]):
            raise ValueError("dataset_revision must be an immutable 40-character commit")
    if config.get("gpu") != "H100" or config.get("cpu") != 4 or config.get("memory_mb") != 16384:
        raise ValueError("Resources must match the fixed H100 / 4 CPU / 16 GiB Modal declarations")
    for key, rate in DEFAULT["rates"].items():
        if key.endswith("per_second") and config["rates"].get(key) != rate:
            raise ValueError("Cost rates must match the verified resource declarations")
    if config["max_seconds_per_stage"] > 1500:
        raise ValueError("Keep 300 seconds before Modal's hard timeout for saving/evaluation")
    return config


def load_config(path=None):
    config = deepcopy(DEFAULT)
    if path:
        supplied = json.loads(Path(path).read_text())
        for key, value in supplied.items():
            if key in ("optimizer", "rates"):
                config[key].update(value)
            else:
                config[key] = value
    return validate(config)


def rehearsal_config():
    config = load_config()
    config.update(depth=2, width=128, short_steps=2, long_steps=2,
                  tokens_per_step=2048, short_microbatch=1, long_microbatch=1,
                  checkpoint_every=1, log_every=1, eval_every=2,
                  eval_examples=2, warmup_steps=1, warmdown_start=3,
                  max_seconds_per_stage=180)
    return validate(config)
