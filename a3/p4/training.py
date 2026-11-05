"""Durable, single-GPU short-to-long FoX pretraining.

The two stages share one token/LR schedule and one resumable token stream.
Only self-produced, checksummed checkpoints are loaded. No Jinhao source is
modified; model definitions are imported from the read-only ``pico`` package.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import random
import subprocess
import time
import uuid
from dataclasses import asdict
from pathlib import Path

UPSTREAM_COMMIT = "92d63d4e8bb4df75c3b71618f31ddde2378b2bcd"


def learning_rate_multiplier(step: int, config: dict) -> float:
    """The same global schedule is used before and after context expansion."""
    total = int(config["short_steps"]) + int(config["long_steps"])
    if not 0 <= step < total:
        raise ValueError("Schedule step is outside the declared training horizon")
    warmup = int(config.get("warmup_steps", 20))
    warmdown = int(config.get("warmdown_start", int(total * 0.7)))
    if warmup < 0 or not 0 <= warmdown < total:
        raise ValueError("Invalid learning-rate schedule")
    multiplier = min(1.0, (step + 1) / max(1, warmup))
    if step >= warmdown:
        multiplier *= 1.0 - 0.95 * (step - warmdown) / max(1, total - warmdown - 1)
    return max(0.05, multiplier) if warmup == 0 else multiplier


def model_state_sha256(state: dict) -> str:
    """Hash names, shapes, dtypes and exact tensor bytes, including BF16."""
    import torch

    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def capture_rng_state() -> dict:
    import numpy as np
    import torch

    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng_state(state: dict) -> None:
    import numpy as np
    import torch

    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"]:
        if not torch.cuda.is_available():
            raise RuntimeError("Cannot resume a CUDA RNG trajectory without CUDA")
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])


def _config_hash(config: dict) -> str:
    return hashlib.sha256(json.dumps(config, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _safe_error(exc: BaseException) -> str:
    message = str(exc)
    for name in ("WANDB_API_KEY", "MODAL_TOKEN_ID", "MODAL_TOKEN_SECRET"):
        secret = os.environ.get(name)
        if secret:
            message = message.replace(secret, "[redacted]")
    return message[:2000]


def _prepared_manifest(root: Path, config: dict) -> dict:
    from p4.artifacts import sha256_file

    manifest = json.loads((root / "manifest.json").read_text())
    if manifest.get("vocab_size") != config["vocab_size"]:
        raise ValueError("Prepared tokenizer vocabulary differs from the run config")
    files = manifest.get("files", {})
    if not isinstance(files, dict):
        raise ValueError("Prepared manifest must map file paths to checksums")
    for required in ("train.bin", "token_bytes.npy", "evaluation.jsonl"):
        if required not in files:
            raise ValueError(f"Prepared manifest does not cover {required}")
    for name, expected in files.items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Unsafe prepared-data path")
        path = root / relative
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"Missing or unsafe prepared-data file: {name}")
        if path.stat().st_size != expected["size_bytes"] or sha256_file(path) != expected["sha256"]:
            raise ValueError(f"Prepared-data integrity failure: {name}")
    return manifest


def _source_hashes() -> dict:
    from p4.artifacts import sha256_file
    import nanochat.gpt
    import pico.variants

    package_root = Path(__file__).parent
    upstream_root = Path(nanochat.gpt.__file__).parent.parent
    paths = {"p4/" + path.name: path for path in package_root.glob("*.py")}
    paths["pico/variants.py"] = Path(pico.variants.__file__)
    for name in ("pyproject.toml", "uv.lock", "nanochat/gpt.py", "nanochat/optim.py",
                 "nanochat/common.py", "nanochat/tokenizer.py", "nanochat/dataset.py"):
        paths["upstream/" + name] = upstream_root / name
    return {name: sha256_file(path) for name, path in sorted(paths.items())}


def _hardware(config: dict, device: str) -> dict:
    import torch

    info = {
        "device": device,
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "cuda_runtime": torch.version.cuda,
        "requested_gpu": config.get("gpu"),
        "requested_cpu_cores": config.get("cpu"),
        "requested_memory_mb": config.get("memory_mb"),
        "platform": platform.platform(),
        "modal_task_id": os.environ.get("MODAL_TASK_ID"),
        "modal_function_id": os.environ.get("MODAL_FUNCTION_ID"),
        "modal_environment": os.environ.get("MODAL_ENVIRONMENT"),
    }
    if str(device).startswith("cuda"):
        props = torch.cuda.get_device_properties(torch.device(device))
        info.update(gpu_name=props.name, gpu_total_memory_bytes=props.total_memory,
                    gpu_compute_capability=[props.major, props.minor])
    return info


def _telemetry(device: str) -> dict:
    import torch

    values = {}
    try:
        import psutil
        process = psutil.Process()
        values["hardware/cpu_rss_bytes"] = process.memory_info().rss
        cpu_times = process.cpu_times()
        now, used = time.monotonic(), cpu_times.user + cpu_times.system
        previous = getattr(_telemetry, "_previous_cpu_sample", None)
        if previous is not None and now > previous[0]:
            values["hardware/cpu_percent"] = max(0.0, 100 * (used - previous[1]) / (now - previous[0]))
        _telemetry._previous_cpu_sample = now, used
        values["hardware/cpu_user_seconds"] = cpu_times.user
        values["hardware/cpu_system_seconds"] = cpu_times.system
    except ImportError:
        pass
    if str(device).startswith("cuda"):
        values["hardware/gpu_allocated_bytes"] = torch.cuda.memory_allocated(device)
        values["hardware/gpu_reserved_bytes"] = torch.cuda.memory_reserved(device)
        values["hardware/gpu_peak_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
        values["peak_allocated_gib"] = torch.cuda.max_memory_allocated(device) / 2**30
        try:
            output = subprocess.check_output([
                "nvidia-smi", "--query-gpu=utilization.gpu,memory.used,power.draw,temperature.gpu",
                "--format=csv,noheader,nounits", "--id=0",
            ], text=True, timeout=3)
            parts = output.strip().split(",")
            for key, value in zip(("gpu_utilization_pct", "gpu_memory_used_mib", "gpu_power_watts",
                                   "gpu_temperature_c"), parts):
                try:
                    values["hardware/" + key] = float(value.strip())
                except ValueError:
                    pass
        except (OSError, subprocess.SubprocessError):
            values["hardware/nvidia_smi_available"] = False
    return values


def _estimated_cost(seconds: float, config: dict, device: str) -> float:
    """Requested-resource estimate, not the invoice or an account-wide limit."""
    if not str(device).startswith("cuda"):
        return 0.0  # Local CPU tests do not use cloud resources.
    rates = config.get("rates", {})
    gpu = float(rates.get("gpu_per_second", rates.get("gpu_usd_per_second", 0.001097)))
    cpu = float(rates.get("cpu_core_per_second", rates.get("cpu_per_core_second", rates.get("cpu_usd_per_core_second", 0.0000131))))
    memory = float(rates.get("memory_gib_per_second", rates.get("memory_per_gib_second", rates.get("memory_usd_per_gib_second", 0.00000222))))
    return seconds * (gpu + cpu * config.get("cpu", 4) + memory * config.get("memory_mb", 16384) / 1024)


def _previous_estimated_cost(root: Path) -> float:
    """Sum each recorded worker attempt once, including partial failed attempts."""
    path = root / "metrics.jsonl"
    if not path.exists():
        return 0.0
    attempts = {}
    for line in path.read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            if row.get("attempt") and "estimated_worker_cost_usd" in row:
                attempts[row["attempt"]] = max(attempts.get(row["attempt"], 0.0), row["estimated_worker_cost_usd"])
    return sum(attempts.values())


class _WandbMirror:
    """JSONL is authoritative; online W&B receives a second copy of metrics."""

    def __init__(self, config: dict, root: Path, stage: str, attempt: str):
        self.run = None
        self.root = root
        self.failed = False
        mode = config.get("wandb_mode", "online")
        if mode == "disabled":
            return
        import wandb

        directory = root / "wandb"
        directory.mkdir(exist_ok=True)
        options = {"symlink": False}
        fields = getattr(wandb.Settings, "model_fields", None)
        if fields is None:
            fields = getattr(wandb.Settings, "__fields__", {})
        if "allow_media_symlink" in fields:
            options["allow_media_symlink"] = False
        self.run = wandb.init(
            project=config.get("wandb_project", "a3-p4-context-extension"),
            entity=config.get("wandb_entity") or None,
            name=f"{config.get('experiment_id', root.name)}-{stage}-{attempt}",
            group=config.get("experiment_id", root.name),
            config=config, mode=mode, dir=str(directory),
            settings=wandb.Settings(**options),
        )
        if self.run is None:
            raise RuntimeError("W&B initialization did not create a run")
        self.run.define_metric("tokens")
        self.run.define_metric("*", step_metric="tokens")

    @property
    def url(self):
        return self.run.url if self.run is not None else None

    def log(self, row: dict, step: int) -> None:
        from p4.artifacts import append_jsonl

        if self.run is None or self.failed:
            return
        try:
            # Let W&B advance its internal row index. The final evaluation can
            # share a training step/token count with the preceding train row;
            # explicitly reusing W&B's `step=` would silently drop that row.
            self.run.log(dict(row, global_step=step))
        except Exception as exc:
            self.failed = True
            append_jsonl(self.root / "wandb-recovery.jsonl", {
                "event": "wandb_log_failure", "message": _safe_error(exc),
                "step": step, "replay_source": "metrics.jsonl",
            })

    def finish(self) -> None:
        from p4.artifacts import append_jsonl

        if self.run is not None:
            try:
                self.run.finish()
            except Exception as exc:
                self.failed = True
                append_jsonl(self.root / "wandb-recovery.jsonl", {
                    "event": "wandb_finish_failure", "message": _safe_error(exc),
                    "replay_source": "metrics.jsonl",
                })
            # W&B core may create debug-core.log as a symlink even when its
            # Python SDK's symlink setting is disabled. Preserve real file
            # bytes and discard only redundant directory aliases before the
            # strict experiment-bundle audit. Never follow arbitrary artifacts.
            directory = self.root / "wandb"
            for path in sorted(directory.rglob("*")):
                if not path.is_symlink():
                    continue
                if path.name not in ("debug-core.log", "latest-run", "debug.log", "debug-internal.log"):
                    raise ValueError(f"Unexpected W&B artifact symlink: {path.name}")
                target = path.resolve()
                row = {"event": "wandb_link_materialized", "path": str(path.relative_to(self.root)),
                       "target_name": target.name}
                if target.is_file():
                    contents = target.read_bytes()
                    temporary = path.with_name(path.name + ".tmp")
                    with temporary.open("wb") as handle:
                        handle.write(contents)
                        handle.flush()
                        os.fsync(handle.fileno())
                    os.replace(temporary, path)
                    row["sha256"] = hashlib.sha256(contents).hexdigest()
                elif target.is_dir() and target.is_relative_to(directory.resolve()):
                    path.unlink()  # The actual run directory remains archived.
                    row["event"] = "wandb_redundant_directory_alias_removed"
                elif not target.exists():
                    path.unlink()
                    row["event"] = "wandb_broken_log_alias_removed"
                else:
                    raise ValueError("W&B directory alias points outside its evidence directory")
                append_jsonl(self.root / "wandb-link-history.jsonl", row)


def train_stage(config: dict, prepared_dir, run_dir, stage="short", commit=lambda: None,
                device="cuda", resume: str | None = None) -> dict:
    """Run one stage or resume one partial stage; never silently reuse results.

    ``resume`` is a CheckpointStore name, not an arbitrary external pickle path.
    For a fresh long stage it defaults to the immutable ``checkpoint1``.
    CPU execution is a correctness test using AdamW, never a research run.
    """
    started = time.monotonic()
    import numpy as np
    import torch
    from pico.variants import build
    from p4.artifacts import CheckpointStore, append_jsonl, atomic_json, sha256_file
    from p4.data import TokenStream
    from p4.evaluation import evaluate_checkpoint

    if stage not in ("short", "long"):
        raise ValueError("stage must be short or long")
    if config.get("variant", "fox") != "fox":
        raise ValueError("Part 4 fixes the architecture to FoX without RoPE")
    is_cuda = str(device).startswith("cuda")
    if is_cuda and not torch.cuda.is_available():
        raise RuntimeError("A CUDA training stage requires an available CUDA GPU")
    if is_cuda and config.get("wandb_mode", "online") != "online" and not config.get("allow_offline_cuda", False):
        raise ValueError("Production CUDA runs require online W&B; rehearsal override must be explicit")
    stage_start = 0 if stage == "short" else int(config["short_steps"])
    stage_end = int(config["short_steps"]) + (int(config["long_steps"]) if stage == "long" else 0)
    sequence_len = int(config[f"{stage}_context"])
    batch = int(config[f"{stage}_microbatch"])
    tokens_per_step = int(config["tokens_per_step"])
    if min(batch, sequence_len, tokens_per_step) <= 0 or tokens_per_step % (batch * sequence_len):
        raise ValueError("tokens_per_step must be a multiple of microbatch * context")
    accumulation = tokens_per_step // (batch * sequence_len)
    for key in ("checkpoint_every", "log_every", "eval_every"):
        if int(config.get(key, 1)) <= 0:
            raise ValueError(f"{key} must be positive")
    if stage_end <= stage_start or float(config.get("max_seconds_per_stage", 1800)) <= 0:
        raise ValueError("Stage must have positive step and time budgets")
    # Validate the whole schedule before any model or tracking resources exist.
    learning_rate_multiplier(stage_start, config)
    prepared, root = Path(prepared_dir), Path(run_dir)
    data_manifest = _prepared_manifest(prepared, config)
    examples = [json.loads(line) for line in (prepared / "evaluation.jsonl").read_text().splitlines() if line.strip()]
    examples = examples[:int(config.get("eval_examples", len(examples)))]
    if not examples:
        raise ValueError("The frozen evaluation set is empty")
    token_bytes = np.load(prepared / "token_bytes.npy", allow_pickle=False)
    if len(token_bytes) != config["vocab_size"]:
        raise ValueError("Token-byte table has the wrong vocabulary size")
    root.mkdir(parents=True, exist_ok=True)
    previous_estimated_cost = _previous_estimated_cost(root)
    config_path = root / "config.json"
    if config_path.exists():
        if _config_hash(json.loads(config_path.read_text())) != _config_hash(config):
            raise ValueError("Existing run config differs; choose a new immutable run directory")
        if stage == "short" and resume is None:
            raise FileExistsError("Run already exists; explicitly resume a checkpoint or choose a new run ID")
    else:
        if stage == "long" or resume is not None:
            raise ValueError("A continuation must use the original run directory/config")
        atomic_json(config_path, config)
    result_path = root / f"{stage}-result.json"
    if result_path.exists() and json.loads(result_path.read_text()).get("status") == "complete":
        raise FileExistsError("Stage already completed; results are never silently rerun")
    attempt = uuid.uuid4().hex[:12]
    metadata_dir = root / "attempts"
    metadata_dir.mkdir(exist_ok=True)
    hardware = _hardware(config, device)
    atomic_json(metadata_dir / f"{stage}-{attempt}-hardware.json", hardware)
    if not (root / "hardware.json").exists():
        atomic_json(root / "hardware.json", hardware)
    source_hashes = _source_hashes()
    provenance = {
        "upstream_commit": UPSTREAM_COMMIT,
        "source_hashes": source_hashes,
        "prepared_manifest_sha256": sha256_file(prepared / "manifest.json"),
        "prepared_files": data_manifest["files"],
        "config_sha256": _config_hash(config),
    }
    provenance_path = root / "provenance.json"
    if provenance_path.exists():
        if json.loads(provenance_path.read_text()) != provenance:
            raise ValueError("Source or prepared data changed since this run began")
    else:
        atomic_json(provenance_path, provenance)
    commit()  # Durable config/data identity exists before initialization/training.
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    if is_cuda:
        torch.cuda.manual_seed_all(config["seed"])
        torch.set_float32_matmul_precision("high")
        torch.cuda.reset_peak_memory_stats(device)
    model = build("fox", seed=config["seed"], device=device, depth=config["depth"],
                  width=config["width"], seq_len=sequence_len, vocab_size=config["vocab_size"])
    if is_cuda:
        recipe = dict(unembedding_lr=.008, embedding_lr=.3, matrix_lr=.02,
                      scalar_lr=.5, weight_decay=.28)
        recipe.update(config.get("optimizer", {}))
        optimizer = model.setup_optimizer(**recipe)
        optimizer_kind = "upstream_muon_adamw"
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
        for group in optimizer.param_groups:
            group["initial_lr"] = group["lr"]
        optimizer_kind = "adamw_cpu_correctness_test_only"
    stream = TokenStream(prepared / "train.bin")
    store = CheckpointStore(root, commit=commit)
    global_step, cumulative_tokens, parent_sha = 0, 0, None
    loaded_state = None
    resume_name = resume or ("checkpoint1" if stage == "long" else None)
    loaded_manifest = None
    if resume_name is not None:
        loaded_state, loaded_manifest = store.load(resume_name, map_location=device)
        if loaded_state["config_sha256"] != _config_hash(config) or loaded_state["provenance"] != provenance:
            raise ValueError("Checkpoint belongs to different code, config or prepared data")
        if loaded_state["optimizer_kind"] != optimizer_kind:
            raise ValueError("Cannot switch optimizer recipe while resuming")
        global_step = int(loaded_state["global_step"])
        cumulative_tokens = int(loaded_state["cumulative_tokens"])
        final_name = "checkpoint1" if stage == "short" else "checkpoint2"
        completed_checkpoint_recovery = global_step == stage_end and resume_name == final_name
        if not stage_start <= global_step < stage_end and not completed_checkpoint_recovery:
            raise ValueError("Resume checkpoint is outside this stage's step interval")
        if cumulative_tokens != global_step * tokens_per_step:
            raise ValueError("Checkpoint token count is inconsistent with global step")
        if stage == "short" and loaded_state["stage"] != "short":
            raise ValueError("Cannot resume a short stage from a long-stage checkpoint")
        if stage == "long" and loaded_state["stage"] == "short" and global_step != stage_start:
            raise ValueError("Long training must begin from completed checkpoint 1")
        model.load_state_dict(loaded_state["model"], strict=True)
        if model_state_sha256(model.state_dict()) != loaded_state["model_state_sha256"]:
            raise ValueError("Loaded learned weights do not match the parent checkpoint")
        optimizer.load_state_dict(loaded_state["optimizer"])
        stream.load_state_dict(loaded_state["data_stream"])
        parent_sha = loaded_state.get("parent_checkpoint_sha256")
        if stage == "long":
            checkpoint1_state, checkpoint1_manifest = store.load("checkpoint1", map_location="cpu")
            del checkpoint1_state
            parent_sha = checkpoint1_manifest["sha256"]
            proof = {
                "attempt": attempt, "loaded_checkpoint": resume_name,
                "loaded_checkpoint_sha256": loaded_manifest["sha256"],
                "checkpoint1_sha256": parent_sha,
                "loaded_model_state_sha256": loaded_state["model_state_sha256"],
                "weights_verified_before_update": True,
                "optimizer_state_loaded": True,
                "global_step_before_update": global_step,
                "tokens_before_update": cumulative_tokens,
                "data_stream_before_update": stream.state_dict(),
                "training_context": sequence_len,
                "window_sizes": [list(window) for window in model.window_sizes],
            }
            atomic_json(metadata_dir / f"{stage}-{attempt}-continuation.json", proof)
            if not (root / "continuation.json").exists():
                atomic_json(root / "continuation.json", proof)
    if global_step != stage_start and loaded_state is None:
        raise ValueError("Missing required continuation checkpoint")
    mirror = _WandbMirror(config, root, stage, attempt)
    append_jsonl(root / "metrics.jsonl", {
        "event": "stage_start", "stage": stage, "attempt": attempt,
        "step": global_step, "tokens": cumulative_tokens,
        "context": sequence_len, "microbatch": batch, "accumulation": accumulation,
        "optimizer_kind": optimizer_kind, "parent_checkpoint_sha256": parent_sha,
        "wandb_url": mirror.url, "wall_time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    })
    atomic_json(metadata_dir / f"{stage}-{attempt}-tracking.json", {"wandb_url": mirror.url})
    commit()
    # SDK/model construction may use RNG. Restore only immediately before work.
    if loaded_state is not None:
        restore_rng_state(loaded_state["rng"])
        # Parameters were copied into the model; the optimizer now owns its
        # loaded states. Release duplicate checkpoint tensor storage promptly.
        loaded_state = None
    train_seconds = 0.0
    starting_step = global_step
    last_checkpoint = loaded_manifest
    last_loss = None

    def prune_recovery_checkpoints() -> None:
        keep = config.get("keep_last_recovery_checkpoints", 2)
        if isinstance(keep, bool) or not isinstance(keep, int) or keep < 1:
            raise ValueError("At least one recovery checkpoint must be retained")
        entries = []
        for path in (root / "checkpoints").glob("*.json"):
            candidate = json.loads(path.read_text())
            name = candidate.get("name", "")
            if name in ("checkpoint1", "checkpoint2"):
                continue
            if name.startswith(("short-step", "long-step", "short-partial-", "long-partial-")):
                entries.append((int(candidate["metadata"]["global_step"]), path.stat().st_mtime_ns, name))
        entries.sort(reverse=True)
        for _, _, name in entries[keep:]:
            # load validates the completed manifest and entire payload hash.
            # Never prune an incomplete or corrupt recovery artifact silently.
            old_state, old_manifest = store.load(name, map_location="cpu")
            del old_state
            checkpoint_path = root / old_manifest["file"]
            manifest_path = root / "checkpoints" / f"{name}.json"
            append_jsonl(root / "checkpoint-history.jsonl", {
                "event": "recovery_checkpoint_pruned", "checkpoint": name,
                "checkpoint_sha256": old_manifest["sha256"],
                "manifest_sha256": sha256_file(manifest_path),
                "global_step": old_manifest["metadata"]["global_step"],
                "reason": f"retain_latest_{keep}_recovery_checkpoints_plus_both_stage_finals",
            })
            commit()  # Persist the audit trail before removing any old payload.
            checkpoint_path.unlink()
            manifest_path.unlink()
            commit()

    def save_checkpoint(name: str) -> dict:
        state_dict = model.state_dict()
        weights_hash = model_state_sha256(state_dict)
        state = {
            "schema_version": 1, "stage": stage, "model": state_dict,
            "model_config": asdict(model.config), "model_state_sha256": weights_hash,
            "optimizer": optimizer.state_dict(), "optimizer_kind": optimizer_kind,
            "global_step": global_step, "cumulative_tokens": cumulative_tokens,
            "data_stream": stream.state_dict(), "rng": capture_rng_state(),
            "config_sha256": _config_hash(config), "provenance": provenance,
            "parent_checkpoint_sha256": parent_sha,
            "schedule": {"next_global_step": global_step,
                         "total_steps": config["short_steps"] + config["long_steps"]},
        }
        saved = store.save(name, state, {
            "stage": stage, "global_step": global_step, "tokens": cumulative_tokens,
            "context": sequence_len, "model_state_sha256": weights_hash,
            "parent_checkpoint_sha256": parent_sha,
        })
        append_jsonl(root / "metrics.jsonl", {
            "event": "checkpoint_saved", "stage": stage, "step": global_step,
            "tokens": cumulative_tokens, "checkpoint": name,
            "sha256": saved["sha256"], "elapsed_seconds": time.monotonic() - started,
        })
        commit()
        prune_recovery_checkpoints()
        return saved

    def run_validation(label: str, contexts: tuple[int, ...], output_name: str) -> dict:
        destination = root / output_name
        if destination.exists():
            # A named final evaluation is published atomically. An explicit
            # recovery can reuse this evidence, but only after its full shape
            # and checkpoint identities are checked below.
            rows = [json.loads(line) for line in destination.read_text().splitlines() if line.strip()]
        else:
            temporary = metadata_dir / f"{label}-{attempt}.tmp"
            rows = evaluate_checkpoint(model, examples, token_bytes, temporary,
                                       label, contexts=contexts,
                                       target_tokens=config.get("eval_target_tokens", 128), device=device)
            os.replace(temporary, destination)
        expected = {(example["id"], length) for example in examples for length in contexts}
        actual = {(row["example_id"], row["context_length"]) for row in rows}
        if len(rows) != len(expected) or actual != expected or any(row["checkpoint"] != label for row in rows):
            raise ValueError("Saved evaluation does not cover the expected checkpoint/example/context cells")
        metrics = {}
        for length in contexts:
            selected = [row for row in rows if row["context_length"] == length]
            nll = sum(row.get("nll_nats", row.get("total_nll_nats", row.get("nll_sum", 0))) for row in selected)
            nbytes = sum(row["target_bytes"] for row in selected)
            if not nbytes or not selected:
                raise ValueError("Evaluation did not produce positive-byte scored examples")
            metrics[f"val/bpb_context_{length}"] = nll / (math.log(2) * nbytes)
        metrics["val/bpb"] = metrics[f"val/bpb_context_{sequence_len}"]
        return metrics

    try:
        while global_step < stage_end:
            elapsed = time.monotonic() - started
            if elapsed >= config.get("max_seconds_per_stage", 1800) or previous_estimated_cost + _estimated_cost(elapsed, config, device) >= config.get("budget_usd", 100):
                last_checkpoint = save_checkpoint(f"{stage}-partial-{global_step:06d}-{attempt}")
                result = {
                    "status": "timeout_partial", "stage": stage, "global_step": global_step,
                    "tokens": cumulative_tokens, "checkpoint": last_checkpoint,
                    "elapsed_seconds": time.monotonic() - started, "wandb_url": mirror.url,
                    "resume_required": True, "training_complete": False,
                    "estimated_worker_cost_usd": _estimated_cost(time.monotonic() - started, config, device),
                    "estimated_cost_usd": previous_estimated_cost + _estimated_cost(time.monotonic() - started, config, device),
                }
                append_jsonl(root / "metrics.jsonl", dict(event="stage_partial", attempt=attempt, **result))
                atomic_json(result_path, result)
                commit()
                return result
            multiplier = learning_rate_multiplier(global_step, config)
            for group in optimizer.param_groups:
                group["lr"] = group["initial_lr"] * multiplier
            if is_cuda:
                torch.cuda.synchronize(device)
            step_started = time.monotonic()
            model.train()
            optimizer.zero_grad(set_to_none=True)
            loss_sum = torch.zeros((), device=device)
            for _ in range(accumulation):
                x, y = stream.next_batch(batch, sequence_len, device)
                loss = model(x, y)
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError("Nonfinite training loss")
                (loss / accumulation).backward()
                loss_sum += loss.detach() / accumulation
            optimizer.step()
            if is_cuda:
                torch.cuda.synchronize(device)
            step_seconds = time.monotonic() - step_started
            train_seconds += step_seconds
            global_step += 1
            cumulative_tokens += tokens_per_step
            last_loss = float(loss_sum.item())
            elapsed = time.monotonic() - started
            metrics = {
                "event": "train_step", "stage": stage, "attempt": attempt,
                "step": global_step, "tokens": cumulative_tokens, "context": sequence_len,
                "train/loss": last_loss, "train/tokens_per_second": tokens_per_step / step_seconds,
                "train/step_seconds": step_seconds, "train/lr_multiplier": multiplier,
                "elapsed_seconds": elapsed,
                "estimated_worker_cost_usd": _estimated_cost(elapsed, config, device),
                "estimated_cost_usd": previous_estimated_cost + _estimated_cost(elapsed, config, device),
                "cost_estimate_scope": "current_worker_entrypoint_only_excludes_startup_and_storage",
                "estimated_gpu_hours": elapsed / 3600 if is_cuda else 0.0,
                "requested_cpu_core_hours": elapsed * config.get("cpu", 4) / 3600 if is_cuda else 0.0,
                "requested_memory_gib_hours": elapsed * config.get("memory_mb", 16384) / 1024 / 3600 if is_cuda else 0.0,
            }
            if global_step % config.get("eval_every", 100) == 0 and global_step != stage_end:
                metrics.update(run_validation(f"validation-step-{global_step}", (sequence_len,),
                                              f"evaluation-{stage}-step-{global_step:06d}-{attempt}.jsonl"))
            if global_step % config.get("log_every", 10) == 0 or global_step == stage_end:
                metrics.update(_telemetry(device))
            append_jsonl(root / "metrics.jsonl", metrics)
            if global_step % config.get("log_every", 10) == 0 or global_step == stage_end:
                mirror.log(metrics, global_step)
                print(json.dumps(metrics, allow_nan=False), flush=True)
                commit()
            if global_step % config.get("checkpoint_every", 50) == 0 and global_step != stage_end:
                last_checkpoint = save_checkpoint(f"{stage}-step{global_step:06d}")
        final_name = "checkpoint1" if stage == "short" else "checkpoint2"
        # Preserve the trained checkpoint before potentially expensive final evaluation.
        if resume_name == final_name and global_step == stage_end:
            last_checkpoint = loaded_manifest
        else:
            last_checkpoint = save_checkpoint(final_name)
        final_metrics = run_validation(final_name, (config["short_context"], config["long_context"]),
                                       f"evaluation-{final_name}.jsonl")
        append_jsonl(root / "metrics.jsonl", dict(event="final_evaluation", stage=stage,
                      step=global_step, tokens=cumulative_tokens, **final_metrics))
        mirror.log(dict(tokens=cumulative_tokens, **final_metrics), global_step)
        result = {
            "status": "complete", "stage": stage, "training_complete": True,
            "global_step": global_step, "steps_this_attempt": global_step - starting_step,
            "tokens": cumulative_tokens, "context": sequence_len,
            "train_loss": last_loss, "train_seconds": train_seconds,
            "elapsed_seconds": time.monotonic() - started,
            "estimated_worker_cost_usd": _estimated_cost(time.monotonic() - started, config, device),
            "estimated_cost_usd": previous_estimated_cost + _estimated_cost(time.monotonic() - started, config, device),
            "cost_estimate_scope": "current_worker_entrypoint_only_excludes_startup_and_storage",
            "checkpoint": last_checkpoint, "parent_checkpoint_sha256": parent_sha,
            "wandb_url": mirror.url, "wandb_replay_required": mirror.failed,
            "parameters": sum(p.numel() for p in model.parameters()), **final_metrics,
        }
        append_jsonl(root / "metrics.jsonl", dict(event="stage_complete", attempt=attempt,
                      step=global_step, **{k: v for k, v in result.items() if k != "checkpoint"}))
        atomic_json(result_path, result)
        commit()
        return result
    except BaseException as exc:
        failure = {
            "status": "failed", "stage": stage, "attempt": attempt,
            "global_step": global_step, "tokens": cumulative_tokens,
            "exception_type": type(exc).__name__, "message": _safe_error(exc),
            "last_durable_checkpoint": last_checkpoint,
            "estimated_worker_cost_usd": _estimated_cost(time.monotonic() - started, config, device),
            "estimated_cost_usd": previous_estimated_cost + _estimated_cost(time.monotonic() - started, config, device),
        }
        # Preserve failure evidence if storage is still available; rethrow the
        # original exception even if failure logging itself cannot be persisted.
        try:
            atomic_json(metadata_dir / f"{stage}-{attempt}-failure.json", failure)
            append_jsonl(root / "metrics.jsonl", dict(event="stage_failure", **failure))
            commit()
        except Exception:
            pass
        raise
    finally:
        mirror.finish()
        if mirror.failed and result_path.exists():
            saved_result = json.loads(result_path.read_text())
            saved_result["wandb_replay_required"] = True
            atomic_json(result_path, saved_result)
        commit()
