"""Call upstream BPB unchanged on immutable, shared best-fit validation batches."""
from pathlib import Path
import gc
import hashlib
import json
import math
import time
import uuid

from p4.artifacts import atomic_json, append_jsonl, sha256_file, CheckpointStore

TOKEN_BUDGET = 262144
EVAL_BATCH = 8
FOX_ID = "fox-seed42-v1"
BASELINE_ID = "baseline-control-seed42-v1"
EVALUATION_ID = "standard-validation-seed42-v1"


def supplemental_sources():
    return {p.name: sha256_file(p) for p in sorted(Path(__file__).parent.glob("*.py"))}


def freeze_validation(prepared, destination):
    """Use the actual upstream best-fit loader; copy its reusable buffers."""
    import numpy as np
    import nanochat.dataset as dataset
    import nanochat.dataloader as loader_module
    import nanochat.loss_eval as metric_module
    from nanochat.tokenizer import RustBPETokenizer

    prepared, destination = Path(prepared), Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    source = prepared / "source/shard_06542.parquet"
    identity = {
        "dataset_revision": json.loads((prepared / "manifest.json").read_text())["dataset_revision"],
        "validation_shard_sha256": sha256_file(source),
        "tokenizer_sha256": sha256_file(prepared / "tokenizer/tokenizer.pkl"),
        "token_bytes_sha256": sha256_file(prepared / "token_bytes.npy"),
        "loader_source_sha256": sha256_file(loader_module.__file__),
        "metric_source_sha256": sha256_file(metric_module.__file__),
        "nominal_tokens_per_cell": TOKEN_BUDGET,
        "batch_size": EVAL_BATCH,
        "contexts": [512, 2048],
        "split": "val",
        "protocol": "upstream tokenizing_distributed_data_loader_bos_bestfit + evaluate_bpb",
        "part2_alignment": "Same procedure and nominal budget; original Jinhao tokenizer/input hashes are unavailable, so exact Part Two input reproduction is not claimed.",
        "cross_context_policy": "Separate standard best-fit streams per length; do not interpret their difference as a matched-target context effect.",
    }
    manifest_path = destination / "validation-manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest["identity"] != identity:
            raise ValueError("Frozen validation identity changed")
        for name, record in manifest["files"].items():
            if sha256_file(destination / name) != record["sha256"]:
                raise ValueError("Frozen validation bytes changed")
        return manifest
    tokenizer = RustBPETokenizer.from_directory(str(prepared / "tokenizer"))
    files, batches = {}, {}
    previous = dataset.DATA_DIR
    try:
        # Storage location only: dataset/loader/metric source code is unchanged.
        dataset.DATA_DIR = str(prepared / "source")
        paths = dataset.list_parquet_files()
        if Path(paths[-1]).name != "shard_06542.parquet":
            raise ValueError("Upstream validation split would select the wrong shard")
        for context in (512, 2048):
            steps = TOKEN_BUDGET // (EVAL_BATCH * context)
            iterator = loader_module.tokenizing_distributed_data_loader_bos_bestfit(
                tokenizer, EVAL_BATCH, context, "val", device="cpu")
            xs, ys, batch_records = [], [], []
            for index in range(steps):
                x, y = next(iterator)
                a, b = x.numpy().astype("<i4", copy=True), y.numpy().astype("<i4", copy=True)
                xs.append(a)
                ys.append(b)
                batch_records.append({"batch": index, "input_sha256": hashlib.sha256(a.tobytes()).hexdigest(),
                                      "target_sha256": hashlib.sha256(b.tobytes()).hexdigest()})
            name = f"validation-{context}.npz"
            np.savez_compressed(destination / name, inputs=np.stack(xs), targets=np.stack(ys))
            files[name] = {"sha256": sha256_file(destination / name),
                           "size_bytes": (destination / name).stat().st_size}
            batches[str(context)] = batch_records
    finally:
        dataset.DATA_DIR = previous
    manifest = {"schema_version": 1, "identity": identity, "files": files, "batches": batches}
    atomic_json(manifest_path, manifest)
    return manifest


class RecordingModel:
    """Collect per-batch evidence without changing upstream evaluate_bpb."""
    def __init__(self, model, token_bytes):
        self.model, self.token_bytes, self.rows = model, token_bytes, []

    def get_device(self):
        return self.model.get_device()

    def __call__(self, x, y, loss_reduction="none"):
        import torch
        losses = self.model(x, y, loss_reduction=loss_reduction)
        targets, losses_flat = y.reshape(-1), losses.reshape(-1)
        valid = targets >= 0
        safe = targets.clamp_min(0)
        sizes = torch.where(valid, self.token_bytes[safe], 0)
        nll = (losses_flat * (sizes > 0)).sum()
        if not bool(torch.isfinite(nll)):
            raise FloatingPointError("Nonfinite standard validation loss")
        self.rows.append({"batch": len(self.rows), "nll_nats_float32_sum": float(nll),
                          "target_bytes": int(sizes.sum()), "scored_non_special_tokens": int((sizes > 0).sum())})
        return losses


def evaluate_frozen(model, arrays, token_bytes, device="cuda"):
    import torch
    from nanochat.loss_eval import evaluate_bpb

    def batches():
        for a, b in zip(arrays["inputs"], arrays["targets"]):
            yield torch.tensor(a, dtype=torch.long, device=device), torch.tensor(b, dtype=torch.long, device=device)
    recorder = RecordingModel(model, token_bytes)
    model.eval()
    with torch.inference_mode():
        value = evaluate_bpb(recorder, batches(), len(arrays["inputs"]), token_bytes)
    nbytes = sum(x["target_bytes"] for x in recorder.rows)
    reconstructed = sum(x["nll_nats_float32_sum"] for x in recorder.rows) / (math.log(2) * nbytes)
    if not math.isclose(value, reconstructed, abs_tol=2e-6, rel_tol=2e-6):
        raise ValueError("Batch evidence does not reconstruct upstream BPB")
    return value, recorder.rows


def evaluate_family(config, prepared, runs, output, family, commit=lambda: None):
    import numpy as np
    import torch
    from pico.variants import build
    from p4.training import _WandbMirror, _hardware, _estimated_cost, model_state_sha256

    if family not in ("fox", "baseline"):
        raise ValueError("Unknown architecture")
    prepared, runs, output = Path(prepared), Path(runs), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    final = output / f"{family}-standard.json"
    if final.exists():
        raise FileExistsError("Named evaluation results are immutable")
    started = time.monotonic()
    validation = freeze_validation(prepared, output)
    atomic_json(output / "supplemental-source-hashes.json", supplemental_sources())
    commit()
    identity = sha256_file(output / "validation-manifest.json")
    run_id = FOX_ID if family == "fox" else BASELINE_ID
    store = CheckpointStore(runs / run_id)
    token_bytes = torch.tensor(np.load(prepared / "token_bytes.npy"), dtype=torch.int64, device="cuda")
    tracking = dict(config, experiment_id=EVALUATION_ID, wandb_project="a3-p4-standard-validation")
    mirror = _WandbMirror(tracking, output, f"{family}-evaluation", uuid.uuid4().hex[:12])
    results = []
    try:
        for checkpoint in ("checkpoint1", "checkpoint2"):
            state, record = store.load(checkpoint)
            if model_state_sha256(state["model"]) != state["model_state_sha256"]:
                raise ValueError("Checkpoint learned-tensor hash failed")
            for context in (512, 2048):
                model = build(family, seed=config["seed"], device="cuda", depth=config["depth"],
                              width=config["width"], seq_len=context, vocab_size=config["vocab_size"])
                model.load_state_dict(state["model"], strict=True)
                with np.load(output / f"validation-{context}.npz", allow_pickle=False) as arrays:
                    bpb, rows = evaluate_frozen(model, arrays, token_bytes)
                label = f"{family}-{checkpoint}-{context}"
                for row in rows:
                    append_jsonl(output / f"{label}-batches.jsonl", row)
                result = {"family": family, "checkpoint": checkpoint, "evaluation_context": context,
                          "training_context": state["model_config"]["sequence_len"], "training_tokens": state["cumulative_tokens"],
                          "bpb": bpb, "nominal_tokens": TOKEN_BUDGET,
                          "scored_non_special_tokens": sum(r["scored_non_special_tokens"] for r in rows),
                          "target_bytes": sum(r["target_bytes"] for r in rows), "batches": len(rows),
                          "checkpoint_sha256": record["sha256"], "learned_weights_unchanged": True,
                          "validation_manifest_sha256": identity, "window_policy": model.window_sizes,
                          "parameters": sum(p.numel() for p in model.parameters())}
                if model_state_sha256(model.state_dict()) != state["model_state_sha256"]:
                    raise ValueError("Evaluation changed checkpoint weights")
                results.append(result)
                mirror.log({"tokens": state["cumulative_tokens"], f"standard/bpb_{context}": bpb,
                            "checkpoint": checkpoint, "family": family}, state["global_step"])
                print(json.dumps(result), flush=True)
                del model
                torch.cuda.empty_cache()
            del state
            gc.collect()
        result = {"schema_version": 1, "status": "complete", "family": family, "results": results,
                  "validation_manifest_sha256": identity, "hardware": _hardware(config, "cuda"),
                  "elapsed_seconds": time.monotonic() - started, "wandb_url": mirror.url,
                  "estimated_worker_cost_usd": _estimated_cost(time.monotonic() - started, config, "cuda"),
                  "cost_scope": "Worker timer only; excludes build/startup/storage and final tracking finalization."}
        atomic_json(final, result)
        commit()
        return result
    finally:
        mirror.finish()
        commit()
