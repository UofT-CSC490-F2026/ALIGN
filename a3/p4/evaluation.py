"""Matched-target document continuation evaluation for the Part 4 FoX model.

An input window of L tokens predicts L shifted targets. Only the final
``target_tokens`` targets count toward the score. Thus the scored targets are
identical when the input window changes, including their preceding target
history. This is a likelihood task for a pretrained model, not a chat test.
"""
from __future__ import annotations

from contextlib import contextmanager
import copy
import hashlib
import json
import math
from pathlib import Path


def _token_hash(tokens):
    return hashlib.sha256(json.dumps(list(tokens), separators=(",", ":")).encode()).hexdigest()


def make_scoring_window(tokens, context, target_tokens=128):
    """Return aligned input/target lists and the start of the common suffix.

    ``context`` denotes model input length, including prior suffix tokens. It
    does not denote the number of tokens before the first scored target.
    """
    if isinstance(context, bool) or not isinstance(context, int) or context < 1:
        raise ValueError("context must be a positive integer")
    if isinstance(target_tokens, bool) or not isinstance(target_tokens, int) or not 0 < target_tokens <= context:
        raise ValueError("target_tokens must be between 1 and context")
    tokens = list(tokens)
    if len(tokens) < context + 1:
        raise ValueError(f"Need at least {context + 1} tokens for a {context}-token input window")
    if any(isinstance(t, bool) or not isinstance(t, int) or t < 0 for t in tokens):
        raise ValueError("Token IDs must be non-negative integers")
    segment = tokens[-(context + 1):]
    return {
        "input_ids": segment[:-1],
        "target_ids": segment[1:],
        "score_start": context - target_tokens,
        "target_suffix_ids": segment[-target_tokens:],
    }


@contextmanager
def evaluation_window_policy(model, context):
    """Use context-derived upstream SSSL windows, then restore the model.

    Both checkpoint models receive this same policy at a given input length.
    Changing only a model's sequence_len without recalculating window_sizes
    would leave its training-time receptive fields in place.
    """
    original_config = getattr(model, "config", None)
    has_windows = hasattr(model, "window_sizes")
    original_windows = getattr(model, "window_sizes", None)
    compute_windows = getattr(model, "_compute_window_sizes", None)
    if has_windows and not callable(compute_windows):
        raise ValueError("Model window_sizes exist but cannot be explicitly recomputed")
    try:
        if callable(compute_windows):
            if original_config is None:
                raise ValueError("Context-derived window policy requires model.config")
            config = copy.copy(original_config)
            config.sequence_len = context
            model.config = config
            model.window_sizes = compute_windows(config)
            policy = {
                "name": "upstream_context_scaled_windows",
                "sequence_len": context,
                "window_pattern": config.window_pattern,
                "window_sizes": [list(w) for w in model.window_sizes],
                "final_layer_full_context": True,
            }
        else:
            # Small reference models can own their full-causal attention policy.
            policy = {"name": "model_owned_full_causal", "sequence_len": context}
        yield policy
    finally:
        if original_config is not None:
            model.config = original_config
        if has_windows:
            model.window_sizes = original_windows
        elif hasattr(model, "window_sizes"):
            delattr(model, "window_sizes")


def evaluate_checkpoint(model, examples, token_bytes, output_path, checkpoint,
                        contexts=(512, 2048), target_tokens=128, device="cpu"):
    """Persist and return one row per example and evaluation input length.

    The output must be new or empty; interrupted/duplicate evaluations must use
    a new file rather than silently appending a second copy. The caller owns
    durable Volume commits after writes. Zero-byte tokens (special tokens) are
    excluded from both NLL and byte denominators, as in upstream nanochat BPB.
    """
    import torch
    from torch.nn import functional as F
    from p4.artifacts import append_jsonl

    contexts = tuple(contexts)
    if not contexts or len(set(contexts)) != len(contexts):
        raise ValueError("contexts must be nonempty and unique")
    if not isinstance(checkpoint, str) or not checkpoint.strip():
        raise ValueError("checkpoint label is required")
    output_path = Path(output_path)
    if output_path.exists() and output_path.stat().st_size:
        raise FileExistsError(f"Refusing duplicate evaluation output: {output_path}")
    examples = list(examples)
    if not examples:
        raise ValueError("Evaluation requires at least one held-out example")
    ids = [example.get("id") for example in examples]
    if any(not isinstance(i, str) or not i for i in ids) or len(set(ids)) != len(ids):
        raise ValueError("Evaluation IDs must be nonempty, unique strings")
    byte_tensor = torch.as_tensor(token_bytes, device=device)
    if byte_tensor.ndim != 1 or byte_tensor.numel() == 0:
        raise ValueError("token_bytes must be a nonempty one-dimensional array")
    if not torch.isfinite(byte_tensor).all() or (byte_tensor < 0).any() or (byte_tensor != byte_tensor.long()).any():
        raise ValueError("Token byte lengths must be finite non-negative integers")
    byte_tensor = byte_tensor.long()
    # Validate the entire request before creating a partial evidence file.
    for example in examples:
        for context in contexts:
            window = make_scoring_window(example["tokens"], context, target_tokens)
            if max(window["input_ids"] + window["target_ids"]) >= byte_tensor.numel():
                raise ValueError(f"Token ID outside tokenizer vocabulary: {example['id']}")
            suffix = torch.tensor(window["target_suffix_ids"], device=device)
            if byte_tensor[suffix].sum().item() <= 0:
                raise ValueError(f"No scored bytes in target suffix: {example['id']}")

    config = getattr(model, "config", None)
    training_context = getattr(config, "sequence_len", None)
    was_training = model.training
    rows = []
    try:
        model.eval()
        with torch.inference_mode():
            for context in contexts:
                with evaluation_window_policy(model, context) as policy:
                    for example in examples:
                        window = make_scoring_window(example["tokens"], context, target_tokens)
                        x = torch.tensor([window["input_ids"]], dtype=torch.long, device=device)
                        y = torch.tensor(window["target_suffix_ids"], dtype=torch.long, device=device)
                        logits = model(x)
                        if logits.ndim != 3 or logits.shape[:2] != x.shape:
                            raise ValueError("Model must return [batch, sequence, vocabulary] logits")
                        losses = F.cross_entropy(logits[0, -target_tokens:].float(), y, reduction="none")
                        lengths = byte_tensor[y]
                        valid = lengths > 0
                        if not torch.isfinite(losses[valid]).all():
                            raise FloatingPointError("Nonfinite held-out continuation loss")
                        nll = losses[valid].double().sum().item()
                        num_bytes = int(lengths.sum().item())
                        row = {
                            "schema_version": 1,
                            "event": "evaluation_example",
                            "checkpoint": checkpoint,
                            "example_id": example["id"],
                            "context_length": context,
                            "input_window_tokens": context,
                            "tokens_before_first_scored_target": context - target_tokens + 1,
                            "target_suffix_tokens": target_tokens,
                            "scored_tokens": int(valid.sum().item()),
                            "target_bytes": num_bytes,
                            "nll_nats": nll,
                            "bpb": nll / (math.log(2) * num_bytes),
                            "target_token_ids": window["target_suffix_ids"],
                            "target_token_byte_counts": lengths.cpu().tolist(),
                            "target_nll_nats": [float(v) if keep else None for v, keep in zip(losses.cpu().tolist(), valid.cpu().tolist())],
                            "input_sha256": _token_hash(window["input_ids"]),
                            "target_sha256": _token_hash(window["target_suffix_ids"]),
                            "window_policy": policy,
                            "model_variant": getattr(model, "variant", None),
                            "training_context_length": training_context,
                            "out_of_training_range": training_context is not None and context > training_context,
                        }
                        append_jsonl(output_path, row)
                        rows.append(row)
    finally:
        model.train(was_training)
    return rows


def aggregate_results(rows):
    """Aggregate BPB by total bytes rather than averaging per-example BPB."""
    groups = {}
    identities = set()
    target_hashes = {}
    for row in rows:
        identity = (row["checkpoint"], row["context_length"], row["example_id"])
        if identity in identities:
            raise ValueError(f"Duplicate evaluation result: {identity}")
        identities.add(identity)
        if row.get("target_sha256"):
            previous = target_hashes.setdefault(row["example_id"], row["target_sha256"])
            if previous != row["target_sha256"]:
                raise ValueError(f"Target changed between evaluations: {row['example_id']}")
        if row["target_bytes"] <= 0 or not math.isfinite(row["nll_nats"]) or row["nll_nats"] < 0:
            raise ValueError("Invalid evaluation numerator or byte denominator")
        key = row["checkpoint"], row["context_length"]
        group = groups.setdefault(key, {"checkpoint": key[0], "context_length": key[1], "examples": 0,
                                       "nll_nats": 0.0, "target_bytes": 0, "scored_tokens": 0})
        group["examples"] += 1
        group["nll_nats"] += row["nll_nats"]
        group["target_bytes"] += row["target_bytes"]
        group["scored_tokens"] += row.get("scored_tokens", 0)
    result = []
    for key in sorted(groups):
        group = groups[key]
        group["bpb"] = group["nll_nats"] / (math.log(2) * group["target_bytes"])
        result.append(group)
    return result
