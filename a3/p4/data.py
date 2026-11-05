"""Frozen ClimbMix subset and exact-cursor token stream for context continuation.

Unlike the Part 2 best-fit cropper, this loader uses a sequential stream with
BOS document separators. This declared change makes data position exactly
restorable and invariant to the batch/context shape. It does not claim to
reproduce the Part 2 data packing or its numerical results.
"""
import json
import os
import hashlib
from pathlib import Path

from .artifacts import atomic_json, sha256_file


class TokenStream:
    def __init__(self, path, cursor=0):
        import numpy as np
        self.path = Path(path)
        if self.path.stat().st_size % 4:
            raise ValueError("Token stream must be little-endian uint32")
        self.tokens = np.memmap(self.path, dtype="<u4", mode="r")
        if len(self.tokens) < 2:
            raise ValueError("Token stream is empty")
        self.cursor = 0
        self.load_state_dict({"cursor": cursor, "token_count": len(self.tokens)})

    def next_batch(self, batch_size, sequence_len, device="cpu"):
        import numpy as np
        import torch
        if batch_size <= 0 or sequence_len <= 0:
            raise ValueError("Positive batch size and sequence length required")
        count = batch_size * sequence_len
        # Each sequence needs one look-ahead label. Adjacent rows deliberately
        # overlap at their boundary token; cursor advances by trained tokens.
        indices = (self.cursor + np.arange(count + 1)) % len(self.tokens)
        values = np.asarray(self.tokens[indices], dtype=np.int64)
        x = torch.from_numpy(values[:-1].reshape(batch_size, sequence_len).copy()).to(device)
        y = torch.from_numpy(values[1:].reshape(batch_size, sequence_len).copy()).to(device)
        self.cursor += count
        return x, y

    def state_dict(self):
        return {"cursor": self.cursor, "token_count": len(self.tokens)}

    def load_state_dict(self, state):
        if state.get("token_count", len(self.tokens)) != len(self.tokens):
            raise ValueError("Token stream length changed")
        cursor = state["cursor"]
        if isinstance(cursor, bool) or not isinstance(cursor, int) or cursor < 0:
            raise ValueError("Invalid token stream cursor")
        self.cursor = cursor


def verify_prepared(root):
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text())
    for name, record in manifest["files"].items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Unsafe prepared artifact path")
        path = root / name
        if path.is_symlink() or not path.is_file() or path.stat().st_size != record["size_bytes"] or sha256_file(path) != record["sha256"]:
            raise ValueError(f"Prepared artifact missing or changed: {name}")
    if "token_count" in manifest and (root / "train.bin").stat().st_size != manifest["token_count"] * 4:
        raise ValueError("Token count does not match the prepared stream")
    return manifest


def prepare_data(root, config, commit=lambda: None):
    """CPU-only preparation. Downloads are pinned to a resolved dataset commit."""
    import numpy as np
    import pyarrow.parquet as pq
    from huggingface_hub import HfApi, hf_hub_download
    from nanochat.tokenizer import RustBPETokenizer

    root = Path(root)
    if (root / "manifest.json").exists():
        existing = verify_prepared(root)
        expected = {key: config[key] for key in ("vocab_size", "train_shards", "train_token_limit",
                                                  "tokenizer_chars", "eval_examples", "long_context", "eval_target_tokens")}
        if existing.get("preparation_config") != expected:
            raise ValueError("Existing preparation differs; use a fresh preparation ID")
        if config.get("dataset_revision") and config["dataset_revision"] != existing["dataset_revision"]:
            raise ValueError("Dataset revision differs; use a fresh preparation ID")
        if existing["token_count"] < (config["short_steps"] + config["long_steps"]) * config["tokens_per_step"] + 1:
            raise ValueError("Cached preparation is too small for the declared no-wrap horizon")
        return existing
    root.mkdir(parents=True, exist_ok=True)
    dataset = "karpathy/climbmix-400b-shuffle"
    revision = config.get("dataset_revision") or HfApi().dataset_info(dataset).sha
    indices = list(range(config["train_shards"])) + [6542]
    source_paths = []
    for index in indices:
        print(f"Preparing pinned dataset shard {index}", flush=True)
        path = Path(hf_hub_download(repo_id=dataset, repo_type="dataset", revision=revision,
                                   filename=f"shard_{index:05d}.parquet",
                                   local_dir=root / "source"))
        source_paths.append(path)
    training, heldout = source_paths[:-1], source_paths[-1]

    def documents(paths):
        for path in paths:
            row = 0
            parquet = pq.ParquetFile(path)
            for batch in parquet.iter_batches(batch_size=128, columns=["text"]):
                for text in batch.column(0).to_pylist():
                    if text:
                        yield path.name, row, text
                    row += 1

    fit_hashes = set()
    fit_index = root / "tokenizer-documents.jsonl"
    def tokenizer_documents():
        chars = 0
        with fit_index.open("w") as index:
            for filename, row, full_text in documents(training):
                full_hash = hashlib.sha256(full_text.encode()).hexdigest()
                fit_hashes.add(full_hash)
                text = full_text[:min(10000, config["tokenizer_chars"] - chars)]
                index.write(json.dumps({"id": f"{filename}:row{row}", "text_sha256": full_hash,
                                        "fit_chars": len(text)}) + "\n")
                chars += len(text)
                yield text
                if chars >= config["tokenizer_chars"]:
                    break

    tokenizer_dir = root / "tokenizer"
    print("Fitting tokenizer on training documents only", flush=True)
    tokenizer = RustBPETokenizer.train_from_iterator(tokenizer_documents(), config["vocab_size"])
    tokenizer.save(str(tokenizer_dir))
    if tokenizer.get_vocab_size() != config["vocab_size"]:
        raise ValueError("Tokenizer vocabulary does not match declared model")
    bos = tokenizer.get_bos_token_id()
    special = {tokenizer.encode_special(s) for s in tokenizer.get_special_tokens()}
    token_bytes = np.asarray([0 if i in special else len(tokenizer.decode_single_token_bytes(i))
                              for i in range(tokenizer.get_vocab_size())], dtype=np.int32)
    np.save(root / "token_bytes.npy", token_bytes)
    count = 0
    with (root / "train.bin.tmp").open("wb") as stream, (root / "training-documents.jsonl").open("w") as index:
        for filename, row, text in documents(training):
            full_hash = hashlib.sha256(text.encode()).hexdigest()
            fit_hashes.add(full_hash)
            tokens = tokenizer.encode(text, prepend=bos)
            tokens = tokens[:config["train_token_limit"] - count]
            index.write(json.dumps({"id": f"{filename}:row{row}", "text_sha256": full_hash,
                                    "start_token": count, "used_tokens": len(tokens)}) + "\n")
            np.asarray(tokens, dtype="<u4").tofile(stream)
            count += len(tokens)
            if count >= config["train_token_limit"]:
                break
        stream.flush()
        os.fsync(stream.fileno())
    if count < (config["short_steps"] + config["long_steps"]) * config["tokens_per_step"] + 1:
        raise ValueError("Prepared subset is too small for a no-wrap training chain")
    (root / "train.bin.tmp").replace(root / "train.bin")
    print(f"Prepared deterministic stream: {count:,} tokens", flush=True)
    examples = []
    heldout_hashes = set()
    duplicate_exclusions = 0
    for filename, row, text in documents([heldout]):
        text_hash = hashlib.sha256(text.encode()).hexdigest()
        if text_hash in fit_hashes or text_hash in heldout_hashes:
            duplicate_exclusions += 1
            continue
        tokens = tokenizer.encode(text, prepend=bos)
        if len(tokens) >= config["long_context"] + 1:
            heldout_hashes.add(text_hash)
            tokens = tokens[:config["long_context"] + 1]
            examples.append({"id": f"{filename}:row{row}", "tokens": tokens,
                             "source_text_sha256": text_hash,
                             "text": tokenizer.decode(tokens),
                             "target_text": tokenizer.decode(tokens[-config["eval_target_tokens"]:])})
        if len(examples) == config["eval_examples"]:
            break
    if len(examples) != config["eval_examples"]:
        raise ValueError("Insufficient held-out long documents")
    print(f"Frozen {len(examples)} held-out evaluation documents", flush=True)
    (root / "evaluation.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in examples))
    files = {}
    for path in [root / "train.bin", root / "token_bytes.npy", root / "evaluation.jsonl",
                 root / "tokenizer-documents.jsonl", root / "training-documents.jsonl", *tokenizer_dir.rglob("*")]:
        if path.is_file():
            files[str(path.relative_to(root))] = {"sha256": sha256_file(path), "size_bytes": path.stat().st_size}
    manifest = {"schema_version": 1, "vocab_size": tokenizer.get_vocab_size(), "token_count": count,
                "dataset": dataset, "dataset_revision": revision,
                "sources": [{"name": p.name, "sha256": sha256_file(p), "size_bytes": p.stat().st_size,
                             "split": "evaluation" if p == heldout else "train"} for p in source_paths],
                "packing": "sequential BOS-separated stream; no best-fit cropping; exact cursor",
                "eval_selection": "first eligible held-out documents; first long_context+1 tokens; common suffix scored",
                "exact_duplicate_documents_excluded": duplicate_exclusions,
                "evaluation_excludes_full_text_hashes_used_for_training_or_tokenizer": True,
                "preparation_config": {key: config[key] for key in ("vocab_size", "train_shards", "train_token_limit",
                                                                    "tokenizer_chars", "eval_examples", "long_context", "eval_target_tokens")},
                "files": files}
    atomic_json(root / "manifest.json", manifest)
    commit()
    return verify_prepared(root)
