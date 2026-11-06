# Our Part 4: baseline context extension with equal-token control

Alternative to Jerry's PR #28, not a replacement for his results. No LaTeX,
Part 5 training, FoX continuation, or question-generation code is included.

## Model and training

Reuse the actual Part 2 baseline weights for seeds 42 and 43: six layers,
width 384, six heads, vocabulary 32768, RoPE and SSSL, 16.384M tokens at 512.
Load into native pinned nanochat GPT (`92d63d4`). For each seed, independently
continue at 512 (control) and 2048 for 1024 updates / 33.554432M additional
tokens. Each final model has 49.938432M cumulative tokens. One H100, BF16,
32768 tokens/update, microbatch 8 or 2 respectively, accumulation 8.

Original checkpoints did not contain optimizer state: both arms restart the
optimizer and data order, at 20% of original base learning rates, with a
20-update warmup and cosine decay to 10%. No rotary interpolation. Longer
context also enlarges local SSSL windows; this is a complete context-setting
intervention, not an isolated RoPE or attention-distance experiment.

## New evaluation

Freeze the first 48 validation documents with at least 2049 tokens (including
BOS), retaining their first 2049 tokens. Score the last 256 targets of those
excerpts, not necessarily the end of the full document. Each of the original,
512-control and 2048-continued checkpoints is scored at both input lengths,
for both seeds. All conditions score the same 12288 target positions and
byte lengths. BPB is total negative log likelihood / (ln(2) * total bytes).
This is teacher-forced natural-text continuation, not retrieval or chat eval.

At 2048 evaluation, mean BPB is 1.43134 (original), 1.35956 (512 control),
and 1.19535 (2048 continuation). The matched-budget reduction is 0.16422 BPB.
At 512 evaluation the corresponding scores are 1.36118, 1.24543, and 1.25232.
Two seeds and 48 selected documents do not establish broad generalization.
The paired bootstrap is for unweighted per-document BPB differences, not
the byte-weighted aggregate table; its intervals do not estimate seed uncertainty.

Monitoring calibration BPB uses a separate packed 512-token subset. Do not
substitute those scores for the primary suffix evaluation. Raw comparison
records include document scores, source/tokenizer/eval hashes and W&B links.

## Reproduction and historical provenance

This draft extracts the original baseline experiment from shared Parts 4/5
helpers into `context_train.py` and `context_eval.py`, removing unused Part 5
paths. Historical JSON remains unchanged; its source hashes refer to those
original helpers, not this extraction. Evaluation tensor arithmetic and the
baseline training update sequence are retained. No fresh numerical GPU check
or training was performed for this draft.

The launcher uses the existing Part 2 Modal image, volume `a3-picochat-pilot`,
secret `wandb-secret`, original `pilot-v1/baseline/seed{42,43}/model.pt` files,
four training shards, validation shard 06542, and the pinned original tokenizer
hash. These are prerequisites, not downloaded by this launcher. The new helper
uses an isolated `part4-baseline-v1-eval` directory for the evaluation suite,
without importing Part 5 question lists. The original experiment reused the
same selected documents/packed batches from a shared suite.

`modal run a3/baseline_context_modal.py` starts paid GPU work. Existing output
directories block reruns. Do not launch it to inspect the PR or delete historical
outputs to bypass the guard. A fresh experiment requires explicit budget and
a new output namespace. Timeouts are safeguards, not guaranteed invoice caps.

## Offline checks

```powershell
python a3/pico/summarize_baseline_context.py
python -m unittest discover -s a3/pico -p test_context_results.py -v
```

These validate historical completion, equal budgets, artifact identity and
byte-weighted BPB arithmetic. They do not validate GPU execution of the extracted
code. The historical W&B audit counted 105 history rows; the later loss-only
export contained 104 points, so use the latter count when plotting loss curves.
