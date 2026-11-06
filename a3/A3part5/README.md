# Part 5: FoX nanochat training and evaluation

Uses nanochat revision `92d63d4e8bb4df75c3b71618f31ddde2378b2bcd` and the
24-layer speedrun capacity: width 1536, 12 query/KV heads, vocabulary 32768,
context 2048. The Part 2 FoX adaptation adds forget gates and removes RoPE;
this is not the paper's optimized attention kernel.

## Training

`run_train.py` invokes native `scripts.base_train`. `fox.py` adapts Part 2
attention to the native block interface and adds activation checkpointing.
The native optimizer, learning-rate schedule, distributed synchronization,
data loader, tokenizer and checkpoint formats are retained. Eligible linear
layers use FP8; attention/embeddings use BF16 and attention scores use FP32.
The global batch is 1,048,576 tokens with microbatch 2 on each of eight H100s
and 32 accumulation steps. The configuration selects 5,571 updates and
5,841,616,896 training tokens for the current model, not an entire corpus pass.

All preparation happens on Modal: eight initial ClimbMix shards, the native
2B-character/32768-vocabulary tokenizer recipe, and 170 training shards plus
the held-out validation shard. Dataset and checkpoint files stay remote.
A pico FoX reference uses the Part 2 recipe: 500 updates, 16.384M tokens, seed
42, six layers, width 384 and context 512, with the shared Part 5 tokenizer.
It is a reproduced reference, not the original Part 2 checkpoint.

The main pipeline prepares data, trains/evaluates the pico reference, records
a scaling prediction, trains nano, and evaluates both saved checkpoints and
both question sets. There is no rehearsal. Checkpoints are saved every 500
updates and at completion. On recovery, training uses the newest checkpoint
file set containing model, metadata and all eight optimizer shards, passing
`--resume-from-step` to the native trainer. Native loader resumption is not
an exact replay of unconsumed packing buffers. A completed horizon is skipped.
If training started but no complete checkpoint exists, execution stops rather
than silently restarting from zero. Native deserialization errors also stop
recovery; no corrupt checkpoint is silently accepted.

## Evaluation and outputs

Canonical final results live under `bpb-default-budget-v1/` in the run directory:

- `bpb.json`: pico/nano at contexts 512 and 2048, each using **41,943,040**
  nominal validation positions (the pinned training validator's default).
  Evaluation batch size remains 8: 10,240 batches at 512; 2,560 at 2048.
  This is a specified matched protocol, not every standalone-evaluator default.
- `evaluation.json`: the four BPB rows, consumed training tokens and scaling
  comparison. The fixed exponent 0.076 is applied to the full-budget pico
  anchor. Recalculations from final evaluation are labelled retrospective;
  the historical pretraining `scaling_prediction.json` is preserved.

Both models share a tokenizer and validation stream at each context. Across
contexts packing changes, and pico at 2048 is extrapolation. Token budget,
capacity and training settings also differ between models, so BPB improvement
cannot be attributed solely to model size or FoX.

The two JSON question specifications contain 20 initial and 10 additional
questions. Both models use greedy base-model completion, context 512 and 48
new tokens, without SFT or a KV cache. Original answers use `answers.json`
and `pico/answers.json`; additional answers use `questions-extra-v1/{pico,nano}/answers.json`.
Recovery validates saved question IDs/prompts/criteria, preserves completed
answers and generates only missing ones. Grades remain manual. The second
question set was selected after reviewing the first; disclose that selection.

Optional CORE evaluation calls native `scripts.base_eval.evaluate_core` on
the final nano checkpoint with `max_per_task=-1`. It writes the measured
aggregate, per-task scores, checkpoint step and context to `core-final.json`.
It does not substitute a constant for an actual evaluation result. Existing
matching CORE results are reused. CORE is opt-in so a BPB/question recovery
cannot accidentally launch an additional benchmark. Partial CORE evaluation
currently restarts the suite; model weights remain unchanged.

Small results are written atomically. Completed BPB cells and question answers
trigger immediate Modal volume commits through the launcher. Recovery skips
complete BPB cells and restarts only an interrupted cell. Do not run overlapping
jobs for the same run ID. Original root-level reduced-budget BPB artifacts from
earlier execution are historical records, not the final results used above.
New main runs use the native training-validation budget and the full-budget
final evaluation. This does not retroactively change existing training logs.

## Commands (repository root)

Use a Python environment with `requirements.txt` installed. The examples reuse
the existing Modal client environment and profile. Secret `a3-p4-wandb` must
contain `WANDB_API_KEY`. Never commit credentials.

```sh
# New training run; use a new run ID. Starts paid GPU work.
MODAL_PROFILE=a3-p4 a3/A3part4/.venv/bin/modal run --detach a3/A3part5/modal_app.py::main --run-id NEW_RUN_ID

# Recover an existing prepared/trained run; resumes native training if needed.
MODAL_PROFILE=a3-p4 a3/A3part4/.venv/bin/modal run --detach a3/A3part5/modal_app.py::main --run-id RUN_ID --resume-prepared

# Evaluate existing checkpoints only: full-budget BPB, scaling, both question sets.
MODAL_PROFILE=a3-p4 a3/A3part4/.venv/bin/modal run --detach a3/A3part5/modal_app.py::evaluate_saved --run-id RUN_ID

# Full-budget BPB and scaling only, preserving completed cells.
MODAL_PROFILE=a3-p4 a3/A3part4/.venv/bin/modal run --detach a3/A3part5/modal_app.py::validate_default --run-id RUN_ID

# Additional questions only.
MODAL_PROFILE=a3-p4 a3/A3part4/.venv/bin/modal run --detach a3/A3part5/modal_app.py::questions --run-id RUN_ID

# Optional full CORE from the saved nano checkpoint; no training.
MODAL_PROFILE=a3-p4 a3/A3part4/.venv/bin/modal run --detach a3/A3part5/modal_app.py::core --run-id RUN_ID
```

Use `--with-core` on `::main` to include CORE after final evaluation. No command
was launched as part of the code cleanup. Entrypoints spawn remote calls and
exit; the jobs continue detached. New run IDs cannot overwrite existing ones.
Modal can preempt workers despite `retries=0`; recovery guards apply on re-entry.
Main GPU training has a 10-hour timeout; single-GPU preparation/reference and
evaluation functions have 6-hour timeouts. CPU orchestration has a separate
12-hour (new run) or 11.5-hour (recovery) limit; these are not dollar guarantees
or promises of completion. Saved-checkpoint evaluation can be relaunched
independently. The workspace budget is the final spending control.

Training logs, configuration, tokenizer hashes, W&B links and checkpoints stay
under `/runs/<run-id>/` on volume `a3-part5-fox`. Cache is in `/cache/`. Local
artifacts, model files, datasets, W&B output and credentials are gitignored.
Native FLOP/MFU estimates are not exact for FP8 conversion, FoX attention or
activation recomputation; do not present them as measured hardware utilization.

## Local checks

```sh
python3 -m unittest discover -s a3/A3part5 -p 'test_*.py'
```

These standard-library tests exercise persistence/recovery, command generation
and fixed evaluation settings. They do not replace numerical CPU/GPU checks.
`fox.verify()` checks forward/gradient agreement with Part 2 using nonzero
attention/MLP projections in disposable models before a new training run.

References: [nanochat](https://github.com/karpathy/nanochat/tree/92d63d4e8bb4df75c3b71618f31ddde2378b2bcd),
[scaling hypothesis](https://arxiv.org/abs/2001.08361).
