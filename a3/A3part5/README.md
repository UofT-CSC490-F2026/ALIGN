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

The main pipeline prepares data, trains/evaluates the pico reference, trains
nano, and evaluates both saved checkpoints and both question sets. The matched
scaling study has its own entry point, described below. Checkpoints are saved every 500
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
- `evaluation.json`: the four measured BPB rows and consumed training tokens.
  Scaling predictions are saved separately by the matched scaling study.

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
jobs for the same run ID. New main runs use the native training-validation budget and the full-budget
final evaluation.

## Commands (repository root)

Use a Python environment with `requirements.txt` installed. The examples reuse
the existing Modal client environment and profile. Secret `a3-p4-wandb` must
contain `WANDB_API_KEY`. Never commit credentials.

```sh
# New training run; use a new run ID. Starts paid GPU work.
MODAL_PROFILE=a3-p4 a3/A3part4/.venv/bin/modal run --detach a3/A3part5/modal_app.py::main --run-id NEW_RUN_ID

# Recover an existing prepared/trained run; resumes native training if needed.
MODAL_PROFILE=a3-p4 a3/A3part4/.venv/bin/modal run --detach a3/A3part5/modal_app.py::main --run-id RUN_ID --resume-prepared

# Evaluate existing checkpoints only: full-budget BPB and both question sets.
MODAL_PROFILE=a3-p4 a3/A3part4/.venv/bin/modal run --detach a3/A3part5/modal_app.py::evaluate_saved --run-id RUN_ID

# Full-budget BPB only, preserving completed cells.
MODAL_PROFILE=a3-p4 a3/A3part4/.venv/bin/modal run --detach a3/A3part5/modal_app.py::validate_default --run-id RUN_ID

# Additional questions only.
MODAL_PROFILE=a3-p4 a3/A3part4/.venv/bin/modal run --detach a3/A3part5/modal_app.py::questions --run-id RUN_ID

# Optional full CORE from the saved nano checkpoint; no training.
MODAL_PROFILE=a3-p4 a3/A3part4/.venv/bin/modal run --detach a3/A3part5/modal_app.py::core --run-id RUN_ID
```

Use `--with-core` on `::main` to include CORE after final evaluation. Entrypoints spawn remote calls and
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

Reference: [nanochat](https://github.com/karpathy/nanochat/tree/92d63d4e8bb4df75c3b71618f31ddde2378b2bcd).

## Matched scaling study

Use `scaling_modal.py` for the empirical scaling analysis. The study estimates
its exponent from the matched small-model measurements.

The study trains depths 4/6/8 with initialization seeds 42 and 43, at context
2048, with the existing saved tokenizer, FoX implementation, native optimizer,
FP8 policy and batch/schedule scaling. Each uses approximately eight training
tokens per native scaling parameter (transformer matrices plus output head).
The training document order is deterministic and shared: seeds vary model
initialization, not dataset order. `scaling_train.py` changes the initialization
seed after native compute initialization, then invokes `run_train.py`; it does
not replace the native training loop. Periodic standard validation and CORE are
disabled for these calibration runs. Each completed model instead receives the
fixed-set evaluation below. Native checkpoints are saved every 250 updates and
at completion.

Preparation separates validation documents by text-hash parity, removes exact
text duplicates and uses the native BOS best-fit loader to freeze two sets of
1,048,576 target positions each. Set A fits the curve; set B checks transfer to
different documents. Both have context 2048, batch 8 and 64 batches. Text and
batch hashes are retained. These sets are not claimed to exclude every example
used by earlier standard validation. Their scores must not be relabelled as
the existing 41,943,040-position standard-validation scores.

Fit `BPB = exp(intercept) * N^slope` by least squares in log space, using set A.
Fit depths 4/6/8 (two seeds each), freeze the nano prediction, and then evaluate
the existing depth-24 checkpoint on the same sets. Set B never enters the fit.
Seed-specific forecasts are sensitivity diagnostics, not calibrated confidence
intervals. There is no intermediate held-out size,
and the final scaling-parameter count is 17.4 times the largest fitted size.
Three sizes limit checks of extrapolation stability. No loss floor
is fitted, and no accuracy threshold is used to hide an unsuccessful forecast.
This is joint model/data scaling along the configured ratio, not independently
identified effects or a claim that ratio eight is compute-optimal.

The nano model already exists and its general performance is known, so this
is explicitly a retrospective study with nano excluded from fitting. The
final nano model is never retrained by this entry point.

```sh
PYTHONPATH=a3 MODAL_PROFILE=a3-p4 a3/A3part4/.venv/bin/modal run --detach a3/A3part5/scaling_modal.py
```

Outputs remain on volume `a3-part5-fox`, under
`/runs/fox-d24-seed42-v1/scaling-study-v1/`:

- `protocol.json`, `evaluation_manifest.json`: frozen settings, source/tokenizer hashes and document identifiers.
- `eval_A.pt`, `eval_B.pt`, `documents_A.parquet`, `documents_B.parquet`: remote-only evaluation inputs.
- `d{depth}-seed{seed}/`: native checkpoints, training logs, W&B link and fixed-set results.
- `nano_prediction.json`: prediction saved before the matched nano evaluation.
- `nano/result.json`, `summary.json`: final measurements, prediction errors and set-B comparisons.
- `status.json`: progress/failure status, separate from the original run.

The detached pipeline runs one small model at a time on one H100 (two-hour
function timeout per model including evaluation). Nano evaluation has a
one-hour timeout; CPU orchestration has a 24-hour timeout. These are bounds,
not runtime or cost estimates. Do not run overlapping study jobs. Recovery
skips completed runs and resumes existing checkpoints only when their model,
metadata and rank-0 optimizer state are available. An attempt without a usable
checkpoint stops rather than silently restarting. Native loader recovery is
approximate. Result sets are committed as they finish, and predictions cannot
be overwritten with different fitting inputs. Old BPB results and question
answers are not changed.

Local checks: `python3 -m unittest discover -s a3/A3part5 -p 'test_*.py'`.
The fit also uses NumPy, already supplied by the pinned nanochat environment.
References: [compute/data scaling](https://arxiv.org/abs/2203.15556) and
[small-scale extrapolation](https://arxiv.org/abs/2202.06387).

## Completed scaling results

All six calibration runs and the existing nano checkpoint evaluation completed.
The following BPB scores are measured, not predictions. Small-model rows are
means over seeds 42 and 43; nano uses seed 42.

| Depth | Scaling parameters | Training tokens per run | Set A BPB | Set B BPB |
|---|---:|---:|---:|---:|
| 4 | 11,536,440 | 92,274,688 | 1.18850 | 1.17554 |
| 6 | 23,206,782 | 185,597,952 | 1.04985 | 1.03555 |
| 8 | 41,959,648 | 335,544,320 | 0.96938 | 0.95440 |
| 24 | 730,253,280 | 5,841,616,896 | 0.71742 | 0.69920 |

Fitting all six individual set-A observations gives
`BPB = exp(2.7443739757599) * N^(-0.15840833477653768)`.
Nano's predicted BPB is **0.61351**, versus **0.71742** measured on set A:
actual minus predicted is **+0.10391**, or **16.94% of the prediction**.
Applying the same curve to set B gives actual **0.69920**, 13.97% above prediction.
Separate seed fits predict 0.60750 and 0.61957; these are sensitivity results,
not a confidence interval. The curve overestimates the improvement at nano size.

The prediction was frozen before the matched nano evaluation, but this remains
a retrospective study: nano was already trained and its general performance
was known. The three-size fit extrapolates 17.4x beyond its largest scaling
parameter count, without an intermediate holdout or fitted loss floor.
The standard-validation nano BPB **0.72369** on 41,943,040 positions remains
unchanged; set-A/B scores use different fixed subsets. Source measurements and
full-precision fits are in `scaling-study-v1/summary.json` on the Modal volume.
