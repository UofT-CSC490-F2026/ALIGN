# Part 4 methods

This is the declared protocol. Completed measurements and interpretation are in
`REPORT.md`; machine-readable evidence is under `results/fox-seed42-v1/`.

## Rationale

The assignment asks for short-context training followed by continuation at 2048.
We use 512 initially, matching the team's inexpensive picochat setting and making
the length increase fourfold. Shortformer studied short-first, longer-second
training and reported efficiency and language-modeling benefits in its setting
([Press, Smith and Lewis, 2021](https://aclanthology.org/2021.acl-long.427/)).
Sequence Length Warmup connects long sequences early in GPT training with
instability and evaluates a length curriculum
([Li, Zhang and He, 2022](https://arxiv.org/abs/2108.06084)). These papers motivate
testing a short-to-long curriculum; they do not establish that this particular
FoX adaptation or the 512/2048 split will improve quality.

## Model, data and optimization

Use nanochat commit `92d63d4e8bb4df75c3b71618f31ddde2378b2bcd` and the
Part Two FoX adaptation: forget gates without RoPE, six layers, width 384,
six heads and a 32,768-token vocabulary. Retain the July baseline's residual/value
pathways and SSSL attention-window schedule. Attention uses the existing dense
reference implementation rather than the FoX authors' optimized kernel.

ClimbMix is pinned to dataset revision
`915333b4f8b8684f39aeaafea600fea6f43fb703`. Prepare training from shards 0--3 and
reserve shard 6542 for evaluation. Fit a fresh BPE tokenizer on 100M training
characters and construct a 40M-token BOS-separated sequential stream. Record
document identities, full-text hashes and byte hashes of generated files.
Exclude exact duplicates used during either tokenizer fitting or training from
the evaluation cohort; this does not guarantee absence of near duplicates.

The sequential loader differs from Part 2's best-fit cropper. Its advantage here
is an exactly restorable token cursor across changed batch/context shapes.
Accordingly, Part 4 does not numerically reproduce the Part 2 pilot, and its BPB
values must not be presented as a direct matched comparison with that pilot.

| Setting | Short stage | Long stage |
|---|---:|---:|
| Input length | 512 | 2048 |
| Global optimizer steps | 1--500 | 501--1000 |
| Tokens per optimizer step | 32,768 | 32,768 |
| Tokens in stage | 16,384,000 | 16,384,000 |
| Cumulative tokens at checkpoint | 16,384,000 | 32,768,000 |
| Microbatch | 8 | 2 |
| Gradient accumulation | 8 | 8 |

The optimizer is upstream Muon/AdamW with the saved configuration. Use one global
schedule: 20-step warmup, constant multiplier through step 700, then linear
warmdown to 0.05 at the last update. Do not restart the scheduler at the context
boundary. Save model, optimizer, global progress, RNG and exact stream cursor.
Before the first long-stage update, verify that the loaded learned tensors match
checkpoint 1 and record its SHA-256 as the parent of checkpoint 2.

Use a single H100, BF16 and bounded runtime. Reducing the microbatch keeps trained
tokens per optimizer update fixed. At fixed microbatch, dense attention matrices
would grow sixteenfold from 512 to 2048; with the proposed fourfold microbatch
reduction, their element count grows fourfold. Actual peak memory and throughput
must be measured rather than inferred from this component alone.

## Custom document-continuation evaluation

Freeze 64 eligible held-out documents before research training. Each example
contains 2049 token IDs. At each evaluation length L, the model receives the last
L input tokens and predicts the shifted targets. Score only the final 128 target
tokens; their IDs are identical between the 512- and 2048-token conditions.
Here L includes preceding target history, rather than denoting L tokens strictly
before the first scored target.

Evaluate both checkpoints at both lengths. At a given length, recompute the same
explicit upstream SSSL inference-window policy for both checkpoints. This avoids
silently comparing their different construction-time receptive-field settings.
Checkpoint 1 at 2048 is an out-of-training-range evaluation. FoX uses no RoPE, so
the intervention does not include a positional-interpolation method.

For scored non-special targets, calculate BPB as total negative log-likelihood
in nats divided by `ln(2) * target_UTF8_bytes`. Aggregate by total bytes rather
than averaging document BPBs without weighting. Retain document IDs, token IDs,
per-target losses, byte counts, window policies and hashes for every cell.

The principal comparisons are:

- 512 versus 2048 available input within checkpoint 1: context extrapolation.
- 512 versus 2048 input within checkpoint 2: use of additional available context
  after long-stage training.
- Checkpoint 1 versus checkpoint 2 at fixed evaluation length: before/after
  changes in continuation quality and short-context retention.

Checkpoint 2 has additional training tokens. Without an equal-token continuation
control at 512, improvements cannot be attributed exclusively to increased
training context. The cohort also selects sufficiently long documents and does
not measure instruction following, chat quality or all language-model abilities.

## Evidence and cost reporting

Periodic durable checkpoints bound recovery loss; checkpoint 1/2 remain immutable.
Each stage's full checkpoints are checked from a fresh Modal worker. Small
evidence is exported locally with a checksum manifest; large weights remain on
Modal unless a full download is requested. W&B history is independently checked for training and evaluation
metrics; JSONL and CSV remain separate evidence copies. CPU and tiny GPU
rehearsals test correctness/persistence and are excluded from quality tables.

Report actual hardware, throughput, memory, timings and stage-specific W&B links.
Per-worker resource-time cost estimates exclude image builds/startup, preparation,
rehearsal, storage and final tracking finalization; these must be accounted for
separately using available account usage. A conservative launch reservation is
not an invoice. Distinguish gross resource consumption, applied credits and net
payment. Part 4 remains within the authorized US$100 ceiling.

## Supplemental standard nanochat validation

After the original context experiment, the team requested a common validation
protocol connecting Parts Two and Four. The follow-up calls the
upstream `nanochat.loss_eval.evaluate_bpb` on batches produced by the
BOS best-fit validation loader. Its nominal budget is 262,144 target positions
per cell, matching Jinhao's pilot budget. Special-token positions are excluded
from BPB by their zero byte weights. Actual non-special token and byte counts
are reported separately.

Freeze exact input/target arrays from the reserved shard at each evaluation
length (512 and 2048), using the original Part Four tokenizer. Reuse those same
arrays for both FoX checkpoints at that length. Record source-shard, tokenizer,
loader, metric, and array hashes. The standard streams at different lengths are
not target-matched to one another, so their difference is not a controlled
context-benefit estimate. Retain the original matched-target task for that purpose.

The original Jinhao tokenizer and validation-input hashes were unavailable.
Accordingly, the follow-up aligns his metric, loader procedure, context and
nominal budget, but does not claim an exact reproduction of his Part Two inputs
or scores. Different training procedures also remain explicitly disclosed.

For continuity across Parts Two and Four, compare the Part Two saved reference
and FoX configuration with the Part Four FoX config and checkpoint metadata.
Verify depth, width, query/KV head counts, vocabulary, seed, precision,
upstream commit, attention-window pattern, parameter count, and variant-source
hash. The Part Four FoX model has the same 73,545,506 parameters and exact
FoX source hash as Jinhao's Part Two run. Its 13,860 parameters above the
unmodified 73,531,646-parameter reference arise from the already selected
FoX adaptation. No capacity correction or retraining is needed.

The supplemental evaluation rescored the existing FoX checkpoints without
changing learned weights. A separate baseline control was completed before
the team clarified that Part Four should report the FoX continuation, not a
new architecture comparison. Its raw artifacts remain saved for provenance,
but its results are excluded from the Part Four quality tables and conclusions.
Implementation and bounded commands are in `standard_eval/README.md`; small
FoX evaluation evidence is in `results/standard-validation-seed42-v1/`.
Existing source and checkpoint records remain immutable.
