# Part 4: baseline context extension

Reproduce the Part 2 baseline at context 512, then continue its saved model and
optimizer at context 2048. Implementation uses the existing `pico.variants.build`
baseline and the Part 2 training loop, with native nanochat data loading,
tokenization, checkpoints, and BPB evaluation.

## Configuration

| Setting | Value |
|---|---|
| nanochat revision | `92d63d4e8bb4df75c3b71618f31ddde2378b2bcd` |
| Architecture | Baseline, RoPE, SSSL attention windows |
| Capacity | 6 layers, width 384, 6 query/KV heads, vocabulary 32,768 |
| Parameters | 73,531,646 (checked before training) |
| Seed / hardware | 42 / one H100 / BF16 / eager model |
| Stage 1 | 500 updates, context 512, microbatch 8, accumulation 8 |
| Stage 2 | 500 additional updates, context 2048, microbatch 2, accumulation 8 |
| Tokens | 32,768/update; 16.384M per stage; 32.768M cumulative |
| Optimizer learning rates | embedding .3, unembedding .008, matrix .02, scalar .5 |
| Weight decay | .28 |

Each stage uses Part 2's 20-update warmup and linear warmdown beginning at 35%
of the stage. Stage 2 **restarts this schedule**, retaining weights and optimizer
state. The native stateful BOS best-fit loader resumes approximately at the next
row group; it does not preserve its unconsumed packing buffer exactly.

Data preparation runs upstream `nanochat.dataset -n 4`: four ClimbMix training
shards and the held-out validation shard. Upstream `scripts.tok_train` trains
the shared tokenizer with `--max-chars=100000000 --vocab-size=32768`.
Downloading four shards does not mean all their tokens are consumed in training.
Tokenizer and data hashes are recorded in `preparation.json`.

## Evaluation and interpretation

Native `nanochat.loss_eval.evaluate_bpb` and the native validation loader evaluate
both checkpoints at both context lengths, with 262,144 nominal validation token
positions per cell (batch 8: 64 batches at 512 or 16 at 2048). BPB also runs every
100 training updates. Lower BPB is better.

CORE is not part of this context-window experiment. The validation budget
matches Part 2; it is not the native trainer's 41,943,040-token default.

The standard loader packs different streams at different sequence lengths.
Compare checkpoints within the same evaluation context. The continued model
has also seen 16.384M more training tokens: this experiment measures the outcome
of context extension plus continued training, not an isolated causal effect of
context length. The matched-target evaluation below additionally holds the
scored text fixed across context lengths.

## Run

From the repository root:

```sh
python3 -m venv a3/A3part4/.venv
a3/A3part4/.venv/bin/pip install -r a3/A3part4/requirements.txt
MODAL_PROFILE=a3-p4 a3/A3part4/.venv/bin/modal run --detach a3/A3part4/modal_app.py --run-id baseline-context-seed42-v1
```

Requires the authorized Modal profile and Modal Secret `a3-p4-wandb` containing
`WANDB_API_KEY`. Remote CPU preparation checks model compatibility and account
access before starting the GPU. A remote pipeline controls both stages and
evaluation, so disconnecting the local detached client does not interrupt them.
New training runs cannot overwrite an existing run ID. There are no configured
application retries. Modal infrastructure preemption can still interrupt work.

One GPU invocation is capped at 3,600 seconds including evaluation. The CPU
pipeline is capped at 5,400 seconds including waiting for that invocation.
These are runtime safeguards; the account's usage limit remains separate.
If evaluation times out, the two completed training checkpoints remain saved;
use evaluation-only recovery below to finish missing BPB comparisons.

## Recover evaluation without training

Once both checkpoints exist, evaluate only missing cells:

```sh
MODAL_PROFILE=a3-p4 a3/A3part4/.venv/bin/modal run --detach a3/A3part4/modal_app.py --run-id baseline-context-seed42-v1 --evaluation-only
```

This mode does not prepare data, train models, or modify checkpoints. It validates
saved BPB rows and preserves completed cells; an interrupted cell restarts from
the beginning. Each completed cell is written atomically and committed before
logging to W&B. A fully evaluated run performs no additional model evaluation.
The existing volume must still contain its original tokenizer and checkpoints.
Do not launch overlapping jobs for the same run ID. Partial training recovery
is not implemented; this entry point requires both completed checkpoints.

The CPU baseline check uses nonzero attention/MLP output projections in disposable
models to exercise those computations. Training initialization is unchanged.

## Outputs

Modal app and volume: `a3-part4-baseline`. W&B project: `a3-part4-baseline`,
grouped by run ID. W&B records loss, validation BPB, throughput and GPU memory;
Modal's app metrics and billing pages show infrastructure usage and charges.

Volume directory `/runs/<run-id>/` contains:

- `status.json`, `preparation.json`, `training.json`: state, provenance, configuration and W&B URL.
- `training.jsonl`, `stage1.json`, `stage2.json`, `training-results.json`: curves and resource measurements.
- `checkpoints/`: native model, optimizer and metadata at steps 500 and 1000, committed before final evaluation.
- `bpb.json`, `result.json`: saved evaluation results. Historical CORE files, if any,
  are left untouched and are not consumed by this pipeline.
- W&B run files alongside the persistent JSON records.

Datasets and tokenizer stay under `/cache/` on the volume. Model and optimizer
files are not downloaded into the repository. `local/` is ignored and can hold
small result downloads or the launch log, for example:

```sh
MODAL_PROFILE=a3-p4 a3/A3part4/.venv/bin/modal volume get a3-part4-baseline runs/baseline-context-seed42-v1/bpb.json a3/A3part4/local/bpb.json
```

This folder contains no LaTeX report. Interpret and write up the results only
after confirming that training and evaluations completed.

## Matched text-continuation evaluation

`continuation_modal.py` compares identical passage endings with different
amounts of preceding text, adapting sliding-window language-model evaluation
from [Shortformer](https://aclanthology.org/2021.acl-long.427/). It is a separate
analysis, not an official benchmark score.

Select the first 256 distinct held-out documents containing at least 2048
native tokenizer tokens; retain their first 2048 tokens. Score only the final
128 tokens, with either 384 or 1920 preceding text tokens, plus BOS. Shifted
input/target tensors have length 512 or 2048 respectively. Both checkpoints
receive exactly the same targets and context condition. Earlier targets are
masked to -1 and scoring uses native `evaluate_bpb`. No generation, training,
new tokenizer, or instruction following is involved.

```sh
PYTHONPATH=a3 MODAL_PROFILE=a3-p4 a3/A3part4/.venv/bin/modal run --detach a3/A3part4/continuation_modal.py
```

Results are under `matched-continuation-v1/` in the existing run directory on
volume `a3-part4-baseline`. The job saves protocol/checkpoint/tokenizer hashes,
selected token sequences and document identifiers, per-document `scores.json`,
`summary.json`, and a separate `status.json`. Scores are atomic and committed
every 16 comparisons; a relaunch reuses completed comparisons. Avoid concurrent
jobs using the same output directory. One H100 has a 30-minute timeout.

Aggregate BPB is byte-weighted. Paired document bootstrap intervals describe
checkpoint differences and context gains on the selected long documents;
they do not capture variability across training seeds. Positive context gain
means lower target BPB with the longer window. The held-out corpus was already
used for monitoring, and this length-filtered sample is not representative of
all text. SSSL window sizes also change with context. Additional training and
the learning-rate restart remain confounders in comparisons of checkpoints.
Existing BPB outputs are unchanged.

Completed results (256 documents; the same 32,768 target positions per cell):

| Checkpoint | Target BPB, window 512 | Target BPB, window 2048 |
|---|---:|---:|
| 1: trained at 512 | 1.35955 | 1.43645 |
| 2: continued at 2048 | 1.26039 | 1.22042 |

Checkpoint 2 improves at both window lengths. The longer window worsens
checkpoint 1 by 0.07690 BPB but improves checkpoint 2 by 0.03997 BPB (3.17%).
For checkpoint 2, 2,000 paired document-bootstrap resamples give a 95% interval
of [0.03367, 0.04645] for the reduction. This supports a benefit from the longer
context setting on these continuations, without isolating context-extension
training from the additional updates or learning-rate restart.

Alignment checks: `python3 -m unittest discover -s a3/A3part4 -p 'test_*.py'`.
