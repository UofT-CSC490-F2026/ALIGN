# Part 4: baseline context extension

Reproduce the Part 2 baseline at context 512, then continue its saved model and
optimizer at context 2048. Implementation uses the existing `pico.variants.build`
baseline and the Part 2 training loop, with native nanochat data loading,
tokenization, checkpoints, BPB, and CORE evaluation.

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

Native `scripts.base_eval.evaluate_core` evaluates both checkpoints on the full
CORE suite (`max_per_task=-1`) at a common evaluation context of 2048. This is
context extrapolation for the 512 checkpoint. Task scores and aggregate CORE
are saved; higher CORE is better.

The standard loader packs different streams at different sequence lengths.
Compare checkpoints within the same evaluation context. The continued model
has also seen 16.384M more training tokens: this experiment measures the outcome
of context extension plus continued training, not an isolated causal effect of
context length. No improvement is assumed before results arrive.

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
Run IDs cannot overwrite existing runs. There are no automatic retries.

One GPU invocation is capped at 3,600 seconds including evaluation. The CPU
pipeline is capped at 5,400 seconds including waiting for that invocation.
These are runtime safeguards; the account's usage limit remains separate.
If evaluation times out, the two completed training checkpoints remain saved;
check `status.json` and available results before scheduling any further work.

## Outputs

Modal app and volume: `a3-part4-baseline`. W&B project: `a3-part4-baseline`,
grouped by run ID. W&B records loss, validation BPB, throughput and GPU memory;
Modal's app metrics and billing pages show infrastructure usage and charges.

Volume directory `/runs/<run-id>/` contains:

- `status.json`, `preparation.json`, `training.json`: state, provenance, configuration and W&B URL.
- `training.jsonl`, `stage1.json`, `stage2.json`, `training-results.json`: curves and resource measurements.
- `checkpoints/`: native model, optimizer and metadata at steps 500 and 1000, committed before final evaluation.
- `bpb.json`, `core-500.json`, `core-1000.json`, `result.json`: saved evaluation results.
- W&B run files alongside the persistent JSON records.

Datasets and tokenizer stay under `/cache/` on the volume. Model and optimizer
files are not downloaded into the repository. `local/` is ignored and can hold
small result downloads or the launch log, for example:

```sh
MODAL_PROFILE=a3-p4 a3/A3part4/.venv/bin/modal volume get a3-part4-baseline runs/baseline-context-seed42-v1/bpb.json a3/A3part4/local/bpb.json
```

This folder contains no LaTeX report. Interpret and write up the results only
after confirming that training and evaluations completed.
