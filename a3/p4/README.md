# Part 4: short-to-long FoX training on Modal

Part 4 code, methods, results and report figures live here.

Seed 42 completed both training stages. See [the write-up](REPORT.md) and
[small research deliverables](results/fox-seed42-v1/) for results. Full models
and data remain on Modal; they are not part of a GitHub push.

The supplemental [standard validation](standard_eval/README.md) uses nanochat's
BPB function and best-fit validation loader to rescore the existing
FoX checkpoints. Its small evidence is under
`results/standard-validation-seed42-v1/`. The original matched-target context
task remains a separate evaluation. Standard validation scores should not be
mixed with its suffix-only BPB values. A baseline control completed before the
team narrowed Part 4's scope remains saved as ancillary evidence, but is not
used in the Part 4 report.

## Account and tools

- Local environment: `a3/p4/.venv` (Python 3.11, Modal 1.6.3, CPU Torch 2.9.1).
- Upstream source: `vendor/nanochat`, pinned to `92d63d4e8bb4df75c3b71618f31ddde2378b2bcd`.
- Modal profile: `a3-p4`; authenticated workspace: `jerry-zeng-uoft`.
- Modal Volume: `a3-p4-context` (v2); App: `a3-p4-context-extension`.
- Modal Secret: `a3-p4-wandb`, containing `WANDB_API_KEY`. Never put the value in this repository or chat.
- W&B project: `a3-p4-context-extension`; each stage records its actual run URL.
- Gross Part 4 ceiling: US$100 including preparation, tests, training and evaluation; credits reduce net payment, not recorded resource consumption.

Use Modal's workspace usage limit as an account guardrail. `local/budget-reservations.jsonl`
also records conservative launch reservations, not invoices. No account upgrades,
scheduled jobs, automatic sweep or automatic paid retries are used.

For a fresh checkout, install a lightweight local controller:

```sh
python3 -m venv a3/p4/.venv
a3/p4/.venv/bin/python -m pip install -r a3/p4/requirements-client.txt
git clone https://github.com/karpathy/nanochat.git a3/p4/vendor/nanochat
git -C a3/p4/vendor/nanochat checkout --detach 92d63d4e8bb4df75c3b71618f31ddde2378b2bcd
a3/p4/.venv/bin/modal setup --profile a3-p4
```

Create the Volume/Secret named above in your own workspace before running jobs.
GPU training dependencies install in the remote image. The larger local Python
environment used during development also supports CPU correctness tests;
`requirements-local.lock.txt` records that optional environment. Neither local
environment is committed to GitHub.

## Running stages

Run these from the repository root. Use the Python launcher so guards execute
**before** Modal initializes cloud functions or builds images:

```sh
a3/p4/.venv/bin/python a3/p4/run.py check
a3/p4/.venv/bin/python a3/p4/run.py prepare
a3/p4/.venv/bin/python a3/p4/run.py rehearsal --experiment-id rehearsal-20261104-01
a3/p4/.venv/bin/python a3/p4/run.py short --experiment-id fox-seed42-v1
a3/p4/.venv/bin/python a3/p4/run.py long --experiment-id fox-seed42-v1
```

`prepare` uses CPU compute and downloads four approximately 92 MB training shards
and one held-out shard from immutable ClimbMix revision
`915333b4f8b8684f39aeaafea600fea6f43fb703`. It fits a 32,768-token BPE tokenizer on
100M training characters, prepares a 40M-token subset and freezes 64 long held-out
documents. Exact duplicate documents used for tokenizer fitting/training are
excluded from evaluation. This is exact-duplicate filtering, not near-duplicate
detection. Full prepared data remains on the persistent Modal Volume. Small
tokenizer/evaluation files and hash manifests are exported locally; the training
stream and bulky document indexes stay on Modal by default.

The loader uses a deterministic continuous stream with BOS document separators,
rather than Part 2's best-fit document cropper. This makes the cursor exactly
restorable when changing context; the run is not a numerical reproduction of the
Part 2 pilot.

`rehearsal` runs a tiny model through actual 512/2048 GPU stages, fresh-worker
checkpoint loading, evaluation and verified local exports. It is a persistence
and correctness test, not a model-quality result. Full training is blocked unless
the current implementation passes this gate and W&B logging works.

The initial main configuration is a six-layer, width-384 FoX model (no RoPE),
500 steps at 512 then 500 steps at 2048. Each step trains 32,768 target tokens:
16.384M per stage, 32.768M total. Both stages share one optimizer/LR schedule and
one data cursor. Checkpoint 2 records checkpoint 1's hash and verifies its learned
weights before the first continuation update. Start with seed 42; additional seeds
require explicit configurations/run IDs and remaining-budget checks.

Inspect/save any config adjustment **before** starting a fresh chain using
`--config path/to/config.json`. Configuration must remain identical across stages.
The memory/throughput rehearsal determines whether adjustments are needed.

## Evidence and dashboards

- Modal: actual job logs, GPU utilization/memory/power/temperature and available
  usage/billing views. Export screenshots/available reports promptly; Starter log
  retention is limited. Modal does not know our model loss or custom evaluation.
- W&B: training loss, validation BPB, tokens, LR, throughput, telemetry and cost
  estimates explicitly logged by the trainer. URLs are saved in run artifacts.
- Volume: `/prepared/climbmix-v1` and `/runs/<experiment-id>`.
- Small local preparation files: `local/prepared/climbmix-v1` (no training stream).
- Local run evidence: `outputs/<experiment-id>` (no model weights by default).
- GitHub-ready small research summaries and figures: `results/<experiment-id>`.
- `local/`, `outputs/`, `vendor/`, `.venv/` and binary data/checkpoints are ignored
  by Git and do not need to be pushed to GitHub.

Each run retains JSONL metrics, fixed evaluation inputs/targets, per-target losses,
config/source/data identities, hardware metadata, checkpoint ancestry, immutable
checkpoint 1/2, and the latest two complete recovery checkpoints. Pruning older
recovery weights is audited in `checkpoint-history.jsonl`; final checkpoints are
never pruned. All other metrics/evaluation evidence remains retained.

`report/` contains CSV tables, JSON summaries, Markdown and figures. The custom
document-continuation evaluation compares checkpoint 1/2 at input windows 512/2048,
scoring identical suffix targets. Both checkpoints use the same inference window
policy at each evaluation length. Checkpoint 1 at 2048 is out-of-training-range.
Checkpoint 2 has additional training tokens, so before/after changes alone do not
isolate context length as their cause.

Every stage is followed by an explicit Volume commit, verification of full
checkpoints in a new CPU worker, and a local hash audit of downloaded small
evidence. `LOCAL_EXPORT_MANIFEST.json` records the export scope and large files
left on Modal. Periodic commits/checkpoints bound
loss after an interruption. A storage failure stops further training. W&B errors
retain replayable files; local JSONL is also an independent evidence copy.

Export/audit without starting a GPU application:

```sh
a3/p4/.venv/bin/python a3/p4/run.py export --experiment-id fox-seed42-v1
a3/p4/.venv/bin/python a3/p4/run.py audit --experiment-id fox-seed42-v1
```

Optional full downloads, still ignored by Git:

```sh
a3/p4/.venv/bin/python a3/p4/run.py export --experiment-id fox-seed42-v1 --full-export
a3/p4/.venv/bin/python a3/p4/run.py export-prepared --full-export
```

If a stage stops early, inspect its status and last durable checkpoint, then
resume that named checkpoint explicitly using `--resume <checkpoint-name>`.
Do not silently restart or change its configuration. Failed launches retain their
budget reservations until actual account usage can be reviewed.

## Offline verification

```sh
PYTHONPATH=a3:a3/p4/vendor/nanochat a3/p4/.venv/bin/python -B -m unittest discover -s a3/p4/tests -v
```

The standard suite requires no cloud access. CPU trajectory tests use AdamW as
explicit correctness fixtures; research GPU training uses upstream Muon/AdamW and
BF16. The optional real offline W&B test is opt-in and uses only a local service.
See `PLAN.md` for the literature rationale and persistence contract.
