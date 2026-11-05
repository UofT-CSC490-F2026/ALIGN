# Part 4 implementation plan

Status: complete for research seed 42. CPU preparation, GPU persistence rehearsal,
both 500-step training stages, fresh-worker checkpoint audits, online W&B history
checks and small evidence exports passed. Full data and weights remain on Modal.
See `REPORT.md` for measured results and `results/fox-seed42-v1/` for Git-ready
tables, figures and metadata. The protocol below records the intended design.

## Scope and access

- All new code, configuration, documentation and local outputs belong under `a3/p4/`.
- Use Modal for this assignment. The user has authorized a US$100 total Part 4 ceiling. Track gross resource usage before credits, including preparation, smoke checks, training and evaluation. Do not treat $100 as a spending target.
- Part 5's selected change is FoX. For Part 4, plan to reuse the existing FoX implementation (forget gates without RoPE), keeping the architecture fixed across both training stages.
- As checked on November 4, 2026, the local Modal SDK/CLI and default authentication configuration are absent. Install an isolated environment under this directory, then use `modal setup` for browser authentication and verify the selected workspace with `modal token info`. Never put credentials in project files or chat.
- Create a dedicated Part 4 Modal App, persistent Volume and W&B project/group. Confirm the account/workspace and available usage before running jobs. Do not reuse Jinhao's pilot result namespace.

## Proposed experiment

1. Pin upstream nanochat to `92d63d4e8bb4df75c3b71618f31ddde2378b2bcd`, matching the existing pilot. Use the Part Two model definitions with a Part Four trainer and entry point.
2. Start with the existing small configuration: 6 layers, width 384, 6 heads, head dimension 64, vocabulary 32,768, BF16 and FoX. Prepare a fresh tokenizer/data snapshot on the user's account unless exact reusable artifacts become available. Record all dataset files, revisions, splits and hashes; keep evaluation documents out of training and tokenizer fitting.
3. Train at context 512 on a declared portion of the training data; preserve the resulting checkpoint as checkpoint 1.
4. Load checkpoint 1, verify its parent hash and learned tensors, expand the context to 2048, and continue training to checkpoint 2. Preserve architecture and tokenizer. Audit context-derived windows/buffers so their behavior is explicit.
5. Choose the token budget, global learning-rate schedule, checkpoint interval and seed count in a saved configuration before the main run. Base them on a bounded memory/throughput pilot. Keep total tokens per optimizer step comparable across contexts by reducing the 2048-token microbatch and adjusting gradient accumulation. Reserve budget for evaluation, exports and failures.
6. Use one GPU training worker at a time, finite wall-time and token limits, and no automatic paid retries. Stop with a final checkpoint before the job timeout. Use unique run IDs and validate metadata before any reuse.

The short-first curriculum is motivated by Shortformer and sequence-length warmup research. Their findings justify investigating the curriculum, not predicting an improvement for this FoX adaptation.

## Evaluation fixed before main training

Implement a custom **held-out document continuation** task suitable for a pretrained model. It does not require instruction following or chat fine-tuning.

- Freeze evaluation document IDs, tokenized examples, target suffixes, hashes and sampling seed before the main training run.
- Evaluate checkpoint 1 and checkpoint 2 with both 512-token and 2048-token available context: four cells using the same scored target spans.
- At each evaluation length, apply the same explicitly recorded inference window policy to both checkpoints. Upstream derives the SSSL windows from construction context, so rebuilding one checkpoint differently must not silently change the comparison. Label checkpoint 1 evaluated at 2048 as out-of-training-range evaluation and record any inference-only context expansion.
- Score only the common target suffix, with an explicit definition of context and target lengths and special-token handling. Save negative log-likelihood, token and byte counts, BPB and paired per-document differences.
- Report the benefit of extra available context within each checkpoint, the before/after checkpoint change, and short-context retention. Add a controlled distant-context perturbation if the initial validation shows it is informative and affordable.
- Save all evaluation inputs/targets and per-example results, not just aggregate scores. Produce CSV tables and standalone figures locally.
- Checkpoint 2 has additional training tokens. State that the basic before/after result cannot isolate context length as the only cause. A token-matched 512-only continuation control is optional if time and budget allow.
- Use an explicit validation-token budget so changing context does not silently change evaluation size. Record BPB versus cumulative tokens, wall time and estimated cost.

## Required evidence and persistence

Each run must contain:

- `config.json`: model/training configuration, seeds, software versions, source hashes, upstream commit, dataset/tokenizer identities, run ID and parent-checkpoint hash.
- `metrics.jsonl`: append-only training and validation history, cumulative tokens, learning rate, step times, throughput, estimated cost and save/evaluation events.
- `hardware.json` plus sampled telemetry: requested resources, actual GPU identity, GPU memory/utilization, CPU/RAM information and runtime versions. GPU utilization is not MFU; label any FLOP-based estimate separately.
- Periodic complete checkpoints: model, optimizer, schedule/global step, cumulative tokens, RNG states and data-loader cursor/state. Do not claim exact recovery unless the loader and all required states are demonstrably restorable.
- Separate immutable checkpoint 1 and checkpoint 2, including checksums and completion manifests.
- Evaluation manifest, example-level outputs, aggregate JSON/CSV, figures and a report containing experiment/dashboard links.
- Job/App identifiers, run status, failure logs, resource-time estimates and available Modal usage/billing evidence. Estimates, gross usage, credits and final net charges must be distinguished.

Write checkpoints to temporary paths, close/validate/hash them, publish their completion manifests, and explicitly commit the Volume. Commit metrics on a bounded interval. Abort further training if durable saving fails. Keep W&B logging as a synchronized view with recoverable local logs; a dashboard alone is not the experiment archive.

Verify the full bundle and checkpoints in a fresh Modal worker after each stage.
Download and checksum-verify small metrics, evaluation outputs, metadata and
figures locally. Keep full datasets and model weights on the persistent Volume
by default, following the user's disk-space preference. Record omitted large
files and their hashes in the local export manifest. Full downloads are optional
and ignored by Git. Do not mark a stage complete until its artifact audit passes.
An abrupt failure can still lose work since the last durable checkpoint; choose
and report a bounded checkpoint interval.

## Gate before full training

1. Local checks: configuration validation, causal masking, evaluation target alignment and artifact manifest auditing.
2. CPU correctness checks on a tiny model, including save/load and resumption state.
3. A short, bounded paid GPU rehearsal at both context lengths: write metrics/checkpoint, commit, end the worker, read from a new worker, resume, evaluate and download.
4. Verify downloaded checksums, checkpoint ancestry, required metric fields, evaluation rows and visible W&B charts. Check Modal hardware charts and capture available usage evidence.
5. Measure memory and throughput, save the final experiment config/cost estimate, then run the main stages within the remaining US$100 ceiling.

## Dashboard and account details

Modal provides operational GPU charts and usage/billing views. It does not automatically produce model-quality curves, evaluation examples, or research checkpoints. Its pricing page currently lists one-day log retention on Starter, so export evidence promptly. W&B will display model metrics explicitly logged by our code; retain JSON/CSV copies regardless.

Workspace budgets cap monthly gross usage before credits, while spend limits cap out-of-pocket charges after credits. Existing workspace usage matters when setting a cap. Environment budgets and programmatic detailed billing reports may require Team/Enterprise; do not depend on these features for a personal account or upgrade the account for this task. Supplement available dashboard exports/screenshots with labeled estimates and local figures.

## Sources

- Modal authentication: https://modal.com/docs/guide/getting-started
- GPU metrics: https://modal.com/docs/guide/gpu-metrics
- Volumes, explicit commits and downloads: https://modal.com/docs/guide/volumes
- Billing/report access: https://modal.com/docs/guide/billing
- Budgets versus spend limits: https://modal.com/docs/guide/budgets
- Pricing and log retention: https://modal.com/pricing
- Shortformer: https://aclanthology.org/2021.acl-long.427/
- Sequence Length Warmup: https://arxiv.org/abs/2108.06084
- Pinned upstream context/window implementation: https://raw.githubusercontent.com/karpathy/nanochat/92d63d4e8bb4df75c3b71618f31ddde2378b2bcd/nanochat/gpt.py
