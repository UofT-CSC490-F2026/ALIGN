# Part 5: FoX nanochat on eight H100s

Uses nanochat revision `92d63d4e8bb4df75c3b71618f31ddde2378b2bcd` and the
official speedrun capacity: 24 layers, width 1536, 12 query/KV heads, vocabulary
32768, context 2048. FoX attention is the Part 2 adaptation (forget gates,
no RoPE), not a claim of reproducing the paper's complete model or kernel.
All source changes are in this directory.

## Native pipeline

`run_train.py` invokes `scripts.base_train` directly. `fox.py` supplies the
FoX attention adapter and block activation checkpointing; native GPT forward,
optimizer, distributed gradient synchronization, schedules, loader, tokenizer,
BPB and checkpoint format are retained. The official FP8 filter converts large
linear layers; small forget gates remain unconverted. Per-GPU microbatch 2
and gradient accumulation preserve the upstream automatically selected global
token batch. Checkpointing recomputes activations to save memory. The dense
Part 2 FoX attention is slower than the official Flash Attention speedrun;
its advertised runtime does not apply here.

Data and tokenizer are prepared remotely, never on the laptop. Download eight
ClimbMix shards, run upstream `scripts.tok_train` with its default 2B-character
budget and 32768 vocabulary, run `scripts.tok_eval`, then download 170 training
shards plus the held-out validation shard. Native packing crops some text.
The stopping rule is data:scaling-parameters ratio 8, not exhaustion of all
shards or validation early stopping. The exact counts and update horizon are
saved in `configuration.json` before training. Report actual consumed tokens,
not 400B or all 170 shards.

## Preparation and evaluation

1. CPU forward and gradient comparison against the Part 2 FoX implementation.
2. Reproduce one Part 2 pico run: seed 42, 6 layers, width 384, 6 heads, context
   512, 500 updates of 32768 tokens, unchanged Part 2 optimizer and schedule.
   This reference uses the shared Part 5 tokenizer because the original Part 2
   checkpoint/tokenizer is unavailable in this workspace. It is explicitly a
   reproduction, not the original artifact. Save weights and question outputs.
3. Record the scaling extrapolation using the measured pico validation BPB.
   The formula and limitations are in `scaling_prediction.json`. This is a
   literature-exponent, single-anchor forecast, not a newly fitted scaling law.
4. Start the main run directly after preparation. The separate rehearsal and
   runtime-estimation gate have been removed from the launch sequence at the
   user's request. Main training starts fresh; rehearsal weights are not used.
5. Train the full native horizon, saving checkpoints every 500 updates and at
   completion. Logs and checkpoint notifications trigger durable volume commits.
6. Release the eight GPUs; use one H100 for native BPB and fixed questions.

Both models are evaluated at 512 and 2048 with batch 8 and 262144 nominal
positions per cell. The scaling comparison uses the common 512 context, same
tokenizer and deterministic validation stream. Pico at 2048 is extrapolation.
Cross-context cells pack different text. The small and large models differ in
data budget and training procedure as well as capacity; report these confounds.
There are no full CORE runs. A fixed pool of 20 questions is completed greedily
for 48 tokens at a common context of 512, without a KV cache. Save every answer
for manual grading; select up to ten nano-only successes with the full pool
disclosed. Do not claim ten successes or emergent abilities before grading.
No SFT is included: both are base models, not instruction-tuned chat models.

## Modal execution and budget

App and volume: `a3-part5-fox`. Profile: `a3-p4`. Existing secret:
`a3-p4-wandb`, containing `WANDB_API_KEY`. No credentials belong in this repo.

```sh
MODAL_PROFILE=a3-p4 a3/A3part4/.venv/bin/modal run --detach a3/A3part5/modal_app.py --run-id fox-d24-seed42-v1
```

The launcher submits the pipeline with `spawn()` and exits; the detached
remote pipeline continues without a local waiting process. Run IDs cannot
overwrite existing runs. No automatic retries, deployments or scaling sweep.
Hard timeouts: eight-GPU rehearsal 20 min; main eight-GPU run 10 h; single-GPU
pico and final evaluation 30 min each; CPU orchestrator 12 h. At checked Modal
rates ($0.001097/H100/s, $0.0000131/CPU-core/s, $0.00000222/GiB/s), requested
compute ceilings total about $370 before preparation/build/storage and actual
CPU/RAM above requests. This reserves substantial space under the $500 cap.
The timeouts bound this launch; they do not authorize another run on failure.
After the first app was cancelled during compilation, an explicitly authorized
recovery reused the saved preparation via `--resume-prepared`. That mode refuses
to overwrite a main training run. Both rehearsal attempts were subsequently
stopped; the current launch goes directly to main training and evaluation.
MFU from upstream is an estimate: it does not fully account for FoX gate work
or activation recomputation and must not be presented as exact hardware MFU.

Persistent outputs under `/vol/runs/<run-id>/`: configuration, tokenizer hashes,
scaling forecast, native console log, `metrics.jsonl`, W&B URL, all-rank model/
optimizer checkpoints, `bpb.json`, `evaluation.json`, `answers.json`, `status.json`,
and `pico/` and `smoke/` records. Cache, datasets and tokenizer stay in `/vol/cache`.
Only small JSON/log files should be downloaded into ignored `local/`.

References: [native speedrun](https://github.com/karpathy/nanochat/blob/92d63d4e8bb4df75c3b71618f31ddde2378b2bcd/runs/speedrun.sh),
[parameter-scaling hypothesis](https://arxiv.org/html/2001.08361v1#S1.E1),
[Modal pricing](https://modal.com/pricing).
