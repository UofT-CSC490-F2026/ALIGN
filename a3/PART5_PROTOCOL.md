# Part 5: native FoX speedrun and prospective scaling comparison

This draft contains our implementation, not Jerry's A3part5 pipeline. It targets
`a3` and depends on the merged Part 2 baseline/FoX reference. No LaTeX, Part 4
experiment, SFT, or completed question-based ability comparison is included.

## Completed experiment

- Pinned nanochat: `92d63d4e8bb4df75c3b71618f31ddde2378b2bcd`.
- FoX without RoPE; native GPT residual/value-embedding pathways retained.
- 24 layers, width 1536, 12 query/KV heads, context 2048, vocabulary 32768.
- 1,384,564,778 total parameters; 730,253,280 scaling parameters.
- Eight H100s, FP8 eligible linears, BF16 attention; FlexAttention score modification.
- Microbatch 16/GPU, accumulation 4, global batch 1,048,576 tokens.
- 5,571 updates / 5,841,616,896 tokens, data-to-scaling-parameter ratio 8.
- 170 ClimbMix train shards plus held-out shard 06542. Not one entire corpus pass.
- Native tokenizer recipe: 2B characters; the exact tokenizer digest is recorded.
- Native training loop, optimizer, schedule and distributed updates preserved by
  checked source substitutions. Save every 250 updates; no automatic retries.

`fox_speedrun.py` implements differentiable cumulative log-forget gates with
FlexAttention, plus a cache that stores cumulative gates for native generation.
`speedrun_train.py` adds durable checkpoint instrumentation and final fixed-set
evaluation without replacing the native training loop.

## Evidence and metrics

`speedrun_results/final.json` was fetched from the completed remote result file.
Final W&B: https://wandb.ai/jinhaozhang123-university-of-toronto/a3-picochat-pilot/runs/137d78el

| Metric | Result |
|---|---:|
| Standard validation BPB (41,943,040 nominal tokens) | 0.72182072 |
| Fixed calibration BPB (524,288 target positions) | 0.69250463 |
| Separate fixed test BPB (524,288 target positions) | 0.70185848 |
| Frozen predicted calibration BPB | 0.58500981 |
| Actual minus prediction | +0.10749481 |

The fixed sets are disjoint token batches from the held-out validation corpus,
not independent external datasets. Only calibration BPB is compared with the
forecast. Standard validation BPB is not interchangeable with fixed-set scores.
The native capped CORE measurement is separate; no full-suite leaderboard claim.

Depth-4/6/8 calibration runs use context 2048 and the same fixed evaluation set.
`predict.json` preserves the pretraining forecast, its timestamp, inputs, and
source/evaluation hashes. The fitted law is BPB = exp(2.875657) * N^(-0.167171)
along D approximately 8N. It is joint model/data scaling, not separately
identified exponents. Three sizes, one seed each, no loss floor, and long
extrapolation make the forecast uncertain. Actual calibration BPB was 18.37%
above prediction. No post-hoc refitting was performed.

The final run is not a matched large-scale baseline-vs-FoX experiment. It does
not establish FoX superiority, and the paired question/ability comparison
remains pending. Historical GPU checks are evidence, not fresh tests of this PR.

## Reproduction and safety

Follow Part 2 README for the local pinned nanochat checkout and Modal setup.
The launcher reuses the existing `a3-picochat-pilot` volume and `wandb-secret`.
It uploads the local nanochat checkout: verify its revision and cleanliness first.
This is a historical, workspace-specific runner, not a fresh-account installer:
`prepare` expects `/vol/july_aligned_cache/tokenizer` and its provenance file
`/vol/results/part2-july-aligned-v1/prepare.json`. Optional shard symlinks reuse
`/vol/final_cache`; missing shards are downloaded. These preparation artifacts
must already exist; their upstream preparation pipeline is outside this PR.

Historical order: prepare, kernel-check, prepare-eval, distributed-check,
h100-check, calibrate, explicitly authorized depth-8 recovery, predict, final.
The initial distributed trial delivered H200; the strict h100-check and final
request H100 explicitly. The depth-8 recovery replaced a canceled calibration;
`predict` intentionally names that completed replacement. On a fresh sweep,
configure a new namespace and the appropriate completed calibration names first.
Do not rerun a successful depth-8 merely to satisfy a historical path name.

Existing run directories block repeat training, and the prediction is frozen
once before final training. The final gate checks model/evaluation hashes and
the H100 timing estimate. The final worker timeout is 13,500 seconds; this is
not an invoice or an account-wide dollar guarantee. The original approved cap
was $250; this PR does not authorize further spending. Never launch paid stages
just to review the PR. No checkpoints, datasets, credentials, or tokenizer
binaries belong in Git. Per-rank optimizer/RNG artifacts are saved, but this
launcher does not implement automatic interrupted-final-run continuation.

The obsolete automatic waiting/launch helper was removed for this PR; stages
are explicit. No training has been rerun during PR preparation.

## Offline checks

```powershell
$env:PYTHONPATH='a3'
$env:NANOCHAT_SOURCE='C:/path/to/pinned/nanochat'
python -m unittest pico.test_speedrun_adapter pico.test_speedrun_results -v
```

These check source-adapter invariants and saved numerical evidence, not GPU
kernel execution. `test_fox_speedrun.run()` is the GPU numerical check, comparing
outputs/gradients to dense attention and checking cache prefill, decoding, and
replication. Historical results are in `kernel-check.json` and `h100-check.json`.
