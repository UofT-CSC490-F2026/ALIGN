# Picochat pilot results

Short fixed-token screening; two seeds are not a significance test.

| Variant | Status | BPB mean | BPB seeds | Delta vs baseline | Tokens/s | Peak GiB | Params |
|---|---|---:|---|---:|---:|---:|---:|
| baseline | complete | 1.36165 | 42: 1.36560, 43: 1.35770 | +0.00000 | 233,814 | 3.16 | 73,531,646 |
| cla2 | complete | 1.39075 | 42: 1.39346, 43: 1.38805 | +0.02910 | 238,817 | 3.15 | 72,646,910 |
| fox | complete | 1.34795 | 42: 1.35234, 43: 1.34357 | -0.01370 | 209,240 | 3.64 | 73,545,506 |

Recorded seed-function elapsed time: 563.0 seconds.
GPU-only estimate for this recorded time: $0.62.
This is NOT the final invoice: container startup, smoke tests, CPU/RAM,
and any failed calls add cost. Credits are not deducted in this estimate.

## Limitations

- CLA2 is adjacent sharing with per-layer value embeddings, not FusedKV-Lite.
- FoX uses dense reference attention; speed is implementation-specific.
- No cached decoding, CORE, SFT, or final-model claims.
- Two seeds, 16.384M tokens each, one short training schedule; do not extrapolate a winner.
- GPU timings include implementation differences and hardware/run variability.
- CUDA deterministic algorithms were not forced; exact reruns can differ slightly.

## W&B runs

- [baseline seed 42](https://wandb.ai/jinhaozhang123-university-of-toronto/a3-picochat-pilot/runs/un4p0zdd)
- [baseline seed 43](https://wandb.ai/jinhaozhang123-university-of-toronto/a3-picochat-pilot/runs/rffq0o0i)
- [cla2 seed 42](https://wandb.ai/jinhaozhang123-university-of-toronto/a3-picochat-pilot/runs/n45utr68)
- [cla2 seed 43](https://wandb.ai/jinhaozhang123-university-of-toronto/a3-picochat-pilot/runs/d7gorc15)
- [fox seed 42](https://wandb.ai/jinhaozhang123-university-of-toronto/a3-picochat-pilot/runs/fp0w9ft9)
- [fox seed 43](https://wandb.ai/jinhaozhang123-university-of-toronto/a3-picochat-pilot/runs/oai19y3t)

## Screening conclusion

Lowest mean validation BPB: **fox** (1.34795).
This nominates a follow-up experiment, not a universal architecture winner.
Retest promising changes at a longer training horizon and with more seeds.

- cla2: paired BPB deltas +0.02786, +0.03035.
- fox: paired BPB deltas -0.01326, -0.01414.
