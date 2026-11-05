# Picochat pilot results

Short fixed-token screening; two seeds are not a significance test.

| Variant | Status | BPB mean | BPB seeds | Delta vs baseline | Tokens/s | Peak GiB | Params |
|---|---|---:|---|---:|---:|---:|---:|
| baseline | complete | 1.36165 | 42: 1.36560, 43: 1.35770 | +0.00000 | 233,814 | 3.16 | 73,531,646 |
| cla2 | complete | 1.39075 | 42: 1.39346, 43: 1.38805 | +0.02910 | 238,817 | 3.15 | 72,646,910 |
| fox | complete | 1.34795 | 42: 1.35234, 43: 1.34357 | -0.01370 | 209,240 | 3.64 | 73,545,506 |
| dcmha | complete | 1.35867 | 42: 1.37780, 43: 1.33954 | -0.00298 | 62,534 | 5.44 | 73,821,950 |
| diff_v2 | complete | 1.37217 | 42: 1.36418, 43: 1.38016 | +0.01052 | 179,554 | 3.25 | 74,430,206 |
| fope | complete | 1.36908 | 42: 1.37600, 43: 1.36216 | +0.00743 | 211,061 | 3.16 | 73,531,646 |

Recorded seed-function elapsed time: 1,596.7 seconds.
GPU-only estimate for this recorded time: $1.75.
This is NOT the final invoice: container startup, smoke tests, CPU/RAM,
and any failed calls add cost. Credits are not deducted in this estimate.

## Limitations

- CLA2 is adjacent sharing with per-layer value embeddings, not FusedKV-Lite.
- FoX/DCMHA use dense reference attention; speed is implementation-specific.
- DIFF V2 has more parameters; this is not parameter- or FLOP-matched.
- FoPE is a head-shared Eq.13 adaptation; no context extrapolation was tested.
- No cached decoding, CORE, SFT, or final-model claims.
- Two seeds, 16.384M tokens each, one short training schedule; do not extrapolate a winner.
- GPU timings include implementation differences and hardware/run variability.
- CUDA deterministic algorithms were not forced; exact reruns can differ slightly.

## W&B runs

- [baseline seed 42](https://wandb.ai/jinhaozhang123-university-of-toronto/a3-picochat-pilot/runs/un4p0zdd)
- [baseline seed 43](https://wandb.ai/jinhaozhang123-university-of-toronto/a3-picochat-pilot/runs/rffq0o0i)
- [cla2 seed 42](https://wandb.ai/jinhaozhang123-university-of-toronto/a3-picochat-pilot/runs/n45utr68)
- [cla2 seed 43](https://wandb.ai/jinhaozhang123-university-of-toronto/a3-picochat-pilot/runs/d7gorc15)
- [dcmha seed 42](https://wandb.ai/jinhaozhang123-university-of-toronto/a3-picochat-pilot/runs/hj1xf29j)
- [dcmha seed 43](https://wandb.ai/jinhaozhang123-university-of-toronto/a3-picochat-pilot/runs/1khk9dh5)
- [diff_v2 seed 42](https://wandb.ai/jinhaozhang123-university-of-toronto/a3-picochat-pilot/runs/ia22f30n)
- [diff_v2 seed 43](https://wandb.ai/jinhaozhang123-university-of-toronto/a3-picochat-pilot/runs/b6e8w9b7)
- [fope seed 42](https://wandb.ai/jinhaozhang123-university-of-toronto/a3-picochat-pilot/runs/hm5t5sc4)
- [fope seed 43](https://wandb.ai/jinhaozhang123-university-of-toronto/a3-picochat-pilot/runs/aglxfghb)
- [fox seed 42](https://wandb.ai/jinhaozhang123-university-of-toronto/a3-picochat-pilot/runs/fp0w9ft9)
- [fox seed 43](https://wandb.ai/jinhaozhang123-university-of-toronto/a3-picochat-pilot/runs/oai19y3t)

## Screening conclusion

Lowest mean validation BPB: **fox** (1.34795).
This nominates a follow-up experiment, not a universal architecture winner.
Retest promising changes at a longer training horizon and with more seeds.

- cla2: paired BPB deltas +0.02786, +0.03035.
- fox: paired BPB deltas -0.01326, -0.01414.
- dcmha: paired BPB deltas +0.01220, -0.01817.
- diff_v2: paired BPB deltas -0.00142, +0.02246.
- fope: paired BPB deltas +0.01040, +0.00446.
