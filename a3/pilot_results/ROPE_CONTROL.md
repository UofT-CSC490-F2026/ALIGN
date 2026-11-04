# FoX with RoPE: follow-up control

Two seeds, 500 steps and 16.384M tokens each. Lower BPB is better.

| Configuration | Seed 42 BPB | Seed 43 BPB | Mean BPB | Mean tokens/s |
|---|---:|---:|---:|---:|
| baseline | 1.36560 | 1.35770 | 1.36165 | 233,814 |
| fox | 1.35234 | 1.34357 | 1.34795 | 209,240 |
| fox_rope | 1.34756 | 1.34840 | 1.34798 | 176,929 |

## Interpretation limits

- baseline: RoPE, no forgetting gates; fox: gates without RoPE; fox_rope: gates with RoPE.
- New CPU/GPU controls verify identical initial FoX parameters and identity rotations recover original FoX outputs.
- Source audit verifies old variant code is unchanged apart from adding the fox_rope branch.
- Existing baseline/FoX measurements are historical, not concurrent timing controls.
- The no-gates/no-RoPE cell remains untested: this is not the full factorial experiment.
- Throughput comparisons include dense-reference versus optimized-attention kernel differences.
- Two seeds do not establish statistical significance or larger-model performance.

## New run provenance

- [fox_rope seed 42](https://wandb.ai/jinhaozhang123-university-of-toronto/a3-picochat-pilot/runs/jvbrasvn), source SHA256 `3fe58d00b6c9d428b9c724da080b1775c5edc93dc493ed62290d2bd1f317bd2a`.
- [fox_rope seed 43](https://wandb.ai/jinhaozhang123-university-of-toronto/a3-picochat-pilot/runs/ooaddmmg), source SHA256 `3fe58d00b6c9d428b9c724da080b1775c5edc93dc493ed62290d2bd1f317bd2a`.

New successful seed-function time: 219.5s; GPU-only estimate $0.24.
This excludes tests, startup, CPU/RAM, and is not the final invoice.
