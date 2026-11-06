# Part 4: native baseline context extension

Original Part 2 baseline weights; upstream GPT with RoPE. Two seeds; same original tokenizer and four ClimbMix training shards. Each continuation adds 33.554432M tokens to the initial 16.384M.

| Checkpoint | Eval 512 BPB | Eval 2048 BPB |
|---|---:|---:|
| Original 512 | 1.36118 | 1.43134 |
| Continued 512 | 1.24543 | 1.35956 |
| Continued 2048 | 1.25232 | 1.19535 |

Both evaluation windows score identical final 256-token suffixes of 48 held-out documents. Optimizer and data order restart in both arms; RoPE retained without interpolation. SSSL windows increase with configured context. Two seeds do not establish generality.

## Paired uncertainty

```json
{
  "means": {
    "0": {
      "512": 1.3611829806740094,
      "2048": 1.431337071583438
    },
    "512": {
      "512": 1.24543475690809,
      "2048": 1.359563374532899
    },
    "2048": {
      "512": 1.2523232533525093,
      "2048": 1.1953468963765967
    }
  },
  "paired_document_stats": {
    "42": {
      "mean_document_delta_bpb": -0.15712864441932745,
      "bootstrap_95_percent_interval": [
        -0.18487255746768783,
        -0.13372490207253418
      ],
      "limitations": "Document bootstrap conditional on this checkpoint; not training-seed uncertainty."
    },
    "43": {
      "mean_document_delta_bpb": -0.18388602671459608,
      "bootstrap_95_percent_interval": [
        -0.21297474787387408,
        -0.15963556866913764
      ],
      "limitations": "Document bootstrap conditional on this checkpoint; not training-seed uncertainty."
    }
  },
  "aggregate_deltas": {
    "42": -0.151105798658147,
    "43": -0.17732715765445795
  },
  "continuation_runtime_cost_estimate": 0.6996506116776313
}
```

## Verified W&B runs

- [context-baseline-512-seed42](https://wandb.ai/jinhaozhang123-university-of-toronto/a3-picochat-pilot/runs/93n85etn): 105 loss points, 2 validation points.
- [context-baseline-2048-seed42](https://wandb.ai/jinhaozhang123-university-of-toronto/a3-picochat-pilot/runs/y7rb15fr): 105 loss points, 2 validation points.
- [context-baseline-512-seed43](https://wandb.ai/jinhaozhang123-university-of-toronto/a3-picochat-pilot/runs/steh4co8): 105 loss points, 2 validation points.
- [context-baseline-2048-seed43](https://wandb.ai/jinhaozhang123-university-of-toronto/a3-picochat-pilot/runs/k28db0y3): 105 loss points, 2 validation points.

Runtime estimate excludes original-checkpoint evaluation, cross-evaluation, startup, and other overhead; not an invoice.
