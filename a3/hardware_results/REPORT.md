# Part 3 hardware results

Two seeds per configuration; throughput/MFU and times averaged across seeds.

| GPU | Variant | Train s | First 5 steps s | Elapsed s | Tokens/s | MFU % | GPU $/run (elapsed) | BPB |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| A10 | baseline | 225.3 | 14.0 | 238.8 | 76,761 | 16.04 | 0.073 | 1.34259 |
| A10 | fox | 308.6 | 5.0 | 325.2 | 53,412 | 11.17 | 0.100 | 1.33859 |
| A100-80GB | baseline | 112.5 | 12.2 | 122.3 | 161,776 | 7.58 | 0.085 | 1.33881 |
| A100-80GB | fox | 122.4 | 3.0 | 132.0 | 135,887 | 6.37 | 0.092 | 1.33014 |
| H100 | baseline | 74.3 | 7.9 | 82.2 | 245,947 | 3.64 | 0.090 | 1.36067 |
| H100 | fox | 72.9 | 1.9 | 79.2 | 228,508 | 3.38 | 0.087 | 1.34926 |

## Provenance and per-seed measurements

- [A10 baseline seed 42](https://wandb.ai/jinhaozhang123-university-of-toronto/a3-picochat-pilot/runs/wzcz67eu): BPB 1.334091, 76,706 tokens/s, estimated MFU 16.03%; GPU NVIDIA A10G; backend `FA3:/vol/hf/hub/models--kernels-community--flash-attn3/snapshots/e29f138fc363b396e5d2706c8a5f6fa7d36f41e0/build/torch29-cxx11-cu128-x86_64-linux/flash_attn_interface.py`; FLOPs/token 146,277,792.
- [A10 fox seed 42](https://wandb.ai/jinhaozhang123-university-of-toronto/a3-picochat-pilot/runs/ntyi74mq): BPB 1.325484, 53,393 tokens/s, estimated MFU 11.16%; GPU NVIDIA A10G; backend `dense-reference-fp32-QK-bf16-PV`; FLOPs/token 146,360,952.
- [A10 fox seed 43](https://wandb.ai/jinhaozhang123-university-of-toronto/a3-picochat-pilot/runs/cp2i516l): BPB 1.351696, 53,431 tokens/s, estimated MFU 11.17%; GPU NVIDIA A10G; backend `dense-reference-fp32-QK-bf16-PV`; FLOPs/token 146,360,952.
- [A10 baseline seed 43](https://wandb.ai/jinhaozhang123-university-of-toronto/a3-picochat-pilot/runs/ipoc49x3): BPB 1.351085, 76,817 tokens/s, estimated MFU 16.05%; GPU NVIDIA A10G; backend `FA3:/vol/hf/hub/models--kernels-community--flash-attn3/snapshots/e29f138fc363b396e5d2706c8a5f6fa7d36f41e0/build/torch29-cxx11-cu128-x86_64-linux/flash_attn_interface.py`; FLOPs/token 146,277,792.
- [A100-80GB baseline seed 42](https://wandb.ai/jinhaozhang123-university-of-toronto/a3-picochat-pilot/runs/yftlg2uj): BPB 1.343229, 160,092 tokens/s, estimated MFU 7.51%; GPU NVIDIA A100-SXM4-80GB; backend `FA3:/vol/hf/hub/models--kernels-community--flash-attn3/snapshots/e29f138fc363b396e5d2706c8a5f6fa7d36f41e0/build/torch29-cxx11-cu128-x86_64-linux/flash_attn_interface.py`; FLOPs/token 146,277,792.
- [A100-80GB fox seed 42](https://wandb.ai/jinhaozhang123-university-of-toronto/a3-picochat-pilot/runs/7lopbmk9): BPB 1.337595, 134,200 tokens/s, estimated MFU 6.30%; GPU NVIDIA A100-SXM4-80GB; backend `dense-reference-fp32-QK-bf16-PV`; FLOPs/token 146,360,952.
- [A100-80GB fox seed 43](https://wandb.ai/jinhaozhang123-university-of-toronto/a3-picochat-pilot/runs/9y27i2p9): BPB 1.322694, 137,574 tokens/s, estimated MFU 6.45%; GPU NVIDIA A100-SXM4-80GB; backend `dense-reference-fp32-QK-bf16-PV`; FLOPs/token 146,360,952.
- [A100-80GB baseline seed 43](https://wandb.ai/jinhaozhang123-university-of-toronto/a3-picochat-pilot/runs/f5r0o41j): BPB 1.334390, 163,460 tokens/s, estimated MFU 7.66%; GPU NVIDIA A100-SXM4-80GB; backend `FA3:/vol/hf/hub/models--kernels-community--flash-attn3/snapshots/e29f138fc363b396e5d2706c8a5f6fa7d36f41e0/build/torch29-cxx11-cu128-x86_64-linux/flash_attn_interface.py`; FLOPs/token 146,277,792.
- [H100 baseline seed 42](https://wandb.ai/jinhaozhang123-university-of-toronto/a3-picochat-pilot/runs/a2m887pi): BPB 1.364781, 226,161 tokens/s, estimated MFU 3.35%; GPU NVIDIA H100 80GB HBM3; backend `FA3:/vol/hf/hub/models--varunneal--flash-attention-3/snapshots/de87b9b5af06dd9984df595bef90b2eba44b181a/build/torch29-cxx11-cu128-x86_64-linux/flash_attention_3/flash_attn_interface.py`; FLOPs/token 146,277,792.
- [H100 fox seed 42](https://wandb.ai/jinhaozhang123-university-of-toronto/a3-picochat-pilot/runs/3i7zorb3): BPB 1.352671, 226,193 tokens/s, estimated MFU 3.35%; GPU NVIDIA H100 80GB HBM3; backend `dense-reference-fp32-QK-bf16-PV`; FLOPs/token 146,360,952.
- [H100 fox seed 43](https://wandb.ai/jinhaozhang123-university-of-toronto/a3-picochat-pilot/runs/zs8mhkj8): BPB 1.345853, 230,823 tokens/s, estimated MFU 3.42%; GPU NVIDIA H100 80GB HBM3; backend `dense-reference-fp32-QK-bf16-PV`; FLOPs/token 146,360,952.
- [H100 baseline seed 43](https://wandb.ai/jinhaozhang123-university-of-toronto/a3-picochat-pilot/runs/syd9k3ss): BPB 1.356566, 265,733 tokens/s, estimated MFU 3.93%; GPU NVIDIA H100 80GB HBM3; backend `FA3:/vol/hf/hub/models--varunneal--flash-attention-3/snapshots/de87b9b5af06dd9984df595bef90b2eba44b181a/build/torch29-cxx11-cu128-x86_64-linux/flash_attention_3/flash_attn_interface.py`; FLOPs/token 146,277,792.

Worker-timer GPU estimate: $1.12; requested CPU/RAM estimate: $0.18.
Excludes startup before worker timer, image build/storage and any other usage; invoice unverified.

MFU is approximate useful-model FLOPs normalized to actual-device dense BF16 peaks (A10G 70/A100 312/H100 989 TFLOP/s), not GPU utilization.
A10G report MFU is corrected from the generic A10 125 TFLOP/s used in raw live logs. Raw JSON is preserved; reported_results.json records both values.
A10G source: https://d1.awsstatic.com/product-marketing/ec2/NVIDIA_AWS_A10G_DataSheet_FINAL_02_17_2022.pdf
FoX reference attention contains FP32 work and redundant masked computation; backend differences confound pure hardware attribution.
Steady rates exclude first five steps; training time includes them and data loading, but excludes evaluation.
Two seeds/runs per configuration are not a robust timing study or statistical significance test.
