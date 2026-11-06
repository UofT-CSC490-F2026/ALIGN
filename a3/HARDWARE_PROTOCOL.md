# Part 3 hardware comparison

Baseline and FoX without RoPE, seeds 42 and 43, on A10, A100-80GB, H100.
Same July baseline, model/data/tokenizer, BF16, 500 steps, 32,768 tokens/step,
batch 8, context 512, accumulation 8, learning-rate schedule, and evaluation
as Part 2. Model is eager; optimizer update kernels are compiled.

Each GPU worker runs baseline42, fox42, fox43, baseline43 to counterbalance
within-container order across seeds. Three workers run independently.
Each worker has a 2400-second hard timeout, no retries, and one container.
Each seed has a 500-second soft deadline. Partial runs cannot enter the
matched-token comparison. No automatic reruns or deployment.

GPU-rate caps: 2400*(.000306+.000694+.001097) = $5.0328.
Requested CPU/RAM for all three caps: 7200*(4*.0000131+16*.00000222)
= $0.6330. Combined runtime-rate estimate $5.67; startup/build/storage are
additional, and actual invoice is not verified. Rates checked Nov 3, 2026:
https://modal.com/pricing

## Measurement definitions

- Training seconds: sum of synchronized optimizer-step timings, including
  data retrieval, forward/backward, and optimizer; excludes validation.
- Steady tokens/s: total tokens in steps 6--500 divided by their total time.
- First-five-step time: reported separately, not assumed to capture all
  compilation or warm-up. No model updates are discarded from training.
- Seed elapsed time: from train-function entry through final evaluation;
  does not include checkpoint serialization/W&B finish. GPU cost based on
  this interval is an estimate, not total billable cost.
- Worker elapsed time: includes checks, all seed calls, saves and W&B finish,
  but excludes worker startup/import overhead before the timer.
- Backend: actual FA3 module path if selected, otherwise torch SDPA automatic
  dispatch for baseline. FoX always uses dense reference FP32 QK / BF16 PV.
  SDPA's selected internal kernel is not individually profiled.
- MFU: 100 * steady_tokens_per_second * estimated_model_flops_per_token /
  dense_BF16_peak_flops. Use nanochat's convention:
  6 * number of Linear weights + 12 * heads * head_dim * sum(effective windows).
  Includes gate projection weights but omits elementwise gates, normalization,
  optimizer FLOPs, and redundant masked dense operations. This is an
  approximate useful-model MFU, NOT measured hardware FLOPs or GPU utilization.
  FoX includes FP32 attention work, so a BF16 denominator is a normalized
  comparison convention, not its attainable mixed-precision peak.
- Dense BF16 peaks (not sparse): A10 125 TFLOP/s; A100 312; H100 SXM 989.
  Actual GPU name, memory, compute capability, CUDA/PyTorch and source hash saved.
  Correction after device inspection: Modal supplied A10G. Its NVIDIA/AWS
  datasheet specifies 70 dense BF16 TFLOP/s (140 sparse), so final report
  MFU uses 70. Original A10G live MFU logs used the generic A10 value 125;
  raw records are retained and corrected report records expose both values.
  Source: https://d1.awsstatic.com/product-marketing/ec2/NVIDIA_AWS_A10G_DataSheet_FINAL_02_17_2022.pdf

Sources:
- https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a10/pdf/a10-datasheet.pdf
- https://www.nvidia.com/en-gb/data-center/a100/
- https://github.com/NVIDIA/exemplar-performance

Results: `hardware_results/`, remote volume `/vol/results/hardware-v1/`.
W&B project `a3-picochat-pilot`, group `hardware-v1`.
Do not merge these timing runs into the original Part 2 pilot averages.

## Execution note

The initial app `ap-jUjJfcLWTPMXnYIIsG9RCg` failed during worker module
import (a local image-building helper was not present remotely), before
any training. Modal repeatedly attempted startup despite `retries=0` for
function execution. It was stopped explicitly. The corrected app is
`ap-a7zXD17ps7jEKvr08rKaVx`. Include initial startup overhead when assessing
the total bill; it is not part of successful-run cost estimates.
