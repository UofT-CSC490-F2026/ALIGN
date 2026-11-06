# Picochat pilot v1

Status: complete. CPU and GPU smoke validation passed for all six configurations.
Baseline and five variants completed both seeds (12 runs, 500 steps each).
Collected results and interpretation are in pilot_results/REPORT.md and sweep.json.
Modal app listing was empty after final collection; no further training launched.
Synthetic smoke-test losses are NOT model-quality results.

CPU checks: finite forward/backward, gradient presence, causal prefix invariance,
checkpoint loading, and baseline agreement with unmodified upstream GPT.
See pilot_results/prepare.json and pilot_results/smoke.json. Pilot results are
stored separately; do not mistake smoke-test values for validation BPB.

User approved a US$40 total ceiling for this pilot on November 3, 2026.
No automatic retries or full-scale training. Single H100 per training function,
6 training functions capped at 2400 seconds each, plus a 600-second GPU smoke test and
1800-second CPU preparation. Reserve the remaining allowance for setup and fixes;
do not repeat the full sweep without checking cumulative spend.

## Common setup

- Upstream nanochat commit: 92d63d4e8bb4df75c3b71618f31ddde2378b2bcd.
- Depth 6, width 384, 6 heads of dimension 64, context 512, SSSL window schedule.
- Custom 32,768-token BPE trained on 100M characters from 4 ClimbMix shards.
- 500 steps x 32,768 tokens = 16,384,000 training tokens per completed run.
- Seeds 42 and 43; same common-parameter initialization and deterministic data order.
- Identical held-out validation shard, 262,144 validation tokens per measurement.
- Upstream Muon/AdamW parameter grouping and fixed common learning rates.
- BF16; eager model execution for all runs. Optimizer uses upstream compiled kernels.
- Each seed has a 1000-second soft deadline. Partial runs cannot be ranked against
  complete runs as matched-token comparisons.
- No SFT, CORE, generation-quality claims, or cached-decoding benchmarks in this pilot.
- Compare BPB, parameters, tokens/s, peak allocated VRAM, wall time, and estimated cost.
- W&B project: a3-picochat-pilot. Secret: wandb-secret. Never store the key in Git.

## Selected implementations

1. **CLA2**: adjacent pairs share projected K/V; each recipient retains its own
   value-embedding addition. This is a nanochat-specific CLA adaptation, **not**
   FusedKV-Lite. It removes receiver K/V projections but does not implement or
   benchmark a shared inference cache. Do not claim measured KV-cache savings.
2. **FoX**: log-sigmoid forget gate accumulated along token positions, added to
   attention logits; no RoPE. Keeps nanochat QK normalization and value embeddings.
   Dense reference attention, not the authors' optimized forgetting kernel.
   This intervention combines forgetting gates with removing RoPE; the pilot
   does not isolate the contribution of either change.
The other screened implementations have been removed from this PR.
Historical JSON records and execution notes are retained as evidence, not
as claims that the current launcher can reproduce the removed variants.

## Sources

- CLA: https://arxiv.org/abs/2405.12981
- FoX: https://arxiv.org/abs/2503.02130
- Pricing checked Nov 3: https://modal.com/pricing

## Current commands (PowerShell, workspace root)

```powershell
$env:PYTHONUTF8='1'
$env:PYTHONIOENCODING='utf-8'
a3\.venv\Scripts\modal.exe run a3\pico_modal.py --stage prepare
a3\.venv\Scripts\modal.exe run a3\pico_modal.py --stage smoke
a3\.venv\Scripts\modal.exe run a3\pico_modal.py --stage sweep
```

For CPU-only preparation while GPU billing access is blocked, set
`$env:A3_CPU_PREP_ONLY='1'`. Reset it to `'0'` before GPU smoke/sweep.

Run sweep only after both smoke stages pass. Results are persisted in the Modal
volume a3-picochat-pilot and downloaded as JSON under a3/pilot_results. Upstream
nanochat source is not edited. Checkpoints are remote and should not be committed.

The current sweep runs baseline, CLA2, and FoX serially (six seed runs).
The `variants` stage runs only CLA2 and FoX. Historical broader-screening
launches described above used an older implementation; no new runs were
performed for this cleanup.
