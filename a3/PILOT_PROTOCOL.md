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

## Precise adaptations

1. **CLA2**: adjacent pairs share projected K/V; each recipient retains its own
   value-embedding addition. This is a nanochat-specific CLA adaptation, **not**
   FusedKV-Lite. It removes receiver K/V projections but does not implement or
   benchmark a shared inference cache. Do not claim measured KV-cache savings.
2. **FoX**: log-sigmoid forget gate accumulated along token positions, added to
   attention logits; no RoPE. Keeps nanochat QK normalization and value embeddings.
   Dense reference attention, not the authors' optimized forgetting kernel.
   This intervention combines forgetting gates with removing RoPE; the pilot
   does not isolate the contribution of either change.
3. **DCMHA**: Appendix D rank-2 query/key compositions before and after softmax.
   Zero initialization of diagonal gates; small nonzero low-rank projections.
   Dense reference implementation. No claim of reproducing DCFormer's throughput.
4. **DIFF V2**: double query heads, shared K/V for paired heads, token/head sigmoid
   lambda, no output RMSNorm. Additional parameters are NOT offset by shrinking
   the MLP; report parameter count as a confound.
5. **FoPE**: paper Eq.13 Fourier-series adaptation, fixed head-shared independent
   cosine/sine coefficients (identity + Normal(0, .1)); frequencies below 2*pi/512
   are represented by cos=1, sin=0. This is NOT the authors' per-head OLMo config.
   No length-generalization conclusion from the 512-token validation alone.

## Sources

- CLA: https://arxiv.org/abs/2405.12981
- FoX: https://arxiv.org/abs/2503.02130
- DCMHA Appendix D: https://arxiv.org/html/2405.08553v2
- DIFF V2: https://huggingface.co/blog/microsoft/diff-attn-v2
- FoPE: https://arxiv.org/abs/2412.17739
- Pricing checked Nov 3: https://modal.com/pricing

## Commands (PowerShell, workspace root)

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

The actual launch runs `--stage single --variant baseline` and `--stage variants`
concurrently (six GPUs maximum) to reduce wall-clock waiting. Token and timeout
limits are unchanged. `variants` maps the five non-baseline configurations.
