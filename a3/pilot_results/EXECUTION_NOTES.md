# Execution notes

- CPU and GPU smoke tests passed on November 3, 2026.
- The first baseline and parallel-variant applications hit a **local result
  deserialization error**, not a numerical training error: `torch.__version__`
  is a TorchVersion object, so returning it required torch on the local client.
- Fixed metadata to `str(torch.__version__)` and JSON-normalized the remote
  return payload. A CPU-only torch wheel was also installed in a3/.venv during
  diagnosis. The remote GPU dependency installation is unchanged.
- Both baseline seeds and both CLA2 seeds completed and were persisted.
- FoX, FoPE and DIFF V2 seed 42 completed and were persisted.
- The interrupted application cancelled DCMHA seed 42 and the seed-43 attempts
  of FoX, FoPE and DIFF V2 before completion. They are excluded from final
  comparisons. W&B retains those interrupted attempts for auditability.
- Recovery skips completed result.json files and reruns only missing seeds.
  The architecture source hash, data, seeds, optimizer and token schedule are
  unchanged by the metadata fix.
- Final results must contain exactly 12 completed records: six configurations
  times two seeds. Use the W&B links in those records, not similarly named
  interrupted runs.
- GPU-only cost derived from successful-run elapsed_seconds is a lower bound
  on total billable usage. Startup, smoke tests, interrupted attempts, CPU/RAM,
  and preparation must also be accounted for. Final Modal invoice not verified.
