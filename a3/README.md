# Assignment 3 - Part 2 implementation

Picochat architecture experiments on nanochat commit
`92d63d4e8bb4df75c3b71618f31ddde2378b2bcd` (July 3, 2026).
The selected interventions are FoX without RoPE and CLA2.
The implementation excludes the assignment LaTeX report, diagrams, and Parts 4--5.

Only baseline, FoX, and CLA2 are implemented and exposed by the launcher.
Historical six-configuration JSON records remain unmodified for provenance;
`REPORT.md` summarizes only the selected configurations. Historical source
hashes refer to the original screening code, not this scope-only cleanup.
See `PILOT_PROTOCOL.md` for implementation details and limitations.

## Reproducing Part 2

From the repository root, clone the external dependency (do not commit it):

```powershell
git clone https://github.com/karpathy/nanochat.git nanochat
git -C nanochat checkout 92d63d4e8bb4df75c3b71618f31ddde2378b2bcd
uv venv a3/.venv
uv pip install --python a3/.venv/Scripts/python.exe modal
```

Authenticate with `a3\.venv\Scripts\modal.exe setup` and create a Modal secret
named `wandb-secret` containing `WANDB_API_KEY`. Never commit credentials.
The remote image installs the pinned CUDA dependencies with
`uv sync --frozen --extra gpu --no-install-project`.
See `PILOT_PROTOCOL.md` for preparation, smoke tests, and budget limits.
After preparation and smoke checks pass, run each selected configuration with
`a3\.venv\Scripts\modal.exe run a3\pico_modal.py --stage single --variant baseline`
(replace `baseline` with `fox` or `cla2`). These commands incur cloud charges;
do not start a new sweep without reviewing cumulative spending.
Existing result files on the Modal volume are reused, so a cached result is not
a fresh rerun. Archive results and choose a new run namespace before any fresh experiment.

Offline result checks need only Python's standard library:

```powershell
python a3/pico/summarize.py a3/pilot_results
```

CPU/GPU correctness checks and their historical outputs are retained
in `pico/tests.py` and `pilot_results/`.
FoX combines forgetting gates with removing RoPE; this experiment does not
isolate the contribution of either change.

## Part 3: hardware benchmark (stacked on Part 2)

`hardware_modal.py` compares baseline and FoX on A10, A100-80GB, and H100,
with seeds 42/43 and the same 500-update recipe. See `HARDWARE_PROTOCOL.md`
for timing definitions, bounded execution, backend differences, and MFU caveats.

Offline validation (no GPU or account access):

```powershell
python a3/pico/summarize_hardware.py
python -m unittest discover -s a3/pico -p test_hardware_results.py -v
```

After completing Part 2 data preparation, an explicitly authorized fresh
benchmark can use `a3\.venv\Scripts\modal.exe run a3\hardware_modal.py`.
This starts paid GPU workers; it is not a validation-only command. Existing
`hardware-v1` results are reused. Inspect provenance and choose a new namespace
in the launcher before a fresh comparison; cached results are not reruns.
The optional `--stage correct-mfu` writes corrected W&B summary fields and
must not be confused with offline reporting.

Historical raw device records are retained unchanged. Their source hashes
identify the original training implementation, before the Part 2 scope cleanup.
The generated report corrects A10G's BF16 peak denominator while preserving
raw MFU values. Current syntax/evidence checks do not constitute fresh GPU tests.
