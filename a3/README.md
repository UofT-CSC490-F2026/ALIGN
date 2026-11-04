# Assignment 3 - Part 2 implementation

Picochat architecture experiments on nanochat commit
`92d63d4e8bb4df75c3b71618f31ddde2378b2bcd` (July 3, 2026).
The selected interventions are FoX without RoPE and CLA2, with a separate
FoX-with-RoPE diagnostic control. This PR excludes the assignment report,
diagrams, and Part 3 hardware experiments.

The broader six-configuration screening code and raw results are retained for
reproducibility in `pico/` and `pilot_results/`. `REPORT.md` summarizes the original
screening; `ROPE_CONTROL.md` summarizes the separate diagnostic control.
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
python a3/pico/summarize_rope_control.py
```

The second command also checks that adding the RoPE control preserved the original
variant source. GPU correctness checks and their historical outputs are retained
in `pico/tests.py`, `pico/rope_control_tests.py`, and `pilot_results/`.
