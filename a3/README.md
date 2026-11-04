# Assignment 3 — Part 2 draft PR

This draft covers the original October 13, 2025 nanochat architecture,
checked against commit `3a5e0bc50b123f37220f1ace6813f080455d19f3`.
The experiments use the July 3, 2026 baseline `92d63d4`; the report includes
a version comparison and three proposals (FoX, CLA, FoPE). Part 2 selects
FoX without RoPE and CLA2, with a separate FoX-with-RoPE diagnostic control.

## Overleaf

1. Upload only `a3.tex`. The TikZ diagram is now embedded; no `figures` folder is required.
2. Select `a3.tex` as the main document and pdfLaTeX as the compiler.
3. Replace the team-name/student-ID placeholder.
4. Compile. The included TikZ diagram renders directly as vector graphics;
   no screenshot, image generation, external tool, or shell escape is needed.
5. Download the eventual completed report as `a3.pdf` into this folder.

The source has not yet been compiled locally: no local LaTeX compiler was found.
Review the rendered layout in Overleaf before submission.

## Optional Mermaid image

`figures/original_architecture.mmd` contains an alternative diagram with explicit
Q/K/V branches. Paste its contents into a Mermaid editor and export a high-resolution
PNG (or a PDF if supported). Upload it to `figures/` in Overleaf, and replace
the embedded `\resizebox`/TikZ diagram block with, for example:

```tex
\includegraphics[width=\linewidth,height=0.78\textheight,keepaspectratio]{figures/original_architecture.png}
```

The Mermaid diagram and TikZ diagram describe the same original model at different
levels of detail. All linear projections are bias-free. Optional KV-head sharing
and inference KV caching are discussed in the text rather than expanded in the diagram.

## Remaining work

- Context extension and final-model results.
- Final proofreading, references, names/IDs, and PDF layout review.

The report includes only the selected baseline/FoX/CLA2 pilot comparisons,
not all five screened candidates. The complete original screening is retained
in `pilot_results/sweep.json` and `pilot_results/REPORT.md`; the additional
control is in `pilot_results/ROPE_CONTROL.md`.
Part 3 hardware experiments and results are intentionally excluded from this PR.

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
