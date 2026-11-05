# Artifact locations

These small summaries and figures can be committed to GitHub.

Full checkpoints, logs and per-example evidence are in Modal Volume `a3-p4-context`, `/runs/fox-seed42-v1`.
`ARTIFACT_MANIFEST.json` records hashes and sizes of the full remote files; it is an index, not the model itself.

Optional full download: `a3/p4/.venv/bin/python a3/p4/run.py export --experiment-id fox-seed42-v1 --full-export`.
Downloaded weights/data remain ignored by Git.
