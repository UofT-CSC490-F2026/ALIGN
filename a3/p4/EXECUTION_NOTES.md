# Part 4 execution notes

November 4, 2026. The research chain completed; see the final completion
record below.

- Workspace usage limit and Part 4 authorized gross ceiling: US$100.
  Conservative launch reservations are recorded locally.
- Initial image build rejected a mutable local log included by a directory mount.
  Fixed by explicitly uploading only Part 4 Python source files and disabling
  automatic source mounts. No preparation/training function ran in that attempt.
- A subsequent image build completed, but Modal rejected registration of the
  H100 function because the account needs a payment method. No training ran.
- CPU preparation can now initialize without registering a GPU function. This
  permits independent preparation if the account allows CPU compute while the
  payment-method requirement is being resolved.
- Additional plotting dependencies are pinned to preserve upstream NumPy 1.26.4.
  An initial plotting-only installation selected a newer NumPy; that image was
  not used for any model training or data preparation.
- Failed-launch reservations are retained as conservative accounting. They are
  not invoices; image-build/setup consumption must be reconciled with Modal usage.

## Completed preparation and tracking check

- CPU-only preparation completed in 74.093 worker seconds at four requested CPU
  cores and 16 GiB memory. The requested-resource timer estimate is about
  US$0.0065, excluding image builds, startup, storage and other account usage; it
  is not a complete bill.
- Prepared 40,000,000 training tokens and 64 frozen evaluation examples using a
  32,768-token tokenizer. All six generated data/tokenizer/index files were
  downloaded locally and checksum-verified. Token IDs, tokenizer round-trip,
  document identities and exact-duplicate exclusion were also checked locally.
- Preparation App: https://modal.com/apps/jerry-zeng-uoft/main/ap-qFYODbjdVkH0Jx30T9KW3R
- W&B secret authenticated successfully as the user's W&B account. A CPU-only
  connection-test run logged a clearly identified setup metric; it was read back
  through the W&B API. This is not a model-quality result.
- W&B project: https://wandb.ai/jerry-zeng-university-of-toronto/a3-p4-context-extension
- Connection-test run: https://wandb.ai/jerry-zeng-university-of-toronto/a3-p4-context-extension/runs/y3bbtdii
- Tracking-test App: https://modal.com/apps/jerry-zeng-uoft/main/ap-nf1qt4nbbDyTtXrN6tsiGA
- At this point in setup, no GPU training or rehearsal had started. Modal's
  payment-method requirement was resolved later by the user.

## Budget and local-storage update

- The user raised the Part 4 gross ceiling from US$50 to US$100, matching the
  workspace usage limit, and confirmed that a payment method was added.
- H100 function registration was verified successfully without invoking a GPU.
- Full prepared data was re-verified on Modal before removing redundant local
  training/index copies and synthetic CPU checkpoints, freeing 191.87 MiB.
- Local preparation is now about 1.9 MiB. Full datasets and model weights stay on
  the persistent Modal Volume by default; small evidence exports preserve the
  complete remote index and verify the selected downloaded files.
- Code, small summaries and figures can go to GitHub. The Python environment,
  datasets, caches and checkpoint binaries remain ignored by Git.
- The GPU persistence rehearsal passed. Its quality numbers are correctness-test
  outputs and are excluded from research-result tables.

## Completed research chain

- Run `fox-seed42-v1` completed 500 steps at 512 and another 500 at 2048:
  16.384M + 16.384M = 32.768M training tokens. The model has 73,545,506 parameters.
- Checkpoint 1: `98e30216bf74de1a817a9b29d576781f6c676be79876f4da7a90e9cabc4ca03e`.
- Checkpoint 2: `c8ca95c3e56113d048a7dc3e0b390464a5130e549e88e62134bed3047584e1f2`.
  Its parent hash and restored training state were verified.
- All four evaluation cells contain the same 64 documents and 8,192 target tokens
  (37,180 bytes). BPB is 1.412895/1.401852 for checkpoint 1 at 512/2048, and
  1.228746/1.189348 for checkpoint 2 at 512/2048.
- Both final checkpoints passed full remote hash/load audits in fresh CPU workers.
  Required online W&B histories were read back successfully. Small local evidence
  exports passed their independent selected-file audits; model binaries stayed
  on Modal.
- Research trainer timers imply US$0.7136 of requested resources, excluding
  platform/setup/finalization overhead. This is not the final invoice. The
  authorized ceiling remains US$100.
- `REPORT.md` explains the extra-training and window-policy confounds, and the
  one-seed/limited-evaluation limitations. No Part 5 training was launched.
