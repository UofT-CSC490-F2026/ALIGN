# Standard nanochat validation of the Part 4 FoX checkpoints

This supplement adds the upstream nanochat BPB metric and BOS best-fit
validation loader alongside the original Part Four matched-target task.
The Part Four report uses only the FoX rescores. A separate baseline control
was completed before the team clarified that its Part Four task is to continue
FoX, not repeat the Part Two architecture comparison. That control remains
saved for provenance but is excluded from the Part Four conclusions.

## Frozen scoring protocol

- Upstream commit: `92d63d4e8bb4df75c3b71618f31ddde2378b2bcd`.
- Call the actual upstream `evaluate_bpb` function, unchanged.
- Generate validation inputs with upstream
  `tokenizing_distributed_data_loader_bos_bestfit`, batch 8, split `val`.
- Use held-out `shard_06542.parquet` at the recorded Part Four dataset revision.
- Freeze 262,144 nominal target positions at each context, 512 and 2048;
  mask special tokens using the original token-byte table.
- Copy the loader's reusable buffers and persist exact input/target arrays
  and hashes on Modal before scoring checkpoints.
- All four checkpoints share the exact frozen arrays at a given context.
  Different contexts have separate best-fit streams, so their BPB difference
  is not presented as a matched-target context effect.
- The original Part Four tokenizer remains attached to the models that were
  trained with it. This matches Jinhao's scoring procedure and budget; it is
  not claimed to reproduce his original tokenized inputs without his tokenizer
  and validation hashes.

## Matched architecture control

Train a fresh July baseline (RoPE, no FoX forget gates), seed 42, depth 6,
width 384, 6 heads, vocabulary 32,768. Retain the original FoX run's:

- Exact tokenizer and 40M-token prepared stream, with identical cursor positions.
- 500 updates at context 512 followed by 500 updates at 2048.
- 32,768 training tokens/update; microbatch/accumulation 8x8 and 2x8.
- Upstream Muon/AdamW, identical learning rates, weight decay, global warmup
  and warmdown schedule, BF16 and H100.
- Full optimizer/RNG/data-cursor continuation at the boundary.

The shared learned parameters are verified to have identical initialization.
The baseline's fixed stage-final checkpoints are evaluated without selecting
checkpoints by validation results. Intermediate validation is omitted from the
control training loop; it does not update weights. Its timing is consequently
not used as an efficiency comparison with the original FoX job.

This tests the full FoX adaptation (forget gates plus removal of RoPE), rather
than isolating the gate's causal effect. One seed is an exploratory comparison,
not evidence of training-seed robustness. FoX is allowed to lose: report every
fixed cell and do not rerun seeds or modify the evaluation to obtain a win.

The primary architecture comparison is checkpoint 2 evaluated at 2048, the
final context. Checkpoint 1 evaluated at 512 supplies the Part Two connection;
both cross-context cells are also reported. These priorities are specified
before reading baseline standard-validation results.

## Execution

Use the existing isolated client environment, profile, Volume and W&B Secret.
Each phase is explicit, has zero automatic retries, and reserves a conservative
amount against the original US$100 Part Four ceiling before cloud registration.

```
a3/p4/.venv/bin/python -B a3/p4/standard_eval/run.py check
a3/p4/.venv/bin/python -B a3/p4/standard_eval/run.py fox-eval
a3/p4/.venv/bin/python -B a3/p4/standard_eval/run.py baseline-short
a3/p4/.venv/bin/python -B a3/p4/standard_eval/run.py baseline-long
a3/p4/.venv/bin/python -B a3/p4/standard_eval/run.py baseline-eval
a3/p4/.venv/bin/python -B a3/p4/standard_eval/run.py verify
a3/p4/.venv/bin/python -B a3/p4/standard_eval/run.py export
```

Remote full checkpoints: `a3-p4-context:/runs/baseline-control-seed42-v1/`.
Remote evaluation and frozen arrays:
`a3-p4-context:/standard_eval/standard-validation-seed42-v1/`.
Small Git-ready exports: `a3/p4/results/standard-validation-seed42-v1/`.
W&B project: `a3-p4-standard-validation`.

The final audit reloads every checkpoint in a fresh CPU worker, checks equal
training tokens/cursors and evaluation identities, and reads back W&B histories.
Weights, raw shards, frozen token arrays and binary W&B logs stay on Modal.
