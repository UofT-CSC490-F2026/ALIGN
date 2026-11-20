# Restored nanochat files

Karpathy deleted midtraining from nanochat (commit `1ddaad1`, 2026-01-31) and later removed two task modules it
imports (commit `950a1dc`, 2026-07-03). The A4 assignment still asks for midtraining, so these three files are
restored from just before those deletions and mounted into the Modal image over the pinned nanochat
commit `92d63d4` (`a4/modal_launch.py`):

| File | Restored from |
|---|---|
| `scripts/mid_train.py` | `1ddaad1^` |
| `tasks/customjson.py` | `950a1dc^` |
| `tasks/spellingbee.py` | `950a1dc^` |

Two edits were needed in `mid_train.py`. First, nanochat's MMLU task changed its arguments after January, so
`MMLU(subset="auxiliary_train", split="train")` became `MMLU(subset="all", split="auxiliary_train")`, the same call
`chat_sft.py` uses. Second, the closing `nanochat.report` block is removed: that module does not exist at the pinned
commit, and the import crashed the non-master ranks while rank 0 was still writing the checkpoint, which killed it and left a
truncated model file. Every other call was checked against the current task classes and helpers.

`scripts/chat_rl.py` is nanochat's file at the pinned commit (identical to master) plus about a dozen lines: a
`--rollouts-dir` option that saves, for every training problem, the question, the reference answer and all sampled
answers with their rewards (Part 3 reviews the right and wrong answers during training). It also gains `--resume-from`
and `--resume-step`, which continue a stopped run from one of its saved checkpoints: the weights are loaded, training
carries on at the next step (a checkpoint already includes its own step's update), the learning-rate schedule continues,
and the training problems already used are skipped. The
optimizer state is not saved by nanochat, so it restarts empty. Modal can preempt GPU jobs, so this is the safeguard.
Nothing else changed.

nanochat is MIT licensed (Andrej Karpathy). Midtraining also reads `identity_conversations.jsonl` from the volume,
which comes from the download line in nanochat's old `speedrun.sh`.
