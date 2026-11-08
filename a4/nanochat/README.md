# Restored nanochat files

Karpathy deleted midtraining from nanochat (commit `1ddaad1`, 2026-01-31) and later removed two task modules it
imports (commit `950a1dc`, 2026-07-03). The A4 assignment still asks for midtraining, so these three files are
restored unchanged from just before those deletions and mounted into the Modal image over the pinned nanochat
commit `92d63d4` (`a4/modal_launch.py`):

| File | Restored from |
|---|---|
| `scripts/mid_train.py` | `1ddaad1^` |
| `tasks/customjson.py` | `950a1dc^` |
| `tasks/spellingbee.py` | `950a1dc^` |

nanochat is MIT licensed (Andrej Karpathy). Midtraining also reads `identity_conversations.jsonl` from the volume,
which comes from the download line in nanochat's old `speedrun.sh`.
