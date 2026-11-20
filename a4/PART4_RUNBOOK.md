# CSC490 A4 Part 4 — Reward-shaping experiments

**Status: code prepared; no new RL runs or paired evaluations performed. All Modal launches require separate manual authorization and are billable.** See PRs #41 (completion), #42 (arithmetic B1), and #43 (RL integration).

## Motivation and fixed methods

Part 3: SFT 89/1319 (6.75%) vs historical correctness-only RL 194/1319 (14.71%) on greedy GSM8K test. Of incorrect sampled training responses, 33% lacked the final `#### NUMBER` line. And 2,815/7,472 training questions had 0/16 correct samples, so binary group-relative rewards supplied no within-group advantage on those questions. The original Part 3 labels are heuristic, not verified causes.

- **A (completion):** +0.15 for a single correctly formatted final `#### NUMBER` line.
- **B1 (arithmetic):** up to +0.20 for distinct verified intermediate calculations whose expression tree and result match GSM8K's annotated reference calculations. No arbitrary arithmetic, answer-only matching or executing generated code. B1 is the frozen production choice; exploratory B2-B5 were not adopted.
- **Modes:** `baseline`, `completion`, `arithmetic`, `combined`. Training reward is original binary correctness + enabled bonuses, and group-centered advantages use the total. Validation/test accuracy still uses **binary final-answer correctness only**.
- Original RL was interrupted and resumed without optimizer state, so compare the historical baseline cautiously; an optional uninterrupted correctness-only rerun is defined in the manifest.

## Local verification (no cloud/GPU)

From repo root, with Python 3.12 and pytest:

```powershell
py -3.12 -m pytest a4/test_rewards.py a4/test_arithmetic_reward.py a4/test_reward_integration.py a4/test_modal_launch.py -q
py -3.12 -m compileall -q a4
py -3.12 a4/plan_experiments.py --mode dry
py -3.12 a4/plan_experiments.py --mode smoke
py -3.12 a4/plan_experiments.py --mode full
```

The planner prints **documentation only**; it does not invoke Modal. `preview_launch.py` can preview a single local command. Verify all commands use `--model-step 466`. Do not paste the printed `modal run` commands into the shell unless intentionally authorizing a billed job.

## Restore Royce's personal Modal profile (mandatory after using Laith's workspace)

The last observed CLI output showed active profile **`allaith2004`**, token workspace **`allaith2004`**, and only Laith's `a4-base` contents and `wandb-secret`. This is NOT authorization to run or transfer from Laith's workspace. First switch profiles and verify **both** the active profile and authenticated token workspace:

```powershell
py -m modal profile list
py -m modal profile activate royce-csc490
py -m modal profile current
py -m modal token info
Get-ChildItem Env:MODAL_* | Select-Object -ExpandProperty Name
```

STOP if `modal token info` still identifies `allaith2004`, if an environment override routes to another workspace, or if the profile is missing. Never print token-secret values or other credentials. A profile change selects credentials; it does not copy volumes or secrets between workspaces.

After confirming `royce-csc490`:

```powershell
py -m modal volume list
py -m modal volume ls a4-base
py -m modal volume ls a4-base /chatsft_checkpoints/a4-sft-default
py -m modal volume ls a4-base /tokenizer
py -m modal secret list
```

The required files are `/chatsft_checkpoints/a4-sft-default/model_000466.pt`, matching `meta_000466.json`, and `/tokenizer/tokenizer.pkl`. The historical RL baseline `chatrl_checkpoints/a4-rl3-resume/model_000466.pt` plus matching metadata is needed only if reevaluating that checkpoint on Royce's workspace. Verify the checkpoint and tokenizer pair by hash/config, not by filename alone.

If the personal volume is empty, arrange a **specific, authorized, one-time** transfer of just these files: download from Laith's volume to a temporary local directory while explicitly using his permitted profile, then **reactivate and verify** `royce-csc490` before uploading to its `a4-base`. Example CLI argument shapes (do not blindly execute until filenames, permission and local free disk space are verified):

```powershell
# SOURCE profile must be explicitly verified as allaith2004, with permission:
# py -m modal volume get a4-base /chatsft_checkpoints/a4-sft-default/model_000466.pt <local-file-path>
# py -m modal volume get a4-base /chatsft_checkpoints/a4-sft-default/meta_000466.json <local-file-path>
# py -m modal volume get a4-base /tokenizer/tokenizer.pkl <local-file-path>
# Then explicitly activate royce-csc490 and verify token info again.
# py -m modal volume put a4-base <local-model-file-path> /chatsft_checkpoints/a4-sft-default/model_000466.pt
# py -m modal volume put a4-base <local-meta-file-path> /chatsft_checkpoints/a4-sft-default/meta_000466.json
# py -m modal volume put a4-base <local-tokenizer-file-path> /tokenizer/tokenizer.pkl
```

`modal volume get/put` operate in the *currently selected workspace and environment*. Avoid `--force` and any overwrite of existing files. Transfers may incur storage/network charges even without GPU usage; check remaining credits. Do not create or copy secret material from Laith's workspace. Create your **own** `wandb-secret` with your authorized W&B key and verify intended W&B project via the dashboard. End on Royce's active profile and check that the source/model paths resolve correctly.

### Remaining readiness checklist (record outcomes before launch)

- [ ] Latest three-PR stack pulled locally; relevant `pytest`, compilation, and no-cloud command planning pass.
- [ ] Active profile and token both say `royce-csc490`; no overriding `MODAL_*` environment.
- [ ] Personal `a4-base` volume has complete verified SFT step-466 model, metadata and matching tokenizer.
- [ ] Local/remote dataset cache and available disk/volume space are sufficient.
- [ ] Personal `wandb-secret` and intended W&B project configured, without disclosing secrets.
- [ ] Credits, current Modal billing/pricing and effective spending/concurrency controls verified; explicit permission before billable run.
- [ ] Single-H100 one-update dry succeeds; inspect logs and COMPLETE marker.
- [ ] Eight-H100 one-update smoke succeeds, with fresh run identifier.
- [ ] A-only, B1-only and combined full runs finish; archive checkpoints and raw logs (optional uninterrupted baseline if budget allows).
- [ ] Paired 1,319-question GSM8K evaluation, uncertainties and error analysis completed; fill the Part 4 assignment report.

## Preflight — required BEFORE any billable operation

1. **Workspace:** Run `py -m modal profile current` and `py -m modal token info` (read-only). Both must refer to **royce-csc490**, never Laith's workspace. Inspect names only of `MODAL_*` environment overrides; avoid revealing token values.
2. **Assets:** In the **royce-csc490** `a4-base` volume, verify `chatsft_checkpoints/a4-sft-default/model_000466.pt`, matching `meta_000466.json`, and `tokenizer/tokenizer.pkl`. Check sizes/hashes and `model_config`, confirm matching tokenizer vocabulary. All three initial experiments must start from the same complete checkpoint. Existence in Laith's volume is **not** existence in Royce's. Do not run if missing; transfer only the authorized files via explicit manual steps.
3. **W&B:** Confirm that the expected `wandb-secret` exists in Royce's workspace, with a valid `WANDB_API_KEY` for the desired account/project; do not reveal or log secret values. Merely listing the secret does not verify that the key works.
4. **Costs:** Check the Modal dashboard's exact workspace credits, account billing behavior, spending/budget/concurrency controls, active jobs, and current H100 prices. Free credits are **not automatically a hard spending cap**. Dry/smoke/full and paired evaluations all use billed GPU time. Do not launch if spending beyond credits is unacceptable and a hard guard is not verified.
5. **Code/integration:** Check PR stack and local tests, and ensure the pinned nanochat revision (`92d63d4e8bb4df75c3b71618f31ddde2378b2bcd`) is compatible with transferred checkpoint and tokenizer. Check training dataset cache and disk room.
6. **Run names:** Launcher refuses existing RL rollout/checkpoint folders; use unique names. A dry/smoke run is real training, not a simulated dry run.

## Experiment matrix and launch

The single source of truth is `a4/part4_experiments.json`, with **SFT step 466 pinned** and the original hyperparameters. Generate exact commands locally with `plan_experiments.py`. Run only after preflight and explicit user authorization, in this order:

1. One-GPU, **one-update dry** for completion or combined, to establish imports/model loading/logging.
2. Eight-GPU, **one-update smoke** for an isolated run name, if the one-GPU dry succeeds.
3. Three **full** one-epoch runs: `a4-p4-completion-v1`, `a4-p4-arithmetic-v1`, `a4-p4-combined-v1`; optional fresh `a4-p4-baseline-v1` if budgets permit. Do not interpret an incomplete experiment's final checkpoint as complete.
4. Paired greedy GSM8K test evaluation (1,319 questions, 512 tokens), using the same scoring and questions for SFT, historical baseline (if its checkpoint is available), and all new runs.

The RL launcher saves distinct checkpoint and rollout folders plus COMPLETE/INCOMPLETE markers. `--max-steps` caps optimizer updates per invocation; dry=1, smoke=1, full=no additional cap. JSONL logs distinguish `correctness`, `completion`, `arithmetic`, `total` and preserve historical `reward` as binary. W&B logs components separately. `a4/eval_models.py` and `a4/dump_answers.py` support evaluation and paired error analysis. The historical `a4-rl3-resume` step 466 checkpoint must also be present in the correct workspace if re-evaluating it.

### After training — analysis required to finish Part 4

Record checkpoint names, full hashes and final optimizer steps. Compare paired test accuracies and 95% bootstrap intervals, raw training correctness curves (not shaped reward as accuracy), missing `####` rate, intermediate calculation matches, and error categories. Report A, B1, A+B1 against the original/historical baseline and any fresh uninterrupted control; mention historical optimizer reset, single seeds, incomplete reference annotation, and unit/role alias risks. Do not claim model improvement before evaluation.

## GPU cost planning (estimates, not billing caps)

The saved planning estimate from 2026-11-14 was **US$0.001097/H100-second** (source: https://modal.com/pricing); recheck it before every manual launch. GPU-only costs at configured timeouts: dry one H100 ×30 min ~US$1.97; smoke eight H100s ×10 min ~US$5.27; full eight H100s ×4 hours ~US$126.37 **per full run**. Three full runs at their timeouts total ~US$379.12 GPU-only. CPU/RAM, builds, storage, transfers and tax may add cost. Timeouts are **not** payment limits.

**No GPU/cloud job is started by these instructions.**
