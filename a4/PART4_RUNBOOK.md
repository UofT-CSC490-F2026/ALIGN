# A4 Part 4: local implementation and manual experiment runbook

No training or evaluation was run for this change. Zero Modal GPU jobs were launched,
and no Modal cloud compute was invoked. The local analysis below measures reward
signals in historical sampled training responses, not a change in GSM8K test accuracy.

## Branch stack and files

- `royce/a4-part4` -> `laith/a4-controls`: `rewards.py`, `test_rewards.py` (Method A).
- `royce/a4-arithmetic` -> `royce/a4-part4`: `arithmetic_reward.py`, `test_arithmetic_reward.py` (Method B).
- `royce/a4-reward-integration` -> `royce/a4-arithmetic`: RL integration, local analysis,
  pure launcher configuration/preview, tests, experiment manifest and this runbook.

PRs: [#41 completion](https://github.com/UofT-CSC490-F2026/ALIGN/pull/41),
[#42 arithmetic](https://github.com/UofT-CSC490-F2026/ALIGN/pull/42),
[#43 integration](https://github.com/UofT-CSC490-F2026/ALIGN/pull/43).
They were created through the connected GitHub tools because `gh` is not installed locally.

Parent updates were incorporated by normal merges; no shared history was rewritten.
Each PR must target its immediate parent. No PR should be merged automatically.

## Local/free validation

From the repository root, with Python 3.12 and pytest installed:

```powershell
py -3.12 -m pytest a4/test_rewards.py a4/test_arithmetic_reward.py a4/test_analyze_reward_signals.py a4/test_reward_integration.py a4/test_modal_launch.py -q
py -3.12 -m compileall -q a4
py -3.12 a4/preview_launch.py --mode smoke --run a4-p4-combined-preview --reward-mode combined
py -3.12 a4/plan_experiments.py --mode smoke
py -3.12 a4/plan_experiments.py --mode full
```

The last two commands **only print documentation**. Do not pipe their output into a
shell or execute it during local validation. Neither preview nor planning imports
Modal, torch, model weights, W&B, or a cloud client. Launcher tests use pure helpers;
production-loop tests select local functions with AST and use in-memory stubs.
The Part 4 suite passed **129 tests**, and `compileall -q a4` passed. The separate
existing `test_fox_cache.py` could not collect because this local Python lacks
`torch`; its actual CPU model checks were not run. No torch installation, weights,
or cloud runtime was needed for the Part 4 suite.

The arithmetic evaluator never uses `eval` or `exec` on generated output.

```powershell
py -3.12 a4/analyze_reward_signals.py C:/Users/wendy/Documents/a4-rollout-analysis/a4-rl3.jsonl C:/Users/wendy/Documents/a4-rollout-analysis/a4-rl3-resume.jsonl --output-dir a4/.reward-analysis/my-new-analysis
```

Use a fresh output directory; existing summaries are never overwritten. The ignored
`.reward-analysis` directory contains JSON, CSV, and an SVG bar chart, without raw
responses. Inputs are opened read-only. All correctness labels come from the original
binary rollout reward (or explicit `correctness` in new-format logs); correctness is
never inferred from a shaped total. Malformed input is flagged and excluded, not
silently repaired. Overlap keys are `(step, problem)`; first occurrence wins, so input
order matters for conflicting re-sampled groups.

## Measured historical signals

The actual local inputs contain 132 and 836 lines. Ten lines are malformed JSON
(two original, eight resumed), and the resumed file has two conflicting overlap keys
at `(61, 980)` and `(61, 988)`. The combined analysis retains 956 records and 15,285
responses. The inputs are subsets of the 8-rank training dataset.

| Configuration | Positive reward rate | Mean total reward | Incorrect responses with partial credit | Mean within-question reward/advantage variance | Groups with zero advantage variance |
|---|---:|---:|---:|---:|---:|
| Baseline | 26.2349% | 0.262349 | 0% | 0.087192 | 41.5272% |
| Completion | 75.6428% | 0.375803 | 66.9800% | 0.098890 | 7.8452% |
| Arithmetic | 26.2349% | 0.262349 | 0% | 0.087192 | 41.5272% |
| Combined | 75.6428% | 0.375803 | 66.9800% | 0.098890 | 7.8452% |

Raw correctness remains **26.2349%** in every configuration. Completion's mean bonus
is 0.113454 overall and 0.100470 among incorrect responses. Its bonus/correctness
Pearson correlation is 0.338124; total-reward/correctness correlation is 0.991496.
Constant bonuses have undefined correlations (`null` in JSON). Variance uses the
population formula per question and then averages equally over retained questions.
Subtracting the group mean leaves the variance unchanged.

After global overlap filtering, original/resumed files retain 130/826 records and
2,080/13,205 responses, with raw correctness rates 25.1923%/26.3991% respectively.
Seventeen groups have a sample count other than 16 (ranging from 3 to 35). The
summary flags this and reports the full distribution. These schema-valid groups
are retained, but may include incomplete or damaged exports; interpret the metrics
as descriptions of the supplied files rather than clean full-dataset estimates.

**Critical limitation:** all historical saved references have zero arithmetic
annotations. The old logger saved only the final reference text segment, while
nanochat stores calculations in separate `python`/`python_output` parts. Therefore
Method B has 0% reference-step coverage on these files. Its zero score is not evidence
that the model has no valid intermediate arithmetic or that Method B fails. A
meaningful retrospective Method B analysis requires full references obtained locally
and matched to these shuffled question texts. No additional dataset was downloaded.
New RL logs reconstruct the original annotated reference from all trusted reference
parts, fixing coverage for future analysis. Tests cover the real GSM8K syntax and
nanochat tool format without any generated code execution.

No claim of training improvement or test-accuracy improvement follows from these
reward statistics. Completion shaping is dense but also rewards many wrong answers.

## Reward and optimizer semantics

`chat_rl.py --reward-mode` accepts `baseline` (default), `completion`, `arithmetic`,
and `combined`. Default weights are 0.15 and 0.20. Completion weight must be finite
and nonnegative; arithmetic weight must be finite and in [0, 0.20]. Method A is kept
intact. Method B checks reference and generated arithmetic using restricted AST
nodes, exact rational arithmetic, bounded expression length/depth/magnitude, and
1e-9 absolute result tolerance. It recognizes ordinary equations and nanochat Python
tool/output pairs. Only matching expression trees earn credit; numeric spellings
and redundant parentheses normalize, but operand order and operations must match.
Repeated reference or generated steps count once. Unrelated correct equations,
wrong results, arbitrary code, numbers alone, and unsupported expressions earn zero.
This conservative method can miss algebraically equivalent rewrites, prose-only
reasoning, or nonstandard equation notation. It is a heuristic, not proof of reasoning.

Training still calls `train_task.reward(conversation, generated_text)` exactly as
before. Baseline bypasses both shaping parsers and returns that reward directly,
without floating-point bonus addition. The total is used by the existing
`advantages = rewards - rewards.mean()` calculation. JSONL `reward` remains binary;
new fields `correctness`, `completion`, `arithmetic`, and `total` are separate.
W&B logs `reward/correctness`, `reward/completion`, `reward/arithmetic`, and
`reward/total`, averaged across ranks, alongside the existing reward metric.
Evaluation retains `task.evaluate`; shaping never enters pass@k or paired accuracy.

`--max-steps N` caps **optimizer updates in this invocation**, including resumes.
It defaults to no additional cap. The loop iterator is capped before any next-step
sampling or evaluation, so a 3-update run performs updates 0, 1, 2 only. Its final
checkpoint is saved after update 2. The original full-horizon LR schedule is unchanged.
Smoke is still real training and real evaluation, and therefore billable.

The launcher sets `--output-tag <run>` instead of writing under the SFT tag. RL dry
and smoke preserve artifacts rather than deleting them. Existing checkpoint or
rollout folders cause refusal before subprocess launch, including incomplete runs;
a fresh name is required. RL launcher auto-resume is intentionally disabled to avoid
mistaking another experiment's checkpoint for this run. A killed/preempted run leaves
an INCOMPLETE marker where possible. Direct script resumes preserve old rollout files
and create a unique resume shard; checkpoint writes refuse existing step files.
Optimizer state is not saved/restored by the original RL implementation, so historical
resumed baseline behavior is not identical to an uninterrupted rerun.

## Workspace and asset prerequisites: manual, read-only checks

Before any **separately authorized manual launch**, run these local/read-only checks
and inspect the results privately. They were not run by this implementation:

```powershell
modal profile current
modal token info
Get-ChildItem Env:MODAL_* | Select-Object -ExpandProperty Name
modal secret list
```

Both profile and token identity must identify **royce-csc490**. STOP if either is
Laith's workspace or any unexpected workspace. Inspect the names of `MODAL_*`
overrides privately, especially `MODAL_PROFILE`, `MODAL_TOKEN_ID`, `MODAL_TOKEN_SECRET`,
and environment selectors. Never print secret values, dump the environment, change
profiles/authentication, create secrets, or write to another workspace as part of
these checks. Confirm the intended Modal environment as well as workspace: the
launcher resolves `a4-base` and `wandb-secret` in that context. If ambiguous, stop.

Check `wandb-secret` exists in royce-csc490 using the names-only list/dashboard. In
the dashboard confirm a nonempty key named `WANDB_API_KEY` without revealing its
value. Confirm the key belongs to the intended W&B account/project and has write
access. Full credential validity cannot be established by checking presence alone;
no W&B API call was made here. Do not use a GPU job to inspect a secret.

Required in **royce-csc490's** existing `a4-base` volume, mounted at `/vol`:

- `chatsft_checkpoints/a4-sft-default/model_<SFT_STEP:06d>.pt` and matching
  `meta_<SFT_STEP:06d>.json`, with a valid `model_config` matching that FoX model.
  Both must be complete; record file hashes and the exact step before all runs.
- `tokenizer/tokenizer.pkl`, exactly matching that checkpoint's vocabulary.
- The GSM8K `main` train/test dataset must be available to nanochat. A populated
  `/vol/hf` cache avoids a small dataset fetch during future manual training.

These requirements were checked against the pinned [GSM8K task](https://github.com/karpathy/nanochat/blob/92d63d4e8bb4df75c3b71618f31ddde2378b2bcd/tasks/gsm8k.py),
[checkpoint loader](https://github.com/karpathy/nanochat/blob/92d63d4e8bb4df75c3b71618f31ddde2378b2bcd/nanochat/checkpoint_manager.py),
and [tokenizer](https://github.com/karpathy/nanochat/blob/92d63d4e8bb4df75c3b71618f31ddde2378b2bcd/nanochat/tokenizer.py) source.

The pinned loader reads model+metadata and `tokenizer.pkl`; RL does not load SFT
optimizer shards or `token_bytes.pt`. Keep `tokenizer/token_bytes.pt` if other
nanochat tasks need it, but it is not used by this RL script. No A3 base checkpoint,
`identity_conversations.jsonl`, extra math dataset, or optimizer shard is required
for RL from an existing SFT checkpoint. The image supplies FoX/pico source, the cache
patch, nanochat at the pinned revision, the runner and reward modules. Do not copy
or overwrite teammate artifacts. No volume assets were inspected or changed here.

The exact SFT step and file availability are **not yet verified**. The manifest's
`sft_model_step: null` uses the pinned loader's latest model file. Before training,
verify the highest model file has matching metadata and freeze that folder, or set
`sft_model_step` to the verified step so generated commands explicitly pin it. The
same checkpoint pair must be used for all new experiments. Do not start from
`a4-sft-math`; it is a stronger but uncontrolled starting point for this comparison.

## Future manual commands: BILLABLE, documentation only

`part4_experiments.json` records the historical baseline, the three primary new
experiments and an optional fresh baseline control. Every planned run starts from
`a4-sft-default` and uses the current original RL defaults explicitly. Unique full
run, checkpoint, rollout and W&B names are provided; smoke adds `-smoke`, dry adds
`-dry`. Use a new version/name if any artifacts already exist.

Generate exact commands without running them:

```powershell
py -3.12 a4/plan_experiments.py --mode smoke
py -3.12 a4/plan_experiments.py --mode full
```

The generated smoke commands cap training at three optimizer updates on **8 H100s**,
with only 8 evaluation examples. Dry uses one H100 and one optimizer update. Full
uses 8 H100s, the full epoch, 400 training-time eval examples and the original
optimizer/generation settings. Each output is marked `DOCUMENTATION ONLY - BILLABLE
IF EXECUTED`. Exact commands for all configurations are also saved in
`part4_manual_commands.txt`; none were executed.

After successful full training, the following paired greedy evaluation is also
**BILLABLE** and must only be run manually after checking workspace and assets.
It uses the same GSM8K test set, 512-token cap, and raw correctness for all models:

```powershell
modal run a4/modal_launch.py --script answers --mode full --run a4-p4-paired-v1 --extra "--models sft:a4-sft-default,rl:a4-rl3-resume,rl:a4-p4-completion-v1,rl:a4-p4-arithmetic-v1,rl:a4-p4-combined-v1 --max-new-tokens 512 --max-problems 1319"
```

This additionally requires a complete `chatrl_checkpoints/a4-rl3-resume` step-466
model+metadata pair in royce-csc490 and each new completed experiment's model pair.
Do not evaluate an INCOMPLETE folder as if training finished. Add the optional fresh
baseline to the paired model list if it was trained. Archive the exact checkpoint
steps, generation settings, environment, revision and hashes with results.

Compute accuracy from the saved `correct` booleans, paired differences by question,
and uncertainty (for example a paired question bootstrap) locally. Analyze failures
by missing final answer/truncation, wrong final value, valid/invalid intermediate
tool output, and matched reference-step coverage, using identical categories for
all experiments. The existing `analyze_rl.py` can support local exploratory error
analysis once paired answer artifacts are available. No post-training performance
or error-analysis results exist yet for the new methods.

## Billing estimate, not a spending guarantee

The configured decorators request exact H100s: one for dry and eight for smoke/full.
Modal's published [pricing](https://modal.com/pricing), checked 2026-11-14, lists
H100 at **US$0.001097 per GPU-second** (US$3.9492/GPU-hour). GPU-only estimates at
these timeout durations are:

| Mode | GPUs | Timeout | GPU-only estimate at timeout |
|---|---:|---:|---:|
| Dry | 1 | 30 min | US$1.9746 |
| Smoke | 8 | 25 min | US$13.1640 |
| Full | 8 | 4 h | US$126.3744 |

Three primary full runs at their timeouts total about **US$379.12 GPU-only**;
adding the optional fresh baseline makes about US$505.50. Runtime is unmeasured;
these are timeout-based estimates, not expected durations or guaranteed invoice
caps. CPU/RAM, image builds, storage, egress, taxes and account-specific pricing
can add costs. Recheck current workspace pricing, remaining credits, budget and
concurrency before launching. Future training, dry/smoke runs, paired evaluation,
and image-building/remote execution are billable operations. Local pytest,
compilation, analysis, command planning and Git/PR preparation launch no Modal jobs.
