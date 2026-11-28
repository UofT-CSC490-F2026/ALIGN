# ALIGN arbiter training

CooperBench supplies coding tasks and tests. It does not supply gold arbiter decisions.
This package turns A2 run bundles plus checkpoint labels into supervised decision
examples. Teacher labels remain provisional until independent review. The current
pilot reserves whole repositories for a test split; Person 3 should approve the
final test partition before reporting research results. The current prompt and
completion follow Person 1's `ArbiterObservation` and `ArbiterDecision` envelope.
The public traces do not contain a reliable repository snapshot, so their
`repository_state` is `{}`; the live harness should send its actual snapshot.

## Offline tests

Run the Person 2 regression suite from the repository root. It uses fake Modal
engines and fake Hugging Face responses; it does not start GPU jobs or download
model weights. The package-specific coverage configuration checks all
`arbiter_training` application code, excluding only the `if __name__ ==
"__main__"` launch guards whose `main()` functions are tested directly.

```powershell
python -m pip install -r arbiter_training/requirements-test.txt
python -m pytest arbiter_training/tests/test_arbiter_training.py arbiter_training/tests/test_arbiter_scripts.py arbiter_training/tests/test_arbiter_modal_offline.py arbiter_training/tests/test_merged_runner_integration.py --cov=arbiter_training --cov-config=arbiter_training/.coveragerc --cov-branch --cov-fail-under=100 --cov-report=term-missing
```

The `Arbiter training tests` workflow enforces 100% line and branch coverage on
pull requests. These offline tests do not replace a live Modal training run or
Person 3's independent model evaluation.

## Runner alignment review (2026-11-26)

The CooperBench Lite split now uses the same revision-qualified task keys as
`scripts/normalize_tasks.py`. Public trace import preserves its original event
UUIDs, so the 21 human reviews still attach to the same proposals. The new
ignored build is `data/runner-aligned-v2/`: 10 imported public runs, 49 observed
labels, and 34 marked synthetic counterfactuals (61 train, 22 validation).
The split still holds out the same repositories. The archived v2 train and validation
SHA256 values are respectively
`a833a6f5022ecb7f27b896b0fa9496387b856e311ab3bfab5822dba484fe2e77`
and `701a613a79050192b7f0b4d565074006714869353e091d0d1b97a886f8ed26ad`.
The archived audit reports no warnings. Regenerating with the current source
categories and synthetic example will produce different hashes.

To reproduce the revised build from the existing pinned teacher labels and
tracked human review file, use new output paths so the historical pilot stays
available:

```powershell
python -m arbiter_training.split_cooperbench --subset lite --output arbiter_training/data/runner-aligned-v3/cooperbench-lite-split.json
python -m arbiter_training.import_traces --split-manifest arbiter_training/data/runner-aligned-v3/cooperbench-lite-split.json --output-dir arbiter_training/data/runner-aligned-v3/public-traces --max-pairs-per-split 5
python -m arbiter_training.merge_review --teacher-labels arbiter_training/data/public-trace-v1/teacher-labels-v2-provisional.jsonl --reviewed-packet arbiter_training/reviews/human-labels-20261125.jsonl --output arbiter_training/data/runner-aligned-v3/human-merged-labels.jsonl
python -m arbiter_training.prepare --tasks arbiter_training/data/runner-aligned-v3/public-traces/tasks.jsonl --events-dir arbiter_training/data/runner-aligned-v3/public-traces --labels arbiter_training/data/runner-aligned-v3/human-merged-labels.jsonl --split-manifest arbiter_training/data/runner-aligned-v3/cooperbench-lite-split.json --output-dir arbiter_training/data/runner-aligned-v3/prepared --history-limit 8
python -m arbiter_training.synthetic --tasks arbiter_training/data/runner-aligned-v3/public-traces/tasks.jsonl --split-manifest arbiter_training/data/runner-aligned-v3/cooperbench-lite-split.json --prepared-dir arbiter_training/data/runner-aligned-v3/prepared --output-dir arbiter_training/data/runner-aligned-v3/combined
python -m arbiter_training.audit --prepared-dir arbiter_training/data/runner-aligned-v3/combined --output arbiter_training/data/runner-aligned-v3/combined/audit.json
```

`prepare.py` uses a runner bundle's `arbiter_observation` event linked by
`trigger_sequence`, retaining the exact decision-time repository state. It
reconstructs observations only for imported public traces, which lack those
events and do not establish repository state. The merged-runner offline
integration test checks an actual bundle and all synthetic tool proposals.
The training contract calls the runner's proposal and decision validators
directly, so both paths use its tool vocabulary and affected-agent checks.
For normalized runner tasks without `metadata.split`, pass the pinned
`--split-manifest` to preparation, review queue generation, and synthetic data
generation. An unknown or held-out task cannot be assigned by hash fallback.
The review queue uses the same recorded observation and deduplicated bundle as
preparation; `inspect_queue.py` displays the observation's repository state.
Prepared runner examples carry `scenario_source: runner_trajectory` for separate
evaluation reporting.

Inference and training now reject inputs exceeding 8,192 tokens; inference
reserves 256 output tokens. With the pinned Qwen tokenizer, all 83 new pilot
prompt-plus-completion records fit (maximum 6,489 tokens). This does not bound
future live runner observations with larger workspaces or concurrent state;
audit those before training or deploying on them. The model prompt offers only
enforceable actions. `restructure` stays in the runner's v1 schema for
compatibility but is rejected by this inference adapter until the team defines
its enforcement semantics. Affected agent IDs are checked against the input.

The archived 2026-11-25 checkpoint was trained on the earlier prompt, task
keys, and synthetic proposals. It remains a pilot baseline; these revised
examples have not been used for another training run. New training is most
useful after labeled runner-generated observations are available, because the
current public trajectories still omit the live coordination state.
One live Modal inference call on a concurrent-runner observation completed on
2026-11-26 using that archived adapter. It produced a valid `continue` for a
safe fixture command; it is a compatibility smoke test, not an accuracy result.

## Original public-trace pilot recipe

The scripts pin both the official CooperBench task revision and a public two-agent
trajectory revision. The source traces were produced by CooperBench, not by our
live interception harness. They are useful for offline training and comparison;
Person 1's harness is still needed for end-to-end intervention experiments.

```powershell
python -m arbiter_training.split_cooperbench --subset lite --output arbiter_training/data/cooperbench-lite-split.json
python -m arbiter_training.import_traces --split-manifest arbiter_training/data/cooperbench-lite-split.json --output-dir arbiter_training/data/public-trace-v1 --max-pairs-per-split 5
python -m arbiter_training.review_queue --tasks arbiter_training/data/public-trace-v1/tasks.jsonl --events-dir arbiter_training/data/public-trace-v1 --output arbiter_training/data/public-trace-v1/teacher-queue-25-10.jsonl --train-limit 25 --validation-limit 10
python -m arbiter_training.pilot_labels --queue arbiter_training/data/public-trace-v1/teacher-queue-25-10.jsonl --output arbiter_training/data/public-trace-v1/teacher-labels-provisional.jsonl
```

The pilot's teacher decisions were drafted from prefix-only checkpoints and saved
as `teacher-labels-provisional.jsonl` under the ignored `data/` directory. Review
the queue and labels before using them as reference decisions. The code in
`synthetic.py` adds marked counterfactual conflict cases; these must be reported
separately from decisions on observed trajectories.

```powershell
python -m arbiter_training.prepare --tasks arbiter_training/data/public-trace-v1/tasks.jsonl --events-dir arbiter_training/data/public-trace-v1 --labels arbiter_training/data/public-trace-v1/teacher-labels-provisional.jsonl --output-dir arbiter_training/data/public-trace-v1/prepared --history-limit 8
python -m arbiter_training.synthetic --tasks arbiter_training/data/public-trace-v1/tasks.jsonl --prepared-dir arbiter_training/data/public-trace-v1/prepared --output-dir arbiter_training/data/public-trace-v1/combined
python -m arbiter_training.audit --prepared-dir arbiter_training/data/public-trace-v1/combined --output arbiter_training/data/public-trace-v1/combined/audit.json
```

## Prepare reviewed examples

`tasks.jsonl` can be A2's `normalize_tasks.py` output or the public-trace importer
output. `--events-dir` must contain `events.jsonl` bundles. Labels are
JSONL in A2 `evaluate.py`'s human-label shape, one per proposed-action event:

```json
{"event_uuid":"...","expected_actions":["continue"],"primary_action":"continue","reviewer":"reviewer-id","rationale":"The proposed action satisfies both agents' constraints."}
```

`primary_action` is optional and defaults to the first acceptable action. Teacher
labels use `"label_source":"teacher_gpt","review_status":"provisional"`; human
reviewed labels identify their reviewer. The training prompt excludes task metadata, setup
specifications, evaluator tests, gold patches, and all events after the proposal.
Put one held-out `task_key` per line in `heldout.txt`.

```powershell
python -m arbiter_training.prepare `
  --tasks PATH_TO_NORMALIZED_TASKS_JSONL `
  --events-dir PATH_TO_RUN_BUNDLES `
  --labels PATH_TO_REVIEWED_LABELS_JSONL `
  --split-manifest PATH_TO_COOPERBENCH_SPLIT_JSON `
  --exclude-task-keys heldout.txt `
  --output-dir arbiter_training/data
```

The output is `train.jsonl`, `validation.jsonl`, and `manifest.json`. For the
public-trace pilot, pass `--history-limit 8` to match the labeling queue. The split
groups all checkpoints from the same task. Keep scenario families and near
duplicates out of Person 3's test set as well.

Run `python -m arbiter_training.audit --prepared-dir PATH` to check provenance,
prompt chronology, duplicate checkpoints, and repository separation. A class
balance warning means the labels are unsuitable for fine-tuning.

## Compare the pretrained models on Modal

The active Modal profile must have GPU access. These commands create model cache
volumes and start H100 jobs; they do not deploy an endpoint.

```powershell
modal run arbiter_training/modal_app.py --mode compare `
  --validation-file arbiter_training/data/validation.jsonl `
  --output-file arbiter_training/data/pretrained-comparison.json
```

The comparison scores accepted actions and valid JSON outputs, separately for
public traces and synthetic cases. The raw predictions are retained for error
review. Set `--limit 10` for a small feasibility pass. These pilot scores are not
independent test results while labels are provisional.
Both models receive the same examples and action contract. Qwen uses its chat
template with thinking disabled; gpt-oss uses Harmony through its tokenizer.

## Fine-tune Qwen3.5-9B

This uses BF16 LoRA on one H100. Qwen3.5's fine-tuning guide advises against
4-bit QLoRA for this model family. The Modal job requires at least 20 training
examples, 5 validation examples, 5 intervention examples, and disjoint task keys.

```powershell
modal volume create align-arbiter-data
modal volume put align-arbiter-data arbiter_training/data/train.jsonl /train.jsonl
modal volume put align-arbiter-data arbiter_training/data/validation.jsonl /validation.jsonl
modal run arbiter_training/modal_app.py --mode train --run-name pilot-001 --steps 100
```

The adapter and metadata go to the `align-arbiter-checkpoints` Modal volume.
Use a new run name for every training attempt. Inference returns the four-field
Person 1 decision contract: `action`, `reason`, `affected_agents`, and
`evidence`.

```powershell
modal run arbiter_training/modal_app.py --mode infer --model qwen `
  --checkpoint-file PATH_TO_CHECKPOINT_JSON
modal run arbiter_training/modal_app.py --mode infer --model qwen `
  --checkpoint-file PATH_TO_CHECKPOINT_JSON `
  --adapter-path /checkpoints/pilot-001/adapter
```

The current A2 example bundle and older pipeline input files are synthetic
contract fixtures. They are insufficient for a meaningful training run.

## Runner contract and second teacher pass

`contract.py` accepts the exact top-level observation fields in PR #50's
`runner/contracts.py`:
`schema_version`, `task_key`, `shared_goal`, `agent_goals`, `constraints`,
`actor_id`, `proposal_sequence`, `proposed_action`, `recent_events`, and
`repository_state`. The decision has `action`, `reason`, `affected_agents`, and
`evidence`. The nested `ProposedAction` has `action: dict` and
`files_touched: list[str]`. Public trace proposals already use this A2 event
shape; `files_touched=[]` means the imported trace did not establish a reliable
file list. The live runner should provide its own bounded observation. The
stacked runner PR #58 handles `query` and `mediate`, while `restructure` remains
unsupported for enforcement and fails closed. Invalid model output raises by
default so the runner cannot silently dispatch a proposal.

The first 35 labels were reconsidered with a fixed decision rubric. One label
was changed and 14 real trace proposals were added. Before human review, all
49 observed decisions were provisional teacher labels. The 34 counterfactuals
remain explicitly tagged. The runner-contract set has 61 train and 22 validation
decisions, with disjoint repositories. The high-impact observed proposals are
exported in a blind 21-row packet at
`data/runner-contract-v1/flagged-review-packet.jsonl` for human review.
On this corrected-format provisional validation set, pretrained Qwen accepted
10/22 decisions (20 valid outputs) and pretrained gpt-oss accepted 7/22
(21 valid outputs). The reviewed rerun and selected adapter are documented
below. These are pilot selection scores, not independent test results.

```powershell
python -m arbiter_training.adjudicate --tasks arbiter_training/data/public-trace-v1/tasks.jsonl --events-dir arbiter_training/data/public-trace-v1 --initial-labels arbiter_training/data/public-trace-v1/teacher-labels-provisional.jsonl --output arbiter_training/data/public-trace-v1/teacher-labels-v2-provisional.jsonl --review-log arbiter_training/data/public-trace-v1/teacher-review-v2.json
python -m arbiter_training.prepare --tasks arbiter_training/data/public-trace-v1/tasks.jsonl --events-dir arbiter_training/data/public-trace-v1 --labels arbiter_training/data/public-trace-v1/teacher-labels-v2-provisional.jsonl --output-dir arbiter_training/data/runner-contract-v1/prepared --history-limit 8
python -m arbiter_training.synthetic --tasks arbiter_training/data/public-trace-v1/tasks.jsonl --prepared-dir arbiter_training/data/runner-contract-v1/prepared --output-dir arbiter_training/data/runner-contract-v1/combined
python -m arbiter_training.audit --prepared-dir arbiter_training/data/runner-contract-v1/combined --output arbiter_training/data/runner-contract-v1/combined/audit.json
python -m arbiter_training.review_packet --prepared-dir arbiter_training/data/runner-contract-v1/combined --audit arbiter_training/data/runner-contract-v1/combined/audit.json --output arbiter_training/data/runner-contract-v1/flagged-review-packet.jsonl
python -m arbiter_training.review_sheet --packet arbiter_training/data/runner-contract-v1/flagged-review-packet.jsonl --output arbiter_training/data/runner-contract-v1/review-sheet.md
```

After a reviewer fills any packet rows, keep the original blind packet and save
the edited copy separately. `merge_review.py` overlays only completed rows,
records the reviewer and rationale, and leaves the remaining teacher labels
provisional. Rerun `prepare.py`, `synthetic.py`, and `audit.py` with that merged
label file before training.

```powershell
python -m arbiter_training.merge_review --teacher-labels arbiter_training/data/public-trace-v1/teacher-labels-v2-provisional.jsonl --reviewed-packet PATH_TO_EDITED_PACKET --output arbiter_training/data/runner-contract-v1/labels-with-human-review.jsonl
```

## Reviewed runner-contract pilot (2026-11-25)

Jinhao reviewed and approved all 21 high-impact observed proposals in the blind
packet. The five-field annotation export is tracked at
`arbiter_training/reviews/human-labels-20261125.jsonl`; it contains no trace
payloads or model predictions. Overlay it on the pinned provisional teacher
labels with `merge_review.py`. The merged observed set has 21 human-reviewed
and 28 provisional teacher labels. The 34 counterfactuals remain separately
marked synthetic examples.

```powershell
python -m arbiter_training.merge_review --teacher-labels arbiter_training/data/public-trace-v1/teacher-labels-v2-provisional.jsonl --reviewed-packet arbiter_training/reviews/human-labels-20261125.jsonl --output arbiter_training/data/runner-contract-v1-reviewed/human-merged-labels.jsonl
python -m arbiter_training.prepare --tasks arbiter_training/data/public-trace-v1/tasks.jsonl --events-dir arbiter_training/data/public-trace-v1 --labels arbiter_training/data/runner-contract-v1-reviewed/human-merged-labels.jsonl --output-dir arbiter_training/data/runner-contract-v1-reviewed/prepared --history-limit 8
python -m arbiter_training.synthetic --tasks arbiter_training/data/public-trace-v1/tasks.jsonl --prepared-dir arbiter_training/data/runner-contract-v1-reviewed/prepared --output-dir arbiter_training/data/runner-contract-v1-reviewed/combined
python -m arbiter_training.audit --prepared-dir arbiter_training/data/runner-contract-v1-reviewed/combined --output arbiter_training/data/runner-contract-v1-reviewed/combined/audit.json
```

The reviewed build has 61 train and 22 validation decisions, with disjoint
repositories and no audit warnings. Human reviews cover 13 train and 8
validation observations. Its train SHA256 is
`292b81f163d39dca2564cfd93a149d920b368b45fc4cf9185024c474b751fd85`;
validation SHA256 is
`a7afc34c103124060adb4ff64dd51e87c67ef6c180949632bbf82939da291685`.
These validation scores remain pilot checkpoint-selection evidence, not Person
3's independent held-out test.

On this reviewed validation set, pretrained Qwen accepted 12/22 decisions
(20 valid four-field outputs); pretrained gpt-oss accepted 7/22 (21 valid
outputs). Qwen was selected for the corrected-format 60-step LoRA run. The
baseline comparison is saved locally at
`data/runner-contract-v1-reviewed/combined/base-comparison.json`.
The BF16 LoRA run used 60 steps and seed 3407. Its best validation loss was
0.5691. The selected adapter is on the `align-arbiter-checkpoints` Modal
volume at `/runner-contract-reviewed-60steps-20261125/adapter`; the tracked
`results/runner-contract-reviewed-60steps-20261125-metadata.json` records
model revision, input contract, dataset hashes, counts, and checkpoint path.
The adapter accepted 17/22 reviewed-set validation decisions with 22/22 valid
outputs, compared with 12/22 for pretrained Qwen and 7/22 for pretrained
gpt-oss. It accepted 10/13 observed trace and 7/9 synthetic decisions. Of the
eight human-reviewed validation observations, it accepted five. The adapter
accepted only 1/3 `block` labels, while pretrained Qwen accepted 3/3; its two
misses were `query` responses that would still withhold the proposed action.
The small, partly provisional validation set cannot establish intervention
safety. Person 3 should evaluate this candidate independently before treating
it as a reliable intervention policy. A compact result breakdown is tracked at
`results/reviewed-validation-20261125.json`; raw predictions remain in the
ignored local comparison reports.

```powershell
$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8'
modal volume put align-arbiter-data arbiter_training/data/runner-contract-v1-reviewed/combined/train.jsonl /runner-contract-v1-reviewed-20261125/train.jsonl
modal volume put align-arbiter-data arbiter_training/data/runner-contract-v1-reviewed/combined/validation.jsonl /runner-contract-v1-reviewed-20261125/validation.jsonl
modal run arbiter_training/modal_app.py --mode train --run-name runner-contract-reviewed-60steps-20261125 --steps 60 --train-volume-path /data/runner-contract-v1-reviewed-20261125/train.jsonl --validation-volume-path /data/runner-contract-v1-reviewed-20261125/validation.jsonl
modal run arbiter_training/modal_app.py --mode compare --validation-file arbiter_training/data/runner-contract-v1-reviewed/combined/validation.jsonl --models qwen --adapter-path /checkpoints/runner-contract-reviewed-60steps-20261125/adapter --output-file arbiter_training/data/runner-contract-v1-reviewed/combined/adapter-comparison.json
modal run arbiter_training/modal_app.py --mode class-smoke --validation-file arbiter_training/data/runner-contract-v1-reviewed/combined/validation.jsonl --model qwen --adapter-path /checkpoints/runner-contract-reviewed-60steps-20261125/adapter
```

The `class-smoke` run completed through `ModalArbiter.decide` on one reviewed
validation observation. Choose a new run name and volume paths for a repeat
training run; the command above records the completed run.

Person 3 can create a blind test queue with `review_queue.py --test-limit N`
once new held-out traces exist. The pinned public trajectory release has no
complete two-agent traces for the reserved test repositories, so a final
independent test requires new runner output or another separate source.
See `PERSON3_HANDOFF.md` for the integration call, annotation procedure, and
independent test protocol.

The earlier top-level-only, 60-step Qwen adapter accepted 16/22 provisional
validation decisions, versus 11/22 for pretrained Qwen and 10/22 for
pretrained gpt-oss on the same 22 decisions. That adapter was trained with a
different nested `ProposedAction` shape and must not be treated as validated
against PR #50. The archived versioned adapter is on Modal at
`/checkpoints/person1-contract-v1-60steps-20261124/adapter` and locally at
`data/checkpoints/person1-contract-v1-60steps-20261124/adapter-files/adapter`.

## Pilot outcome (2026-11-24)

This table archives the earlier two-field decision format. The current scripts
emit Person 1's four-field format, so rerunning them will produce different
prompts and scores.

The imported public set contains 10 two-agent runs and 553 proposals. We labeled
35 observed checkpoints and added 20 marked counterfactuals. The prepared split
contains 40 train and 15 validation decisions across disjoint repositories.
All labels are provisional teacher judgments, and Person 3's independent test
set has not been evaluated.

| Model | Accepted validation decisions | Observed trace decisions | Block decisions |
| --- | ---: | ---: | ---: |
| Pretrained Qwen3.5-9B | 8/15 | 6/10 | 3/3 |
| Pretrained gpt-oss-20b, final channel | 8/15 | 7/10 | 1/3 |
| Qwen3.5-9B, four-step LoRA smoke | 6/15 | 5/10 | 3/3 |
| Qwen3.5-9B, 20-step LoRA pilot | 10/15 | 8/10 | 3/3 |

The 20-step adapter is stored on `align-arbiter-checkpoints` at
`/qwen-pilot-20steps-20261124/adapter` and downloaded under
`data/checkpoints/qwen-pilot-20steps-20261124/adapter-files/adapter`.
Neither pretrained nor fine-tuned model got a `query` or `mediate` validation
decision correct. Do not deploy the pilot as the intervention policy. Review the
labels, expand those action classes with real trajectories, and rerun an
independent held-out evaluation first.
