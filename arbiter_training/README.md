# ALIGN arbiter training

CooperBench supplies coding tasks and tests. It does not supply gold arbiter decisions.
This package turns A2 run bundles plus checkpoint labels into supervised decision
examples. Teacher labels remain provisional until independent review. The current
pilot reserves whole repositories for a test split; Person 3 should approve the
final test partition before reporting research results. The current prompt and
completion follow Person 1's `ArbiterObservation` and `ArbiterDecision` envelope.
The public traces do not contain a reliable repository snapshot, so their
`repository_state` is `{}`; the live harness should send its actual snapshot.

## Build the public-trace pilot

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

## Person 1 contract and second teacher pass

`contract.py` accepts the exact top-level observation fields Person 1 sent:
`schema_version`, `task_key`, `shared_goal`, `agent_goals`, `constraints`,
`actor_id`, `proposal_sequence`, `proposed_action`, `recent_events`, and
`repository_state`. The decision has `action`, `reason`, `affected_agents`, and
`evidence`. Public trace terminal calls are mapped into the A2 tool-call shape
with `kind`, `tool_name`, `arguments`, and `content`. Person 1's precise
`ProposedAction` definition is still needed to verify that nested mapping.

The first 35 labels were reconsidered with a fixed decision rubric. One label
was changed and 14 real trace proposals were added. The resulting 49 observed
decisions are still provisional teacher labels. The 34 counterfactuals remain
explicitly tagged. The current combined set has 61 train and 22 validation
decisions, with disjoint repositories. The high-impact observed proposals are
exported in a blind 21-row packet at
`data/person1-contract-v1/flagged-review-packet.jsonl` for later human review.

```powershell
python -m arbiter_training.adjudicate --tasks arbiter_training/data/public-trace-v1/tasks.jsonl --events-dir arbiter_training/data/public-trace-v1 --initial-labels arbiter_training/data/public-trace-v1/teacher-labels-provisional.jsonl --output arbiter_training/data/public-trace-v1/teacher-labels-v2-provisional.jsonl --review-log arbiter_training/data/public-trace-v1/teacher-review-v2.json
python -m arbiter_training.prepare --tasks arbiter_training/data/public-trace-v1/tasks.jsonl --events-dir arbiter_training/data/public-trace-v1 --labels arbiter_training/data/public-trace-v1/teacher-labels-v2-provisional.jsonl --output-dir arbiter_training/data/person1-contract-v1/prepared --history-limit 8
python -m arbiter_training.synthetic --tasks arbiter_training/data/public-trace-v1/tasks.jsonl --prepared-dir arbiter_training/data/person1-contract-v1/prepared --output-dir arbiter_training/data/person1-contract-v1/combined
python -m arbiter_training.audit --prepared-dir arbiter_training/data/person1-contract-v1/combined --output arbiter_training/data/person1-contract-v1/combined/audit.json
python -m arbiter_training.review_packet --prepared-dir arbiter_training/data/person1-contract-v1/combined --audit arbiter_training/data/person1-contract-v1/combined/audit.json --output arbiter_training/data/person1-contract-v1/flagged-review-packet.jsonl
```

Person 3 can create a blind test queue with `review_queue.py --test-limit N`
once new held-out traces exist. The pinned public trajectory release has no
complete two-agent traces for the reserved test repositories, so a final
independent test requires new runner output or another separate source.
See `PERSON3_HANDOFF.md` for the integration call, annotation procedure, and
independent test protocol.

The Person 1 contract-shaped, 60-step Qwen adapter accepted 16/22 provisional
validation decisions, versus 11/22 for pretrained Qwen and 10/22 for
pretrained gpt-oss on the same 22 decisions. It improved `query` to 3/6 and
`mediate` to 2/3, so those cases still need independent evaluation. The
versioned adapter is on Modal at
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
