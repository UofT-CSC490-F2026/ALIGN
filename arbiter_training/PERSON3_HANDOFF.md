# Arbiter training handoff

## Interface for Person 1

`contract.py` accepts the top-level `ArbiterObservation` fields Person 1 sent and
requires an `ArbiterDecision` with `action`, `reason`, `affected_agents`, and
`evidence`. `ModalArbiter.decide(observation)` returns that decision; its
`decide_many` method batches offline calls. `ArbiterModel` in `modal_app.py`
keeps the selected model loaded across calls within an ephemeral Modal app.
No endpoint has been deployed.

```python
from arbiter_training.inference import ModalArbiter
from arbiter_training.modal_app import app

with app.run():
    arbiter = ModalArbiter(
        model="qwen",
        adapter_path="/checkpoints/person1-contract-v1-60steps-20261124/adapter",
    )
    decision = arbiter.decide(observation)
```

Pass `model="qwen"` with no adapter for the pretrained Qwen baseline or
`model="gpt-oss"` for the pretrained gpt-oss baseline. The nested
`ProposedAction` type and a real observation sample from Person 1 still need to
be checked against the public-trace mapping. Public traces have no trustworthy
repository snapshot; their `repository_state` is `{}`. Live observations should
include the actual repository state.
`ModalArbiter` returns `query` with an `invalid_model_output` evidence marker
if a model response does not parse; set `invalid_policy="raise"` during offline
scoring so malformed outputs remain visible as errors.
The ephemeral `class-smoke` run returned a valid four-field decision through
this interface on one validation observation.

## Training data and provenance

The pinned CooperBench task revision is
`b612b1a35af722751454813d9e5a7888f065fc9e`. The pinned public trajectory
revision is `bf6cc8469a7b853f70e8f043af145b5053952293`. The 10 imported
two-agent runs contain 553 proposed actions. A second teacher pass has 49
observed decision labels, and 34 additional counterfactuals are marked
`synthetic_counterfactual`. All 83 labels remain provisional; no independent
human reviewer has approved them.

The current Person 1 shaped data are in `data/person1-contract-v1/combined/`:
61 training and 22 validation decisions. The training action counts are 35
`continue`, 10 `query`, 8 `mediate`, and 8 `block`; validation has 10, 6, 3,
and 3 respectively. Repositories do not cross the split. The audit report is
`audit.json`. The second-pass decisions and reasons are in
`data/public-trace-v1/teacher-review-v2.json`.

The 60-step BF16 LoRA adapter is on Modal volume `align-arbiter-checkpoints`
at `/person1-contract-v1-60steps-20261124/adapter` and locally under
`data/checkpoints/person1-contract-v1-60steps-20261124/adapter-files/adapter`.
The train and validation SHA256 values are respectively
`535c37b42758948e46f89164250c9d5a38a4f79f4e8482b66c1a4ce8dfa6691c`
and `21006a63666c10f69e0d94e8622c81a7b380849e4744c6aac67ef19003073a62`.
All 83 prompt-plus-completion records are below the 8192-token training limit
(maximum 6457 tokens with the pinned Qwen tokenizer). The best validation loss
is 0.6122. In provisional validation decisions,
pretrained Qwen accepted 11/22, pretrained gpt-oss accepted 10/22, and the
fine-tuned Qwen adapter accepted 16/22. The adapter got 3/6 `query`, 2/3
`mediate`, 3/3 `block`, and 8/10 `continue` decisions. Observed public-trace
decisions were 8/13; synthetic counterfactuals were 8/9. These are pilot
selection results, not independent test scores.
There are no labeled `restructure` or `escalate` examples in this pilot, so
neither action has measured performance. `query` and `mediate` remain weak on
the small provisional set. The model has not been tested on live repository
snapshots, because public trace observations use an empty `repository_state`.

The blind packet `data/person1-contract-v1/flagged-review-packet.jsonl`
contains 21 high-impact observed proposals with blank annotation fields.
Reviewers should fill `expected_actions`, `primary_action`, `rationale`, and
`reviewer`, then export only these fields and `event_uuid` to a human label
JSONL. The packet omits teacher decisions and model predictions. Synthetic
cases should be reviewed separately from observed events.

## Independent test plan for Person 3

The provisional CooperBench Lite split reserves all 27 task pairs from
`pallets_click_task` and `typst_task` as test repositories. Neither repo appears
in the training or validation examples. The current pinned public trajectory
release has **zero complete two-agent traces** for these two repositories. Use
Person 1's live runner or another separate source to collect held-out
checkpoints; do not derive final test scores from this pilot validation set.

1. Freeze the test task keys and repository split before observing candidate
   model predictions. Keep all checkpoints from a task together.
2. Annotate a blind mix of safe, uncertain, conflicting, and destructive
   proposals using only each event's prefix. Allow more than one acceptable
   action when the distinction is genuinely ambiguous. Use two reviewers for
   high-impact interventions and resolve disagreements before scoring.
3. Score valid JSON, accepted action, per-action recall and precision, and
   source-specific results. Review whether `reason`, `affected_agents`, and
   `evidence` are grounded in the visible prefix. Report observed trajectories
   and counterfactuals separately. Treat `query` and `mediate` as essential
   rather than allowing overall accuracy to hide their misses.
4. In end-to-end runs, compare no arbiter, monitoring only, rule-based,
   pretrained, and fine-tuned conditions on the same tasks and seeds. Measure
   completed goals, conflicts and resolution, harmful actions, latency, and
   cost. Keep the test set outside checkpoint selection and prompt revision.

Once the held-out runs and human labels exist, prepare a separate test file
without mixing it into training or validation:

```powershell
python -m arbiter_training.prepare_test --tasks PATH_TO_TEST_TASKS_JSONL --events-dir PATH_TO_TEST_BUNDLES --labels PATH_TO_HUMAN_LABELS_JSONL --output arbiter_training/data/heldout/test.jsonl
modal run arbiter_training/modal_app.py --mode compare --validation-file arbiter_training/data/heldout/test.jsonl --models qwen,gpt-oss --output-file arbiter_training/data/heldout/base-comparison.json
modal run arbiter_training/modal_app.py --mode compare --validation-file arbiter_training/data/heldout/test.jsonl --models qwen --adapter-path /checkpoints/person1-contract-v1-60steps-20261124/adapter --output-file arbiter_training/data/heldout/adapter-comparison.json
```

`prepare_test.py` rejects non-test tasks and provisional teacher labels.

The training scripts reproduce the repository split and labels, but the final
held-out reference labels and end-to-end results belong to Person 3. The
checkpoint is a research candidate until those results are available.
