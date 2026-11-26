# Arbiter training handoff

## 2026-11-26 runner review update

The current code in PR #51 now uses the merged runner's recorded
`arbiter_observation` for labeled runner proposals. An offline merged-runner
trial verified exact observation reuse, including repository state. Public
CooperBench traces still require reconstruction and have empty repository
state. Task keys now match A2 normalization, synthetic proposals use runner
tool shapes, and inference rejects unknown affected agents and reserved
`restructure` decisions. The model context limit is checked before inference
or training. The rebuilt ignored pilot data are under
`data/runner-aligned-v2/combined/` (61 train, 22 validation); all 83 examples
fit the pinned Qwen 8,192-token training limit, with a maximum of 6,489.
The older checkpoint below has not been retrained on this revision. It should
be reported as a historical public-trace pilot, not a runner-trained model.
An additional live Modal smoke on 2026-11-26 passed a recorded concurrent-runner
observation with `workspaces`, `shared_git`, `pending_proposals`, and
`in_flight_actions` through the existing Qwen adapter. It returned a valid
`continue` for a safe `echo fixture` proposal. The Modal run was
`ap-a8DwbnMlL4Q9TJKn89USJV`. This verifies model loading and one live-format
inference call, not detection quality or full end-to-end outcomes.


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
        adapter_path="/checkpoints/runner-contract-reviewed-60steps-20261125/adapter",
    )
    decision = arbiter.decide(observation)
```

Pass `model="qwen"` with no adapter for the pretrained Qwen baseline or
`model="gpt-oss"` for the pretrained gpt-oss baseline. The nested
`ProposedAction` now matches PR #50's `runner/contracts.py`: `action: dict`
and `files_touched: list[str]`. A real live observation sample is still needed
to check the richer action payload and repository snapshot. Public traces have
no trustworthy repository snapshot; their `repository_state` is `{}` and
`files_touched=[]`. Live observations should include the actual bounded state.
`ModalArbiter` raises on invalid model output by default. An explicit
`invalid_policy="query"` instead returns a query with an
`invalid_model_output` evidence marker. The stacked runner PR #58 handles
`query` and `mediate`; `restructure` remains unsupported for enforcement and
fails closed. The earlier ephemeral `class-smoke` run returned a valid
four-field decision through this interface on one validation observation.

## PR #50 schema alignment (2026-11-25)

The current preparation code uses the exact nested proposal fields in PR #50,
`action` and `files_touched`. The corrected-format provisional set is under
`data/runner-contract-v1/combined/`: 61 training and 22 validation decisions,
with no repository overlap. Its train and validation SHA256 values are
`d8c46bc7c92aae33b3dfa3278430505e2fdbd8d8be7b0ef1cba878fd77431e70`
and `30048795854ebda7353a0b5a041b8b2deea7df1055c3ad31c24cc5cb40b28c7f`.
On those provisional validation labels, pretrained Qwen accepted 10/22 with
20 valid outputs; pretrained gpt-oss accepted 7/22 with 21 valid outputs.
The 21-item blind human review packet and numbered worksheet are in
`data/runner-contract-v1/`. Jinhao completed all 21 reviews. Their annotations
are tracked in `reviews/human-labels-20261125.jsonl`; the reviewed build is in
`data/runner-contract-v1-reviewed/combined/`. It still has 61 train and 22
validation decisions with disjoint repositories, now including 13 human
training labels and 8 human validation labels. The train and validation SHA256
values are `292b81f163d39dca2564cfd93a149d920b368b45fc4cf9185024c474b751fd85`
and `a7afc34c103124060adb4ff64dd51e87c67ef6c180949632bbf82939da291685`.
The reviewed data audit has no warnings. On the reviewed validation set,
pretrained Qwen accepted 12/22 with 20 valid outputs; pretrained gpt-oss
accepted 7/22 with 21 valid outputs. These remain pilot validation scores,
not held-out test results. The new runner-format 60-step BF16 LoRA adapter is
at `/checkpoints/runner-contract-reviewed-60steps-20261125/adapter` on the
`align-arbiter-checkpoints` Modal volume. It accepted 17/22 decisions with
22 valid outputs; 10/13 observed trace decisions and 7/9 synthetic cases.
It accepted 5/8 human-reviewed validation observations. Its best validation
loss was 0.5691. Metadata and a compact score breakdown are tracked in
`results/runner-contract-reviewed-60steps-20261125-metadata.json` and
`results/reviewed-validation-20261125.json`. A `class-smoke` run completed
through `ModalArbiter.decide` with the new adapter on one validation
observation.

The adapter accepted only 1/3 `block` decisions versus 3/3 for pretrained
Qwen. Both adapter misses were `query` decisions, which withhold the proposal
in the current runner but do not match the reviewed block labels. Do not use
this pilot score to claim intervention safety. Person 3 needs a separate
held-out test and live-run outcome analysis. The adapter below is archived
from the older proposal format.

## Training data and provenance

The pinned CooperBench task revision is
`b612b1a35af722751454813d9e5a7888f065fc9e`. The pinned public trajectory
revision is `bf6cc8469a7b853f70e8f043af145b5053952293`. The 10 imported
two-agent runs contain 553 proposed actions. A second teacher pass has 49
observed decision labels, and 34 additional counterfactuals are marked
`synthetic_counterfactual`. Jinhao reviewed 21 observed decisions; 28 observed
teacher labels and all 34 synthetic counterfactual labels remain provisional.

The archived top-level-only data are in `data/person1-contract-v1/combined/`:
61 training and 22 validation decisions. The training action counts are 35
`continue`, 10 `query`, 8 `mediate`, and 8 `block`; validation has 10, 6, 3,
and 3 respectively. Repositories do not cross the split. The audit report is
`audit.json`. The second-pass decisions and reasons are in
`data/public-trace-v1/teacher-review-v2.json`.

The archived 60-step BF16 LoRA adapter is on Modal volume `align-arbiter-checkpoints`
at `/person1-contract-v1-60steps-20261124/adapter` and locally under
`data/checkpoints/person1-contract-v1-60steps-20261124/adapter-files/adapter`.
The train and validation SHA256 values are respectively
`535c37b42758948e46f89164250c9d5a38a4f79f4e8482b66c1a4ce8dfa6691c`
and `21006a63666c10f69e0d94e8622c81a7b380849e4744c6aac67ef19003073a62`.
All 83 prompt-plus-completion records are below the 8192-token training limit
(maximum 6457 tokens with the pinned Qwen tokenizer). The best validation loss
is 0.6122. In provisional validation decisions,
pretrained Qwen accepted 11/22, pretrained gpt-oss accepted 10/22, and the
fine-tuned Qwen adapter accepted 16/22. Its nested `ProposedAction` differs
from PR #50; this score does not validate runner-format inputs. The adapter got 3/6 `query`, 2/3
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
modal run arbiter_training/modal_app.py --mode compare --validation-file arbiter_training/data/heldout/test.jsonl --models qwen --adapter-path /checkpoints/runner-contract-reviewed-60steps-20261125/adapter --output-file arbiter_training/data/heldout/adapter-comparison.json
```

`prepare_test.py` rejects non-test tasks and provisional teacher labels.

The training scripts reproduce the repository split and labels, but the final
held-out reference labels and end-to-end results belong to Person 3. The
checkpoint is a research candidate until those results are available.
