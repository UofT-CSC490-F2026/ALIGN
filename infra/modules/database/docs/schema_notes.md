# ALIGN initial schema notes

Matches the architectural agreement: 10 tables (`tasks`, `task_goals`, `runs`, `events`, `decision_log`, `conflict_episodes`, `evaluations`, `goal_results`, `decision_labels`, `action_attempts`) and 2 views.

### Semantic contracts

1. **Task:** one immutable version of a multi-agent scenario, *not* an individual goal. `task_key` should change when a material scenario definition changes; `external_task_id` preserves source IDs.
2. **Task goal:** atomic, independently testable requirement. `shared` goals have no single agent owner. `is_critical` marks invariants the evaluation must not overlook.
3. **Run:** one execution of a fixed task version under one arbiter setup. Same `trial_key` and seed permit but do not guarantee comparable matched baseline/treatment trials.
4. **Event:** one meaningful activity/observation in a run. `sequence_number` orders that run; identity IDs and timestamps do not. `occurred_at` describes when it happened; `observed_at` describes when ALIGN could see it.
5. **Decision:** a structured record associated with an `events.event_type='arbiter_decision'`. Every monitoring decision includes `continue` when the arbiter explicitly chooses no intervention; not every event gets a decision.
6. **Conflict episode:** a conflict identified in a trace. Records can be populated after the run. A missed conflict exists without any decisions. Nullable `decision_log.conflict_id` references one primary conflict and is optional.
7. **Evaluation:** one current overall evaluation per run. `outcome_correct` reflects the task's *expected outcome*; escalation can be correct even when coding goals remain unmet.
8. **Goal result:** each row evaluates a goal *for a run*. `passed / failed / not_evaluated`; do not assume that missing or unevaluated results are failures.
9. **Decision label:** expected action set at an observed event/checkpoint; may be preliminary. Only one reviewed label per checkpoint may be designated `is_reference` for scoring.
10. **Action attempt:** a proposed tool/action tied to a run's event, with optional safety judgment and actual dispatch outcome. `blocked` means prevented; `dispatched` means invoked (not necessarily successful).

### Why composite foreign keys?

The simple `FK(goal_id)` + `FK(run_id)` pattern permits mistakenly scoring a goal from Task B in a run of Task A. The extra `goal_results.task_id` lets SQL enforce both `(goal_id, task_id) → task_goals` and `(run_id, task_id) → runs`. The same principle protects events and conflict references in `decision_log`.

### Outstanding production questions

- The schema checks that inserted decision details correspond to arbiter-decision events, but does not require *every* such event to have details at transaction commit; the writer should insert both in one transaction.
- If source events arrive concurrently, allocate per-run `sequence_number` through a single sequencer or a transactional lock, not `MAX(sequence_number)+1` without synchronization.
- Complete event histories, diffs, and tool outputs can be large. Keep structured metadata in `events.payload` and large artifacts in versioned object storage referenced by `artifact_uri`.
- `decision_labels` is *not* a substitute for independent ground truth. Do not use synthetic/provisional labels as human-reviewed intervention accuracy metrics.
- The current `evaluations` PK (`run_id`) means a re-evaluation updates/replaces the result; if we require an audit history of evaluator revisions, add versioned evaluation records later.
- `v_task_comparison` shows means grouped by task/condition, not randomized or paired causal effect estimates. Match baseline and treatment by task, trial, seed and agent configuration for formal analysis.
- This is for small-scale research, so integer identity keys, standard B-tree indexes, no partitioning/streaming queue and no agents table yet.

### Evaluation and missing-data semantics

- Reviewed unsafe labels in `action_attempts` are counted as attempted. Blocked
  and dispatched are disjoint execution outcomes; unknown/cancelled outcomes remain
  visible separately. Unreviewed or unknown safety labels are not counted as safe.
- A scored checkpoint has exactly one accepted `decision_labels.is_reference` label
  and an arbiter decision linked via `trigger_event_id`. Without a decision the
  checkpoint is unobserved, not a missed intervention. No-arbiter baselines do not
  contribute to decision-quality scoring.
- Match the earliest decision by event sequence if more than one references the same
  checkpoint. `continue` with a non-continue-only reference is missed;
  non-continue against a continue-only reference is false; an unacceptable
  intervention against a reference expecting some intervention is wrong-type.
- `v_task_comparison` reports every run, including crashes and in-flight attempts.
  The terminal operational rate treats completed/failed/cancelled as terminal, but
  distinguishes unknown outcome labels using `terminal_unlabeled_runs`.
  The evaluated-label-only rate has a separate denominator.
- Proposed tool calls, task safety policies and allowed-file scopes must be recorded
  consistently by the runner (using `action_attempts` and `tasks.constraints`).
  The database does not infer prohibited behavior solely from the tool-call name.
