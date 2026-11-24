# ALIGN runner architecture (initial proposal)

Base: final-project / A2. This PR introduces contracts and local event recording only;
it does not execute models or containers yet.

## Ownership and interfaces

- Runner (Person 1): environment, agent loop, context, action gate, execution and event traces.
- Training (Person 2): offline decision examples and model implementing `Arbiter.decide`.
- Evaluation (Person 3): independent labels, test runner and experiment matrix.
- A5 (Person 4): system coverage, reviews and integration checks.

`runner/contracts.py` is the proposed shared boundary. Observations contain only
decision-time evidence: no future results, gold patches, hidden tests, expected labels,
or unrestricted database rows. `repository_state` and `recent_events` must be
bounded and sanitized. Record the actual observation used by the model on a separate
system event linked to the proposal, so training can replay it exactly. For `none`
runs also record the same observation, without invoking a model.

## Execution flow

1. Load normalized task and project an agent-visible goal/context.
2. Prepare isolated workspace(s) from pinned base commit.
3. Obtain a model-proposed tool action; record proposed_action.
4. Build and record the observation from the prior state.
5. In none mode dispatch; monitor mode records a decision but does not enforce it;
   other conditions validate and enforce a decision before side effects.
6. Record arbiter_decision (if applicable) and action_result, referencing proposal sequence.
7. Return tool output or intervention feedback to the agent. Bound steps and retries.
8. Trusted evaluator runs per-goal tests and emits test_result and run_end.
9. Stage with scripts/run_artifacts.py, import with process_runs.py, evaluate with evaluate.py.

## Reuse CooperBench

Candidate upstream: https://github.com/cooperbench/CooperBench/tree/63b9d44d9f39a02fccf5bf0052db48a917a011fd
(pinned for investigation; compatibility with A2's dataset revision is not verified).

Mini-SWE-Agent v2 has an explicit action dispatch seam:
src/cooperbench/agents/mini_swe_agent_v2/agents/default.py, DefaultAgent.execute_actions.
It parses send_message before calling env.execute, so wrapping Docker execution alone
does not intercept every action and may also capture trusted internal commands.
Investigate a minimal subclass/adapter with a per-action hook before both side
effects, and retain upstream environment, prompts, communication, and evaluation where feasible.
Upstream coop uses separate agent workspaces with optional shared Git remote, not a
single shared filesystem. Preserve that behavior for CooperBench comparisons.
Do not serialize blocking peer waits while holding a global action lock.
Start with a deterministic fake backend, then one real task.

## Action semantics and compatibility

Use existing lowercase action names and conditions. Initially enforce continue, block,
mediate (cancel original proposal and send revision feedback), and optionally escalate
(stop). Query/restructure are reserved until implemented; never silently dispatch
unsupported decisions. Monitor-mode proposed blocks must be represented separately from
enforced blocks: current process_runs.py assumes arbiter_decision action=block is enforced
and rejects a dispatched result. Resolve this contract with evaluation before monitor runs.
Existing action_result accepts dispatched/cancelled/unknown, not blocked. A block with
cancelled is imported as blocked. Dispatch does not imply tool success; log exit code separately.
Do not treat uncertain files_touched as authoritative: shell commands can obscure paths.
Enforce sandbox restrictions independently of any model or heuristic, in every condition.

Keep arbiter_condition=llm for pretrained and SFT; distinguish in arbiter_config.
Record model/revision, adapter ID, prompt version and reasoning configuration.
Record unknown metrics as unknown, not success or zero. Conflict evidence must originate
from independent observations, not an arbiter's own predictions.

## Proposed next PRs

- Environment and agent adapter: investigate actual CooperBench task/image compatibility,
  container isolation, trusted test setup, and model provider. No host shell execution
  of model-generated commands; no host credentials, Docker socket or hidden tests in agents.
- Orchestration: bounded two-agent execution, decision-time observations, raw baseline traces.
- Intervention gate: actual cancellation/feedback, timeout and invalid output handling,
  monitoring compatibility, tests with fake environments.
- Integration: model adapter, trusted tests, real trace through A2 import/evaluation.

No cloud deployment is required for this core. Run from repository root:

```bash
python -m unittest discover -s scripts/tests -p 'test_runner_core.py' -v
```

Known limitations: local in-memory recorder (no durable streaming); prefix validation
is quadratic and should be profiled; no automatic redaction; no runtime enforcement,
agent execution, upstream dependency, or model training is implemented yet.
