# ALIGN runner architecture

See [architecture decision](ARCHITECTURE.md) and [quickstart](QUICKSTART.md).
The concurrent implementation is pending Phase 6 local verification; earlier
turn-based and PostgreSQL verification does not establish concurrent correctness.

## Ownership and interfaces

- Runner: environment, independent agent workers, shared Git broker, action
  gate, coordination state and A2 events.
- Training: shared observation/decision contract, training prompt and inference.
- Evaluation: independent conflict/constraint labels, task selection, scoring.
- Quality: regression, Docker/DB integration, profiling and review.

The top-level v1 observation fields in contracts.py remain unchanged.
repository_state now includes workspaces, shared_git, pending_proposals,
in_flight_actions, submitted_agents, and coordination_version. Workspace
snapshots are last observed state, not a live atomic read of another worker.
The queue event records arrival; proposed_action is recorded at decision time,
and arbiter_observation contains exactly the sanitized input sent to the arbiter.
All referenced recent events precede that canonical proposal event.

## Execution and semantics

Agents work independently on isolated local branches. Local bash and messages
pass through the arbiter; publication/fetch are explicit structured tools
mediated by a separate network-disabled Git container. Approved local actions
may run concurrently. Authorization is serialized; no global state lock is held
during model calls or tools. In-flight actions are visible to subsequent decisions.
A changed coordination version can cancel a stale enforced approval; report
authorization_stale separately from model-selected blocks. Monitor is advisory
only, but synchronous inference can change timing. Compare all conditions on
the same concurrent protocol and do not equate monitor timing with no arbiter.

continue dispatches, block withholds, query requests clarification and a new
proposal, mediate sends revision feedback and notifies affected active peers,
escalate stops authorization and tears down workspaces. Query attempts are
bounded by per-agent step and trial limits. Restructure is reserved: its
four-field decision format does not unambiguously specify a new ownership or
sequencing plan. Fail closed until the team agrees on semantics or excludes it.
No action is silently remapped.

A2 v1 action_result records dispatched/cancelled/unknown. The arbiter_decision
event records enforced decisions; monitor uses arbiter_advice. The trusted
broker merges final commit ancestry once; the evaluator checks both features
in an isolated container using pinned CooperBench test logic. Missing tests
remain not_evaluated. Independent conflict labels/constraint audits are not
inferred from arbiter decisions.

## Limits

No unrestricted Git remote, shared mutable filesystem, forced remote writes,
or cross-agent branch mutation. These are environment restrictions, not
arbiter effects. The trial deadline bounds scheduling and local subprocesses;
cleanup has a separate bounded grace period. An already issued external model
request may continue remotely and incur usage after local cancellation. Such
usage is unknown; late responses cannot authorize actions or append events.
Configured-secret redaction is exact-value replacement, not arbitrary secret
discovery. Use disposable hosts and trusted images; Docker is not a complete
security boundary. The event recorder remains in-memory and validates prefixes;
process crashes are not guaranteed to preserve evidence. Normal deadline
termination preserves recorded events and marks incomplete actions honestly.

No hosted infrastructure is needed for scripted verification. Phase 6 must
exercise concurrency, shared Git, evaluator isolation, successful tests,
deadlines, redaction, callable inference mocks and disposable PostgreSQL.
