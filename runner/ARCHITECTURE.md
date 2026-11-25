# Concurrent ALIGN architecture

The pinned CooperBench Mini-SWE-Agent v2 coop runner creates threads, configures
GitConnector/MessagingConnector, and intercepts messaging in DefaultAgent
execute_actions before forwarding shell commands to Docker. Its Git connector
exposes a direct Git server remote (receive-pack); wrapping only Docker exec
would miss message handling and would allow model-generated subprocesses to
reach the remote without a separately enforced operation boundary. Its default
evaluator backend does not enforce the desired network/resource/deadline policy.

Decision: retain ALIGN's hardened contracts, agent/model loop, A2 event writer,
task loading and evaluator algorithms; reuse the pinned task images and trusted
upstream test logic, not the upstream execution or Git server. Use concurrent
agent workers with isolated Docker workspaces and a host-mediated Git broker
in a third isolated container. Agent local Git commands are allowed through
the bash gate. Publication/fetch are explicit proposed tools, not arbitrary
network requests; the broker accepts only fixed actor refs, fast-forward
updates and validated base ancestry. No host mounts, socket, or network access.
Agents can merge fetched branches locally. The broker merges submitted commit
ancestry before evaluation, avoiding duplicate application of integrated work.

Tradeoffs: this is an ALIGN-specific controlled protocol, not a reproduction of
upstream CooperBench concurrency, unrestricted Git transport, or shared mutable
filesystem races. Non-fast-forward and cross-agent ref writes are disallowed by
the environment in every condition, so do not claim measured prevention of
those operations by the arbiter. Bundle transfer and final integration are
additional overhead; validate supported task images and Git versions locally.
Docker is not an absolute security boundary; use disposable hosts without secrets.

## Implementation notes

- Live CLI uses concurrent execution across all arbitration conditions; mock
  alternating is retained as a deterministic reference.
- Arbiter calls are serialized separately from coordination snapshots. Shared
  Git operations compare the published refs at authorization and again under
  the broker lock. A changed ref cancels the attempt with `authorization_stale`.
  Isolated local work can continue when a peer finishes an unrelated action.
  Report stale cancellations separately from the selected intervention.
- Evaluator sandbox uses docker CLI subprocess timeouts, explicit resource limits,
  no network, and identical image ID/base commit checks. Shared final commits
  are merged in the broker and the pinned upstream per-feature test routine is
  called on the merged tree. Legacy independent patches use a private injected
  backend rather than mutating upstream module globals.
- Prompt hash is recorded. The HTTP prompt is aligned with Person 2's contract;
  if the training package is present its canonical prompt is imported. Until
  branches are integrated, a snapshot of the same prompt provides compatibility.
- Query consumes a turn and asks for missing facts/reproposal. Restructure is
  not implemented because its decision schema does not specify a replacement
  assignment; agree on the experimental action space with Person 2.
- Trial expiry prevents new authorization and removes known containers. External
  calls already issued may continue remotely; their usage is not fabricated.
  Cleanup is bounded separately and reports failure. A queued proposal stopped
  before arbitration receives `proposal_cancelled`; a canonical proposal has
  one terminal `action_result`. Completed tool results are preserved even if
  the following workspace refresh fails. Process-crash journaling is out of scope.
