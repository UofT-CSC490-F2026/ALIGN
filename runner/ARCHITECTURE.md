# Concurrent ALIGN architecture (implementation pending Phase 6 validation)

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
