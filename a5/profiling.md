# Part 1. Profiling five ALIGN functions

We selected functions that define benchmark tasks, decide whether a proposed action can proceed, record and validate run events, and reconstruct conflicts for evaluation. Together they cover both the execution path and the evidence used to assess the arbiter.

We profiled application commit `e3dcfb3` locally on macOS arm64 with Python 3.11.5, following the [Python profiler documentation](https://docs.python.org/3/library/profile.html). We used `cProfile` around repeated calls and sorted the resulting `pstats` table by cumulative time. Means below divide each function's cumulative time by its call count. Inputs were prepared before profiling. These are profiled times on controlled local inputs. The gate used a deterministic local arbiter.

| Function and reason for selection | Work per call | Calls | Cumulative time | Mean per call |
|---|---|---:|---:|---:|
| `validate_events` rejects malformed evidence before a run bundle can be used. | Validate the repository's eight-event example run. | 1,000 | 118 ms | 118 μs |
| `normalized_cooperbench` determines which feature pairs become experimental tasks and assigns each agent its goal. | Build all 66 pairs from one synthetic task with 12 features in a temporary file tree. | 10 | 349 ms | 34.9 ms |
| `ArbitrationGate.evaluate` checks the arbiter response before the runner acts on a proposal. Its overhead is paid on each arbitrated action. | Evaluate one proposal with six history events and a fixed local block decision. | 1,000 | 382 ms | 382 μs |
| `EventWriter.emit` records agent activity in the ordered history used by the arbiter and evaluator. | Append a proposal as each of 20 traces grows to 100 events. | 1,980 | 2,411 ms | 1.22 ms |
| `conflict_evidence` matches conflict resolutions to their onset and rejects invalid links. Those links determine which episodes count as resolved. | Scan the eight-event fixture containing one conflict onset and one resolution. | 10,000 | 71 ms | 7.1 μs |

## Findings and proposed improvements

1. **Run validation.** UUID validation took 48 of 118 ms. All eight events share one run ID. Checking its format on the first event and comparing later IDs with that validated value would reduce UUID parses from 16 to 9 per fixture while retaining the per-event identity checks.
2. **Task normalization.** Ten passes made 1,980 text reads, consuming 92 ms. Each pair rereads the same setup script and two feature descriptions. Reading the setup script once per task and each description once per feature would reduce reads from 198 to 13 per pass on this workload. Keep the cache local to one normalization call so later runs see file changes.
3. **Arbitration.** Deep copying accounted for 357 of 382 ms (93%) with the local arbiter. The copies protect the runner's observation and the returned decision from mutation. A candidate for benchmarking is to prepare an immutable history snapshot once and share it across decisions while continuing to isolate mutable proposal fields. This requires a compatible arbiter interface and a test that agent or arbiter mutations cannot alter recorded history. Live inference latency was outside the workload.
4. **Event recording.** We built 20 traces at each length, starting each with a run-start event before profiling and then appending identical proposals. At final lengths of 10, 50, and 100 events, the mean time per append was 0.12, 0.43, and 1.22 ms, respectively. The writer validates the entire trace on every append while holding its lock. At 100 events, validation accounted for 2.281 of 2.411 seconds. We would validate each new event against retained state, including seen event IDs and proposal links, and keep full-trace validation when writing the bundle. A regression test should verify that duplicate IDs and invalid links are still rejected.
5. **Conflict evaluation.** The function spent 60 of 71 ms in its own body. Only two of the eight events concern conflicts, yet it fetches sequence and payload fields for every event. An early event-type filter could skip those lookups for ordinary events while preserving the order and validation of conflict events. At 7.1 μs per fixture this is a lower priority than the repeated task-file reads. A longer, conflict-sparse trace would be the appropriate follow-up workload.

These are proposed changes. The measurements describe the existing implementation.

From the repository root, run `python3 -m a5.profile_validate_events` and `python3 -m a5.profile_pipeline`. The saved profiler tables are in [the validation profile](validate_events_profile.txt) and [the four additional profiles](pipeline_profiles.txt).
