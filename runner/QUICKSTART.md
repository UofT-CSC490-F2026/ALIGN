# Runner handoff

Run these PowerShell commands from the repository root. The concurrent mock
workflow, loopback model contract, and local PostgreSQL replay are verified.

```powershell
py -m unittest discover -s scripts/tests -p 'test_runner*.py' -v
py -m unittest discover -s scripts/tests -p 'test_*.py' -v
py scripts/normalize_tasks.py custom --scenarios-dir scripts/examples --revision v1 --output-dir normalized/fixture
py scripts/load_tasks.py --input-dir normalized/fixture
$runs = foreach ($condition in 'none','monitor','rule_based','llm') {
    py -m runner.cli --tasks normalized/fixture/tasks.jsonl --output-dir local-runs --protocol concurrent --condition $condition --mock-action block | ConvertFrom-Json
}
$runs | ForEach-Object { py scripts/run_artifacts.py verify --run-dir $_.bundle }
```

Mock trials exercise the artifact and evaluation interfaces with synthetic evidence.
The included A2 scenario is a non-runnable fixture. Each trial writes
`<output>/<run UUID>/bundle/{manifest.json,events.jsonl}`.

## Training and observation export

Choose `--condition none` for unarbitrated trajectories. The runner records the
same version 1 decision-time observation schema with or without an arbiter. To
extract JSONL payloads from a verified bundle without reserializing through
PowerShell's JSON converter:

```powershell
$bundle = $runs[0].bundle
py scripts/run_artifacts.py verify --run-dir $bundle
if ($LASTEXITCODE -ne 0) { throw 'Bundle verification failed' }
py -c "import json,sys; from pathlib import Path; p=Path(sys.argv[1]); rows=map(json.loads,(p/'events.jsonl').read_text(encoding='utf-8').splitlines()); (p.parent/'observations.jsonl').write_text(''.join(json.dumps(e['payload'])+chr(10) for e in rows if e['event_type']=='arbiter_observation'),encoding='utf-8')" $bundle
```

Each payload's `proposal_sequence` points to the proposal in its source run.
Keep the run key with the export for joins. Use independent labels keyed to the
proposal UUID. Keep labels, later action results, evaluator-only metadata and
hidden tests out of model inputs. Split by underlying task or repository before
training so alternate conditions for the same task cannot cross splits.

## Local PostgreSQL replay

Set `ALIGN_DATABASE_URL` to your disposable local database after applying the
migrations in [the A2 setup](../scripts/README.md#load-normalized-tasks-into-postgresql-pr-3).
The concurrent four-condition fixture was verified on disposable PostgreSQL 16
with both migrations applied by Flyway. Set `ALIGN_TEST_DATABASE_URL` to the same
disposable service to enable the three PostgreSQL regression tests.

```powershell
py scripts/load_tasks.py --input-dir normalized/fixture --apply
foreach ($run in $runs) {
    py scripts/process_runs.py --run-dir $run.bundle --apply
    py scripts/process_runs.py --run-dir $run.bundle --apply
    py scripts/evaluate.py --run-key $run.run_key --evaluator-version runner-smoke --apply
}
```

An identical import reports `inserted: false` on replay. These mock runs have
zero evaluated goals. Use `--protocol concurrent --condition llm --mock-action
escalate` to exercise early termination without goal test evidence.
`arbiter_advice` in monitor mode contributes no enforcement decision rows.

## Live configuration and remaining validation

Install optional pinned CooperBench evaluator with
`pip install -r runner/requirements-cooperbench.txt` in an isolated environment.
The requirements file is the sole source of the intended Git commit. Live runs
verify the installed CooperBench VCS origin and commit via PEP 610 direct_url.json
before creating containers or calling models. Mock runs require no installation
and record a null upstream revision.
Prepare the pinned A2 dataset locally. Preload a trusted compatible task image on
a disposable Docker host, with clean /workspace/repo at the normalized base SHA.
Use a dedicated Docker build context for offline dependencies. The following
local setup was verified for pinned dataset revision `b612b1a35af722751454813d9e5a7888f065fc9e`,
Click task 2956 at base `5c1239b0116b66492cdd0848144ab3c78a04495a`.

```powershell
$base = 'akhatua/cooperbench-pallets-click@sha256:497bb01824e4f2bc5a53db80446a61f3edaca5b5e4e5bf3100d423b11e11eb9a'
docker pull $base
$ctx = New-Item -ItemType Directory -Force click-wheelhouse
py -m pip download --no-deps --dest $ctx.FullName flit_core==3.12.0
@("FROM $base", 'COPY flit_core-3.12.0-py3-none-any.whl /opt/align-wheels/',
  'ENV PIP_NO_INDEX=1 PIP_FIND_LINKS=/opt/align-wheels') |
    Set-Content -Encoding ascii (Join-Path $ctx.FullName Dockerfile)
docker build --network=none --pull=false -t align-click2956:offline $ctx.FullName
'{"pallets_click_task:task2956":"align-click2956:offline"}' | Set-Content -Encoding ascii images.json
```

The runner resolves that image once and uses the same immutable ID for agents,
broker and evaluator. Keep the dataset, gold patches and hidden tests out of the
build context and agent workspaces. Use a disposable Docker host. Phase 6
checked the real Docker isolation settings and cleanup on Click task 2956.
Other images may require different trusted offline dependencies.

Evaluator-only positive controls for Click 2956 used the pinned feature 1 and
8 reference patches. Feature 1 alone passed its suite and failed feature 8's.
The combined solutions passed both. When the second suite was missing or timed
out, the first pass was retained and feature 8 remained `not_evaluated`.
These controls verify the test pipeline and are separate from agent scores.
In a separate scripted run, one agent published a commit and its peer fetched,
merged, and extended it. The broker emitted each change once. Applying the
known solutions to that merged result in the trusted evaluator passed both
suites.

```powershell
$common = @('--live', '--tasks', 'normalized/heldout/tasks.jsonl',
    '--task-key', 'YOUR_NORMALIZED_TASK_KEY', '--limit', '1', '--seed', '42',
    '--max-steps', '20', '--output-dir', 'real-runs', '--images', 'images.json',
    '--dataset-dir', 'PATH_TO_PINNED_DATASET', '--agent-endpoint',
    'https://AGENT_HOST/v1/chat/completions', '--agent-model', 'AGENT_MODEL')
$pretrained = @('--arbiter-endpoint', 'https://BASE_HOST/v1/chat/completions',
    '--arbiter-model', 'BASE_MODEL', '--training', 'none', '--checkpoint', 'BASE_ID')
$fineTuned = @('--arbiter-endpoint', 'https://SFT_HOST/v1/chat/completions',
    '--arbiter-model', 'SFT_MODEL', '--training', 'sft', '--checkpoint', 'SFT_ID')

py -m runner.cli @common --condition none          # Also generates training trajectories
py -m runner.cli @common --condition monitor @pretrained
py -m runner.cli @common --condition rule_based
py -m runner.cli @common --condition llm @pretrained
py -m runner.cli @common --condition llm @fineTuned
```

For an evaluation matrix, reuse the same task keys, seed, agent model, and
limits. Use `none` and `rule_based` with the base command. For `monitor` and
`llm`, also supply `--arbiter-endpoint` and `--arbiter-model`. Use
`--training none --checkpoint BASE_ID` for a pretrained `llm` run and
`--training sft --checkpoint SFT_ID` for a fine-tuned `llm` run. The checkpoint
flag records provenance. The endpoint and model arguments select the served
checkpoint. Keys, if needed, are supplied via
ALIGN_AGENT_API_KEY and ALIGN_ARBITER_API_KEY. Never put keys in URLs/config.
Use `--task-key` (repeatable) to select tasks, `--limit` to cap the selection and
`--max-steps` to bound proposal turns per agent, including submission. The seed is a trial identifier only;
the remote model may not implement seeded generation. No automatic API retries.

## Contracts and limitations

The prior verified implementation used an alternating loop. The new live path
uses concurrent separate workspaces and host-mediated shared Git. Query and
mediation withhold the proposal and request a revision. Restructure remains
unsupported and fails closed.
Monitor mode logs arbiter_advice (not arbiter_decision) to avoid treating advisory
blocks as enforced in A2. Person 3 must score advisory events separately.
Model paths are hints, not verified command effects. No complete constraint audit,
conflict gold labels, dollar-cost accounting, or inference server is implemented.
Token counts are recorded when all attempted live calls report usage, including
on failed runs with complete usage. Capture decision latency from the decision
or advice payload, runtime from A2 evaluation, and failures from `run_end` and
`arbiter_error`. For tasks with no
declared constraints, constraint satisfaction is vacuous when the evaluator succeeds;
nonempty constraints still need an independent audit.
Unknown evidence stays unknown. Person 3 must supply independent conflict/constraint
evidence and metrics; Person 2 must serve the actual trained checkpoint.
The adapter sends `model`, `messages`, `temperature` and `max_tokens`. It expects
`finish_reason: stop` and a JSON object in `choices[0].message.content`. Loopback
tests cover the wire contract and bearer auth. Confirm the GPT-OSS and Qwen
endpoints support these fields, and that reasoning content is separate from the
final JSON. HTTP timeout limits individual socket waits. The trial deadline
stops local authorization even if an external request returns late. Real endpoint
validation remains outstanding.

## Concurrent protocol

The live commands above now select `align-concurrent-shared-git-v1` for every
condition. Use `--protocol concurrent` to exercise the concurrent scheduler
with free mocks; default mocks remain alternating for deterministic regressions.
Live alternating mode is disallowed to avoid accidentally mixing protocols.
Use `--trial-timeout 600` (seconds), `--tool-timeout`, `--model-timeout` and
`--max-steps` consistently across conditions.

A live task needs the same preloaded immutable image for the agents, Git broker
and evaluator. The new evaluator enforces network isolation and resource limits
and streams patches into the container; no upstream default Docker sandbox is
used. Ensure all test build dependencies are already in the trusted image
(the earlier Click smoke needed flit_core). No network install is permitted.
The Git broker supports structured `git_publish` and `git_fetch` tools;
agents can inspect or merge `refs/remotes/align/agent1` and
`refs/remotes/align/agent2` locally. It merges actual final commit ancestry
before running both trusted feature tests. No force-push is allowed.
See ARCHITECTURE.md for design tradeoffs.

Optional Person 2 backend (only after training branch is available and paid
inference explicitly authorized): add `--arbiter-backend modal --arbiter-model qwen`
instead of HTTP arbiter endpoint; for SFT use `--training sft --checkpoint`
with the actual adapter path. The runner calls ModalArbiter.decide directly
inside Person 2's app context. No Modal usage was launched in this implementation
pass. Callable arbiter token totals remain unknown unless its interface provides
complete usage; do not compare unknown totals as zeros.

Known configured secret values in ALIGN_AGENT_API_KEY, ALIGN_ARBITER_API_KEY,
ALIGN_DATABASE_URL and env names listed in ALIGN_TRACE_SECRET_ENV (comma-separated)
are redacted before event persistence. Extra credentials must be configured
explicitly; this is not a general credential scanner.

The endpoint commands are configuration templates. No paid inference, live
Modal workload, or held-out experiment was run during this verification.
