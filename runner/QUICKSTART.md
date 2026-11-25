# Runner handoff

All runner branches are draft/unmerged. Hosted CI intentionally skipped using commit
messages containing [skip ci]; run the tests locally from the repository root.

```bash
python -m unittest discover -s scripts/tests -p 'test_runner*.py' -v
python scripts/normalize_tasks.py custom --scenarios-dir scripts/examples --revision v1 --output-dir normalized/fixture
python -m runner.cli --tasks normalized/fixture/tasks.jsonl --output-dir local-runs --condition llm --mock-action block
```

Mock trials are NOT real evaluation results. The included A2 scenario is a
non-runnable fixture. Each trial writes <output>/<run UUID>/bundle/{manifest.json,events.jsonl}.
Use only real traces for experiments; import with the existing A2 commands.

## Live smoke test (explicit opt-in; may incur model charges)

Install optional pinned CooperBench evaluator with
`pip install -r runner/requirements-cooperbench.txt` in an isolated environment.
The requirements file is the sole source of the intended Git commit. Live runs
verify the installed CooperBench VCS origin and commit via PEP 610 direct_url.json
before creating containers or calling models. Mock runs require no installation
and record a null upstream revision.
Prepare the pinned A2 dataset locally. Preload a trusted compatible task image on
a disposable Docker host, with clean /workspace/repo at the normalized base SHA.
Create an image map such as {"pallets_click_task:task1":"YOUR_PRELOADED_IMAGE"}.
Do not mount the dataset, host directory, Docker socket or credentials into agent
containers. Verify image contents and upstream evaluator isolation before running;
the upstream evaluator is separate and runs generated code, so use a disposable
host without secrets. The installed upstream evaluator resolves its own image name;
ensure it resolves to the same trusted base image and revision.

```bash
python -m runner.cli --live --tasks normalized/heldout/tasks.jsonl --output-dir real-runs --condition none --limit 1 --images images.json --dataset-dir /path/to/dataset --agent-endpoint https://YOUR_HOST/v1/chat/completions --agent-model YOUR_MODEL
```

For LLM conditions add --arbiter-endpoint and --arbiter-model; use --training sft
--checkpoint ID to distinguish adapters. Keys, if needed, are supplied via
ALIGN_AGENT_API_KEY and ALIGN_ARBITER_API_KEY. Never put keys in URLs/config.
Use --max-steps and --limit to bound calls. The seed is a trial identifier only;
the remote model may not implement seeded generation. No automatic API retries.

## Contracts and limitations

This implementation uses an ALIGN-specific alternating, separate-workspace loop,
explicit bash/message tools and CooperBench's pinned merge/test evaluator; it is
not a reproduction of upstream concurrent Mini-SWE-Agent or shared Git protocol.
Mediation cancels the current action and requests revision; restructure/query are
not enforced and produce a failed trial rather than silently executing.
Monitor mode logs arbiter_advice (not arbiter_decision) to avoid treating advisory
blocks as enforced in A2. Person 3 must score advisory events separately.
Model paths are hints, not verified command effects. No complete constraint audit,
conflict gold labels, cost accounting, or inference server is implemented.
Unknown evidence stays unknown. Person 3 must supply independent conflict/constraint
evidence and metrics; Person 2 must serve the actual trained checkpoint.
The OpenAI-compatible adapter expects strict JSON content; test GPT-OSS/Qwen
endpoint chat templates before experiments. HTTP timeout is per request, not a
global process deadline. No live tests have been run in this development pass.
