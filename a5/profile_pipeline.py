"""Run from the repository root: python3 -m a5.profile_pipeline"""
import cProfile
import json
import platform
import pstats
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

from runner.arbitration import ArbitrationGate, ConstantArbiter

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))  # These CLIs import their sibling modules.
from evaluate import conflict_evidence
from normalize_tasks import normalized_cooperbench
from process_runs import expected_rows


def measure(function, args, calls, workload):
    print(f"\n{function.__qualname__}: {calls} calls; {workload}")
    profiler = cProfile.Profile()
    with profiler:
        for _ in range(calls):
            function(*args)
    pstats.Stats(profiler).strip_dirs().sort_stats("cumulative").print_stats(12)


def main():
    print(f"Python {platform.python_version()} on {platform.system()} {platform.machine()}")
    events = [json.loads(line) for line in (SCRIPTS / "examples/example_run.events.jsonl")
              .read_text().splitlines()]
    with TemporaryDirectory() as directory:
        dataset = Path(directory)
        task = dataset / "pallets_click_task/task2068"
        task.mkdir(parents=True)
        for name in ("Dockerfile", "runner.sh", "run_tests.sh", "combined.patch"):
            (task / name).touch()
        (task / "setup.sh").write_text(f'BASE_COMMIT="{"a" * 40}"\n')
        for number in range(1, 13):
            feature = task / f"feature{number}"
            feature.mkdir()
            (feature / "feature.md").write_text(f"Implement feature {number}.")
            for name in ("feature.patch", "tests.patch"):
                (feature / name).touch()
        args = (dataset, "b" * 40, "all")
        records = normalized_cooperbench(*args)
        assert len(records) == 66 and all(len(r["goals"]) == 2 for r in records)
        measure(normalized_cooperbench, args, 10, "one synthetic task, all 66 pairs of 12 features")

    gate = ArbitrationGate("rule_based", ConstantArbiter("block"))
    observation = dict(schema_version=1, task_key="fixture", actor_id="agent1",
                       shared_goal="Integrate both changes", agent_goals={"agent1": "A", "agent2": "B"},
                       constraints={}, proposal_sequence=9, proposed_action=events[1]["payload"],
                       recent_events=events[1:7], repository_state={"changed_files": ["README.md"]})
    action, _, _, error = gate.evaluate(observation)
    assert action == "block" and error is None
    measure(gate.evaluate, (observation,), 1000, "one proposal, six history events, fixed local decision")

    rows = expected_rows(events)
    assert len(rows) == 8 and rows[2]["payload"]["trigger_sequence"] == 2
    measure(expected_rows, (events,), 10000, "eight-event example run")
    onsets, resolutions = conflict_evidence(events)
    assert set(onsets) == set(resolutions) == {6}
    measure(conflict_evidence, (events,), 10000, "eight-event example run, one resolved conflict")


if __name__ == "__main__":
    main()
