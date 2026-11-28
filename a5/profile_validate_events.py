"""From the repository root: python3 -m a5.profile_validate_events"""
import cProfile
import json
import platform
import pstats
from pathlib import Path

from scripts.run_artifacts import validate_events


def main():
    fixture = Path(__file__).resolve().parents[1] / "scripts/examples/example_run.events.jsonl"
    events = [json.loads(line) for line in fixture.read_text().splitlines()]
    header = validate_events(events)
    assert header["complete"] and header["event_count"] == len(events)

    repetitions = 1000
    profiler = cProfile.Profile()
    with profiler:
        for _ in range(repetitions):
            validate_events(events)

    print(f"Python {platform.python_version()} on {platform.system()} {platform.machine()}")
    print(f"{fixture.name}: {len(events)} events per validation, {repetitions} validations")
    pstats.Stats(profiler).strip_dirs().sort_stats("cumulative").print_stats(15)


if __name__ == "__main__":
    main()
