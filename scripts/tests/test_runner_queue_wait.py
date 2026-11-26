"""Deterministically cover the concurrent coordinator's empty-queue retry."""

import unittest
from queue import Empty, Queue as StandardQueue
from unittest.mock import patch

try:
    from runner.arbitration import ArbitrationGate
    from runner.concurrent import run_concurrent
    from runner.environment import MockEnvironment, ScriptedAgent
    from runner.event_writer import EventWriter
except ModuleNotFoundError:
    # PR #51 can also be checked before the stacked runner PRs reach its base.
    RUNNER_AVAILABLE = False
else:
    RUNNER_AVAILABLE = True


class EmptyOnceQueue:
    """Force one coordinator retry regardless of worker scheduling."""

    def __init__(self):
        self._queue = StandardQueue()
        self._first_get = True

    def put(self, item):
        self._queue.put(item)

    def get(self, timeout=None):
        if self._first_get:
            self._first_get = False
            raise Empty
        return self._queue.get(timeout=timeout)


@unittest.skipUnless(RUNNER_AVAILABLE, "Runner stack is not present in this checkout")
class ConcurrentQueueWaitTests(unittest.TestCase):
    def test_coordinator_retries_empty_completion_queue(self):
        task = {
            "task_key": "fixture", "shared_goal": "Both agents submit",
            "constraints": {},
            "goals": [
                {"owner_type": "agent", "owner_agent_id": "agent1", "description": "First"},
                {"owner_type": "agent", "owner_agent_id": "agent2", "description": "Second"},
            ],
        }
        agents = {actor: ScriptedAgent([]) for actor in ("agent1", "agent2")}
        writer = EventWriter(task_key="fixture", arbiter_condition="none",
                             agent_config={}, arbiter_config={})
        with patch("runner.concurrent.Queue", EmptyOnceQueue):
            events = run_concurrent(task, agents, MockEnvironment(), writer,
                                    gate=ArbitrationGate("none"))
        self.assertEqual(events[-1]["payload"]["status"], "completed")
        self.assertEqual(events[-1]["payload"]["termination_reason"], "agents_submitted")


if __name__ == "__main__":
    unittest.main()
