import threading
import unittest
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from runner.arbitration import ArbitrationGate, ConstantArbiter
from runner.concurrent import run_concurrent
from runner.deadline import TrialDeadline
from runner.environment import MockEnvironment
from runner.event_writer import EventWriter
from runner.redaction import Redactor
from test_runner_orchestration import task


def proposal(command):
    return {"action": {"tool": "bash", "command": command}, "files_touched": []}


def wait(event):
    if not event.wait(5):
        raise AssertionError("Expected synchronization signal")


class Agent:
    def __init__(self, *steps):
        self.steps = iter(steps)
        self.feedback = []

    def propose(self, context):
        step = next(self.steps, None)
        return step(context) if callable(step) else step

    def receive(self, result):
        self.feedback.append(result)


class Arbiter:
    def __init__(self, decide):
        self.decide = decide


class ConcurrentTests(unittest.TestCase):
    def run_case(self, agents, env, condition="none", arbiter=None, **kwargs):
        fixture = task()
        writer = kwargs.pop("writer", None) or EventWriter(
            task_key=fixture["task_key"], arbiter_condition=condition,
            agent_config={}, arbiter_config={})
        events = run_concurrent(fixture, agents, env, writer,
                                gate=ArbitrationGate(condition, arbiter), **kwargs)
        self.assertEqual([e["sequence_number"] for e in events], list(range(1, len(events)+1)))
        self.assertTrue(env.closed)
        return events

    def test_independent_reasoning_and_tools_with_serial_decisions(self):
        for condition in ("none", "monitor", "rule_based", "llm"):
            with self.subTest(condition=condition):
                reasoning, tools = threading.Barrier(2), threading.Barrier(2)
                deciding = threading.Lock()
                called = []

                class Environment(MockEnvironment):
                    def execute(self, actor, action):
                        tools.wait(5)
                        return super().execute(actor, action)

                def propose(context):
                    reasoning.wait(5)
                    return proposal(context["actor_id"])

                def decide(obs):
                    self.assertTrue(deciding.acquire(blocking=False))
                    try:
                        called.append(deepcopy(obs))
                        return ConstantArbiter("block" if condition=="monitor" else "continue").decide(obs)
                    finally:
                        deciding.release()

                env = Environment()
                agents = {actor: Agent(propose) for actor in ("agent1", "agent2")}
                events = self.run_case(agents, env, condition,
                                       None if condition=="none" else Arbiter(decide))
                self.assertEqual(events[-1]["payload"]["status"], "completed")
                self.assertCountEqual([actor for actor, _ in env.executed], agents)
                observations = [e["payload"] for e in events if e["event_type"]=="arbiter_observation"]
                if condition!="none":
                    self.assertEqual(observations, called)
                self.assertTrue(observations[1]["repository_state"]["in_flight_actions"])
                for obs in observations:
                    seq = obs["proposal_sequence"]
                    self.assertEqual(events[seq-1]["actor_id"], obs["actor_id"])
                    self.assertTrue(all(e["sequence_number"]<seq for e in obs["recent_events"]))

    def test_local_completion_does_not_cancel_independent_approval(self):
        running, deciding, completed = (threading.Event() for _ in range(3))

        class Environment(MockEnvironment):
            def execute(self, actor, action):
                if actor=="agent1":
                    running.set()
                    wait(deciding)
                return super().execute(actor, action)

        def second(_):
            wait(running)
            return proposal("independent")

        def submitted(_):
            completed.set()

        def decide(obs):
            if obs["actor_id"]=="agent2":
                deciding.set()
                wait(completed)
            return ConstantArbiter().decide(obs)

        env = Environment()
        events = self.run_case({"agent1":Agent(proposal("first"),submitted), "agent2":Agent(second)},
                               env, "llm", Arbiter(decide))
        self.assertEqual(events[-1]["payload"]["status"], "completed")
        self.assertEqual(len(env.executed), 2)
        self.assertFalse(any(e["event_type"]=="authorization_stale" for e in events))

    def test_broker_race_is_cancelled_then_reproposed(self):
        class Environment(MockEnvironment):
            def __init__(self):
                super().__init__()
                self.tip = "old"

            def shared_state(self):
                return {"published":{"agent1":self.tip}}

            def execute(self, actor, action, *, expected_refs):
                if self.tip=="old":
                    self.tip = "new"
                    return {"stale":True,"exit_code":1}
                if expected_refs!={"agent1":"new"}:
                    raise AssertionError("Reproposal must use the new ref")
                return super().execute(actor, action)

        fetch = {"action":{"tool":"git_fetch","peer":"agent1"},"files_touched":[]}
        env = Environment()
        events = self.run_case({"agent1":Agent(),"agent2":Agent(fetch,fetch)}, env, "llm", ConstantArbiter())
        self.assertEqual(events[-1]["payload"]["status"], "completed")
        self.assertEqual(len(env.executed), 1)
        self.assertEqual([e["payload"]["outcome"] for e in events if e["event_type"]=="action_result"],
                         ["cancelled","dispatched"])
        self.assertEqual(sum(e["event_type"]=="authorization_stale" for e in events), 1)

    def test_escalation_marks_running_action_unknown(self):
        running, cleanup = threading.Event(), threading.Event()

        class Environment(MockEnvironment):
            def execute(self, actor, action):
                running.set()
                wait(cleanup)
                return {"output":"late response", "exit_code":0}

            def close(self):
                cleanup.set()
                super().close()

        def second(_):
            wait(running)
            return proposal("escalate")

        def decide(obs):
            action = "escalate" if obs["actor_id"]=="agent2" else "continue"
            return ConstantArbiter(action).decide(obs)

        events = self.run_case({"agent1":Agent(proposal("running")),"agent2":Agent(second)},
                               Environment(), "llm", Arbiter(decide))
        self.assertEqual(events[-1]["payload"]["termination_reason"], "escalated")
        outcomes = {events[e["trigger_sequence"]-1]["actor_id"]:e["payload"]["outcome"]
                    for e in events if e["event_type"]=="action_result"}
        self.assertEqual(outcomes,{"agent1":"unknown","agent2":"cancelled"})

    def test_deadline_cancels_both_deciding_and_queued_proposals(self):
        deciding, queued, release, returned = (threading.Event() for _ in range(4))
        deadline = TrialDeadline(10)

        class Writer(EventWriter):
            def emit(self, actor_type, event_type, payload, **kwargs):
                seq = super().emit(actor_type,event_type,payload,**kwargs)
                if event_type=="proposal_queued" and kwargs["actor_id"]=="agent2":
                    queued.set()
                    deadline.cancel()
                return seq

        def decide(obs):
            deciding.set()
            wait(queued)
            wait(release)
            returned.set()
            return ConstantArbiter().decide(obs)

        def second(_):
            wait(deciding)
            return proposal("queued")

        env = MockEnvironment()
        writer = Writer(task_key=task()["task_key"],arbiter_condition="llm",agent_config={},arbiter_config={})
        self.addCleanup(release.set)
        events = self.run_case({"agent1":Agent(proposal("deciding")),"agent2":Agent(second)}, env,
                               "llm", Arbiter(decide), writer=writer, deadline=deadline)
        release.set()
        wait(returned)
        self.assertEqual(events[-1]["payload"]["failure_category"], "timeout")
        self.assertEqual(len(env.executed), 0)
        self.assertTrue(any(e["event_type"]=="proposal_cancelled" for e in events))
        self.assertEqual([e["payload"]["outcome"] for e in events if e["event_type"]=="action_result"], ["cancelled"])
        self.assertEqual(writer.snapshot(), events)

    def test_query_mediation_and_messages_use_safe_reproposal(self):
        delivered = threading.Event()
        secret = "fixture-secret-with-entropy"

        class Environment(MockEnvironment):
            def execute(self, actor, action):
                result = super().execute(actor, action)
                return {**result,"output":secret}

        class Sender(Agent):
            def receive(self, result):
                super().receive(result)
                if result["output"]=="Message delivered":
                    delivered.set()

        def peer_waits(_):
            wait(delivered)
            return proposal("peer-inspection")

        def decide(obs):
            actions = {"needs context":"query","needs agreement":"mediate"}
            command = obs["proposed_action"]["action"].get("command")
            decision = ConstantArbiter(actions.get(command,"continue"),secret).decide(obs)
            if command=="needs agreement":
                decision["affected_agents"] = ["agent1","agent2"]
            return decision

        message = {"action":{"tool":"message","recipient":"agent2","content":"ready"},"files_touched":[]}
        sender = Sender(proposal("needs context"),proposal("needs agreement"),message,proposal("revised"))
        receiver = Agent(peer_waits)
        env = Environment()
        writer = EventWriter(task_key=task()["task_key"],arbiter_condition="llm",agent_config={},
                             arbiter_config={},redactor=Redactor([secret]))
        events = self.run_case({"agent1":sender,"agent2":receiver},env,"llm",Arbiter(decide),writer=writer)
        self.assertEqual(events[-1]["payload"]["status"],"completed")
        self.assertIn("missing facts",sender.feedback[0]["output"])
        self.assertIn("reconcile",sender.feedback[1]["output"])
        self.assertTrue(any("Message from agent1: ready"==r["output"] for r in receiver.feedback))
        self.assertTrue(any("Arbiter mediation" in r["output"] for r in receiver.feedback))
        self.assertCountEqual([(actor,a["command"]) for actor,a in env.executed],
                              [("agent1","revised"),("agent2","peer-inspection")])
        with TemporaryDirectory() as tmp:
            trace, bundle = Path(tmp)/"events.jsonl", Path(tmp)/"bundle"
            writer.write_bundle(trace,bundle)
            for path in (trace,bundle/"events.jsonl",bundle/"manifest.json"):
                self.assertNotIn(secret.encode(),path.read_bytes())

    def test_reasoning_timeout_rejects_late_secret_proposal(self):
        started, release, returned = (threading.Event() for _ in range(3))
        deadline = TrialDeadline(10)
        secret = "another-fixture-secret"

        def reasoning(_):
            started.set()
            wait(release)
            returned.set()
            return proposal(secret)

        def other(_):
            wait(started)
            deadline.cancel()
            raise RuntimeError(secret)

        env = MockEnvironment()
        writer = EventWriter(task_key=task()["task_key"],arbiter_condition="none",agent_config={},
                             arbiter_config={},redactor=Redactor([secret]))
        self.addCleanup(release.set)
        events = self.run_case({"agent1":Agent(reasoning),"agent2":Agent(other)},env,
                               writer=writer,deadline=deadline)
        self.assertFalse(env.executed)
        self.assertNotIn(secret,str(events))
        release.set()
        wait(returned)
        self.assertEqual(writer.snapshot(),events)

    def test_completed_action_survives_state_refresh_timeout(self):
        class Environment(MockEnvironment):
            def state_for(self, actor):
                if actor=="agent1" and self.executed:
                    raise TimeoutError("Workspace refresh timed out")
                return {}

        env = Environment()
        events = self.run_case({"agent1":Agent(proposal("completed")), "agent2":Agent()},env)
        results = [e["payload"]["outcome"] for e in events if e["event_type"]=="action_result"]
        self.assertEqual(results,["dispatched"])
        self.assertEqual(events[-1]["payload"]["failure_category"],"timeout")

    def test_queued_peer_is_visible_at_first_decision(self):
        both_queued = threading.Barrier(2)

        class ArbitrationLock:
            def __init__(self):
                self.lock = threading.Lock()

            def __enter__(self):
                both_queued.wait(5)
                self.lock.acquire()

            def __exit__(self,*exc):
                self.lock.release()

        agents = {actor:Agent(proposal(actor)) for actor in ("agent1","agent2")}
        with patch("runner.concurrent.Lock",side_effect=[threading.Lock(),ArbitrationLock()]):
            events = self.run_case(agents,MockEnvironment(),"llm",ConstantArbiter())
        self.assertEqual(events[-1]["payload"]["status"],"completed")
        obs = next(e["payload"] for e in events if e["event_type"]=="arbiter_observation")
        pending = obs["repository_state"]["pending_proposals"]
        self.assertEqual(set(pending),set(agents)-{obs["actor_id"]})
        sequence = next(iter(pending.values()))["queue_sequence"]
        self.assertLess(sequence,obs["proposal_sequence"])
        self.assertEqual(events[sequence-1]["event_type"],"proposal_queued")

    def test_peer_observation_marks_snapshot_refresh_in_progress(self):
        refreshing, observed = threading.Event(), threading.Event()

        class Environment(MockEnvironment):
            def state_for(self, actor):
                if actor == "agent1" and self.executed:
                    refreshing.set()
                    wait(observed)
                return {"actor": actor}

        def second(_):
            wait(refreshing)
            return proposal("inspect peer")

        def decide(obs):
            if obs["actor_id"] == "agent2":
                state = obs["repository_state"]
                self.assertEqual(state["workspace_snapshot_status"]["agent1"], "refreshing")
                self.assertNotIn("agent1", state["in_flight_actions"])
                observed.set()
            return ConstantArbiter().decide(obs)

        events = self.run_case(
            {"agent1": Agent(proposal("edit")), "agent2": Agent(second)},
            Environment(), "llm", Arbiter(decide),
        )
        self.assertEqual(events[-1]["payload"]["status"], "completed")
