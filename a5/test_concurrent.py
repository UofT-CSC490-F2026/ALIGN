"""A5 depth tests for runner.concurrent.run_concurrent. All collaborators are local."""
from threading import Event
from unittest import TestCase
from unittest.mock import Mock, patch

from runner.arbitration import ArbitrationGate, ConstantArbiter
from runner.concurrent import run_concurrent
from runner.environment import MockEnvironment, ScriptedAgent
from runner.event_writer import EventWriter


def shell(command):
    return {"action": {"tool": "bash", "command": command}, "files_touched": []}


def of_type(events, kind):
    return [e for e in events if e["event_type"] == kind]


def wait(signal):
    if not signal.wait(5):
        raise AssertionError("Timed out waiting for the other worker")


class ConcurrentDepthTests(TestCase):
    def setUp(self):
        self.task = {"task_key": "fixture", "shared_goal": "both", "constraints": {},
                     "goals": [{"goal_key": f"g{i}", "owner_type": "agent",
                                "owner_agent_id": f"agent{i}", "description": "edit"}
                               for i in (1, 2)]}
        self.env = MockEnvironment()
        self.a = ScriptedAgent([])
        self.b = ScriptedAgent([])

    def trial(self, condition="none", arbiter=None, **options):
        self.writer = EventWriter(task_key="fixture", arbiter_condition=condition,
                                  agent_config={}, arbiter_config={})
        events = run_concurrent(self.task, {"agent1": self.a, "agent2": self.b},
                                self.env, self.writer, gate=ArbitrationGate(condition, arbiter),
                                **options)
        self.assertEqual(events[-1]["event_type"], "run_end")
        return events

    def assert_outcomes(self, events, expected):
        proposals = of_type(events, "proposed_action")
        actual = [(e["trigger_sequence"], e["payload"]["outcome"])
                  for e in of_type(events, "action_result")]
        self.assertEqual(len(proposals), len(expected))
        self.assertEqual(actual, [(p["sequence_number"], result)
                                  for p, result in zip(proposals, expected)])

    def test_evaluation_waits_for_both_submissions(self):
        submitted = Event()
        evaluator = Mock(return_value={})
        original_emit = EventWriter.emit

        def signal_submission(writer, actor, kind, payload, **fields):
            seq = original_emit(writer, actor, kind, payload, **fields)
            if kind == "agent_submitted" and fields["actor_id"] == "agent1":
                submitted.set()
            return seq

        def late_step(_):
            wait(submitted)
            evaluator.assert_not_called()
            return shell("last edit")

        turns = iter([late_step, lambda _: None])
        self.b.propose = lambda context: next(turns)(context)
        self.env.patches = lambda: {"actions": list(self.env.executed)}
        with patch.object(EventWriter, "emit", signal_submission):
            events = self.trial(evaluator=evaluator)
        self.assertEqual(events[-1]["payload"]["status"], "completed")
        evaluator.assert_called_once_with(self.task, {"actions": [("agent2", shell("last edit")["action"])]})

    def test_monitor_block_advice_still_dispatches_once(self):
        self.a = ScriptedAgent([shell("inspect")])
        events = self.trial("monitor", ConstantArbiter("block"))
        self.assertEqual(self.env.executed, [("agent1", shell("inspect")["action"])])
        self.assertEqual([e["payload"]["action"] for e in of_type(events, "arbiter_advice")], ["block"])
        self.assertFalse(of_type(events, "arbiter_decision"))
        self.assert_outcomes(events, ["dispatched"])

    def test_enforced_block_allows_safe_reproposal(self):
        self.a = ScriptedAgent([shell("original"), shell("revised")])
        arbiter = Mock()
        arbiter.decide.side_effect = lambda obs: ConstantArbiter(
            "block" if obs["proposed_action"] == shell("original") else "continue").decide(obs)
        events = self.trial("rule_based", arbiter)
        self.assertEqual(events[-1]["payload"]["status"], "completed")
        self.assertEqual(self.env.executed, [("agent1", shell("revised")["action"])])
        self.assert_outcomes(events, ["cancelled", "dispatched"])

    def test_enforced_arbiter_error_cancels_proposal_once(self):
        self.a = ScriptedAgent([shell("inspect")])
        arbiter = Mock()
        arbiter.decide.return_value = {"action": "invalid"}
        events = self.trial("rule_based", arbiter)
        self.assertEqual(events[-1]["payload"]["status"], "failed")
        self.assertEqual(self.env.executed, [])
        self.assertEqual([e["payload"]["error_type"] for e in of_type(events, "arbiter_error")], ["ValueError"])
        self.assert_outcomes(events, ["cancelled"])

    def test_monitor_error_records_error_and_continues(self):
        self.a = ScriptedAgent([shell("inspect")])
        arbiter = Mock()
        arbiter.decide.side_effect = RuntimeError("fixture")
        events = self.trial("monitor", arbiter)
        self.assertEqual(events[-1]["payload"]["status"], "completed")
        self.assertEqual([e["payload"]["error_type"] for e in of_type(events, "arbiter_error")], ["RuntimeError"])
        self.assertEqual(self.env.executed, [("agent1", shell("inspect")["action"])])
        self.assert_outcomes(events, ["dispatched"])

    def test_mailbox_delivers_two_messages_once_in_order(self):
        receiver_ready, both_sent = Event(), Event()
        messages = [{"action": {"tool": "message", "recipient": "agent2", "content": text},
                     "files_touched": []} for text in ("first", "second")]
        self.a = ScriptedAgent(messages)
        take_step = self.a.propose

        def send(context):
            wait(receiver_ready)
            proposal = take_step(context)
            if proposal is None:
                both_sent.set()
            return proposal

        self.a.propose = send
        self.b = ScriptedAgent([shell("step one"), shell("step two")])
        receive_step = self.b.propose

        def receive(context):
            receiver_ready.set()
            wait(both_sent)
            return receive_step(context)

        self.b.propose = receive
        events = self.trial()
        self.assertEqual(events[-1]["payload"]["status"], "completed")
        self.assertEqual([item["output"] for item in self.b.feedback],
                         ["mock execution", "Message from agent1: first",
                          "Message from agent1: second", "mock execution"])

    def test_query_feedback_stays_with_requesting_agent(self):
        received = Event()
        self.a = ScriptedAgent([shell("ambiguous")])
        record = self.a.receive

        def acknowledge(result):
            record(result)
            received.set()

        self.a.receive = acknowledge
        self.b = ScriptedAgent([shell("peer step")])
        peer_step = self.b.propose

        def peer(context):
            wait(received)
            return peer_step(context)

        self.b.propose = peer
        arbiter = Mock()
        arbiter.decide.side_effect = lambda obs: ConstantArbiter(
            "query" if obs["actor_id"] == "agent1" else "continue").decide(obs)
        events = self.trial("rule_based", arbiter)
        self.assertEqual(events[-1]["payload"]["status"], "completed")
        self.assertEqual(len(self.a.feedback), 1)
        self.assertIn("missing facts", self.a.feedback[0]["output"])
        self.assertEqual([item["output"] for item in self.b.feedback], ["mock execution"])
        self.assertEqual(self.env.executed, [("agent2", shell("peer step")["action"])])

    def test_escalation_skips_patch_collection_and_evaluation(self):
        self.a = ScriptedAgent([shell("needs review")])
        self.env.patches = Mock()
        evaluator = Mock()
        events = self.trial("rule_based", ConstantArbiter("escalate"), evaluator=evaluator)
        self.assertEqual(events[-1]["payload"]["termination_reason"], "escalated")
        self.env.patches.assert_not_called()
        evaluator.assert_not_called()
        self.assertFalse(of_type(events, "test_result"))

    def test_missing_goal_result_is_recorded_as_not_evaluated(self):
        events = self.trial(evaluator=lambda *_: {"goal_results": {"g2": "passed"}})
        self.assertEqual(events[-1]["payload"]["status"], "completed")
        self.assertCountEqual([e["payload"] for e in of_type(events, "test_result")],
                              [{"goal_key": "g2", "result": "passed"},
                               {"goal_key": "g1", "result": "not_evaluated"}])

    def test_feedback_failure_preserves_dispatched_result(self):
        self.a = ScriptedAgent([shell("edit")])
        self.a.receive = Mock(side_effect=ValueError("feedback failed"))
        events = self.trial()
        self.assertEqual(self.env.executed, [("agent1", shell("edit")["action"])])
        self.assertEqual(events[-1]["payload"]["failure_category"], "agent_error")
        self.assert_outcomes(events, ["dispatched"])

    def test_cleanup_failure_preserves_original_agent_error(self):
        self.a.propose = Mock(side_effect=ValueError("agent failed"))
        self.env.close = Mock(side_effect=OSError("cleanup failed"))
        events = self.trial()
        end = events[-1]["payload"]
        self.assertEqual((end["status"], end["failure_category"], end["failure_reason"]),
                         ("failed", "agent_error", "ValueError"))
        self.assertTrue(end["cleanup_failed"])
        self.env.close.assert_called_once()

    def test_usage_failure_preserves_completed_trial(self):
        events = self.trial(usage=Mock(side_effect=ValueError("missing accounting")))
        end = events[-1]["payload"]
        self.assertEqual((end["status"], end["termination_reason"]), ("completed", "agents_submitted"))
        self.assertTrue(end["usage_error"])
        self.assertNotIn("input_tokens", end)
        self.assertNotIn("output_tokens", end)
