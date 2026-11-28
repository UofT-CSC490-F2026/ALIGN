import unittest
from runner.arbitration import ArbitrationGate,ConstantArbiter
from runner.environment import MockEnvironment,ScriptedAgent
from runner.orchestrator import run_task
from test_runner_orchestration import task,writer,proposal

class InterventionTests(unittest.TestCase):
    def run_case(self,condition,action):
        env=MockEnvironment()
        agents={a:ScriptedAgent([proposal()]) for a in ("agent1","agent2")}
        events=run_task(task(),agents,env,writer(condition),gate=ArbitrationGate(condition,ConstantArbiter(action)))
        return env,agents,events
    def test_block_never_dispatches(self):
        env,agents,events=self.run_case("llm","block")
        self.assertEqual(env.executed,[])
        self.assertEqual(len([e for e in events if e["event_type"]=="arbiter_decision"]),2)
        self.assertIn("block",agents["agent1"].feedback[0]["output"])
    def test_continue_dispatches_once(self):
        env,_,events=self.run_case("llm","continue")
        self.assertEqual(len(env.executed),2)
        self.assertEqual(events[-1]["payload"]["status"],"completed")
    def test_monitor_advisory_only(self):
        env,_,events=self.run_case("monitor","block")
        self.assertEqual(len(env.executed),2)
        self.assertFalse(any(e["event_type"]=="arbiter_decision" for e in events))
        self.assertEqual(len([e for e in events if e["event_type"]=="arbiter_advice"]),2)
    def test_mediation_cancels(self):
        env,agents,_=self.run_case("llm","mediate")
        self.assertEqual(env.executed,[])
        self.assertIn("Revise",agents["agent1"].feedback[0]["output"])
    def test_escalation_stops_trial(self):
        env,_,events=self.run_case("llm","escalate")
        self.assertEqual(env.executed,[])
        self.assertEqual(len([e for e in events if e["event_type"]=="proposed_action"]),1)
    def test_unsupported_action_fails_closed(self):
        env,_,events=self.run_case("llm","restructure")
        self.assertEqual(env.executed,[])
        self.assertEqual(events[-1]["payload"]["status"],"failed")
