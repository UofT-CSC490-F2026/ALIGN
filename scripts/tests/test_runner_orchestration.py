import unittest
from runner.environment import MockEnvironment, ScriptedAgent
from runner.event_writer import EventWriter
from runner.orchestrator import run_task

def task():
    return {"task_key":"fixture","shared_goal":"both","constraints":{},
            "goals":[{"goal_key":f"g{i}","owner_type":"agent","owner_agent_id":f"agent{i}","description":"edit"} for i in (1,2)]}

def writer(condition="none"):
    return EventWriter(task_key="fixture",arbiter_condition=condition,agent_config={},arbiter_config={},seed=42)

def proposal():
    return {"action":{"tool":"bash","command":"echo hello"},"files_touched":[]}

class OrchestrationTests(unittest.TestCase):
    def test_baseline_trace(self):
        env=MockEnvironment()
        events=run_task(task(),{a:ScriptedAgent([proposal()]) for a in ("agent1","agent2")},env,writer())
        self.assertEqual(len(env.executed),2)
        self.assertTrue(env.closed)
        self.assertEqual(events[-1]["payload"]["status"],"completed")
        obs=[e for e in events if e["event_type"]=="arbiter_observation"]
        self.assertEqual(len(obs),2)
        self.assertEqual(obs[0]["payload"]["repository_state"]["executed_count"],0)
        self.assertEqual(obs[1]["payload"]["repository_state"]["executed_count"],1)
    def test_step_limit(self):
        env=MockEnvironment()
        events=run_task(task(),{a:ScriptedAgent([proposal()]) for a in ("agent1","agent2")},env,writer(),max_steps=1)
        self.assertEqual(events[-1]["payload"]["failure_category"],"timeout")
        self.assertTrue(env.closed)
