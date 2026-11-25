"""Fault injection and interaction regressions; no model or Docker calls."""
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
from runner.arbitration import ArbitrationGate, ConstantArbiter
from runner.environment import MockEnvironment, ScriptedAgent
from runner.orchestrator import dispatch, run_task, validate_proposal
from scripts.run_artifacts import verify_bundle, load_events
from scripts.process_runs import expected_rows, derived_end
from test_runner_orchestration import task, writer, proposal


class HardeningTests(unittest.TestCase):
    def test_gate_rejects_invalid_condition_and_missing_arbiter(self):
        with self.assertRaisesRegex(ValueError,'Invalid condition'):
            ArbitrationGate('unexpected')
        with self.assertRaisesRegex(ValueError,'Arbiter required'):
            ArbitrationGate('monitor')

    def run_agents(self, agents=None, env=None, gate=None, **kwargs):
        env=env or MockEnvironment()
        agents=agents or {a:ScriptedAgent([proposal()]) for a in ('agent1','agent2')}
        w=writer('llm' if gate else 'none')
        events=run_task(task(),agents,env,w,gate=gate or ArbitrationGate('none'),**kwargs)
        self.assertTrue(env.closed)
        self.assertEqual(sum(e['event_type']=='run_end' for e in events),1)
        return env,agents,w,events

    def test_invalid_proposals(self):
        for p in [[],{}, {'action':{},'files_touched':[]},
                  {'action':{'tool':'bash','command':' '},'files_touched':[]},
                  {'action':{'tool':'bash','command':'echo ok','extra':True},'files_touched':[]},
                  {'action':{'tool':'message','recipient':[],'content':'hi'},'files_touched':[]},
                  {'action':{'tool':'git_publish','force':True},'files_touched':[]},
                  {'action':{'tool':'git_fetch','peer':'other'},'files_touched':[]},
                  {'action':{'tool':'bash','command':'inspect'},'files_touched':[1]},
                  {'action':{'tool':'bash','command':'\x00'},'files_touched':[]}]:
            with self.subTest(p=p),self.assertRaises(ValueError):
                validate_proposal(p)

    def test_git_publication_validation_returns_independent_proposal(self):
        publication={'action':{'tool':'git_publish'},'files_touched':[]}
        validated=validate_proposal(publication)
        self.assertEqual(validated,publication)
        validated['action']['tool']='other'
        self.assertEqual(publication['action']['tool'],'git_publish')

    def test_alternating_runner_rejects_self_message_and_invalid_roles(self):
        message = {'action':{'tool':'message','recipient':'agent1','content':'self'},'files_touched':[]}
        env,_,_,events = self.run_agents(agents={'agent1':ScriptedAgent([message]),'agent2':ScriptedAgent([])})
        self.assertEqual(events[-1]['payload']['failure_category'],'agent_error')
        self.assertFalse(env.executed)
        with self.assertRaisesRegex(ValueError,'Cannot message self'):
            dispatch(MockEnvironment(),{},'agent1',message['action'])
        _,_,_,events = self.run_agents(agents={'agent1':ScriptedAgent([])})
        self.assertEqual(events[-1]['payload']['failure_category'],'agent_error')

    def test_configured_secret_in_alternating_proposal_never_dispatches(self):
        secret='configured-shell-secret'
        action={'action':{'tool':'bash','command':secret},'files_touched':[]}
        with patch.dict(os.environ,{'ALIGN_AGENT_API_KEY':secret}):
            env,_,_,events=self.run_agents(agents={'agent1':ScriptedAgent([action]),'agent2':ScriptedAgent([])})
        self.assertFalse(env.executed)
        self.assertEqual(events[-1]['payload']['failure_category'],'agent_error')
        self.assertNotIn(secret,str(events))

    def test_alternating_run_preserves_partial_evidence_and_usage_failure(self):
        first = task()['goals'][0]['goal_key']
        def evaluator(task,patches):
            return {'goal_results':{first:'passed'},'evaluation_error':True}
        def usage():
            raise RuntimeError('private details')
        _,_,_,events = self.run_agents(evaluator=evaluator,usage=usage)
        self.assertEqual([e['payload']['result'] for e in events if e['event_type']=='test_result'],
                         ['passed','not_evaluated'])
        self.assertEqual(events[-1]['payload']['failure_category'],'infrastructure_error')
        self.assertTrue(events[-1]['payload']['usage_error'])
        _,_,_,invalid = self.run_agents(evaluator=lambda *_:{'goal_results':{first:'mislabeled'}})
        self.assertEqual(invalid[-1]['payload']['failure_category'],'infrastructure_error')
        self.assertFalse(any(e['event_type']=='test_result' for e in invalid))

    def test_successful_alternating_evaluation_retains_both_goal_results(self):
        def evaluate(task,patches):
            return {'goal_results':{goal['goal_key']:'passed' for goal in task['goals']},
                    'shared_goal_satisfied':True,'critical_constraints_passed':True,'merge_status':'clean'}
        _,_,_,events=self.run_agents(evaluator=evaluate)
        self.assertEqual(events[-1]['payload']['status'],'completed')
        self.assertTrue(events[-1]['payload']['shared_goal_satisfied'])
        self.assertEqual([e['payload']['result'] for e in events if e['event_type']=='test_result'],['passed']*2)

    def test_cleanup_failure_is_failed_bundle(self):
        class Broken(MockEnvironment):
            def close(self):
                super().close()
                raise RuntimeError('private credential')
        _,_,_,events=self.run_agents(env=Broken())
        self.assertEqual(events[-1]['payload']['failure_category'],'infrastructure_error')
        self.assertNotIn('private credential',json.dumps(events))

    def test_state_and_agent_exceptions(self):
        class Broken(MockEnvironment):
            def state(self): raise RuntimeError('secret')
        _,_,_,events=self.run_agents(env=Broken())
        self.assertEqual(events[-1]['payload']['failure_category'],'infrastructure_error')
        class BrokenAgent(ScriptedAgent):
            def propose(self, context): raise TimeoutError('secret')
        _,_,_,events=self.run_agents(agents={'agent1':BrokenAgent([]),'agent2':ScriptedAgent([])})
        self.assertEqual(events[-1]['payload']['failure_category'],'timeout')

    def test_dispatch_exception_has_unknown_result_and_usage(self):
        class Broken(MockEnvironment):
            def execute(self,*args): raise RuntimeError('secret')
        _,_,_,events=self.run_agents(env=Broken(),usage=lambda:{'input_tokens':7,'output_tokens':3})
        result=next(e for e in events if e['event_type']=='action_result')
        self.assertEqual(result['payload']['outcome'],'unknown')
        self.assertEqual(events[-1]['payload']['input_tokens'],7)

    def test_malformed_arbiter_and_exceptions_fail_closed(self):
        class Bad:
            def __init__(self,value): self.value=value
            def decide(self,obs):
                if isinstance(self.value,Exception): raise self.value
                return self.value
        for value in [{},{'action':[],'reason':'','affected_agents':[],'evidence':{}},RuntimeError('secret'),TimeoutError('secret')]:
            with self.subTest(value=value):
                env,_,_,events=self.run_agents(gate=ArbitrationGate('llm',Bad(value)))
                self.assertEqual(env.executed,[])
                self.assertEqual(events[-1]['payload']['status'],'failed')
                self.assertEqual(next(e for e in events if e['event_type']=='action_result')['payload']['outcome'],'cancelled')
        env,_,_,events=self.run_agents(gate=ArbitrationGate('monitor',Bad(RuntimeError())))
        self.assertEqual(len(env.executed),2)
        self.assertEqual(events[-1]['payload']['status'],'completed')

    def test_message_delivery_and_departed_peer(self):
        msg={'action':{'tool':'message','recipient':'agent2','content':'coordinate'},'files_touched':[]}
        agents={'agent1':ScriptedAgent([msg,msg]),'agent2':ScriptedAgent([])}
        env,agents,_,events=self.run_agents(agents=agents)
        self.assertEqual(env.executed,[])
        self.assertEqual(len(agents['agent2'].feedback),1)
        self.assertIn('coordinate',agents['agent2'].feedback[0]['output'])
        self.assertEqual(agents['agent1'].feedback[1]['exit_code'],1)

    def test_mediation_revision_and_no_future_observation(self):
        class Once:
            calls=0
            def decide(self,obs):
                self.calls+=1
                return ConstantArbiter('mediate' if self.calls==1 else 'continue').decide(obs)
        class Revising(ScriptedAgent):
            def propose(self,context):
                p=super().propose(context)
                if p and self.feedback:
                    p['action']['command']='echo revised'
                return p
        agents={'agent1':Revising([proposal(),proposal()]),'agent2':ScriptedAgent([])}
        env,agents,_,events=self.run_agents(agents=agents,gate=ArbitrationGate('llm',Once()))
        self.assertEqual(env.executed,[('agent1',{'tool':'bash','command':'echo revised'})])
        for event in events:
            if event['event_type']=='arbiter_observation':
                obs=event['payload']
                self.assertTrue(all(e['sequence_number']<obs['proposal_sequence'] for e in obs['recent_events']))
                self.assertEqual(events[obs['proposal_sequence']-1]['actor_id'],obs['actor_id'])

    def test_escalation_skips_evaluator(self):
        def forbidden(*args): self.fail('evaluator invoked')
        env,_,_,events=self.run_agents(gate=ArbitrationGate('llm',ConstantArbiter('escalate')),evaluator=forbidden)
        self.assertEqual(env.executed,[])
        self.assertEqual(events[-1]['payload']['termination_reason'],'escalated')

    def test_gate_cannot_dispatch_same_proposal_twice(self):
        def duplicate(writer,obs,execute):
            execute()
            execute()
        env,_,_,events=self.run_agents(gate=duplicate)
        self.assertEqual(len(env.executed),1)
        self.assertEqual(events[-1]['payload']['status'],'failed')

    def test_all_conditions_bundle_replay_and_tamper(self):
        for condition in ('none','monitor','rule_based','llm'):
            with self.subTest(condition=condition),TemporaryDirectory() as tmp:
                env=MockEnvironment()
                w=writer(condition)
                agents={a:ScriptedAgent([proposal()]) for a in ('agent1','agent2')}
                events=run_task(task(),agents,env,w,gate=ArbitrationGate(condition,ConstantArbiter()))
                root=Path(tmp)
                w.write_bundle(root/'events.jsonl',root/'bundle')
                self.assertTrue(verify_bundle(root/'bundle')['complete'])
                replay,_=load_events((root/'bundle/events.jsonl').read_bytes())
                self.assertEqual(expected_rows(events),expected_rows(replay))
                self.assertEqual(derived_end(replay)[0],'completed')
                with (root/'bundle/events.jsonl').open('ab') as stream: stream.write(b'bad')
                with self.assertRaises(ValueError): verify_bundle(root/'bundle')
