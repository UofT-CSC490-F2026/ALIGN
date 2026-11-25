"""Exercise the real HTTP client against an ephemeral loopback server."""
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import patch
from urllib.error import HTTPError
from runner.cli import main
from runner.environment import MockEnvironment
from runner.model_adapter import ChatClient, ChatAgent, ChatArbiter
from scripts.run_artifacts import load_events
from test_runner_orchestration import task


class ModelTests(unittest.TestCase):
    def setUp(self):
        self.status=200
        self.delay=0
        self.requests=[]
        self.response={'choices':[{'finish_reason':'stop','message':{'content':'{"done":true}'}}],
                       'usage':{'prompt_tokens':9,'completion_tokens':4}}
        owner=self
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                request=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                owner.requests.append((dict(self.headers),request))
                time.sleep(owner.delay)
                self.send_response(owner.status)
                if owner.status==302:
                    self.send_header('Location','http://127.0.0.1:1/leak')
                self.end_headers()
                response=owner.response(request) if callable(owner.response) else owner.response
                raw=response if isinstance(response,bytes) else json.dumps(response).encode()
                try: self.wfile.write(raw)
                except (BrokenPipeError,ConnectionResetError,ConnectionAbortedError): pass
            def log_message(self,*args): pass
        self.server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
        self.thread=threading.Thread(target=self.server.serve_forever,daemon=True)
        self.thread.start()
        self.client=ChatClient(f'http://127.0.0.1:{self.server.server_port}/v1/chat/completions','checkpoint/model',key_env='ALIGN_TEST_HTTP_KEY',timeout=1)

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()

    def test_format_auth_and_usage(self):
        with patch.dict(os.environ,{'ALIGN_TEST_HTTP_KEY':'test-key-only'}):
            self.assertIsNone(ChatAgent(self.client).propose({'goal':'edit'}))
        headers,body=self.requests[0]
        self.assertEqual(headers['Authorization'],'Bearer test-key-only')
        self.assertEqual(body['model'],'checkpoint/model')
        self.assertEqual(body['temperature'],0)
        self.assertEqual(body['max_tokens'],1024)
        self.assertEqual(body['messages'][0]['role'],'system')
        self.assertEqual((self.client.input_tokens,self.client.output_tokens),(9,4))

    def test_no_auth_when_unset_and_exact_observation(self):
        with patch.dict(os.environ,{},clear=True):
            obs={'schema_version':1,'actor_id':'agent1','proposed_action':{'action':{}}}
            ChatArbiter(self.client).decide(obs)
        headers,body=self.requests[0]
        self.assertNotIn('Authorization',headers)
        self.assertEqual(json.loads(body['messages'][1]['content']),obs)

    def test_invalid_responses(self):
        original=self.response
        values=[b'not-json',[],{}, {'choices':[]},
                {'choices':[{'finish_reason':'length','message':{'content':'{}'}}]},
                {'choices':[{'finish_reason':'stop','message':{'content':'[]'}}]},
                {'choices':[{'finish_reason':'stop','message':{'content':'{"x":NaN}'}}]},
                {'choices':[{'finish_reason':'stop','message':{'content':'{"action":"block","action":"continue"}'}}]}]
        for value in values:
            with self.subTest(value=value):
                self.response=value
                with self.assertRaises((ValueError,KeyError,IndexError,TypeError)):
                    self.client.generate([])
        self.response=original

    def test_missing_usage_unknown(self):
        self.response.pop('usage')
        self.client.generate([])
        self.assertFalse(self.client.usage_known)

    def test_http_error_and_redirect_rejected(self):
        for status in (401,429,500,302):
            self.status=status
            with self.subTest(status=status),self.assertRaises(HTTPError):
                self.client.generate([])
        self.assertFalse(self.client.usage_known)
        self.assertEqual(len(self.requests),4)

    def test_timeout(self):
        self.delay=.15
        self.client.timeout=.02
        with self.assertRaises(TimeoutError): self.client.generate([])
        self.assertFalse(self.client.usage_known)

    def test_endpoint_validation(self):
        for url in ('https://user:key@example.com/v1','https://example.com/v1?key=secret','http://example.com/v1'):
            with self.assertRaises(ValueError): ChatClient(url,'model',key_env='KEY')

    def test_cli_endpoints_checkpoint_and_recorded_observation(self):
        step={}
        def respond(request):
            context=json.loads(request['messages'][1]['content'])
            if request['model']=='arbiter-sft':
                content={'action':'continue','reason':'fixture','affected_agents':[],'evidence':{}}
            else:
                actor=context['context']['actor_id']
                step[actor]=step.get(actor,0)+1
                content=({'action':{'tool':'bash','command':'echo test'},'files_touched':[]}
                         if step[actor]==1 else {'done':True})
            return {'choices':[{'finish_reason':'stop','message':{'content':json.dumps(content)}}],
                    'usage':{'prompt_tokens':9,'completion_tokens':4}}
        self.response=respond
        fixture=task()
        verified_revision='a'*40
        with TemporaryDirectory() as tmp:
            images=Path(tmp)/'images.json'
            images.write_text('{"fixture_task:task0":"preloaded"}',encoding='utf-8')
            with patch('runner.cli.load_tasks',return_value=[fixture]), \
                 patch('runner.cli.installed_revision',return_value=verified_revision), \
                 patch('runner.cli.cooperbench_identity',return_value=('fixture_task',0,[1,2])), \
                 patch('runner.cli.prepare_environment',return_value=MockEnvironment()), \
                 patch('runner.cli.evaluate_patches',return_value={}), \
                 patch.dict(os.environ,{'ALIGN_AGENT_API_KEY':'agent-secret','ALIGN_ARBITER_API_KEY':'arbiter-secret'}):
                code=main(['--live','--tasks','fixture','--output-dir',tmp,'--images',str(images),
                           '--dataset-dir',tmp,'--condition','llm','--training','sft','--checkpoint','revision-123',
                           '--agent-endpoint',self.client.endpoint,'--agent-model','agent',
                           '--arbiter-endpoint',self.client.endpoint,'--arbiter-model','arbiter-sft'])
            self.assertEqual(code,0)
            raw=next(Path(tmp).glob('*/bundle/events.jsonl')).read_bytes()
            events,header=load_events(raw)
            self.assertEqual(header['agent_config']['upstream_revision'],verified_revision)
            self.assertEqual(header['arbiter_config']['checkpoint'],'revision-123')
            self.assertEqual(header['arbiter_config']['training'],'sft')
            self.assertNotIn(b'-secret',raw)
            recorded=[e['payload'] for e in events if e['event_type']=='arbiter_observation']
            sent=[json.loads(body['messages'][1]['content']) for headers,body in self.requests
                  if body['model']=='arbiter-sft' and headers['Authorization']=='Bearer arbiter-secret']
            self.assertEqual(recorded,sent)
            self.assertEqual(len(recorded),2)
            self.assertEqual(events[-1]['payload']['input_tokens'],54)
            self.assertEqual(events[-1]['payload']['output_tokens'],24)
