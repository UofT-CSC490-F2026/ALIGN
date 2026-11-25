"""Docker command contract mocks plus real local Git patch round trip.

These tests do not assert that Docker images or upstream evaluation work.
"""
from pathlib import Path
import shlex
import subprocess
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
from runner.environment import DockerEnvironment


class DockerContractTests(unittest.TestCase):
    def test_initialization_failure_removes_started_containers(self):
        calls=[]
        def docker(args,**kwargs):
            calls.append(args)
            if args[0]=='exec':
                return subprocess.CompletedProcess(args,0,'wrong-base\n','')
            return subprocess.CompletedProcess(args,0,'','')
        with patch.object(DockerEnvironment,'_docker',side_effect=docker):
            with self.assertRaisesRegex(ValueError,'base commit'):
                DockerEnvironment('trusted-image','a'*40)
        self.assertEqual(sum(a[0]=='run' for a in calls),1)
        self.assertEqual(sum(a[:2]==['rm','-f'] for a in calls),1)
        run=calls[0]
        self.assertIn('--network=none',run)
        self.assertIn('--pull=never',run)
        self.assertFalse(any('mount' in a or 'volume' in a for a in run))

    def test_cleanup_attempts_all_and_reports_failure(self):
        env=DockerEnvironment.__new__(DockerEnvironment)
        env.containers={'agent1':'first','agent2':'second'}
        with patch.object(env,'_docker',return_value=subprocess.CompletedProcess([],1,'','denied')) as call:
            with self.assertRaises(RuntimeError): env.close()
        self.assertEqual(call.call_count,2)
        self.assertEqual(len(env.containers),2)

    def test_patch_roundtrip_new_deleted_and_committed_files(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp)
            def git(*args):
                return subprocess.run(['git',*args],cwd=root,text=True,capture_output=True,check=True)
            git('init')
            git('config','user.email','fixture@example.invalid')
            git('config','user.name','Fixture')
            git('config','core.autocrlf','false')
            (root/'deleted.txt').write_text('delete me\n')
            (root/'committed.txt').write_text('before\n')
            git('add','-A')
            git('commit','-m','base [skip ci]')
            base=git('rev-parse','HEAD').stdout.strip()
            (root/'committed.txt').write_text('after\n')
            git('add','-A')
            git('commit','-m','change [skip ci]')
            (root/'deleted.txt').unlink()
            (root/'new.txt').write_text('new file\n')
            env=DockerEnvironment.__new__(DockerEnvironment)
            env.containers={'agent1':'fake'}
            env.base_commit=base
            def execute(actor,command,check=False,output_limit=None):
                return git(*shlex.split(command)[1:])
            env._exec=execute
            generated=env.patches()['agent1']
            self.assertIn('new file mode',generated)
            self.assertIn('deleted file mode',generated)
            self.assertIn('+after',generated)
            # Apply to a separate clean clone: patch is relative to the base,
            # rather than just the agent's latest commit or working tree.
            clone=root/'replay'
            git('clone','--no-hardlinks',str(root),str(clone))
            subprocess.run(['git','checkout',base],cwd=clone,check=True,capture_output=True)
            subprocess.run(['git','apply','--binary','-'],cwd=clone,input=generated,text=True,check=True,capture_output=True)
            self.assertEqual((clone/'committed.txt').read_text(),'after\n')
            self.assertEqual((clone/'new.txt').read_text(),'new file\n')
            self.assertFalse((clone/'deleted.txt').exists())
