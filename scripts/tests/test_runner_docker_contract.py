"""Docker command contract mocks plus real local Git patch round trip.

These tests do not assert that Docker images or upstream evaluation work.
"""
from pathlib import Path
import shlex
import subprocess
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch
from runner.environment import DockerEnvironment


class DockerContractTests(unittest.TestCase):
    def test_legacy_workspace_state_and_action_timeout(self):
        env=DockerEnvironment.__new__(DockerEnvironment)
        env.containers={'agent1':'sandbox'}
        env.base_commit='a'*40
        for action in ({'tool':'message'}, {'tool':'bash','command':None}):
            with self.assertRaisesRegex(ValueError,'Expected bash action'):
                env.execute('agent1',action)
        with patch.object(env,'_exec',side_effect=[Mock(stdout=' M file'),Mock(stdout='tracked diff')]):
            self.assertEqual(env.state(),{'agent1':{'git_status':' M file','diff':'tracked diff'}})
        completed=subprocess.CompletedProcess([],17,'out','err')
        with patch.object(env,'_exec',return_value=completed):
            self.assertEqual(env.execute('agent1',{'tool':'bash','command':'inspect'}),
                             {'output':'outerr','exit_code':17,'output_truncated':False})
        with patch.object(env,'_exec',side_effect=subprocess.TimeoutExpired('docker',1)), \
             patch.object(env,'close') as close:
            with self.assertRaisesRegex(TimeoutError,'trial aborted'):
                env.execute('agent1',{'tool':'bash','command':'inspect'})
            close.assert_called_once()

    def test_legacy_rejects_bad_configuration_and_dirty_image(self):
        base='a'*40
        for image,commit,timeout,reason in (('',base,1,'Trusted image'),
                                            ('--help',base,1,'Trusted image'),
                                            ('image','no base',1,'Trusted image'),
                                            ('image',base,0,'Positive timeout')):
            with self.subTest(image=image,commit=commit,timeout=timeout), self.assertRaisesRegex(ValueError,reason):
                DockerEnvironment(image,commit,timeout=timeout)
        with patch.object(DockerEnvironment,'_exec',side_effect=[Mock(stdout=base),Mock(stdout=' M file')]), \
             patch.object(DockerEnvironment,'_docker',return_value=Mock(returncode=0)) as docker:
            with self.assertRaisesRegex(ValueError,'workspace must be clean'):
                DockerEnvironment('image',base)
            self.assertEqual(docker.call_args.args[0][:2],['rm','-f'])

    def test_cleanup_is_retryable_after_a_docker_timeout(self):
        env=DockerEnvironment.__new__(DockerEnvironment)
        env.containers={'agent1':'first','agent2':'second'}
        removed=Mock(returncode=1,stderr='No such container: second')
        error=subprocess.TimeoutExpired('docker',1)
        with patch.object(env,'_docker',side_effect=[error,removed,Mock(returncode=0)]):
            with self.assertRaisesRegex(RuntimeError,'Could not remove'):
                env.close()
            self.assertEqual(env.containers,{'agent1':'first'})
            env.close()
            self.assertEqual(env.containers,{})

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


class SharedGitContractTests(unittest.TestCase):
    def setUp(self):
        from runner.deadline import TrialDeadline
        from runner.shared_git import SharedDockerEnvironment

        self.base, self.tip = "a" * 40, "b" * 40
        self.head = self.base
        self.fast_forward = True

        def reply(args, **kwargs):
            command = args[-1]
            output = b""
            status = 0
            if command == "git rev-parse HEAD":
                output = self.head.encode()
            elif "rev-parse HEAD" in command:
                output = self.base.encode()
            elif "rev-parse refs/align" in command:
                output = self.tip.encode()
            elif "merge-base --is-ancestor" in command and not self.fast_forward:
                status = 1
            return subprocess.CompletedProcess(args, status, output, b"")

        docker = patch("runner.environment.bounded_run", side_effect=reply)
        self.docker = docker.start()
        self.addCleanup(docker.stop)
        self.env = SharedDockerEnvironment("fixture", self.base, deadline=TrialDeadline())
        self.addCleanup(self.env.close)

    def test_publish_then_fetch_keeps_peer_branch_separate(self):
        env = self.env
        refs = env.shared_state()["published"]
        fetch = {"tool": "git_fetch", "peer": "agent1"}
        self.assertEqual(env.execute("agent2", fetch, expected_refs=refs)["exit_code"], 0)
        self.assertEqual(env.execute("agent1", {"tool": "git_publish"}, expected_refs=refs)["published"], refs)
        self.head = self.tip
        with patch("runner.shared_git.subprocess.run", return_value=subprocess.CompletedProcess([], 0, b"bundle", b"")) as binary:
            published = env.execute("agent1", {"tool": "git_publish"}, expected_refs=refs)["published"]
            self.assertEqual(published, {"agent1": self.tip, "agent2": self.base})
            self.assertTrue(env.execute("agent2", fetch, expected_refs=refs)["stale"])
            self.assertEqual(env.execute("agent2", fetch, expected_refs=published)["exit_code"], 0)
            self.assertEqual(binary.call_args.kwargs["input"], b"bundle")

    def test_rejected_operations_leave_public_refs_unchanged(self):
        env = self.env
        refs = env.shared_state()["published"]
        self.fast_forward = False
        rejected = env.execute("agent1", {"tool": "git_publish"}, expected_refs=refs)
        self.assertEqual(rejected["exit_code"], 1)
        self.assertEqual(env.shared_state()["published"], refs)
        for action in ({"tool": "git_fetch", "peer": "agent1"}, {"tool": "unknown"}):
            with self.subTest(action=action), self.assertRaises(ValueError):
                env.execute("agent1", action, expected_refs=refs)
        self.assertEqual(env.execute("agent1", {"tool": "bash", "command": "pwd"})["exit_code"], 0)
        self.assertEqual(env.state_for("agent1")["head"], self.base)

    def test_final_merge_returns_joint_patch_or_conflict(self):
        env = self.env
        with patch.object(env, "_binary", return_value=b"joint diff"):
            self.assertEqual(env.patches(), {"_merge_status": "clean", "agent1": "joint diff", "agent2": "joint diff"})
        with patch.object(env, "_import_broker"), patch.object(env, "_broker_exec", side_effect=[
                Mock(returncode=0), Mock(returncode=1), Mock(stdout="unmerged file")]):
            self.assertEqual(env.patches(), {"_merge_status": "conflicts", "agent1": "", "agent2": ""})
