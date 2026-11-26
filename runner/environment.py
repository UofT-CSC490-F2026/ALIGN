"""Isolated execution and deterministic substitutes."""
import re
import subprocess
from copy import deepcopy
from typing import Protocol
from uuid import uuid4

from .docker_utils import (
    ACTORS, COMMAND_CAPTURE_BYTES, MAX_PATCH_BYTES, agent_feedback,
    bounded_run, isolated_run_args, require_complete,
)

class Agent(Protocol):
    def propose(self, context: dict) -> dict | None: ...
    def receive(self, feedback: dict) -> None: ...

class Environment(Protocol):
    def execute(self, actor: str, action: dict) -> dict: ...
    def state(self) -> dict: ...
    def patches(self) -> dict: ...
    def close(self) -> None: ...

class ScriptedAgent:
    def __init__(self, actions):
        self.actions = iter(deepcopy(actions))
        self.feedback = []
    def propose(self, context):
        return next(self.actions, None)
    def receive(self, feedback):
        self.feedback.append(deepcopy(feedback))

class MockEnvironment:
    def __init__(self):
        self.executed = []
        self.closed = False
    def execute(self, actor, action):
        if self.closed:
            raise RuntimeError("Environment closed")
        self.executed.append((actor, deepcopy(action)))
        return {"output":"mock execution", "exit_code":0}
    def state(self):
        return {"executed_count":len(self.executed)}
    def patches(self):
        return {"agent1":"","agent2":""}
    def close(self):
        self.closed = True

class DockerEnvironment:
    """Separate workspaces, trusted preloaded image, no mounts/network/credentials.
    Use disposable hosts: Docker is not an absolute security boundary.
    """
    def __init__(self, image, base_commit, *, timeout=30):
        if not image or image.startswith("-") or not re.fullmatch("[0-9a-f]{40}",base_commit):
            raise ValueError("Trusted image and pinned base commit required")
        if type(timeout) is not int or timeout<1:
            raise ValueError("Positive timeout required")
        self.base_commit, self.timeout, self.containers = base_commit, timeout, {}
        try:
            for actor in ACTORS:
                name = "align-"+uuid4().hex
                self.containers[actor] = name
                self._docker(isolated_run_args(name, image, lifetime="3600"), timeout=60, check=True)
                if self._exec(actor,"git rev-parse HEAD",check=True).stdout.strip()!=base_commit:
                    raise ValueError("Task image base commit mismatch")
                if self._exec(actor,"git status --porcelain",check=True).stdout.strip():
                    raise ValueError("Task image workspace must be clean")
        except BaseException:
            self.close()
            raise
    @staticmethod
    def _docker(args, *, timeout, check=False, output_limit=COMMAND_CAPTURE_BYTES):
        result = bounded_run(["docker", *args], timeout=timeout, limit=output_limit)
        result.stdout = result.stdout.decode("utf-8", errors="replace")
        result.stderr = result.stderr.decode("utf-8", errors="replace")
        if check:
            result.check_returncode()
        return result

    def _exec(self, actor, command, *, check=False, output_limit=COMMAND_CAPTURE_BYTES):
        return self._docker(
            ["exec", "-w", "/workspace/repo", self.containers[actor], "bash", "-lc", command],
            timeout=self.timeout, check=check, output_limit=output_limit,
        )
    def execute(self,actor,action):
        if action.get("tool")!="bash" or not isinstance(action.get("command"),str):
            raise ValueError("Expected bash action")
        try:
            result = self._exec(actor, action["command"])
        except subprocess.TimeoutExpired as exc:
            self.close()
            raise TimeoutError("Command timed out; trial aborted") from exc
        return agent_feedback(result)
    def state(self):
        return {a:{"git_status":self._exec(a,"git status --porcelain",check=True).stdout[:4000],
                   "diff":self._exec(a,f"git diff --no-ext-diff {self.base_commit}",check=True).stdout[:8000]}
                for a in self.containers}
    def patches(self):
        out={}
        for a in self.containers:
            self._exec(a,"git add -A",check=True)
            out[a] = require_complete(self._exec(
                a, f"git diff --cached --no-ext-diff --binary {self.base_commit}",
                check=True, output_limit=MAX_PATCH_BYTES,
            )).stdout
        return out
    def close(self):
        for a,name in list(self.containers.items()):
            try:
                r=self._docker(["rm","-f",name],timeout=30)
                if r.returncode==0 or "No such container" in r.stderr:
                    self.containers.pop(a,None)
            except (OSError,subprocess.TimeoutExpired):
                pass
        if self.containers:
            raise RuntimeError("Could not remove all trial containers")
