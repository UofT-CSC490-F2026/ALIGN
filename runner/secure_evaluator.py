"""Isolated Docker adapter for the pinned CooperBench test algorithms.

The upstream Docker backend's thread timeout is intentionally not reused.
No evaluator code runs on the host. Inputs are streamed, not interpolated into
shell commands, and the exact agent image is used after verifying its identity.
"""
from dataclasses import dataclass
import subprocess
from uuid import uuid4

from .shared_git import image_id
from .docker_utils import (EVALUATOR_CAPTURE_BYTES, bounded_run,
                           isolated_run_args, require_complete)


@dataclass
class Result:
    returncode: int
    stdout: str
    stderr: str

    def stdout_read(self):
        return self.stdout

    def stderr_read(self):
        return self.stderr


class SecureSandbox:
    def __init__(self, image, base_commit, deadline, timeout, workdir="/workspace"):
        self.name = "align-eval-" + uuid4().hex
        self.deadline, self.timeout, self.workdir = deadline, timeout, workdir
        self.closed = False
        try:
            self._run(isolated_run_args(self.name,image,workdir=workdir),check=True,cap=60)
            check = self.exec("git","-C","/workspace/repo","rev-parse","HEAD")
            if check.returncode or check.stdout.strip() != base_commit:
                raise ValueError("Evaluator image base commit mismatch")
            clean = self.exec("git","-C","/workspace/repo","status","--porcelain")
            if clean.returncode or clean.stdout.strip():
                raise ValueError("Evaluator image workspace must be clean")
            if self.exec("mkdir","-p","/patches").returncode:
                raise RuntimeError("Evaluator patch directory creation failed")
        except BaseException:
            self.terminate()
            raise

    def _run(self,args,*,data=None,check=False,cap=None):
        try:
            return require_complete(bounded_run(
                ["docker", *args], data=data, check=check,
                timeout=self.deadline.remaining(cap or self.timeout),
                limit=EVALUATOR_CAPTURE_BYTES,
            ))
        except subprocess.TimeoutExpired as exc:
            self.terminate()
            raise TimeoutError("Evaluator Docker command timed out") from exc

    def exec(self,*args):
        if self.closed:
            raise RuntimeError("Evaluator sandbox closed")
        r=self._run(["exec","-w",self.workdir,self.name,*args])
        return Result(r.returncode,r.stdout.decode("utf-8",errors="replace"),
                      r.stderr.decode("utf-8",errors="replace"))

    def write_patch(self,filename,content):
        # All names come from trusted evaluator code, not model text.
        if not filename.endswith(".patch") or "/" in filename or "\\" in filename:
            raise ValueError("Invalid patch filename")
        if self.closed:
            raise RuntimeError("Evaluator sandbox closed")
        self._run(["exec","-i",self.name,"bash","-c",'cat > "$1"',"align-write",
                   "/patches/"+filename],data=content.encode("utf-8"),check=True)

    def terminate(self):
        if self.closed:
            return
        r=subprocess.run(["docker","rm","-f",self.name],capture_output=True,text=True,timeout=30)
        if r.returncode and "No such container" not in r.stderr:
            raise RuntimeError("Evaluator container cleanup failed")
        self.closed=True


class SecureBackend:
    def __init__(self, expected_image, base_commit, deadline):
        self.expected_image,self.base_commit,self.deadline=expected_image,base_commit,deadline
        self.sandboxes=[]

    def create_sandbox(self,image,timeout=600,workdir="/workspace"):
        resolved=image_id(image,self.deadline)
        if resolved!=self.expected_image:
            raise ValueError("Agent and evaluator image IDs differ")
        sandbox=SecureSandbox(resolved,self.base_commit,self.deadline,timeout,workdir)
        self.sandboxes.append(sandbox)
        return sandbox

    def close(self):
        failed=False
        for sandbox in self.sandboxes:
            try:
                sandbox.terminate()
            except Exception:
                failed=True
        if failed:
            raise RuntimeError("Evaluator cleanup incomplete")
