"""Trial-local Git broker. All transfers cross the trusted structured-tool boundary.

No network or host mounts are needed. Agent-generated Git bundles are parsed
only inside isolated containers. The broker owns refs and permits fast-forward
publication to the calling agent's branch; it never executes model shell text.
"""
from copy import deepcopy
import re
import subprocess
from threading import Lock
from tempfile import TemporaryFile
from uuid import uuid4

from .environment import DockerEnvironment

ACTORS = ("agent1", "agent2")
MAX_BUNDLE = 64 * 1024 * 1024
BARE = "/tmp/align-shared.git"
MERGE_DIR = "/tmp/align-final"


def isolated_run_args(name, image, *, workdir=None):
    args = ["run", "-d", "--pull=never", "--name", name, "--network=none",
            "--cap-drop=ALL", "--security-opt=no-new-privileges",
            "--pids-limit=256", "--memory=4g", "--cpus=2"]
    if workdir:
        args += ["-w", workdir]
    return args + ["--entrypoint=sleep", image, "infinity"]


def image_id(image, deadline):
    if not isinstance(image,str) or not image or image.startswith("-"):
        raise ValueError("Invalid image reference")
    result = subprocess.run(["docker", "image", "inspect", "--format", "{{.Id}}", image],
                            capture_output=True, text=True, check=True,
                            timeout=deadline.remaining(30))
    value = result.stdout.strip()
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", value):
        raise ValueError("Expected preloaded immutable Docker image ID")
    return value


class SharedDockerEnvironment(DockerEnvironment):
    """Independent local branches plus a network-disabled, host-mediated bare repo."""
    def __init__(self, image, base_commit, *, deadline, timeout=30):
        self.deadline = deadline
        self._git_lock = Lock()
        self._close_lock = Lock()
        self._published = {a: base_commit for a in ACTORS}
        self._broker = None
        super().__init__(image, base_commit, timeout=timeout)
        try:
            for actor in ACTORS:
                self._exec(actor, f"git checkout -B align-{actor} && "
                           f"git config user.name {actor} && "
                           "git config user.email agent@align.invalid", check=True)
            self._broker = "align-git-" + uuid4().hex
            self._docker(isolated_run_args(self._broker, image), timeout=60, check=True)
            self._broker_exec(
                f"git clone --bare --no-local /workspace/repo {BARE} && "
                f"git -C {BARE} update-ref refs/heads/agent1 {base_commit} && "
                f"git -C {BARE} update-ref refs/heads/agent2 {base_commit}", check=True)
            if self._broker_exec(f"git -C {BARE} rev-parse HEAD", check=True).stdout.strip() != base_commit:
                raise ValueError("Git broker base commit mismatch")
        except BaseException:
            self.close()
            raise

    def _docker(self, args, *, timeout, check=False):
        # Cleanup must still be possible after the trial deadline.
        if args[:2] != ["rm", "-f"]:
            timeout = self.deadline.remaining(timeout)
        try:
            return super()._docker(args, timeout=timeout, check=check)
        except subprocess.TimeoutExpired as exc:
            raise TimeoutError("Docker command timed out") from exc

    def _broker_exec(self, command, *, check=False):
        return self._docker(["exec", self._broker, "bash", "-lc", command],
                            timeout=self.timeout, check=check)

    def _binary(self, container, command, data=None):
        args = ["docker", "exec", "-i", container, "bash", "-lc", command]
        try:
            with TemporaryFile() as output:
                subprocess.run(args, input=data, stdout=output, stderr=subprocess.DEVNULL,
                               timeout=self.deadline.remaining(self.timeout), check=True)
                output.seek(0)
                result=output.read(MAX_BUNDLE+1)
        except subprocess.TimeoutExpired as exc:
            raise TimeoutError("Git transfer timed out") from exc
        if len(result)>MAX_BUNDLE:
            raise ValueError("Git transfer exceeded bundle limit")
        return result

    def _export(self, actor):
        # Explicit base ancestry check; a submitted branch must derive from the task base.
        return self._binary(self.containers[actor],
            f"cd /workspace/repo && git merge-base --is-ancestor {self.base_commit} HEAD && "
            f"git bundle create - HEAD ^{self.base_commit}")

    def _import_broker(self, actor, *, final=False):
        ref = f"refs/align/{'final' if final else 'staged'}/{actor}"
        head=self._exec(actor,"git rev-parse HEAD",check=True).stdout.strip()
        if head==self.base_commit:
            self._broker_exec(f"git -C {BARE} update-ref {ref} {self.base_commit}",check=True)
            return self.base_commit
        data = self._export(actor)
        self._binary(self._broker,
            f"cat > /tmp/align-transfer.bundle && git -C {BARE} fetch --no-tags "
            f"/tmp/align-transfer.bundle +HEAD:{ref} && rm -f /tmp/align-transfer.bundle", data)
        tip = self._broker_exec(f"git -C {BARE} rev-parse {ref}", check=True).stdout.strip()
        if not re.fullmatch(r"[0-9a-f]{40}", tip):
            raise ValueError("Invalid imported commit")
        self._broker_exec(f"git -C {BARE} merge-base --is-ancestor {self.base_commit} {tip}", check=True)
        return tip

    def shared_state(self):
        return {"base_commit": self.base_commit, "published": dict(self._published)}

    def state_for(self, actor):
        if actor not in ACTORS:
            raise ValueError("Unknown actor")
        return {
            "head": self._exec(actor, "git rev-parse HEAD", check=True).stdout.strip(),
            "branch": self._exec(actor, "git branch --show-current", check=True).stdout.strip(),
            "git_status": self._exec(actor, "git status --porcelain", check=True).stdout[:4000],
            "diff": self._exec(actor, f"git diff --no-ext-diff --no-textconv {self.base_commit}",
                               check=True).stdout[:8000],
            "fetched_refs": self._exec(actor, "git for-each-ref --format='%(refname) %(objectname)' refs/remotes/align",
                                       check=True).stdout[:2000],
        }

    def execute(self, actor, action, *, expected_refs=None):
        self.deadline.remaining()
        if actor not in ACTORS:
            raise ValueError("Unknown actor")
        tool = action.get("tool")
        if tool == "bash":
            # The supervisor owns cleanup; a worker must not remove peer containers.
            r = self._exec(actor, action["command"])
            return {"output": (r.stdout + r.stderr)[:12000], "exit_code": r.returncode}
        if tool not in {"git_publish", "git_fetch"}:
            raise ValueError("Unsupported environment tool")
        with self._git_lock:
            self.deadline.remaining()
            target = actor if tool == "git_publish" else action["peer"]
            if target not in ACTORS or (tool == "git_fetch" and target == actor):
                raise ValueError("Invalid Git target")
            old = self._published[target]
            if expected_refs is None or expected_refs.get(target) != old:
                return {"output": "Shared branch changed since authorization; inspect and repropose.",
                        "exit_code": 1, "stale": True, "published": self.shared_state()["published"]}
            if tool == "git_publish":
                tip = self._import_broker(actor)
                # No force-push or cross-agent publication, even in the no-arbiter condition.
                check = self._broker_exec(f"git -C {BARE} merge-base --is-ancestor {old} {tip}")
                if check.returncode:
                    return {"output": "Non-fast-forward publication rejected; fetch and integrate first.",
                            "exit_code": 1}
                self._broker_exec(f"git -C {BARE} update-ref refs/heads/{actor} {tip} {old}", check=True)
                self._published = {**self._published, actor: tip}
                return {"output": f"Published {actor} at {tip}", "exit_code": 0,
                        "published": self.shared_state()["published"]}
            if old==self.base_commit:
                self._exec(actor,f"git update-ref refs/remotes/align/{target} {old}",check=True)
                return {"output":f"Fetched align/{target} at {old}; inspect or merge it locally.",
                        "exit_code":0,"published":self.shared_state()["published"]}
            bundle = self._binary(self._broker, f"git -C {BARE} bundle create - refs/heads/{target} ^{self.base_commit}")
            self._binary(self.containers[actor],
                "cd /workspace/repo && cat > /tmp/align-transfer.bundle && "
                f"git fetch --no-tags /tmp/align-transfer.bundle +refs/heads/{target}:refs/remotes/align/{target} "
                "&& rm -f /tmp/align-transfer.bundle", bundle)
            return {"output": f"Fetched align/{target} at {old}; inspect or merge it locally.",
                    "exit_code": 0, "published": self.shared_state()["published"]}

    def patches(self):
        """Merge actual commit ancestry in the broker, then evaluate the merged tree once."""
        with self._git_lock:
            self.deadline.remaining()
            for actor in ACTORS:
                self._exec(actor, "git add -A && git -c core.hooksPath=/dev/null "
                           "-c user.name=ALIGN -c user.email=align@invalid "
                           "commit --allow-empty --no-gpg-sign -m 'ALIGN final snapshot'", check=True)
                self._import_broker(actor, final=True)
            self._broker_exec(
                f"git clone --no-local {BARE} {MERGE_DIR} && "
                f"git -C {MERGE_DIR} fetch {BARE} "
                "refs/align/final/agent1:refs/heads/align-final-agent1 "
                "refs/align/final/agent2:refs/heads/align-final-agent2 && "
                f"git -C {MERGE_DIR} checkout align-final-agent1", check=True)
            merged = self._broker_exec(
                f"git -C {MERGE_DIR} -c core.hooksPath=/dev/null "
                "-c user.name=ALIGN -c user.email=align@invalid "
                "merge --no-edit --no-gpg-sign align-final-agent2")
            if merged.returncode:
                if self._broker_exec(f"git -C {MERGE_DIR} ls-files -u", check=True).stdout.strip():
                    return {"_merge_status": "conflicts", "agent1": "", "agent2": ""}
                raise RuntimeError("Shared Git final merge failed")
            patch = self._binary(self._broker,
                f"git -C {MERGE_DIR} diff --binary --no-ext-diff --no-textconv {self.base_commit} HEAD").decode("utf-8")
            # An identical full merged patch avoids double-application of integrated changes.
            return {"_merge_status": "clean", "agent1": patch, "agent2": patch}

    def close(self):
        # During failed base initialization the broker may not exist yet.
        with self._close_lock:
            failures = []
            if self._broker:
                try:
                    r = super()._docker(["rm", "-f", self._broker], timeout=30)
                    if r.returncode and "No such container" not in r.stderr:
                        raise RuntimeError("Git broker removal failed")
                    self._broker = None
                except Exception as exc:
                    failures.append(type(exc).__name__)
            try:
                super().close()
            except Exception as exc:
                failures.append(type(exc).__name__)
            if failures:
                raise RuntimeError("Could not remove all trial resources")
