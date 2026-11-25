"""Reuse trusted CooperBench merge/test evaluator; not upstream agent orchestration."""
from importlib import metadata
import json
import re
from pathlib import Path
from tempfile import TemporaryDirectory
from .task_loader import cooperbench_identity
from .environment import DockerEnvironment

def installed_revision():
    """Verify the installed VCS dependency against the sole requirements pin."""
    requirements = Path(__file__).with_name("requirements-cooperbench.txt")
    pins = []
    for line in requirements.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        match = re.fullmatch(
            r"git\+(https://github\.com/cooperbench/CooperBench\.git)@([0-9a-f]{40})",
            line, flags=re.IGNORECASE)
        if match:
            pins.append(match.groups())
    if len(pins) != 1:
        raise RuntimeError("Expected exactly one CooperBench Git commit pin in requirements-cooperbench.txt")
    repository, expected = pins[0]
    try:
        raw = metadata.distribution("cooperbench").read_text("direct_url.json")
        source = json.loads(raw) if raw else {}
        vcs = source.get("vcs_info", {})
        installed = vcs.get("commit_id")
        url = source.get("url", "")
        if (vcs.get("vcs") != "git" or not isinstance(installed, str)
                or not isinstance(url, str)
                or url.rstrip("/").removesuffix(".git").lower()
                != repository.removesuffix(".git").lower()
                or installed.lower() != expected.lower()):
            raise ValueError("Unverified or mismatched VCS origin")
    except (metadata.PackageNotFoundError, OSError, ValueError, TypeError, AttributeError) as exc:
        raise RuntimeError(
            "CooperBench installation does not match the pinned Git revision. "
            "Install with python -m pip install -r runner/requirements-cooperbench.txt"
        ) from exc
    return installed.lower()


def prepare_environment(task,image,*,timeout=30):
    cooperbench_identity(task)
    return DockerEnvironment(image,task["base_commit"],timeout=timeout)

def evaluation_evidence(task,result):
    _,_,features=cooperbench_identity(task)
    goals={}
    for i,f in enumerate(features,1):
        evidence=result.get(f"feature{i}",{})
        passed=evidence.get("passed")
        # Upstream fills in false with no exit code when it skips merged tests.
        exit_code=evidence.get("exit_code")
        counts=[evidence.get(key) for key in ("tests_passed","tests_failed") if key in evidence]
        observed=(type(passed) is bool and
                  ("exit_code" not in evidence or type(exit_code) is int and exit_code>=0) and
                  (not counts or any(type(count) is int and count>0 for count in counts)))
        if result.get("error") or evidence.get("feature_id",f)!=f or not observed:
            goals[f"feature{f}"]="not_evaluated"
        else:
            goals[f"feature{f}"]="passed" if passed else "failed"
    both,merge=result.get("both_passed"),result.get("merge",{}).get("status")
    missing_tests=merge in {"clean","identical"} and "not_evaluated" in goals.values()
    evaluation_error=bool(result.get("error")) or missing_tests
    shared=None
    if not evaluation_error:
        if merge=="conflicts":
            shared=False
        elif merge in {"clean","identical"} and type(both) is bool:
            shared=both
    return {"goal_results":goals,"shared_goal_satisfied":shared,
            "critical_constraints_passed":True if not task["constraints"] and not evaluation_error else None,
            "merge_status":merge,
            "evaluation_error":evaluation_error}

def evaluate_patches(task,patches,dataset_dir,*,timeout=600):
    from cooperbench.eval.sandbox import test_merged
    repo,number,features=cooperbench_identity(task)
    with TemporaryDirectory() as tmp:
        paths=[Path(tmp)/f"agent{i}.patch" for i in (1,2)]
        for i,path in enumerate(paths,1):
            path.write_text(patches[f"agent{i}"],encoding="utf-8")
        result=test_merged(repo,number,*features,patch1=paths[0],patch2=paths[1],
                           timeout=timeout,backend="docker",dataset_dir=dataset_dir)
    return evaluation_evidence(task,result)
