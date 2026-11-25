"""Reuse trusted CooperBench merge/test evaluator; not upstream agent orchestration."""
from pathlib import Path
from tempfile import TemporaryDirectory
from .task_loader import cooperbench_identity
from .environment import DockerEnvironment

UPSTREAM_REVISION="63b9d44d9f39a02fccf5bf0052db48a917a011fd"

def prepare_environment(task,image,*,timeout=30):
    cooperbench_identity(task)
    return DockerEnvironment(image,task["base_commit"],timeout=timeout)

def evaluation_evidence(task,result):
    _,_,features=cooperbench_identity(task)
    goals={}
    for i,f in enumerate(features,1):
        passed=result.get(f"feature{i}",{}).get("passed")
        goals[f"feature{f}"]="not_evaluated" if result.get("error") or type(passed) is not bool else "passed" if passed else "failed"
    both,merge=result.get("both_passed"),result.get("merge",{}).get("status")
    shared=both and merge in {"clean","identical"} if not result.get("error") and type(both) is bool and merge else None
    return {"goal_results":goals,"shared_goal_satisfied":shared,
            "critical_constraints_passed":None,"merge_status":merge,
            "evaluation_error":bool(result.get("error"))}

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
