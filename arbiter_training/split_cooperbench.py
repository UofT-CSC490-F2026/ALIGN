"""Pin and split CooperBench Lite by repository before labeling trajectories."""

import argparse
import json
from collections import Counter
from pathlib import Path

from huggingface_hub import hf_hub_download
from scripts.normalize_tasks import cooperbench_task_key

DATASET = "CooperBench/cooperbench-dataset"
REVISION = "b612b1a35af722751454813d9e5a7888f065fc9e"
VALIDATION_REPOS = {"huggingface_datasets_task", "pillow_task"}
TEST_REPOS = {"pallets_click_task", "typst_task"}


def source_json(filename):
    path = hf_hub_download(repo_id=DATASET, repo_type="dataset", revision=REVISION,
                           filename=filename)
    return json.loads(Path(path).read_text(encoding="utf-8"))


def make_split(subset):
    source = source_json(f"subsets/{subset}.json")
    report = source_json("gold_conflict_report.json")
    conflict_map = {
        (r["repo"], r["task_id"], min(r["f1"], r["f2"]), max(r["f1"], r["f2"])):
        r["has_conflict"] for r in report["all_results"]
    }
    rows = []
    for task in source["tasks"]:
        repo, task_id = task["repo"], task["task_id"]
        split = "test" if repo in TEST_REPOS else "validation" if repo in VALIDATION_REPOS else "train"
        for pair in task["pairs"]:
            f1, f2 = sorted(pair)
            rows.append({"repo": repo, "task_id": task_id, "f1": f1, "f2": f2,
                         "task_key": cooperbench_task_key(repo, task_id, f1, f2, REVISION),
                         "split": split,
                         # Used only for split audits, never copied to arbiter input.
                         "gold_patches_merge_conflict": conflict_map[(repo, task_id, f1, f2)]})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset", choices=["lite", "flash"], default="lite")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = make_split(args.subset)
    repos = {name: {r["repo"] for r in rows if r["split"] == name}
             for name in ("train", "validation", "test")}
    if any(repos[a] & repos[b] for a, b in (("train", "validation"),
                                            ("train", "test"), ("validation", "test"))):
        raise ValueError("Repository leakage across splits")
    counts = {name: {"pairs": sum(r["split"] == name for r in rows),
                     "merge_conflicts": sum(r["split"] == name and
                                            r["gold_patches_merge_conflict"] for r in rows),
                     "repos": sorted(repos[name])}
              for name in repos}
    result = {"source": DATASET, "revision": REVISION, "subset": args.subset,
              "split_policy": "repository-held-out; gold merge status used only for audit",
              "counts": counts, "pairs": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(counts, indent=2))


if __name__ == "__main__":
    main()
