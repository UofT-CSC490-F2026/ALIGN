"""Plots and exploratory analysis for Part 3 from the saved RL training answers and the saved test-set answers.

    python a4/analyze_rl.py <rollouts-merged.json> <answers dir> <output dir> [digitized Karpathy curve .json]

rollouts-merged.json holds, for every training problem, the 16 answers the model sampled and their rewards.
The answers dir holds sft.jsonl and rl.jsonl, the greedy test-set answers written by dump_answers.py.
"""
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).parent
FINAL = re.compile(r"#### (\-?[0-9\.\,]+)")
NUMBER = re.compile(r"\d[\d,]*\.?\d*")
TOOL_RESULT = re.compile(r"<\|output_start\|>(.*?)<\|output_end\|>", re.S)
PYTHON = re.compile(r"<\|python_start\|>(.*?)<\|python_end\|>", re.S)
REPEAT = re.compile(r"(\d)\1{11,}")
COMMON = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 24, 52, 60, 100, 365, 1000}  # constants a solution may use without the question saying them


def number(text):
    try:
        return float(text.replace(",", "").rstrip("."))
    except ValueError:
        return None


def final_answer(text):
    match = FINAL.search(text)
    return number(match.group(1)) if match else None


def failure_kind(question, reference, answer):
    """One label for a wrong answer. The order matters: the first test that fits wins."""
    if "####" not in answer:
        if REPEAT.search(answer):
            return "no final answer: repeats a digit"
        tail = answer.rfind("<|python_start|>")
        if tail >= 0 and "<|python_end|>" not in answer[tail:]:
            return "no final answer: stops inside a calculator call"
        if answer.rstrip().endswith((".", "!", "?")):
            return "no final answer: stops after a full sentence"
        return "no final answer: stops mid-sentence"
    predicted, truth = final_answer(answer), final_answer(reference)
    if predicted is None or truth is None:
        return "wrong final answer: unreadable"
    results = [number(r) for r in TOOL_RESULT.findall(answer)]
    given = {number(n) for n in NUMBER.findall(question)}
    if predicted == -truth and truth != 0:
        return "wrong final answer: sign flipped"
    if abs(predicted - truth) <= 2:
        return "wrong final answer: off by 1 or 2"
    if truth and predicted and any(abs(predicted / truth - k) < 1e-9 or abs(predicted / truth - 1 / k) < 1e-9
                                   for k in (2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 100)):
        return "wrong final answer: off by a whole factor"
    if predicted in given:
        return "wrong final answer: repeats a number from the question"
    if len(results) > 1 and predicted in results[:-1]:
        return "wrong final answer: reports an earlier step's result"
    reasoning = TOOL_RESULT.sub(" ", PYTHON.sub(" ", answer))
    shown = {number(n) for n in NUMBER.findall(reasoning)} | set(results)
    operands = [number(n) for e in PYTHON.findall(answer) for n in NUMBER.findall(e)]
    if truth in shown | set(operands):
        return "wrong final answer: reaches the right value, then reports a different one"
    if any(o not in given and o not in results and o not in COMMON for o in operands):
        return "wrong final answer: uses a number found nowhere in the question or earlier steps"
    if any(g not in shown and g not in set(operands) for g in given):
        return "wrong final answer: never uses a number the question gives"
    if predicted != int(predicted):
        return "wrong final answer: not a whole number"
    return "wrong final answer: wrong setup or operation (all numbers present)"


def main(rollouts_path, answers_dir, out_dir, karpathy_path=None):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rollouts = json.load(open(rollouts_path))
    curves = json.load(open(HERE / "rl_eval_curves.json"))
    karpathy = json.load(open(karpathy_path)) if karpathy_path else None
    report = {}

    # ---- training reward per step and per problem
    by_step = defaultdict(list)
    pass_rates = []
    for row in rollouts:
        rewards = [s["reward"] for s in row["samples"]]
        by_step[row["step"]].append(np.mean(rewards))
        pass_rates.append((row["step"], row["problem"], float(np.mean(rewards)), sum(r > 0 for r in rewards)))
    steps = sorted(by_step)
    reward = np.array([np.mean(by_step[s]) for s in steps])
    smooth = np.convolve(reward, np.ones(20) / 20, mode="valid")
    report["training_reward"] = {"first_20_steps": float(reward[:20].mean()), "last_20_steps": float(reward[-20:].mean()),
                                 "overall": float(reward.mean())}
    kinds = Counter(r[3] for r in pass_rates)
    report["problems_by_samples_correct_out_of_16"] = {str(k): kinds[k] for k in range(17)}
    report["problem_groups"] = {"never solved (0 of 16)": kinds[0], "sometimes (1-15 of 16)": sum(kinds[k] for k in range(1, 16)),
                                "always solved (16 of 16)": kinds[16]}

    # ---- failures during training, by stage of training
    stages = [(0, 117), (117, 234), (234, 351), (351, 467)]
    kinds_all = Counter()
    kinds_by_stage = {st: Counter() for st in stages}
    lengths_by_stage = {st: [] for st in stages}
    answered_by_stage = {st: [0, 0] for st in stages}
    for row in rollouts:
        stage = next(st for st in stages if st[0] <= row["step"] < st[1])
        for sample in row["samples"]:
            text, ok = sample["text"], sample["reward"] > 0
            lengths_by_stage[stage].append(len(text))
            answered_by_stage[stage][0] += "####" in text
            answered_by_stage[stage][1] += 1
            if not ok:
                label = failure_kind(row["question"], row["reference"], text)
                kinds_all[label] += 1
                kinds_by_stage[stage][label] += 1
    total_wrong = sum(kinds_all.values())
    report["training_failure_kinds"] = {k: {"count": v, "share_of_wrong": round(v / total_wrong, 4)} for k, v in kinds_all.most_common()}
    report["by_stage"] = {f"steps {a}-{b - 1}": {
        "mean_answer_chars": float(np.mean(lengths_by_stage[(a, b)])),
        "share_with_final_answer": answered_by_stage[(a, b)][0] / answered_by_stage[(a, b)][1],
        "wrong_kinds": dict(kinds_by_stage[(a, b)].most_common())} for a, b in stages}

    # ---- which kinds of problems are hard: by question features
    feature = defaultdict(lambda: [0, 0])
    for row in rollouts:
        n = len(NUMBER.findall(row["question"]))
        key = "0-1" if n <= 1 else "2-3" if n <= 3 else "4-5" if n <= 5 else "6+"
        feature[key][0] += float(np.mean([s["reward"] for s in row["samples"]]))
        feature[key][1] += 1
    report["reward_by_numbers_in_question"] = {k: {"problems": v[1], "mean_reward": round(v[0] / v[1], 4)} for k, v in sorted(feature.items())}
    length_bins = defaultdict(lambda: [0, 0])
    for row in rollouts:
        words = len(row["question"].split())
        key = "<30 words" if words < 30 else "30-49" if words < 50 else "50-79" if words < 80 else "80+"
        length_bins[key][0] += float(np.mean([s["reward"] for s in row["samples"]]))
        length_bins[key][1] += 1
    report["reward_by_question_length"] = {k: {"problems": v[1], "mean_reward": round(v[0] / v[1], 4)} for k, v in length_bins.items()}

    # ---- cluster the questions by topic and report accuracy per cluster
    from sklearn.cluster import KMeans
    from sklearn.feature_extraction.text import TfidfVectorizer
    questions = [r["question"] for r in rollouts]
    vectoriser = TfidfVectorizer(stop_words="english", max_features=4000, min_df=5)
    matrix = vectoriser.fit_transform(questions)
    labels = KMeans(n_clusters=8, n_init=5, random_state=0).fit_predict(matrix)
    terms = np.array(vectoriser.get_feature_names_out())
    clusters = []
    for c in range(8):
        members = np.where(labels == c)[0]
        centre = np.asarray(matrix[members].mean(axis=0)).ravel()
        mean_reward = float(np.mean([np.mean([s["reward"] for s in rollouts[i]["samples"]]) for i in members]))
        no_answer = float(np.mean(["####" not in s["text"] for i in members for s in rollouts[i]["samples"]]))
        clusters.append({"cluster": c, "problems": int(len(members)), "top_words": list(terms[np.argsort(-centre)[:8]]),
                         "mean_reward": round(mean_reward, 4), "share_without_final_answer": round(no_answer, 4),
                         "example": rollouts[members[0]]["question"][:160]})
    report["question_clusters"] = sorted(clusters, key=lambda c: c["mean_reward"])

    # ---- test set: SFT versus RL, greedy answers
    sft = [json.loads(l) for l in open(Path(answers_dir) / "sft.jsonl")]
    rl = [json.loads(l) for l in open(Path(answers_dir) / "rl.jsonl")]
    both = sum(a["correct"] and b["correct"] for a, b in zip(sft, rl))
    report["test_set"] = {"sft_correct": sum(a["correct"] for a in sft), "rl_correct": sum(b["correct"] for b in rl),
                          "both": both, "sft_only": sum(a["correct"] and not b["correct"] for a, b in zip(sft, rl)),
                          "rl_only": sum(b["correct"] and not a["correct"] for a, b in zip(sft, rl)),
                          "neither": sum(not a["correct"] and not b["correct"] for a, b in zip(sft, rl))}
    for name, rows in [("sft", sft), ("rl", rl)]:
        kinds_test = Counter(failure_kind(r["question"], r["reference"], r["answer"]) for r in rows if not r["correct"])
        report[f"test_failure_kinds_{name}"] = dict(kinds_test.most_common())

    # ---- figures
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    ax = axes[0]
    ax.plot(steps, reward, color="#c7c0e8", lw=0.8, label="per step")
    ax.plot(steps[19:], smooth, color="#5b4bb7", lw=2, label="20-step average (ours)")
    if karpathy:
        k = karpathy["reward_smoothed"]
        ax.plot(k["step"], k["value"], "o--", color="#d9822b", label="Karpathy d20 (smoothed, read off plot)")
    ax.set_title("Training reward"); ax.set_xlabel("step"); ax.set_ylabel("mean reward"); ax.legend(fontsize=8)
    ax = axes[1]
    for name, series in curves.items():
        if name == "note":
            continue
        xs = sorted(int(s) for s in series)
        ax.plot(xs, [series[str(s)][0] for s in xs], "o-", label=name.split(" (")[0], alpha=0.9 if name.startswith("a4-rl3") else 0.45)
    if karpathy:
        ax.plot(karpathy["pass_at_1"]["step"], karpathy["pass_at_1"]["value"], "s--", color="#d9822b", label="Karpathy d20")
    ax.set_title("pass@1 (400 test problems)"); ax.set_xlabel("step"); ax.legend(fontsize=8)
    ax = axes[2]
    for name, series in curves.items():
        if name == "note":
            continue
        xs = sorted(int(s) for s in series)
        ax.plot(xs, [series[str(s)][7] for s in xs], "o-", label=name.split(" (")[0], alpha=0.9 if name.startswith("a4-rl3") else 0.45)
    ax.set_title("pass@8 (400 test problems)"); ax.set_xlabel("step"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out / "rl_curves.png", dpi=150); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    labels_sorted = list(report["training_failure_kinds"])
    axes[0].barh(range(len(labels_sorted)), [report["training_failure_kinds"][k]["count"] for k in labels_sorted], color="#5b4bb7")
    axes[0].set_yticks(range(len(labels_sorted))); axes[0].set_yticklabels(labels_sorted, fontsize=8); axes[0].invert_yaxis()
    axes[0].set_title("Wrong answers during training, by kind")
    groups = report["problem_groups"]
    axes[1].bar(list(groups), list(groups.values()), color=["#b3202f", "#d9822b", "#2f7a46"])
    axes[1].set_title("Training problems by how many of 16 samples were right"); axes[1].tick_params(axis="x", labelsize=8)
    fig.tight_layout(); fig.savefig(out / "rl_failures.png", dpi=150); plt.close(fig)

    json.dump(report, open(out / "rl_analysis.json", "w"), indent=2)
    print(json.dumps({k: report[k] for k in ("training_reward", "problem_groups", "test_set")}, indent=2))
    print("failure kinds (training):")
    for k, v in report["training_failure_kinds"].items():
        print(f"  {v['count']:6d}  {100 * v['share_of_wrong']:5.1f}%  {k}")


if __name__ == "__main__":
    main(*sys.argv[1:5])
