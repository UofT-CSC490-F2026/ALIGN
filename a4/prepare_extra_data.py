"""Build the extra math data for SFT and midtraining as a JSONL file in nanochat's CustomJSON format.

    python a4/prepare_extra_data.py extra_math.jsonl

Sources: MetaMathQA (GSM_AnsAug and GSM_Rephrased rows) and Orca-Math 200K. Every answer is rewritten to end with
"#### <number>", the format nanochat's GSM8K task grades. Anything that shares a 13-word span (numbers blanked) with a
GSM8K test question is dropped, along with exact duplicates of test questions.
"""
import json
import random
import re
import sys

from datasets import load_dataset

SEED = 0
METAMATH_PER_TYPE = 40_000  # GSM_AnsAug and GSM_Rephrased, so 80K rows
ORCA_ROWS = 40_000
NGRAM = 13
NUMBER = r"-?\d[\d,]*\.?\d*"


def words(text):
    """Lowercase word tokens with every number replaced by '#', so changed numbers do not hide a copy."""
    return re.findall(r"[a-z#]+", re.sub(r"\d[\d,.]*", "#", text.lower()))


def grams(text, n=NGRAM):
    tokens = words(text)
    return {tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)}


def metamath(rows):
    """Keep the GSM_AnsAug and GSM_Rephrased rows and drop the trailing 'The answer is: N' line."""
    for row in rows:
        body, marker, answer = row["response"].rpartition("\n#### ")
        if not marker or not re.fullmatch(NUMBER, answer.split("\n")[0].strip()):
            continue
        yield row["type"], row["query"].strip(), f"{body}\n#### {answer.split(chr(10))[0].strip()}"


def orca(rows):
    """Orca-Math answers are prose. Keep those whose last number sits in the closing sentence, and make it the answer."""
    for row in rows:
        answer = row["answer"].strip()
        found = list(re.finditer(NUMBER, answer))
        if not found or len(answer) - found[-1].end() > 40:
            continue
        number = found[-1].group().rstrip(".,").replace(",", "")
        if re.fullmatch(r"-?\d+(\.\d+)?", number):
            yield "orca", row["question"].strip(), f"{answer}\n#### {number}"


def main(output):
    rng = random.Random(SEED)
    test = load_dataset("openai/gsm8k", "main", split="test")
    test_questions = {q.strip().lower() for q in test["question"]}
    test_grams = set().union(*(grams(q) for q in test["question"]))

    pools = {"GSM_AnsAug": [], "GSM_Rephrased": [], "orca": []}
    meta = load_dataset("meta-math/MetaMathQA", split="train").filter(lambda r: r["type"] in pools)
    for kind, question, answer in metamath(meta):
        pools[kind].append((question, answer))
    for kind, question, answer in orca(load_dataset("microsoft/orca-math-word-problems-200k", split="train")):
        pools[kind].append((question, answer))

    kept, report = [], {}
    for kind, limit in [("GSM_AnsAug", METAMATH_PER_TYPE), ("GSM_Rephrased", METAMATH_PER_TYPE), ("orca", ORCA_ROWS)]:
        rng.shuffle(pools[kind])
        taken, dropped = 0, 0
        for question, answer in pools[kind]:
            if taken == limit:
                break
            if question.lower() in test_questions or grams(question) & test_grams:
                dropped += 1
                continue
            kept.append([{"role": "user", "content": question}, {"role": "assistant", "content": answer}])
            taken += 1
        report[kind] = {"usable": len(pools[kind]), "dropped_as_overlap": dropped, "kept": taken}
    rng.shuffle(kept)
    with open(output, "w", encoding="utf-8") as f:
        for conversation in kept:
            f.write(json.dumps(conversation) + "\n")
    print(json.dumps(report, indent=2), f"\ntotal {len(kept)} rows -> {output}")


if __name__ == "__main__":
    main(sys.argv[1])
