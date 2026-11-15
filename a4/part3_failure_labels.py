"""Dependency-free copy of Laith's Part 3 heuristics for local research.

Source: a4/analyze_rl.py. Labels are hypotheses, not verified causal diagnoses.
Do not use these labels as correctness targets or production rewards.
"""
import re

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
