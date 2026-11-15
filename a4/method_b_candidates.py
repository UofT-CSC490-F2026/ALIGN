"""Offline-only Method B candidates. NOT imported by the production RL path.

All candidates reuse the production restricted arithmetic parser. Reference trees
must match; B2 swaps children only at Add/Mult nodes (no reassociation/distribution).
Fixed weights are selected before measuring associations with training correctness.
"""
import re
from fractions import Fraction
from arithmetic_reward import (arithmetic_reward, completion_calculations,
                               reference_calculations, _TOOL, _validated_equation, _NUMERIC)
from rewards import has_valid_final_answer, FINAL_ANSWER_PATTERN

B_WEIGHT = .20
SMALL_WEIGHT = .05
# Structural constants, not a claim of semantic grounding or unit conversion.
STRUCTURAL_CONSTANTS = frozenset(map(Fraction, (0, 1, 2, 100)))
QUANTITY = re.compile(r'(?<![\w.])[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?')
WORDS = dict(zip(('zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty').split(), range(21)))
WORDS.update(half=Fraction(1, 2), twice=2, double=2, dozen=12, pair=2)


def commutative_key(tree):
    op = tree[0]
    if op == 'number':
        return tree
    if op == 'negative':
        return op, commutative_key(tree[1])
    left, right = commutative_key(tree[1]), commutative_key(tree[2])
    if op in ('Add', 'Mult') and repr(left) > repr(right):
        left, right = right, left
    return op, left, right


def canonical_steps(steps):
    return {(commutative_key(tree), value) for tree, value in steps}


def leaf_values(tree):
    if tree[0] == 'number':
        return {tree[1]}
    if tree[0] == 'negative':
        # Negative subexpression is structural; leaves still need provenance.
        return leaf_values(tree[1])
    return leaf_values(tree[1]) | leaf_values(tree[2])


def question_quantities(question):
    quantities = set()
    for token in QUANTITY.findall(question):
        try:
            if len(token) <= 120:
                quantities.add(Fraction(token.replace(',', '')))
        except (ValueError, ZeroDivisionError):
            pass
    quantities.update(Fraction(WORDS[w]) for w in re.findall(r'\b[a-z]+\b', question.lower()) if w in WORDS)
    return quantities


def grounded_matches(question, matched):
    """Numeric provenance closure, NOT semantic grounding or chronological proof.

    Only already-reference-aligned steps are considered. A step's literal operands
    must be question quantities, fixed structural constants, or outputs from other
    accepted steps. Closure allows reordering, but no invented dependency roots.
    """
    available = question_quantities(question) | STRUCTURAL_CONSTANTS
    pending, accepted = set(matched), set()
    while pending:
        ready = {step for step in pending if leaf_values(step[0]) <= available}
        if not ready:
            break
        accepted.update(ready)
        available.update(value for _, value in ready)
        pending.difference_update(ready)
    return accepted


def final_fraction(text):
    if not has_valid_final_answer(text):
        return None
    token = FINAL_ANSWER_PATTERN.search(text).group(1)
    if len(token) > 120 or not _NUMERIC.fullmatch(token):
        return None
    try:
        value = Fraction(token.replace(',', ''))
        return value if abs(value) <= 10**12 else None
    except (ValueError, ZeroDivisionError):
        return None


def last_tool(text):
    start = text.rfind('<|python_start|>')
    match = _TOOL.match(text, start) if start >= 0 else None
    return match, _validated_equation(*match.groups()) if match else None


def candidate_signals(question, reference, text):
    strict_ref = reference_calculations(reference)
    strict_generated = completion_calculations(text)
    reference_steps = canonical_steps(strict_ref)
    generated = canonical_steps(strict_generated)
    matched = reference_steps & generated
    grounded = grounded_matches(question, matched)
    denominator = len(reference_steps)
    b1 = arithmetic_reward(reference, text)  # Production scorer, unchanged.
    b2 = B_WEIGHT * len(matched) / denominator if denominator else 0.
    b3 = B_WEIGHT * len(grounded) / denominator if denominator else 0.
    final = final_fraction(text)
    match, tool_step = last_tool(text)
    consistent = tool_step is not None and final is not None and tool_step[1] == final
    relevant_last = tool_step is not None and (commutative_key(tool_step[0]), tool_step[1]) in reference_steps
    # Ungated consistency is only counted as a diagnostic, never awarded.
    b4 = SMALL_WEIGHT if consistent and relevant_last else 0.
    repeated = re.search(r'(\d)\1{11,}', text) is not None
    incomplete = '<|python_start|>' in text and match is None
    b5 = -SMALL_WEIGHT if repeated or incomplete else 0.
    numbers = question_quantities(question)
    outputs = {value for _, value in strict_generated}
    operands = set().union(*(leaf_values(tree) for tree, _ in strict_generated)) if strict_generated else set()
    return dict(B1=b1, B2=b2, B3=b3, B4=b4, B5=b5,
                diagnostics=dict(reference_steps=len(reference_steps), strict_reference_steps=len(strict_ref),
                                 generated_steps=len(generated), matched_steps=len(matched), grounded_steps=len(grounded),
                                 ungated_consistency=consistent, relevant_last=relevant_last,
                                 repeated_digit=repeated, incomplete_tool=incomplete,
                                 termination_cause='unknown',
                                 numeric_question_values=len(numbers),
                                 unused_numeric_values=len(numbers - operands - outputs),
                                 ungrounded_operand_values=len(operands - numbers - outputs - STRUCTURAL_CONSTANTS)))
