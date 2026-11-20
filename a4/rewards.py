
"""Additional GSM8K rewards for CSC490 A4 Part 4."""

import re

FINAL_ANSWER_PATTERN = re.compile(
    r"(?m)^[ \t]*####[ \t]*(-?\d[\d,]*(?:\.\d+)?)[ \t]*$"
)


def has_valid_final_answer(text: str) -> bool:
    """Whether the response ends with exactly one valid final answer."""
    if not isinstance(text, str):
        return False

    matches = list(FINAL_ANSWER_PATTERN.finditer(text))
    return (
        len(matches) == 1
        and text[matches[0].end():].strip() == ""
    )


def completion_reward(text: str, weight: float = 0.15) -> float:
    """Small bonus for producing a valid GSM8K final answer."""
    if weight < 0:
        raise ValueError("Reward weight must be nonnegative")

    return weight if has_valid_final_answer(text) else 0.0
