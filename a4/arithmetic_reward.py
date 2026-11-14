"""Conservative, deterministic GSM8K intermediate-arithmetic shaping reward.

Reference equations come from GSM8K's <<expression=result>> annotations.
Only an explicit, numerically valid equation with the *same expression and result*
in a completion earns credit. This is a heuristic, not a proof of reasoning.
"""

from __future__ import annotations

import ast
import re
from fractions import Fraction

# Basic numeric equations: 3 * 4 = 12, 5 + 2.5 = 7.5, 1,200/3 = 400.
# The intentionally conservative extractor does not claim to support every
# algebraic notation appearing in GSM8K or natural-language completions.
_NUMBER = r"-?\d[\d,]*(?:\.\d+)?"
_TERM = rf"\(?{_NUMBER}\)?"
_EQUATION = re.compile(
    rf"(?<![\w.])(?P<expression>{_TERM}(?:[ \t]*[+*/×÷\-][ \t]*{_TERM})+)"
    rf"[ \t]*=[ \t]*(?P<result>{_NUMBER})(?![\w.])"
)
_ANNOTATION = re.compile(r"<<([^<>]+)>>")
_MAX_EXPRESSION_LENGTH = 120
_MAX_ABSOLUTE_VALUE = 10**12


def _normalize(expression: str) -> str:
    return (
        expression.replace(",", "")
        .replace("×", "*")
        .replace("÷", "/")
        .replace("−", "-")
        .replace(" ", "")
        .replace("\t", "")
    )


def _evaluate(expression: str) -> Fraction | None:
    """Evaluate arithmetic safely with exact rational arithmetic, no eval()."""
    expr = _normalize(expression)
    if not expr or len(expr) > _MAX_EXPRESSION_LENGTH:
        return None
    try:
        tree = ast.parse(expr, mode="eval")

        def walk(node: ast.AST, depth: int = 0) -> Fraction:
            if depth > 30:
                raise ValueError("expression too deep")
            if isinstance(node, ast.Expression):
                value = walk(node.body, depth + 1)
            elif isinstance(node, ast.Constant) and type(node.value) in (int, float):
                value = Fraction(str(node.value))
            elif isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
                value = walk(node.operand, depth + 1)
                if isinstance(node.op, ast.USub):
                    value = -value
            elif isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
                lhs = walk(node.left, depth + 1)
                rhs = walk(node.right, depth + 1)
                if isinstance(node.op, ast.Add):
                    value = lhs + rhs
                elif isinstance(node.op, ast.Sub):
                    value = lhs - rhs
                elif isinstance(node.op, ast.Mult):
                    value = lhs * rhs
                else:
                    value = lhs / rhs
            else:
                raise ValueError("unsupported syntax")
            if abs(value) > _MAX_ABSOLUTE_VALUE:
                raise ValueError("magnitude too large")
            return value

        return walk(tree)
    except (SyntaxError, TypeError, ValueError, ZeroDivisionError, OverflowError):
        return None


def _validated_equation(expression: str, result: str) -> tuple[str, Fraction] | None:
    value = _evaluate(expression)
    try:
        claimed = Fraction(result.replace(",", ""))
    except (ValueError, ZeroDivisionError):
        return None
    if value is None or value != claimed:
        return None
    return _normalize(expression), value


def reference_calculations(reference_solution: str) -> set[tuple[str, Fraction]]:
    """Unique, valid annotated equations in a GSM8K reference answer."""
    steps: set[tuple[str, Fraction]] = set()
    for annotation in _ANNOTATION.findall(reference_solution):
        if "=" not in annotation:
            continue
        expression, result = annotation.rsplit("=", 1)
        equation = _validated_equation(expression, result.strip())
        if equation is not None:
            steps.add(equation)
    return steps


def completion_calculations(completion: str) -> set[tuple[str, Fraction]]:
    """Unique, valid explicit equations stated by the model."""
    steps: set[tuple[str, Fraction]] = set()
    for match in _EQUATION.finditer(completion):
        equation = _validated_equation(match.group("expression"), match.group("result"))
        if equation is not None:
            steps.add(equation)
    return steps


def arithmetic_score(reference_solution: str, completion: str) -> float:
    """Fraction of unique supported reference equations explicitly reproduced."""
    if not isinstance(reference_solution, str) or not isinstance(completion, str):
        raise TypeError("reference_solution and completion must be strings")
    reference = reference_calculations(reference_solution)
    if not reference:
        return 0.0
    matched = reference & completion_calculations(completion)
    return len(matched) / len(reference)


def arithmetic_reward(reference_solution: str, completion: str, weight: float = 0.20) -> float:
    """Shaping bonus in [0, weight]; correctness remains a separate reward."""
    if not 0 <= weight <= 1:
        raise ValueError("weight must be in [0, 1]")
    return weight * arithmetic_score(reference_solution, completion)
