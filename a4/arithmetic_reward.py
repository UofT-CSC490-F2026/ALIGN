"""Conservative reference-aligned arithmetic shaping. Never executes model code."""
from __future__ import annotations
import ast
import math
import re
from fractions import Fraction

_NUMBER = r"[+-]?(?:(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
_NUMERIC = re.compile(rf"{_NUMBER}\Z")
_ANNOTATION = re.compile(r"<<([^<>]+)>>")
_TOOL = re.compile(r"<\|python_start\|>(.*?)<\|python_end\|>\s*<\|output_start\|>(.*?)<\|output_end\|>", re.S)
_CHUNK = r"(?:\d[\d,.]*(?:[eE][+-]?\d+)?|\.\d+(?:[eE][+-]?\d+)?|[+*/()\- \t])+"
_EQUATION = re.compile(rf"(?<![\w.])({_CHUNK})=[ \t]*({_NUMBER})(?![\w.]|\d)")


def _normalize(text):
    return text.replace('\u00d7', '*').replace('\u00f7', '/').replace('\u2212', '-').strip()


def _parse(expression):
    expression = _normalize(expression)
    if not expression or len(expression) > 120 or not re.fullmatch(r'[\d.,eE+*/()\s-]+', expression):
        raise ValueError('expression length')
    for token in re.findall(r"[\d.,]+(?:[eE][+-]?\d+)?", expression):
        if not _NUMERIC.fullmatch(token) or ('e' in token.lower() and abs(int(token.lower().split('e')[1])) > 18):
            raise ValueError('invalid numeric literal')
    expression = expression.replace(',', '')
    tree = ast.parse(expression, mode='eval')

    def walk(node, depth=0):
        if depth > 30:
            raise ValueError('expression depth')
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            value = Fraction(ast.get_source_segment(expression, node))
            key = ('number', value)
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value, key = walk(node.operand, depth + 1)
            if isinstance(node.op, ast.USub):
                value, key = -value, ('negative', key)
            if key[0] == 'number' or (key[0] == 'negative' and key[1][0] == 'number'):
                key = ('number', value)
        elif isinstance(node, ast.BinOp) and type(node.op) in (ast.Add, ast.Sub, ast.Mult, ast.Div):
            left, lk = walk(node.left, depth + 1)
            right, rk = walk(node.right, depth + 1)
            if isinstance(node.op, ast.Add):
                value = left + right
            elif isinstance(node.op, ast.Sub):
                value = left - right
            elif isinstance(node.op, ast.Mult):
                value = left * right
            else:
                value = left / right
            key = (type(node.op).__name__, lk, rk)
        else:
            raise ValueError('unsupported arithmetic')
        if abs(value) > 10**12 or value.denominator > 10**18:
            raise ValueError('numeric bounds')
        return value, key

    value, key = walk(tree.body)
    if not any(isinstance(n, ast.BinOp) for n in ast.walk(tree)):
        raise ValueError('not a calculation')
    return key, value


def _validated_equation(expression, result):
    try:
        if len(result) > 120 or not _NUMERIC.fullmatch(result.strip()):
            return None
        if 'e' in result.lower() and abs(int(result.lower().split('e')[1])) > 18:
            return None
        key, value = _parse(expression)
        claimed = Fraction(result.strip().replace(',', ''))
        if abs(value - claimed) > Fraction(1, 10**9):
            return None
        return key, value
    except (SyntaxError, TypeError, ValueError, ZeroDivisionError, OverflowError, RecursionError):
        return None


def reference_calculations(reference_solution):
    steps = set()
    for annotation in _ANNOTATION.findall(reference_solution):
        if annotation.count('=') == 1:
            equation = _validated_equation(*annotation.split('='))
            if equation is not None:
                steps.add(equation)
    return steps


def completion_calculations(completion):
    steps = set()
    for match in _TOOL.finditer(completion):
        equation = _validated_equation(*match.groups())
        if equation is not None:
            steps.add(equation)
    ordinary = _TOOL.sub(' ', completion)
    ordinary = re.sub(r'<\|python_start\|>.*', '', ordinary, flags=re.S)
    ordinary = _normalize(ordinary)
    for match in _EQUATION.finditer(ordinary):
        expression, result = match.groups()
        # An operator immediately after a variable is not a standalone numeric
        # expression: never accept the numeric suffix of "x + 3*4=12".
        if expression.lstrip().startswith(('+', '-', '*', '/')) and ordinary[:match.start()].rstrip()[-1:].isalnum():
            continue
        equation = _validated_equation(expression, result)
        if equation is not None:
            steps.add(equation)
    return steps


def arithmetic_score(reference_solution: str, completion: str) -> float:
    """Unique matching expression trees / unique valid annotated reference steps.

    Numeric spellings and parentheses normalize; operand order is preserved.
    Algebraically equivalent rewrites are deliberately not inferred.
    """
    if not isinstance(reference_solution, str) or not isinstance(completion, str):
        raise TypeError('reference_solution and completion must be strings')
    reference = reference_calculations(reference_solution)
    return len(reference & completion_calculations(completion)) / len(reference) if reference else 0.0


def arithmetic_reward(reference_solution: str, completion: str, weight: float = 0.20) -> float:
    """Bonus in [0, weight], with weight bounded by the method's 0.20 cap."""
    if not math.isfinite(weight) or not 0 <= weight <= 0.20:
        raise ValueError('weight must be finite and in [0, 0.20]')
    return weight * arithmetic_score(reference_solution, completion)
