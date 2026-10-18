"""
Bounded annotation templates with readable references and an arithmetic-only interpreter.
"""

from __future__ import annotations

import ast
import math
import operator
import re
from functools import lru_cache
from typing import Any

TOKEN = re.compile(r"\{\{(.*?)}}", re.DOTALL)
REFERENCE = re.compile(r"\[([^\[\]\n]{1,160})]")
FORMAT = re.compile(r"(?:\d{1,2})?(?:\.\d)?[fge%]|d")
FUNCTIONS = {"abs": abs, "min": min, "max": max, "round": round,
             "sqrt": math.sqrt, "floor": math.floor, "ceil": math.ceil}
OPERATORS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
             ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod,
             ast.Pow: operator.pow}


@lru_cache(maxsize=256)
def parse(expression: str) -> tuple[ast.Expression, tuple[str, ...], str]:
    """
    Parse a short expression once; never execute Python source or access application objects.
    """
    if len(expression) > 2048:
        raise ValueError("Formula is too long")
    references = []

    def substitute(match: re.Match) -> str:
        """
        Replace human-readable references with interpreter-owned local identifiers.
        """
        references.append(match.group(1))
        return f"_v{len(references) - 1}"

    source = REFERENCE.sub(substitute, expression.strip())
    body, separator, formatting = source.rpartition(":")
    if separator:
        source, formatting = body.strip(), formatting.strip()
        if not FORMAT.fullmatch(formatting):
            raise ValueError("Use numeric formats such as .1f, .2f, d or .0%")
    else:
        formatting = ""
    tree = ast.parse(source, mode="eval")
    if len(list(ast.walk(tree))) > 96 or len(references) > 32:
        raise ValueError("Formula is too complex")
    allowed = (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Call, ast.Name, ast.Load, ast.Constant,
               ast.UAdd, ast.USub, *OPERATORS)
    for node in ast.walk(tree):
        if not isinstance(node, allowed):
            raise TypeError("Unsupported formula syntax")
        if (isinstance(node, ast.Name) and node.id not in FUNCTIONS
                and not (node.id.startswith("_v") and node.id[2:].isdigit() and int(node.id[2:]) < len(references))):
            raise ValueError("Choose a variable using Add variable")
        if isinstance(node, ast.Constant):
            numeric(node.value)
    return tree, tuple(references), formatting


def numeric(value: Any) -> float | int:
    """
    Admit finite, bounded real arithmetic operands, excluding strings and arbitrary objects.
    """
    if type(value) not in (int, float) or not math.isfinite(value) or abs(value) > 1e15:
        raise ValueError("Formula requires a finite numeric value")
    return value


def evaluate(expression: str, values: dict[str, Any], bindings: dict[str, str] | None = None) -> str:
    """
    Evaluate approved syntax and format one result; unavailable inputs propagate as --.
    """
    tree, references, formatting = parse(expression)
    operands = [values.get((bindings or {}).get(name, name)) for name in references]
    if any(value is None for value in operands):
        return "--"

    def visit(node: ast.AST, depth: int = 0) -> Any:
        """
        Interpret only constants, approved references, arithmetic and named numeric functions.
        """
        if depth > 12:
            raise ValueError("Formula is too deeply nested")
        if isinstance(node, ast.Name) and node.id.startswith("_v") and node.id[2:].isdigit():
            index = int(node.id[2:])
            if index < len(operands):
                return operands[index]
        if isinstance(node, ast.Constant):
            return numeric(node.value)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = numeric(visit(node.operand, depth + 1))
            return value if isinstance(node.op, ast.UAdd) else -value
        if isinstance(node, ast.BinOp) and type(node.op) in OPERATORS:
            left, right = numeric(visit(node.left, depth + 1)), numeric(visit(node.right, depth + 1))
            if isinstance(node.op, ast.Pow) and abs(right) > 16:
                raise ValueError("Formula exponent exceeds 16")
            return numeric(OPERATORS[type(node.op)](left, right))
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id in FUNCTIONS and not node.keywords and 1 <= len(node.args) <= 8):
            args = [numeric(visit(arg, depth + 1)) for arg in node.args]
            return numeric(FUNCTIONS[node.func.id](*args))
        raise ValueError("Use variables, numbers, arithmetic, abs/min/max/round/sqrt/floor/ceil")

    result = visit(tree.body)
    if formatting:
        return format(int(numeric(result)) if formatting == "d" else numeric(result), formatting)
    if type(result) is float:
        return f"{numeric(result):.6g}"
    if type(result) in (str, bool, int):
        return str(result)[:256]
    raise ValueError("Unsupported readout value")


def render(text: str, values: dict[str, Any], bindings: dict[str, str] | None = None) -> str:
    """
    Replace up to 64 formula tokens, showing errors without interrupting image presentation.
    """
    def substitute(match: re.Match) -> str:
        """
        Keep malformed, missing and invalid math localized to the affected readout.
        """
        try:
            return evaluate(match.group(1), values, bindings)
        except (ValueError, TypeError, SyntaxError, ArithmeticError, RecursionError):
            return "#ERR"

    return TOKEN.sub(substitute, text, count=128)[:8192]


def render_item(item: dict[str, Any], values: dict[str, Any]) -> dict[str, Any]:
    """
    Resolve display text once, preserving a table's original cells even for string-valued inputs.
    """
    if "{{" not in item["text"]:
        return item
    if item["kind"] == "table":
        from .tables import cells

        text = "\n".join("\t".join(render(value, values, item["bindings"]).replace("\n", " ").replace("\t", " ")
                                 for value in row) for row in cells(item["text"]))[:8192]
    else:
        text = render(item["text"], values, item["bindings"])
    return {**item, "text": text}
