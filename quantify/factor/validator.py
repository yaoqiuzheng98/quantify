"""Syntax validation for Qlib factor expressions.

Catches the obvious mistakes (unbalanced parentheses, unknown fields, unknown
operators, empty expression) *before* we pay the cost of evaluating them with
Qlib. The authoritative check is still Qlib's own parser in
:func:`quantify.factor.evaluator`, but failing fast here keeps the LLM loop
cheap and gives the model precise, actionable error messages.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from quantify.factor.qlib_data import QLIB_FIELDS

# Qlib Expression operators (element-wise + rolling + pair-wise + cross-section).
# Mirrors qlib.data.ops; kept as a allowlist so typos surface early.
QLIB_OPERATORS: frozenset[str] = frozenset(
    {
        # element-wise / unary
        "Abs",
        "Sign",
        "Log",
        "Power",
        "Mask",
        "Not",
        "Exp",
        # binary arithmetic / logic helpers
        "Add",
        "Sub",
        "Mul",
        "Div",
        "Greater",
        "Less",
        "And",
        "Or",
        "Gt",
        "Ge",
        "Lt",
        "Le",
        "Eq",
        "Ne",
        "If",
        # rolling (single series, window)
        "Ref",
        "Mean",
        "Sum",
        "Std",
        "Var",
        "Skew",
        "Kurt",
        "Max",
        "Min",
        "Med",
        "Mad",
        "Rank",
        "Count",
        "Delta",
        "Slope",
        "Rsquare",
        "Resi",
        "WMA",
        "EMA",
        "Quantile",
        "IdxMax",
        "IdxMin",
        # pair rolling (two series, window)
        "Corr",
        "Cov",
        # cross-sectional
        "CSRank",
        "CSZScore",
    }
)

_FIELD_RE = re.compile(r"\$([A-Za-z_][A-Za-z0-9_]*)")
_IDENT_RE = re.compile(r"(?<![\w$])([A-Za-z_][A-Za-z0-9_]*)\s*\(")
_ALLOWED_CHARS_RE = re.compile(r"^[\sA-Za-z0-9_$+\-*/%.,()<>=!&|]+$")


@dataclass
class ValidationResult:
    expression: str
    ok: bool
    error: str | None = None
    fields: tuple[str, ...] = ()
    operators: tuple[str, ...] = ()


def _check_parens(expr: str) -> str | None:
    depth = 0
    for ch in expr:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth < 0:
                return "括号不匹配：右括号多于左括号"
    if depth != 0:
        return "括号不匹配：左括号未闭合"
    return None


def validate_expression(expression: str) -> ValidationResult:
    """Lightweight static validation of a single Qlib expression string."""
    expr = (expression or "").strip()
    if not expr:
        return ValidationResult(expr, False, "表达式为空")

    if not _ALLOWED_CHARS_RE.match(expr):
        bad = sorted({ch for ch in expr if not _ALLOWED_CHARS_RE.match(ch)})
        return ValidationResult(expr, False, f"包含非法字符: {''.join(bad)!r}")

    if (paren_err := _check_parens(expr)) is not None:
        return ValidationResult(expr, False, paren_err)

    fields = tuple(dict.fromkeys(_FIELD_RE.findall(expr)))
    if not fields:
        return ValidationResult(expr, False, "表达式未引用任何数据字段（形如 $close）")

    unknown_fields = [f for f in fields if f.lower() not in {x.lower() for x in QLIB_FIELDS}]
    if unknown_fields:
        return ValidationResult(
            expr,
            False,
            f"未知字段 {unknown_fields}；可用字段: {sorted(QLIB_FIELDS)}",
        )

    operators = tuple(dict.fromkeys(_IDENT_RE.findall(expr)))
    unknown_ops = [o for o in operators if o not in QLIB_OPERATORS]
    if unknown_ops:
        return ValidationResult(
            expr,
            False,
            f"未知算子 {unknown_ops}；可用算子参考 Qlib Ops（如 Ref/Mean/Std/Corr/Rank）",
        )

    return ValidationResult(expr, True, None, fields, operators)
