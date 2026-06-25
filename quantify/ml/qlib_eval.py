"""Qlib expression evaluator for runtime factor computation.

Parses Qlib-style factor expressions (e.g. ``Mean(Div(Sub($close, Ref($close, 1)), $close), 20)``)
and evaluates them from raw numpy arrays, without needing Qlib installed.

This is used by ML strategies to compute factor values at runtime inside the
BacktestEngine, where only ``attribute_history()`` / ``get_fundamentals()``
are available.

Supported operators (28):
    Binary:  Add, Sub, Mul, Div, Power, Corr, Cov, Ref, Gt, Lt, And, Or
    Unary:   Abs, Sign, Log, Sqrt, Mean, Std, Sum, Min, Max, Skew, Kurt,
             Var, Median, Rank, EMA, WMA, Delta, Slope, Resi, Quantile, IdxMax
    Special: If(cond, true, false)
    Terminals: $open, $high, $low, $close, $volume, $amount, $vwap,
               $turn, $pe, $pb, $ps, $total_mv, $circ_mv, numeric constants
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Parser: expression string → AST
# ---------------------------------------------------------------------------


class _Node:
    """AST node."""


class _Func(_Node):
    __slots__ = ("name", "args")

    def __init__(self, name: str, args: list[_Node]):
        self.name = name
        self.args = args


class _Field(_Node):
    __slots__ = ("name",)

    def __init__(self, name: str):
        self.name = name


class _Const(_Node):
    __slots__ = ("value",)

    def __init__(self, value: float):
        self.value = value


class _Neg(_Node):
    """Unary minus: -1 * expr"""

    __slots__ = ("expr",)

    def __init__(self, expr: _Node):
        self.expr = expr


def parse_expression(expr: str) -> _Node:
    """Parse a Qlib expression string into an AST.

    Handles:
    - Function calls: Func(arg1, arg2, ...)
    - Field references: $close, $volume
    - Numeric constants: 0, 1, 1e-8, 0.05
    - Unary minus: -1 * expr  (converted to Neg)
    """
    parser = _Parser(expr.strip())
    return parser.parse()


class _Parser:
    """Recursive descent parser for Qlib expressions."""

    def __init__(self, text: str):
        self.text = text
        self.pos = 0

    def peek(self) -> str:
        if self.pos >= len(self.text):
            return ""
        return self.text[self.pos]

    def skip_ws(self):
        while self.pos < len(self.text) and self.text[self.pos] in " \t\n\r":
            self.pos += 1

    def parse(self) -> _Node:
        self.skip_ws()
        node = self._parse_expr()
        self.skip_ws()
        return node

    def _parse_expr(self) -> _Node:
        self.skip_ws()

        # Handle leading "-1 * " pattern (common in factor library)
        if self.peek() == "-":
            self.pos += 1
            self.skip_ws()
            # Check if it's "-1 * expr"
            if self._peek_number():
                num = self._parse_number()
                self.skip_ws()
                if self.peek() == "*":
                    self.pos += 1
                    self.skip_ws()
                    inner = self._parse_expr()
                    return _Neg(inner)
                return _Const(-num)
            return _Neg(self._parse_expr())

        # Field reference: $close
        if self.peek() == "$":
            return self._parse_field()

        # Number
        if self._peek_number():
            return _Const(self._parse_number())

        # Function call: Name(args)
        return self._parse_func()

    def _peek_number(self) -> bool:
        c = self.peek()
        return c.isdigit() or c == "."

    def _parse_number(self) -> float:
        start = self.pos
        while self.pos < len(self.text) and self.text[self.pos] in "0123456789.eE+-":
            # Don't consume +/- unless they're part of a number (after e/E)
            if self.text[self.pos] in "+-" and self.pos > start and self.text[self.pos - 1] not in "eE":
                break
            self.pos += 1
        return float(self.text[start : self.pos])

    def _parse_field(self) -> _Field:
        assert self.peek() == "$"
        self.pos += 1  # skip $
        start = self.pos
        while self.pos < len(self.text) and (self.text[self.pos].isalnum() or self.text[self.pos] == "_"):
            self.pos += 1
        name = self.text[start : self.pos]
        return _Field(name)

    def _parse_func(self) -> _Node:
        start = self.pos
        while self.pos < len(self.text) and (self.text[self.pos].isalnum() or self.text[self.pos] == "_"):
            self.pos += 1
        name = self.text[start : self.pos]
        self.skip_ws()
        if self.peek() != "(":
            # Not a function — could be a bare constant like "0"
            try:
                return _Const(float(name))
            except ValueError:
                raise SyntaxError(f"Unexpected token: {name} at pos {start}")
        self.pos += 1  # skip (
        args = self._parse_args()
        self.skip_ws()
        if self.peek() != ")":
            raise SyntaxError(f"Expected ')' at pos {self.pos}, got '{self.peek()}'")
        self.pos += 1  # skip )
        return _Func(name, args)

    def _parse_args(self) -> list[_Node]:
        args = []
        self.skip_ws()
        if self.peek() == ")":
            return args
        args.append(self._parse_expr())
        self.skip_ws()
        while self.peek() == ",":
            self.pos += 1
            self.skip_ws()
            args.append(self._parse_expr())
            self.skip_ws()
        return args


# ---------------------------------------------------------------------------
# Evaluator: AST → numpy computation
# ---------------------------------------------------------------------------


def _safe_div(a, b):
    return np.divide(a, b, out=np.full_like(a, 0.0, dtype=float), where=np.abs(b) > 1e-10)


def _rolling_mean(x, w):
    s = pd.Series(x)
    return s.rolling(w, min_periods=1).mean().to_numpy()


def _rolling_std(x, w):
    s = pd.Series(x)
    return s.rolling(w, min_periods=1).std().to_numpy()


def _rolling_sum(x, w):
    s = pd.Series(x)
    return s.rolling(w, min_periods=1).sum().to_numpy()


def _rolling_min(x, w):
    s = pd.Series(x)
    return s.rolling(w, min_periods=1).min().to_numpy()


def _rolling_max(x, w):
    s = pd.Series(x)
    return s.rolling(w, min_periods=1).max().to_numpy()


def _rolling_skew(x, w):
    s = pd.Series(x)
    return s.rolling(w, min_periods=max(3, min(w, 5))).skew().to_numpy()


def _rolling_kurt(x, w):
    s = pd.Series(x)
    return s.rolling(w, min_periods=max(4, min(w, 5))).kurt().to_numpy()


def _rolling_median(x, w):
    s = pd.Series(x)
    return s.rolling(w, min_periods=1).median().to_numpy()


def _rolling_rank(x, w):
    s = pd.Series(x)
    return s.rolling(w, min_periods=1).rank(pct=True).to_numpy()


def _rolling_var(x, w):
    s = pd.Series(x)
    return s.rolling(w, min_periods=1).var().to_numpy()


def _rolling_quantile(x, w, q):
    s = pd.Series(x)
    return s.rolling(w, min_periods=1).quantile(q).to_numpy()


def _rolling_corr(x, y, w):
    sx, sy = pd.Series(x), pd.Series(y)
    return sx.rolling(w, min_periods=max(3, min(w, 5))).corr(sy).to_numpy()


def _rolling_cov(x, y, w):
    sx, sy = pd.Series(x), pd.Series(y)
    return sx.rolling(w, min_periods=max(3, min(w, 5))).cov(sy).to_numpy()


def _ref(x, n):
    s = pd.Series(x)
    return s.shift(n).to_numpy()


def _delta(x, n):
    s = pd.Series(x)
    return (s - s.shift(n)).to_numpy()


def _ema(x, n):
    s = pd.Series(x)
    return s.ewm(span=n, adjust=False).mean().to_numpy()


def _wma(x, n):
    s = pd.Series(x)
    weights = np.arange(1, n + 1, dtype=float)
    weights /= weights.sum()
    return (
        s.rolling(n, min_periods=1)
        .apply(lambda v: np.average(v, weights=weights[-len(v) :]), raw=True)
        .to_numpy()
    )


def _slope(x, w):
    """Rolling linear regression slope of x against time index."""
    s = pd.Series(x)

    def _slope_fn(v):
        if len(v) < 2:
            return 0.0
        x_arr = np.arange(len(v), dtype=float)
        x_mean = x_arr.mean()
        y_mean = v.mean()
        denom = ((x_arr - x_mean) ** 2).sum()
        if denom < 1e-12:
            return 0.0
        return ((x_arr - x_mean) * (v - y_mean)).sum() / denom

    return s.rolling(w, min_periods=2).apply(_slope_fn, raw=True).to_numpy()


def _resi(x, w):
    """Rolling linear regression residual (last point - fitted line)."""
    s = pd.Series(x)

    def _resi_fn(v):
        if len(v) < 2:
            return 0.0
        x_arr = np.arange(len(v), dtype=float)
        x_mean, y_mean = x_arr.mean(), v.mean()
        denom = ((x_arr - x_mean) ** 2).sum()
        if denom < 1e-12:
            return 0.0
        slope = ((x_arr - x_mean) * (v - y_mean)).sum() / denom
        intercept = y_mean - slope * x_mean
        fitted = slope * x_arr + intercept
        return v[-1] - fitted[-1]

    return s.rolling(w, min_periods=2).apply(_resi_fn, raw=True).to_numpy()


def _idxmax(x, w):
    """Rolling index of max value (position from start of window)."""
    s = pd.Series(x)
    return s.rolling(w, min_periods=1).apply(lambda v: np.argmax(v), raw=True).to_numpy()


# Operator dispatch table
_UNARY_OPS = {
    "Abs": np.abs,
    "Sign": np.sign,
    "Log": lambda x: np.log(np.abs(x) + 1e-10),
    "Sqrt": lambda x: np.sqrt(np.abs(x)),
}

_ROLLING_UNARY = {
    "Mean": _rolling_mean,
    "Std": _rolling_std,
    "Sum": _rolling_sum,
    "Min": _rolling_min,
    "Max": _rolling_max,
    "Skew": _rolling_skew,
    "Kurt": _rolling_kurt,
    "Var": _rolling_var,
    "Median": _rolling_median,
    "Rank": _rolling_rank,
    "IdxMax": _idxmax,
}

_ROLLING_BINARY = {
    "Corr": _rolling_corr,
    "Cov": _rolling_cov,
}


def evaluate(node: _Node, data: dict[str, np.ndarray]) -> np.ndarray:
    """Evaluate an AST node against a data dict mapping field names → 1-D arrays.

    Parameters
    ----------
    node : _Node
        Parsed expression AST (from ``parse_expression``).
    data : dict[str, np.ndarray]
        Maps field name (without $) → 1-D numpy array of historical values.
        Must contain at least all fields referenced in the expression.

    Returns
    -------
    np.ndarray
        1-D array of computed values, same length as inputs.
        The last element is the current day's factor value.
    """
    if isinstance(node, _Const):
        # Return array of constant, same length as any data field
        ref = next(iter(data.values())) if data else np.array([node.value])
        return np.full_like(ref, node.value, dtype=float)

    if isinstance(node, _Field):
        if node.name not in data:
            raise KeyError(f"Field '${node.name}' not in data")
        return data[node.name].astype(float)

    if isinstance(node, _Neg):
        return -evaluate(node.expr, data)

    if isinstance(node, _Func):
        name = node.name
        args = [evaluate(a, data) for a in node.args]

        # Binary arithmetic
        if name == "Add":
            return args[0] + args[1]
        if name == "Sub":
            return args[0] - args[1]
        if name == "Mul":
            return args[0] * args[1]
        if name == "Div":
            return _safe_div(args[0], args[1])
        if name == "Power":
            return np.power(args[0], args[1])

        # Comparison / logical
        if name == "Gt":
            return (args[0] > args[1]).astype(float)
        if name == "Lt":
            return (args[0] < args[1]).astype(float)
        if name == "And":
            return ((args[0] != 0) & (args[1] != 0)).astype(float)
        if name == "Or":
            return ((args[0] != 0) | (args[1] != 0)).astype(float)

        # If(cond, true, false)
        if name == "If":
            cond = args[0] != 0
            return np.where(cond, args[1], args[2])

        # Simple unary
        if name in _UNARY_OPS:
            return _UNARY_OPS[name](args[0])

        # Rolling unary: Func(x, window)
        if name in _ROLLING_UNARY:
            window = int(args[1][0]) if len(args) > 1 else 20
            return _ROLLING_UNARY[name](args[0], window)

        # Rolling binary: Func(x, y, window)
        if name in _ROLLING_BINARY:
            window = int(args[2][0]) if len(args) > 2 else 20
            return _ROLLING_BINARY[name](args[0], args[1], window)

        # Shift / delta
        if name == "Ref":
            n = int(args[1][0]) if len(args) > 1 else 1
            return _ref(args[0], n)
        if name == "Delta":
            n = int(args[1][0]) if len(args) > 1 else 1
            return _delta(args[0], n)

        # EMA / WMA
        if name == "EMA":
            n = int(args[1][0]) if len(args) > 1 else 5
            return _ema(args[0], n)
        if name == "WMA":
            n = int(args[1][0]) if len(args) > 1 else 5
            return _wma(args[0], n)

        # Slope / Resi
        if name == "Slope":
            w = int(args[1][0]) if len(args) > 1 else 20
            return _slope(args[0], w)
        if name == "Resi":
            w = int(args[1][0]) if len(args) > 1 else 20
            return _resi(args[0], w)

        # Quantile(x, w, q) — 3 args
        if name == "Quantile":
            w = int(args[1][0]) if len(args) > 1 else 20
            q = float(args[2][0]) if len(args) > 2 else 0.5
            return _rolling_quantile(args[0], w, q)

        raise ValueError(f"Unknown operator: {name}")

    raise TypeError(f"Unknown node type: {type(node)}")


def eval_expression(expr: str, data: dict[str, np.ndarray]) -> float:
    """Parse and evaluate a Qlib expression, returning the last (current) value.

    Parameters
    ----------
    expr : str
        Qlib expression string, e.g. ``"Mean($close, 20)"``.
    data : dict[str, np.ndarray]
        Maps field name → 1-D array of historical values.

    Returns
    -------
    float
        The last element of the computed array (today's factor value).
    """
    ast = parse_expression(expr)
    result = evaluate(ast, data)
    if len(result) == 0:
        return float("nan")
    val = float(result[-1])
    return val if np.isfinite(val) else float("nan")
