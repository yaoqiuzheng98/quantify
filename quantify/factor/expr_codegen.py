"""Translate a Qlib factor expression into Python (numpy) code.

The generated code computes the factor value from raw OHLCV arrays
(``close``, ``open``, ``high``, ``low``, ``volume``, ``amount``,
``turnover_rate``) that are available via ``attribute_history`` in the
backtest engine.

Supported operators (subset of Qlib):
  - Binary: Add, Sub, Mul, Div, Power, Greater, Less, Gt, Ge, Lt, Le, Eq, Ne, And, Or
  - Unary: Abs, Sign, Log, Not
  - Rolling (single series + window): Ref, Mean, Sum, Std, Var, Max, Min, Med,
    Skew, Kurt, Delta, Rank, Count, IdxMax, IdxMin, EMA, WMA, Slope, Rsquare, Resi
  - Pair rolling: Corr, Cov
  - Conditional: If

Unsupported operators raise ``CodegenError`` so the caller can skip the factor.
"""

from __future__ import annotations

import re


class CodegenError(Exception):
    """Raised when an expression cannot be translated to Python."""


# ── Tokenizer ─────────────────────────────────────────────────────────────

_TOKEN_RE = re.compile(
    r"""
    (?P<NUMBER>\d+\.?\d*(?:[eE][+-]?\d+)?)  |   # 123, 1.5, 1e-8, 1.5e+10
    (?P<FIELD>\$[A-Za-z_]\w*)      |   # $close, $volume
    (?P<IDENT>[A-Za-z_]\w*)        |   # Mean, Sub, ...
    (?P<COMMA>,)                   |
    (?P<LPAREN>\()                 |
    (?P<RPAREN>\))                 |
    (?P<WS>\s+)
    """,
    re.VERBOSE,
)


def _tokenize(expr: str) -> list[tuple[str, str]]:
    tokens: list[tuple[str, str]] = []
    pos = 0
    while pos < len(expr):
        m = _TOKEN_RE.match(expr, pos)
        if not m:
            raise CodegenError(f"Cannot tokenize at position {pos}: {expr[pos:pos+20]}")
        kind = m.lastgroup
        val = m.group()
        if kind != "WS":
            tokens.append((kind, val))
        pos = m.end()
    return tokens


# ── Parser → AST ──────────────────────────────────────────────────────────

# AST nodes are tuples: (op, *args)
# - ("num", 3.14)
# - ("field", "close")
# - ("call", "Mean", [arg1, arg2, ...])
# - ("binop", "+", left, right)   — from Add/Sub/Mul/Div/Power/Greater/Less/...
# - ("unary", "-", arg)           — from Sub(0, x) → 0 - x
# - ("if", cond, then, else)


class _Parser:
    def __init__(self, tokens: list[tuple[str, str]]):
        self.tokens = tokens
        self.pos = 0

    def _peek(self) -> tuple[str, str] | None:
        return self.tokens[self.pos] if self.pos < len(self.tokens) else None

    def _consume(self, kind: str) -> str:
        tok = self._peek()
        if tok is None or tok[0] != kind:
            raise CodegenError(f"Expected {kind}, got {tok}")
        self.pos += 1
        return tok[1]

    def parse(self) -> tuple:
        node = self._parse_expr()
        if self.pos != len(self.tokens):
            raise CodegenError(f"Trailing tokens at {self.pos}: {self.tokens[self.pos:]}")
        return node

    def _parse_expr(self) -> tuple:
        return self._parse_atom()

    def _parse_atom(self) -> tuple:
        tok = self._peek()
        if tok is None:
            raise CodegenError("Unexpected end of expression")

        kind, val = tok

        if kind == "NUMBER":
            self.pos += 1
            return ("num", float(val))

        if kind == "FIELD":
            self.pos += 1
            field_name = val[1:]  # strip $
            return ("field", field_name)

        if kind == "IDENT":
            # Could be a function call
            self.pos += 1
            if self._peek() and self._peek()[0] == "LPAREN":
                self._consume("LPAREN")
                args: list[tuple] = []
                if self._peek() and self._peek()[0] != "RPAREN":
                    args.append(self._parse_expr())
                    while self._peek() and self._peek()[0] == "COMMA":
                        self._consume("COMMA")
                        args.append(self._parse_expr())
                self._consume("RPAREN")
                return ("call", val, args)
            raise CodegenError(f"Bare identifier: {val}")

        raise CodegenError(f"Unexpected token: {tok}")


# ── AST → Python code ─────────────────────────────────────────────────────

# Binary operators that map to Python infix
_BINOP_MAP = {
    "Add": "+", "Sub": "-", "Mul": "*", "Div": "/", "Power": "**",
    "Greater": ">", "Less": "<", "Gt": ">", "Ge": ">=", "Lt": "<",
    "Le": "<=", "Eq": "==", "Ne": "!=", "And": "&", "Or": "|",
}

# Rolling operators: (python_func_name, n_args, window_is_last_arg)
_ROLLING_OPS = {
    "Ref": ("_ref", 2, True),
    "Mean": ("_rolling_mean", 2, True),
    "Sum": ("_rolling_sum", 2, True),
    "Std": ("_rolling_std", 2, True),
    "Var": ("_rolling_var", 2, True),
    "Max": ("_rolling_max", 2, True),
    "Min": ("_rolling_min", 2, True),
    "Med": ("_rolling_median", 2, True),
    "Skew": ("_rolling_skew", 2, True),
    "Kurt": ("_rolling_kurt", 2, True),
    "Delta": ("_delta", 2, True),
    "Rank": ("_rolling_rank", 2, True),
    "Count": ("_rolling_count", 2, True),
    "IdxMax": ("_rolling_idxmax", 2, True),
    "IdxMin": ("_rolling_idxmin", 2, True),
    "EMA": ("_ema", 2, True),
    "WMA": ("_wma", 2, True),
    "Slope": ("_rolling_slope", 2, True),
    "Rsquare": ("_rolling_rsquare", 2, True),
    "Resi": ("_rolling_resi", 2, True),
}

_PAIR_OPS = {
    "Corr": ("_rolling_corr", 3, True),   # Corr(x, y, N)
    "Cov": ("_rolling_cov", 3, True),
}

_UNARY_OPS = {
    "Abs": "np.abs", "Sign": "np.sign", "Log": "np.log", "Not": "~",
}


def _ast_to_py(node: tuple) -> str:
    """Recursively convert AST node to a Python expression string."""
    tag = node[0]

    if tag == "num":
        return repr(node[1])

    if tag == "field":
        return node[1]  # bare variable name like 'close', 'volume'

    if tag == "binop":
        op, left, right = node[1], node[2], node[3]
        return f"({_ast_to_py(left)} {op} {_ast_to_py(right)})"

    if tag == "unary":
        op, arg = node[1], node[2]
        return f"({op}{_ast_to_py(arg)})"

    if tag == "if":
        cond, then, else_ = node[1], node[2], node[3]
        return f"np.where({_ast_to_py(cond)}, {_ast_to_py(then)}, {_ast_to_py(else_)})"

    if tag == "call":
        name = node[1]
        args = node[2]

        # Binary operators
        if name in _BINOP_MAP:
            if len(args) != 2:
                raise CodegenError(f"{name} expects 2 args, got {len(args)}")
            left = _ast_to_py(args[0])
            right = _ast_to_py(args[1])
            op = _BINOP_MAP[name]
            return f"({left} {op} {right})"

        # Unary operators
        if name in _UNARY_OPS:
            if len(args) != 1:
                raise CodegenError(f"{name} expects 1 arg, got {len(args)}")
            arg = _ast_to_py(args[0])
            py_func = _UNARY_OPS[name]
            return f"{py_func}({arg})"

        # If (3 args)
        if name == "If":
            if len(args) != 3:
                raise CodegenError(f"If expects 3 args, got {len(args)}")
            cond = _ast_to_py(args[0])
            then = _ast_to_py(args[1])
            else_ = _ast_to_py(args[2])
            return f"np.where({cond}, {then}, {else_})"

        # Rolling operators (single series + window)
        if name in _ROLLING_OPS:
            py_func, n_args, _ = _ROLLING_OPS[name]
            if len(args) != n_args:
                raise CodegenError(f"{name} expects {n_args} args, got {len(args)}")
            series = _ast_to_py(args[0])
            window = _ast_to_py(args[1])
            return f"{py_func}({series}, {window})"

        # Pair rolling operators
        if name in _PAIR_OPS:
            py_func, n_args, _ = _PAIR_OPS[name]
            if len(args) != n_args:
                raise CodegenError(f"{name} expects {n_args} args, got {len(args)}")
            x = _ast_to_py(args[0])
            y = _ast_to_py(args[1])
            window = _ast_to_py(args[2])
            return f"{py_func}({x}, {y}, {window})"

        raise CodegenError(f"Unsupported operator: {name}")

    raise CodegenError(f"Unknown AST node: {node}")


def _transform_binops(node: tuple) -> tuple:
    """Rewrite Add/Sub/Mul/.../Gt/... calls as binop nodes for readability."""
    tag = node[0]

    if tag == "num" or tag == "field":
        return node

    if tag == "call":
        name = node[1]
        args = [_transform_binops(a) for a in node[2]]

        if name in _BINOP_MAP:
            if len(args) == 2:
                return ("binop", _BINOP_MAP[name], args[0], args[1])
            # Sub(0, x) with 1 arg → unary minus
            if name == "Sub" and len(args) == 1:
                return ("unary", "-", args[0])

        if name == "If" and len(args) == 3:
            return ("if", args[0], args[1], args[2])

        return ("call", name, args)

    return node


def expression_to_python(expr: str) -> str:
    """Convert a Qlib expression string to a Python expression string.

    The generated code references variables ``close``, ``open_``, ``high``,
    ``low``, ``volume``, ``amount``, ``turnover_rate`` (numpy arrays) and
    helper functions ``_ref``, ``_rolling_mean``, etc.

    Raises ``CodegenError`` if the expression cannot be translated.
    """
    tokens = _tokenize(expr)
    parser = _Parser(tokens)
    ast = parser.parse()
    ast = _transform_binops(ast)
    return _ast_to_py(ast)


# ── Helper function source code (to be included in generated strategy) ────

HELPER_FUNCS = '''
import numpy as np

def _ref(arr, n):
    """Ref(arr, n) = arr shifted forward by n (value n bars ago)."""
    n = int(n)
    if n <= 0:
        return arr
    out = np.full_like(arr, np.nan, dtype=float)
    if n < len(arr):
        out[n:] = arr[:-n]
    return out

def _rolling_mean(arr, n):
    n = int(n)
    if n <= 0 or len(arr) < n:
        return np.array([np.nan] * len(arr))
    out = np.full_like(arr, np.nan, dtype=float)
    cumsum = np.cumsum(np.nan_to_num(arr))
    out[n - 1:] = (cumsum[n - 1:] - np.concatenate([[0], cumsum[:-n]])) / n
    return out

def _rolling_sum(arr, n):
    n = int(n)
    if n <= 0 or len(arr) < n:
        return np.array([np.nan] * len(arr))
    out = np.full_like(arr, np.nan, dtype=float)
    cumsum = np.cumsum(np.nan_to_num(arr))
    out[n - 1:] = cumsum[n - 1:] - np.concatenate([[0], cumsum[:-n]])
    return out

def _rolling_std(arr, n):
    n = int(n)
    if n <= 1 or len(arr) < n:
        return np.array([np.nan] * len(arr))
    out = np.full_like(arr, np.nan, dtype=float)
    for i in range(n - 1, len(arr)):
        out[i] = np.std(arr[i - n + 1:i + 1], ddof=1)
    return out

def _rolling_var(arr, n):
    n = int(n)
    if n <= 1 or len(arr) < n:
        return np.array([np.nan] * len(arr))
    out = np.full_like(arr, np.nan, dtype=float)
    for i in range(n - 1, len(arr)):
        out[i] = np.var(arr[i - n + 1:i + 1], ddof=1)
    return out

def _rolling_max(arr, n):
    n = int(n)
    if n <= 0 or len(arr) < n:
        return np.array([np.nan] * len(arr))
    out = np.full_like(arr, np.nan, dtype=float)
    for i in range(n - 1, len(arr)):
        out[i] = np.max(arr[i - n + 1:i + 1])
    return out

def _rolling_min(arr, n):
    n = int(n)
    if n <= 0 or len(arr) < n:
        return np.array([np.nan] * len(arr))
    out = np.full_like(arr, np.nan, dtype=float)
    for i in range(n - 1, len(arr)):
        out[i] = np.min(arr[i - n + 1:i + 1])
    return out

def _rolling_median(arr, n):
    n = int(n)
    if n <= 0 or len(arr) < n:
        return np.array([np.nan] * len(arr))
    out = np.full_like(arr, np.nan, dtype=float)
    for i in range(n - 1, len(arr)):
        out[i] = np.median(arr[i - n + 1:i + 1])
    return out

def _rolling_skew(arr, n):
    n = int(n)
    if n <= 2 or len(arr) < n:
        return np.array([np.nan] * len(arr))
    out = np.full_like(arr, np.nan, dtype=float)
    for i in range(n - 1, len(arr)):
        out[i] = float(pd.Series(arr[i - n + 1:i + 1]).skew())
    return out

def _rolling_kurt(arr, n):
    n = int(n)
    if n <= 3 or len(arr) < n:
        return np.array([np.nan] * len(arr))
    out = np.full_like(arr, np.nan, dtype=float)
    for i in range(n - 1, len(arr)):
        out[i] = float(pd.Series(arr[i - n + 1:i + 1]).kurt())
    return out

def _delta(arr, n):
    """Delta(arr, n) = arr - Ref(arr, n)."""
    n = int(n)
    return arr - _ref(arr, n)

def _rolling_rank(arr, n):
    """Rank(arr, n) = percentile rank of last value in window of n."""
    n = int(n)
    if n <= 0 or len(arr) < n:
        return np.array([np.nan] * len(arr))
    out = np.full_like(arr, np.nan, dtype=float)
    for i in range(n - 1, len(arr)):
        window = arr[i - n + 1:i + 1]
        out[i] = (window < arr[i]).sum() / (n - 1) if n > 1 else 0.5
    return out

def _rolling_count(arr, n):
    """Count(arr, n) = number of True values in window of n."""
    n = int(n)
    if n <= 0 or len(arr) < n:
        return np.array([np.nan] * len(arr))
    out = np.full_like(arr, np.nan, dtype=float)
    bool_arr = arr.astype(bool)
    cumsum = np.cumsum(bool_arr.astype(float))
    out[n - 1:] = cumsum[n - 1:] - np.concatenate([[0], cumsum[:-n]])
    return out

def _rolling_idxmax(arr, n):
    n = int(n)
    if n <= 0 or len(arr) < n:
        return np.array([np.nan] * len(arr))
    out = np.full_like(arr, np.nan, dtype=float)
    for i in range(n - 1, len(arr)):
        out[i] = np.argmax(arr[i - n + 1:i + 1])
    return out

def _rolling_idxmin(arr, n):
    n = int(n)
    if n <= 0 or len(arr) < n:
        return np.array([np.nan] * len(arr))
    out = np.full_like(arr, np.nan, dtype=float)
    for i in range(n - 1, len(arr)):
        out[i] = np.argmin(arr[i - n + 1:i + 1])
    return out

def _ema(arr, n):
    """EMA(arr, n) = exponential moving average with span n."""
    n = int(n)
    if n <= 0 or len(arr) == 0:
        return np.array([np.nan] * len(arr))
    alpha = 2.0 / (n + 1)
    out = np.full_like(arr, np.nan, dtype=float)
    out[0] = arr[0]
    for i in range(1, len(arr)):
        out[i] = alpha * arr[i] + (1 - alpha) * out[i - 1]
    return out

def _wma(arr, n):
    """WMA(arr, n) = weighted moving average, weights = 1,2,...,n."""
    n = int(n)
    if n <= 0 or len(arr) < n:
        return np.array([np.nan] * len(arr))
    weights = np.arange(1, n + 1, dtype=float)
    out = np.full_like(arr, np.nan, dtype=float)
    for i in range(n - 1, len(arr)):
        out[i] = np.dot(arr[i - n + 1:i + 1], weights) / weights.sum()
    return out

def _rolling_slope(arr, n):
    """Slope(arr, n) = linear regression slope of arr over window n."""
    n = int(n)
    if n <= 1 or len(arr) < n:
        return np.array([np.nan] * len(arr))
    out = np.full_like(arr, np.nan, dtype=float)
    x = np.arange(n, dtype=float)
    x_mean = x.mean()
    x_var = ((x - x_mean) ** 2).sum()
    for i in range(n - 1, len(arr)):
        y = arr[i - n + 1:i + 1]
        y_mean = y.mean()
        out[i] = ((x - x_mean) * (y - y_mean)).sum() / x_var
    return out

def _rolling_rsquare(arr, n):
    """Rsquare(arr, n) = R-squared of linear regression over window n."""
    n = int(n)
    if n <= 2 or len(arr) < n:
        return np.array([np.nan] * len(arr))
    out = np.full_like(arr, np.nan, dtype=float)
    x = np.arange(n, dtype=float)
    x_mean = x.mean()
    ss_xx = ((x - x_mean) ** 2).sum()
    for i in range(n - 1, len(arr)):
        y = arr[i - n + 1:i + 1]
        y_mean = y.mean()
        ss_yy = ((y - y_mean) ** 2).sum()
        ss_xy = ((x - x_mean) * (y - y_mean)).sum()
        if ss_yy > 0:
            out[i] = (ss_xy ** 2) / (ss_xx * ss_yy)
        else:
            out[i] = 0.0
    return out

def _rolling_resi(arr, n):
    """Resi(arr, n) = residual of last point from linear regression over window n."""
    n = int(n)
    if n <= 2 or len(arr) < n:
        return np.array([np.nan] * len(arr))
    out = np.full_like(arr, np.nan, dtype=float)
    x = np.arange(n, dtype=float)
    x_mean = x.mean()
    ss_xx = ((x - x_mean) ** 2).sum()
    for i in range(n - 1, len(arr)):
        y = arr[i - n + 1:i + 1]
        y_mean = y.mean()
        slope = ((x - x_mean) * (y - y_mean)).sum() / ss_xx
        intercept = y_mean - slope * x_mean
        predicted = slope * x[-1] + intercept
        out[i] = arr[i] - predicted
    return out

def _rolling_corr(x, y, n):
    """Corr(x, y, n) = rolling Pearson correlation over window n."""
    n = int(n)
    if n <= 2 or len(x) < n:
        return np.array([np.nan] * len(max(x, y, key=len)))
    out = np.full_like(x, np.nan, dtype=float)
    for i in range(n - 1, len(x)):
        xi = x[i - n + 1:i + 1]
        yi = y[i - n + 1:i + 1]
        if np.std(xi) > 0 and np.std(yi) > 0:
            out[i] = np.corrcoef(xi, yi)[0, 1]
    return out

def _rolling_cov(x, y, n):
    """Cov(x, y, n) = rolling covariance over window n."""
    n = int(n)
    if n <= 2 or len(x) < n:
        return np.array([np.nan] * len(max(x, y, key=len)))
    out = np.full_like(x, np.nan, dtype=float)
    for i in range(n - 1, len(x)):
        xi = x[i - n + 1:i + 1]
        yi = y[i - n + 1:i + 1]
        out[i] = np.cov(xi, yi, ddof=1)[0, 1]
    return out
'''
