"""2D vectorized Qlib expression evaluator.

Evaluates factor expressions on 2D arrays (time × stocks) instead of 1D
arrays (time only).  This allows computing factor values for all stocks
in one pass, giving ~100x speedup over per-stock evaluation.

All rolling operations work along axis 0 (time).  The last row of the
result contains today's factor values for all stocks.
"""

from __future__ import annotations

import numpy as np

from .qlib_eval import _Field, _Const, _Func, _Neg


# ---------------------------------------------------------------------------
# 2D rolling functions — operate on (time × stocks) arrays, roll along axis 0
# ---------------------------------------------------------------------------


def _safe_div_2d(a, b):
    return np.divide(a, b, out=np.zeros_like(a, dtype=float), where=np.abs(b) > 1e-10)


def _rolling_mean_2d(x, w):
    n = x.shape[0]
    out = np.empty_like(x, dtype=float)
    csum = np.cumsum(np.nan_to_num(x, nan=0.0), axis=0)
    padded = np.vstack([np.zeros((1, x.shape[1])), csum])
    for i in range(n):
        s = max(0, i - w + 1)
        out[i] = (csum[i] - padded[s]) / (i - s + 1)
    return out


def _rolling_sum_2d(x, w):
    n = x.shape[0]
    out = np.empty_like(x, dtype=float)
    csum = np.cumsum(np.nan_to_num(x, nan=0.0), axis=0)
    padded = np.vstack([np.zeros((1, x.shape[1])), csum])
    for i in range(n):
        s = max(0, i - w + 1)
        out[i] = csum[i] - padded[s]
    return out


def _rolling_std_2d(x, w):
    n = x.shape[0]
    out = np.empty_like(x, dtype=float)
    xn = np.nan_to_num(x, nan=0.0)
    csum = np.cumsum(xn, axis=0)
    csum2 = np.cumsum(xn**2, axis=0)
    padded = np.vstack([np.zeros((1, x.shape[1])), csum])
    padded2 = np.vstack([np.zeros((1, x.shape[1])), csum2])
    for i in range(n):
        s = max(0, i - w + 1)
        cnt = i - s + 1
        if cnt < 2:
            out[i] = 0.0
        else:
            mean = (csum[i] - padded[s]) / cnt
            mean2 = (csum2[i] - padded2[s]) / cnt
            var = np.maximum((mean2 - mean * mean) * cnt / (cnt - 1), 0.0)  # ddof=1
            out[i] = np.sqrt(var)
    return out


def _rolling_min_2d(x, w):
    from numpy.lib.stride_tricks import sliding_window_view

    n = x.shape[0]
    out = np.empty_like(x, dtype=float)
    for i in range(min(w - 1, n)):
        s = max(0, i - w + 1)
        out[i] = np.min(x[s : i + 1], axis=0)
    if n >= w:
        windows = sliding_window_view(x, w, axis=0)
        out[w - 1 :] = windows.min(axis=-1)
    return out


def _rolling_max_2d(x, w):
    from numpy.lib.stride_tricks import sliding_window_view

    n = x.shape[0]
    out = np.empty_like(x, dtype=float)
    for i in range(min(w - 1, n)):
        s = max(0, i - w + 1)
        out[i] = np.max(x[s : i + 1], axis=0)
    if n >= w:
        windows = sliding_window_view(x, w, axis=0)
        out[w - 1 :] = windows.max(axis=-1)
    return out


def _rolling_var_2d(x, w):
    return _rolling_std_2d(x, w) ** 2


def _rolling_skew_2d(x, w):
    n = x.shape[0]
    out = np.empty_like(x, dtype=float)
    for i in range(n):
        s = max(0, i - w + 1)
        v = x[s : i + 1]  # (cnt, stocks)
        cnt = i - s + 1
        if cnt < 3:
            out[i] = 0.0
        else:
            n_ = cnt
            m = v.mean(axis=0)
            m3 = ((v - m) ** 3).sum(axis=0) / n_
            m2 = ((v - m) ** 2).sum(axis=0) / n_
            mask = m2 > 1e-24
            out[i] = np.where(
                mask, (m3 / np.where(mask, m2**1.5, 1.0)) * np.sqrt(n_ * (n_ - 1)) / (n_ - 2), 0.0
            )
    return out


def _rolling_kurt_2d(x, w):
    n = x.shape[0]
    out = np.empty_like(x, dtype=float)
    for i in range(n):
        s = max(0, i - w + 1)
        v = x[s : i + 1]
        cnt = i - s + 1
        if cnt < 4:
            out[i] = 0.0
        else:
            n_ = cnt
            m = v.mean(axis=0)
            sd = v.std(axis=0, ddof=1)
            mask = sd > 1e-12
            g2 = (
                ((n_ + 1) * n_ / ((n_ - 1) * (n_ - 2) * (n_ - 3)))
                * ((v - m) ** 4).sum(axis=0)
                / np.where(mask, sd**4, 1.0)
            )
            g2 -= 3 * (n_ - 1) ** 2 / ((n_ - 2) * (n_ - 3))
            out[i] = np.where(mask, g2, 0.0)
    return out


def _rolling_median_2d(x, w):
    n = x.shape[0]
    out = np.empty_like(x, dtype=float)
    for i in range(n):
        s = max(0, i - w + 1)
        out[i] = np.median(x[s : i + 1], axis=0)
    return out


def _rolling_rank_2d(x, w):
    """Rolling rank (pct) of last row within window, per column."""
    n = x.shape[0]
    out = np.empty_like(x, dtype=float)
    for i in range(n):
        s = max(0, i - w + 1)
        v = x[s : i + 1]  # (cnt, stocks)
        out[i] = (v <= x[i]).sum(axis=0) / v.shape[0]
    return out


def _rolling_quantile_2d(x, w, q):
    n = x.shape[0]
    out = np.empty_like(x, dtype=float)
    for i in range(n):
        s = max(0, i - w + 1)
        out[i] = np.quantile(x[s : i + 1], q, axis=0)
    return out


def _rolling_corr_2d(x, y, w):
    n = x.shape[0]
    out = np.empty_like(x, dtype=float)
    for i in range(n):
        s = max(0, i - w + 1)
        vx, vy = x[s : i + 1], y[s : i + 1]
        cnt = i - s + 1
        if cnt < 3:
            out[i] = 0.0
        else:
            mx, my = vx.mean(axis=0), vy.mean(axis=0)
            dx, dy = vx - mx, vy - my
            denom = np.sqrt((dx**2).sum(axis=0) * (dy**2).sum(axis=0))
            out[i] = np.where(denom > 1e-12, (dx * dy).sum(axis=0) / np.where(denom > 1e-12, denom, 1.0), 0.0)
    return out


def _rolling_cov_2d(x, y, w):
    n = x.shape[0]
    out = np.empty_like(x, dtype=float)
    for i in range(n):
        s = max(0, i - w + 1)
        vx, vy = x[s : i + 1], y[s : i + 1]
        cnt = i - s + 1
        if cnt < 3:
            out[i] = 0.0
        else:
            mx, my = vx.mean(axis=0), vy.mean(axis=0)
            out[i] = ((vx - mx) * (vy - my)).sum(axis=0) / (cnt - 1)
    return out


def _ref_2d(x, n):
    out = np.zeros_like(x, dtype=float)
    if n < x.shape[0]:
        out[n:] = x[: x.shape[0] - n]
    return out


def _delta_2d(x, n):
    out = np.zeros_like(x, dtype=float)
    if n < x.shape[0]:
        out[n:] = x[n:] - x[: x.shape[0] - n]
    return out


def _ema_2d(x, n):
    alpha = 2.0 / (n + 1)
    out = np.empty_like(x, dtype=float)
    out[0] = x[0]
    for i in range(1, x.shape[0]):
        out[i] = alpha * x[i] + (1 - alpha) * out[i - 1]
    return out


def _wma_2d(x, n):
    weights = np.arange(1, n + 1, dtype=float)
    weights /= weights.sum()
    out = np.empty_like(x, dtype=float)
    for i in range(x.shape[0]):
        s = max(0, i - n + 1)
        v = x[s : i + 1]
        w = weights[-len(v) :]
        out[i] = (v * w[:, None]).sum(axis=0)
    return out


def _slope_2d(x, w):
    n = x.shape[0]
    out = np.empty_like(x, dtype=float)
    for i in range(n):
        s = max(0, i - w + 1)
        v = x[s : i + 1]
        cnt = i - s + 1
        if cnt < 2:
            out[i] = 0.0
        else:
            x_arr = np.arange(cnt, dtype=float)
            x_mean = x_arr.mean()
            y_mean = v.mean(axis=0)
            denom = ((x_arr - x_mean) ** 2).sum()
            out[i] = np.where(
                denom > 1e-12, ((x_arr - x_mean)[:, None] * (v - y_mean)).sum(axis=0) / denom, 0.0
            )
    return out


def _resi_2d(x, w):
    n = x.shape[0]
    out = np.empty_like(x, dtype=float)
    for i in range(n):
        s = max(0, i - w + 1)
        v = x[s : i + 1]
        cnt = i - s + 1
        if cnt < 2:
            out[i] = 0.0
        else:
            x_arr = np.arange(cnt, dtype=float)
            x_mean = x_arr.mean()
            y_mean = v.mean(axis=0)
            denom = ((x_arr - x_mean) ** 2).sum()
            if denom < 1e-12:
                out[i] = 0.0
            else:
                slope = ((x_arr - x_mean)[:, None] * (v - y_mean)).sum(axis=0) / denom
                intercept = y_mean - slope * x_mean
                out[i] = v[-1] - (slope * x_arr[-1] + intercept)
    return out


def _idxmax_2d(x, w):
    n = x.shape[0]
    out = np.empty_like(x, dtype=float)
    for i in range(n):
        s = max(0, i - w + 1)
        out[i] = np.argmax(x[s : i + 1], axis=0).astype(float)
    return out


# ---------------------------------------------------------------------------
# Operator dispatch
# ---------------------------------------------------------------------------

_UNARY_OPS_2D = {
    "Abs": np.abs,
    "Sign": np.sign,
    "Log": lambda x: np.log(np.abs(x) + 1e-10),
    "Sqrt": lambda x: np.sqrt(np.abs(x)),
}

_ROLLING_UNARY_2D = {
    "Mean": _rolling_mean_2d,
    "Std": _rolling_std_2d,
    "Sum": _rolling_sum_2d,
    "Min": _rolling_min_2d,
    "Max": _rolling_max_2d,
    "Skew": _rolling_skew_2d,
    "Kurt": _rolling_kurt_2d,
    "Var": _rolling_var_2d,
    "Median": _rolling_median_2d,
    "Rank": _rolling_rank_2d,
    "IdxMax": _idxmax_2d,
}

_ROLLING_BINARY_2D = {
    "Corr": _rolling_corr_2d,
    "Cov": _rolling_cov_2d,
}


def evaluate_2d(node, data: dict[str, np.ndarray]) -> np.ndarray:
    """Evaluate an AST node on 2D arrays (time × stocks).

    Returns (time × stocks) array.  Last row = today's values for all stocks.
    """
    if isinstance(node, _Const):
        ref = next(iter(data.values()))
        return np.full_like(ref, node.value, dtype=float)

    if isinstance(node, _Field):
        if node.name not in data:
            raise KeyError(f"Field '${node.name}' not in data")
        return data[node.name].astype(float)

    if isinstance(node, _Neg):
        return -evaluate_2d(node.expr, data)

    if isinstance(node, _Func):
        name = node.name
        args = [evaluate_2d(a, data) for a in node.args]

        if name == "Add":
            return args[0] + args[1]
        if name == "Sub":
            return args[0] - args[1]
        if name == "Mul":
            return args[0] * args[1]
        if name == "Div":
            return _safe_div_2d(args[0], args[1])
        if name == "Power":
            return np.power(args[0], args[1])
        if name == "Gt":
            return (args[0] > args[1]).astype(float)
        if name == "Lt":
            return (args[0] < args[1]).astype(float)
        if name == "And":
            return ((args[0] != 0) & (args[1] != 0)).astype(float)
        if name == "Or":
            return ((args[0] != 0) | (args[1] != 0)).astype(float)
        if name == "If":
            cond = args[0] != 0
            return np.where(cond, args[1], args[2])
        if name in _UNARY_OPS_2D:
            return _UNARY_OPS_2D[name](args[0])
        if name in _ROLLING_UNARY_2D:
            window = int(args[1][0, 0]) if len(args) > 1 else 20
            return _ROLLING_UNARY_2D[name](args[0], window)
        if name in _ROLLING_BINARY_2D:
            window = int(args[2][0, 0]) if len(args) > 2 else 20
            return _ROLLING_BINARY_2D[name](args[0], args[1], window)
        if name == "Ref":
            n = int(args[1][0, 0]) if len(args) > 1 else 1
            return _ref_2d(args[0], n)
        if name == "Delta":
            n = int(args[1][0, 0]) if len(args) > 1 else 1
            return _delta_2d(args[0], n)
        if name == "EMA":
            n = int(args[1][0, 0]) if len(args) > 1 else 5
            return _ema_2d(args[0], n)
        if name == "WMA":
            n = int(args[1][0, 0]) if len(args) > 1 else 5
            return _wma_2d(args[0], n)
        if name == "Slope":
            w = int(args[1][0, 0]) if len(args) > 1 else 20
            return _slope_2d(args[0], w)
        if name == "Resi":
            w = int(args[1][0, 0]) if len(args) > 1 else 20
            return _resi_2d(args[0], w)
        if name == "Quantile":
            w = int(args[1][0, 0]) if len(args) > 1 else 20
            q = float(args[2][0, 0]) if len(args) > 2 else 0.5
            return _rolling_quantile_2d(args[0], w, q)
        raise ValueError(f"Unknown operator: {name}")

    raise TypeError(f"Unknown node type: {type(node)}")
