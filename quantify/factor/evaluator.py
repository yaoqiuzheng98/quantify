"""Evaluate Qlib factor expressions with statistical gates + Alphalens.

Flow for a single expression
----------------------------
1. ``D.features`` computes the factor and ``$close`` over the universe/window.
2. Statistical-quality gates reject degenerate factors cheaply (too sparse,
   constant, all-NaN) before paying for Alphalens.
3. Alphalens ``get_clean_factor_and_forward_returns`` bins the factor and
   computes forward returns; we derive IC / Rank-IC / IR / t-stat, the
   long-short quantile spread and top-quantile turnover.
4. :class:`FactorEvaluation` carries the verdict + a text report that is fed
   back to the LLM for the next iteration.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import date

import numpy as np
import pandas as pd

from quantify.factor.qlib_data import init_qlib, list_instruments
from quantify.factor.validator import validate_expression
from quantify.utils.logger import log


@dataclass
class QualityThresholds:
    """Gates a factor must clear to be accepted into the library."""

    min_coverage: float = 0.6  # fraction of (date,asset) cells that are finite
    min_abs_ic: float = 0.02  # |mean IC| on the primary period
    min_abs_icir: float = 0.3  # |IC_IR| on the primary period
    min_abs_rank_ic: float = 0.02  # |mean Rank-IC| on the primary period
    max_alphalens_loss: float = 0.35  # alphalens max_loss (data dropped binning)


@dataclass
class PeriodMetrics:
    period: str
    ic_mean: float | None = None
    ic_std: float | None = None
    icir: float | None = None
    ic_tstat: float | None = None
    rank_ic_mean: float | None = None
    rank_icir: float | None = None
    quantile_spread: float | None = None
    turnover: float | None = None


@dataclass
class FactorEvaluation:
    expression: str
    passed: bool
    reason: str = ""
    coverage: float | None = None
    n_obs: int = 0
    primary_period: str | None = None
    periods: dict[str, PeriodMetrics] = field(default_factory=dict)

    # convenience copies of the primary period's metrics (for DB storage)
    ic_mean: float | None = None
    ic_std: float | None = None
    icir: float | None = None
    ic_tstat: float | None = None
    rank_ic_mean: float | None = None
    rank_icir: float | None = None
    quantile_spread: float | None = None
    turnover: float | None = None

    def to_dict(self) -> dict:
        data = asdict(self)
        data["periods"] = {k: asdict(v) for k, v in self.periods.items()}
        return data

    def to_feedback_text(self) -> str:
        """Compact, LLM-friendly summary of this evaluation."""
        head = f"表达式: {self.expression}\n结果: {'通过' if self.passed else '未通过'}"
        if self.reason:
            head += f"（原因: {self.reason}）"
        if not self.periods:
            return head
        lines = [head, f"覆盖率={_fmt(self.coverage)}  样本数={self.n_obs}  主周期={self.primary_period}"]
        lines.append("各周期指标: period | IC | Rank-IC | IC_IR | t | 多空收益差 | 换手")
        for key, pm in self.periods.items():
            lines.append(
                f"  {key}: IC={_fmt(pm.ic_mean)} RankIC={_fmt(pm.rank_ic_mean)} "
                f"IR={_fmt(pm.icir)} t={_fmt(pm.ic_tstat)} "
                f"spread={_fmt(pm.quantile_spread)} turn={_fmt(pm.turnover)}"
            )
        return "\n".join(lines)


def _fmt(value: float | None) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "NA"
    return f"{value:.4f}"


def _resolve_universe(
    universe: str | list[str] | None,
    start_date: str,
    end_date: str,
) -> list[str]:
    """Resolve a universe spec into Qlib instrument codes.

    - ``None`` / ``"all"``  -> every dumped instrument
    - an index code (e.g. ``"000300.SH"``) -> point-in-time constituent union
    - an explicit list of Tushare codes -> converted to Qlib codes
    """
    from quantify.factor.qlib_data import ts_code_to_qlib

    if universe is None or (isinstance(universe, str) and universe.lower() in {"all", ""}):
        return list_instruments()
    if isinstance(universe, list):
        return [ts_code_to_qlib(c) for c in universe]
    # treat as an index code -> constituent union over the window
    from quantify.backtest.universe import index_constituents_union

    members = index_constituents_union(universe, start_date, end_date)
    if not members:
        log.warning(f"Universe {universe!r} resolved to 0 constituents; falling back to all instruments.")
        return list_instruments()
    return [ts_code_to_qlib(c) for c in members]


def _return_columns(factor_data: pd.DataFrame) -> list[str]:
    return [c for c in factor_data.columns if c not in {"factor", "factor_quantile", "group"}]


def _ic_frame(factor_data: pd.DataFrame, ret_col: str, method: str) -> pd.Series:
    """Per-date cross-sectional correlation between factor and a return column."""

    def _corr(group: pd.DataFrame) -> float:
        if len(group) < 3:  # Need at least 3 stocks for meaningful correlation
            return np.nan
        return group["factor"].corr(group[ret_col], method=method, min_periods=3)

    return factor_data.groupby(level="date", group_keys=False).apply(_corr).dropna()


def _quantile_spread(factor_data: pd.DataFrame, ret_col: str) -> float | None:
    if "factor_quantile" not in factor_data.columns:
        return None
    by_q = factor_data.groupby("factor_quantile", observed=True)[ret_col].mean()
    if by_q.empty:
        return None
    return float(by_q.loc[by_q.index.max()] - by_q.loc[by_q.index.min()])


def evaluate_expression(
    expression: str,
    *,
    universe: str | list[str] | None = None,
    start_date: str = "2018-01-01",
    end_date: str = "2023-12-31",
    periods: tuple[int, ...] = (1, 5, 10),
    quantiles: int = 5,
    primary_period: int | None = None,
    thresholds: QualityThresholds | None = None,
) -> FactorEvaluation:
    """Compute and grade one Qlib factor expression."""
    thresholds = thresholds or QualityThresholds()

    syntax = validate_expression(expression)
    if not syntax.ok:
        return FactorEvaluation(expression, False, f"语法校验失败: {syntax.error}")

    primary = primary_period if primary_period is not None else periods[0]
    primary_key = f"{primary}D"

    try:
        factor_data, coverage, n_obs = _compute_factor_data(
            expression, universe, start_date, end_date, periods, quantiles, thresholds
        )
    except _GateError as exc:
        return FactorEvaluation(expression, False, str(exc), coverage=exc.coverage, n_obs=exc.n_obs)
    except Exception as exc:  # noqa: BLE001 - surface qlib/alphalens errors as a failed eval
        return FactorEvaluation(expression, False, f"求值/Alphalens 失败: {exc}")

    from alphalens.performance import quantile_turnover

    ret_cols = _return_columns(factor_data)
    period_metrics: dict[str, PeriodMetrics] = {}
    for col in ret_cols:
        ic = _ic_frame(factor_data, col, "pearson")
        rank_ic = _ic_frame(factor_data, col, "spearman")
        ic_mean = float(ic.mean()) if not ic.empty else None
        ic_std = float(ic.std()) if len(ic) > 1 else None
        icir = (ic_mean / ic_std) if (ic_mean is not None and ic_std not in (None, 0)) else None
        ic_tstat = (icir * np.sqrt(len(ic))) if (icir is not None and len(ic) > 0) else None
        rank_mean = float(rank_ic.mean()) if not rank_ic.empty else None
        rank_std = float(rank_ic.std()) if len(rank_ic) > 1 else None
        rank_icir = (rank_mean / rank_std) if (rank_mean is not None and rank_std not in (None, 0)) else None

        turnover = None
        if "factor_quantile" in factor_data.columns:
            top_q = int(factor_data["factor_quantile"].max())
            try:
                to = quantile_turnover(factor_data["factor_quantile"], top_q)
                turnover = float(to.mean()) if to is not None and len(to) else None
            except Exception:  # noqa: BLE001
                turnover = None

        period_metrics[col] = PeriodMetrics(
            period=col,
            ic_mean=ic_mean,
            ic_std=ic_std,
            icir=icir,
            ic_tstat=ic_tstat,
            rank_ic_mean=rank_mean,
            rank_icir=rank_icir,
            quantile_spread=_quantile_spread(factor_data, col),
            turnover=turnover,
        )

    pm = period_metrics.get(primary_key) or next(iter(period_metrics.values()), None)
    evaluation = FactorEvaluation(
        expression=expression,
        passed=False,
        coverage=coverage,
        n_obs=n_obs,
        primary_period=pm.period if pm else primary_key,
        periods=period_metrics,
    )
    if pm is not None:
        evaluation.ic_mean = pm.ic_mean
        evaluation.ic_std = pm.ic_std
        evaluation.icir = pm.icir
        evaluation.ic_tstat = pm.ic_tstat
        evaluation.rank_ic_mean = pm.rank_ic_mean
        evaluation.rank_icir = pm.rank_icir
        evaluation.quantile_spread = pm.quantile_spread
        evaluation.turnover = pm.turnover

    evaluation.passed, evaluation.reason = _grade(evaluation, thresholds)
    return evaluation


class _GateError(Exception):
    """Raised when a statistical-quality gate rejects the factor early."""

    def __init__(self, message: str, *, coverage: float | None = None, n_obs: int = 0) -> None:
        super().__init__(message)
        self.coverage = coverage
        self.n_obs = n_obs


def _compute_factor_data(
    expression: str,
    universe: str | list[str] | None,
    start_date: str,
    end_date: str,
    periods: tuple[int, ...],
    quantiles: int,
    thresholds: QualityThresholds,
) -> tuple[pd.DataFrame, float, int]:
    from alphalens.utils import get_clean_factor_and_forward_returns
    from qlib.data import D

    init_qlib()
    instruments = _resolve_universe(universe, start_date, end_date)
    if not instruments:
        raise _GateError("股票池为空（请先 dump-data 并确认 universe）")

    raw = D.features(instruments, [expression, "$close"], start_time=start_date, end_time=end_date)
    if raw is None or raw.empty:
        raise _GateError("Qlib 求值返回空（检查表达式与数据范围）")
    raw.columns = ["factor", "close"]

    total_cells = len(raw)
    finite = np.isfinite(raw["factor"].to_numpy(dtype="float64")).sum()
    coverage = float(finite / total_cells) if total_cells else 0.0
    if coverage < thresholds.min_coverage:
        raise _GateError(
            f"覆盖率过低 {coverage:.2%} < {thresholds.min_coverage:.0%}", coverage=coverage, n_obs=int(finite)
        )
    factor_series = raw["factor"].replace([np.inf, -np.inf], np.nan).dropna()
    if factor_series.nunique() <= 1:
        raise _GateError("因子为常数/无区分度", coverage=coverage, n_obs=int(finite))

    # Qlib returns a (instrument, datetime) MultiIndex; alphalens wants (date, asset).
    factor = factor_series.copy()
    factor.index = factor.index.set_names(["asset", "date"])
    factor = factor.reorder_levels(["date", "asset"]).sort_index()

    prices = raw["close"].copy()
    prices.index = prices.index.set_names(["asset", "date"])
    prices = prices.reorder_levels(["date", "asset"]).unstack("asset").sort_index()

    factor_data = get_clean_factor_and_forward_returns(
        factor,
        prices,
        quantiles=quantiles,
        periods=tuple(periods),
        max_loss=thresholds.max_alphalens_loss,
        filter_zscore=None,  # Disable lookahead-biased outlier filtering
    )
    return factor_data, coverage, int(len(factor_data))


def _grade(evaluation: FactorEvaluation, thresholds: QualityThresholds) -> tuple[bool, str]:
    if not evaluation.periods:
        return False, "无有效周期指标"
    ic = evaluation.ic_mean
    icir = evaluation.icir
    rank_ic = evaluation.rank_ic_mean
    if ic is None or icir is None:
        return False, "IC 指标缺失"
    if abs(ic) < thresholds.min_abs_ic:
        return False, f"|IC|={abs(ic):.4f} < {thresholds.min_abs_ic}"
    if abs(icir) < thresholds.min_abs_icir:
        return False, f"|IC_IR|={abs(icir):.4f} < {thresholds.min_abs_icir}"
    if rank_ic is not None and abs(rank_ic) < thresholds.min_abs_rank_ic:
        return False, f"|RankIC|={abs(rank_ic):.4f} < {thresholds.min_abs_rank_ic}"
    return True, "通过 IC/IR/RankIC 全部门槛"


def evaluation_window_default() -> tuple[str, str]:
    """A sensible default eval window from the dumped calendar (last ~6y)."""
    from quantify.factor.qlib_data import calendar_bounds

    lo, hi = calendar_bounds()
    if lo is None or hi is None:
        return "2018-01-01", "2023-12-31"
    start = max(lo, date(hi.year - 6, hi.month, hi.day))
    return start.isoformat(), hi.isoformat()
