"""Multi-factor composition: pick factors from the library, standardize,
weight-combine them into a single score, select top-N stocks per day and
evaluate the resulting long-only portfolio.

This is a lightweight vectorized backtest on top of the Qlib factor panels —
it does **not** go through the event-driven ``backtest`` engine (which is
strategy-level).  The goal is a quick read on how well a basket of mined
factors works together before handing the selection to a real strategy.

Flow
----
1. Load passing factors from ``factor_library`` (ranked by |ICIR|).
2. For each factor, compute the raw value panel (date × asset) via Qlib.
3. Cross-sectional z-score standardize per day.
4. Align direction with ``sign(ic_mean)`` so that "high score = bullish".
5. Weight-combine (equal / IC / ICIR) into a composite score panel.
6. Each day, pick top-N stocks by composite score, equal-weight.
7. Compute portfolio daily returns, cumulative curve, and summary metrics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd

from quantify.database.factor_store import FactorRecord, list_factors
from quantify.utils.logger import log

WeightMethod = Literal["equal", "ic", "icir"]


@dataclass
class ComposeConfig:
    """Configuration for multi-factor composition."""

    universe: str | list[str] | None = None
    start_date: str | None = None
    end_date: str | None = None
    max_factors: int = 10
    top_n: int = 50
    weight: WeightMethod = "icir"
    min_icir: float = 0.3
    rebalance_freq: int = 5  # re-select every N trading days (1=daily)
    max_corr: float = 0.7  # drop a candidate if |corr| with an already-picked factor exceeds this


@dataclass
class ComposeResult:
    """Outcome of a composition run."""

    selected: list[FactorRecord] = field(default_factory=list)
    weights: dict[str, float] = field(default_factory=dict)
    daily_returns: pd.Series | None = None
    cumulative: pd.Series | None = None
    n_obs: int = 0
    metrics: dict[str, float] = field(default_factory=dict)
    holdings: pd.DataFrame | None = None  # (date, asset) -> weight


# ---------------------------------------------------------------------------
# factor panel
# ---------------------------------------------------------------------------


def compute_factor_panel(
    expression: str,
    universe: str | list[str] | None,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """Compute a single factor expression over the universe.

    Returns a ``DataFrame`` indexed by date, columns = asset codes, values =
    the raw factor value (NaN where missing).
    """
    from qlib.data import D

    from quantify.factor.evaluator import _resolve_universe
    from quantify.factor.qlib_data import init_qlib

    init_qlib()
    instruments = _resolve_universe(universe, start_date, end_date)
    if not instruments:
        return pd.DataFrame()

    raw = D.features(instruments, [expression], start_time=start_date, end_time=end_date)
    if raw is None or raw.empty:
        return pd.DataFrame()

    s = raw.iloc[:, 0].replace([np.inf, -np.inf], np.nan)
    s.index = s.index.set_names(["asset", "date"])
    s = s.reorder_levels(["date", "asset"]).sort_index()
    return s.unstack("asset")


def _cross_sectional_zscore(panel: pd.DataFrame) -> pd.DataFrame:
    """Standardize each row (cross-section) to zero mean / unit std."""
    mean = panel.mean(axis=1)
    std = panel.std(axis=1)
    # avoid division by zero
    std = std.replace(0, np.nan)
    return panel.sub(mean, axis=0).div(std, axis=0)


# ---------------------------------------------------------------------------
# factor selection
# ---------------------------------------------------------------------------


def _select_factors(
    factors: list[FactorRecord],
    max_factors: int,
    min_icir: float,
) -> list[FactorRecord]:
    """Pick top factors by |ICIR|, respecting the min_icir floor."""
    eligible = [f for f in factors if f.icir is not None and abs(f.icir) >= min_icir]
    eligible.sort(key=lambda f: abs(f.icir or 0), reverse=True)
    return eligible[:max_factors]


def _decorrelate(
    panels: dict[str, pd.DataFrame],
    selected: list[FactorRecord],
    max_corr: float,
) -> list[FactorRecord]:
    """Greedily drop factors that are too correlated with already-kept ones.

    Correlation is measured on the flattened (date, asset) factor values.
    """
    if max_corr >= 1.0 or len(selected) <= 1:
        return selected

    kept: list[FactorRecord] = []
    kept_series: list[pd.Series] = []
    for fac in selected:
        panel = panels.get(fac.expression)
        if panel is None or panel.empty:
            continue
        flat = panel.stack(dropna=False)
        too_correlated = False
        for prev in kept_series:
            aligned = pd.concat([flat, prev], axis=1, join="inner").dropna()
            if len(aligned) > 10 and abs(aligned.iloc[:, 0].corr(aligned.iloc[:, 1])) > max_corr:
                too_correlated = True
                break
        if not too_correlated:
            kept.append(fac)
            kept_series.append(flat)
    return kept


# ---------------------------------------------------------------------------
# composition
# ---------------------------------------------------------------------------


def _compute_weights(factors: list[FactorRecord], method: WeightMethod) -> dict[str, float]:
    """Return a dict mapping expression -> weight (sums to 1)."""
    raw: dict[str, float] = {}
    for f in factors:
        if method == "equal":
            raw[f.expression] = 1.0
        elif method == "ic":
            raw[f.expression] = abs(f.ic_mean or 0)
        else:  # icir
            raw[f.expression] = abs(f.icir or 0)
    total = sum(raw.values())
    if total <= 0:
        n = len(factors) or 1
        return {f.expression: 1.0 / n for f in factors}
    return {k: v / total for k, v in raw.items()}


def _compose_score(
    panels: dict[str, pd.DataFrame],
    factors: list[FactorRecord],
    weights: dict[str, float],
) -> pd.DataFrame:
    """Combine standardized factor panels into a single composite score."""
    composite = None
    for fac in factors:
        panel = panels.get(fac.expression)
        if panel is None or panel.empty:
            continue
        z = _cross_sectional_zscore(panel)
        # align direction: if ic_mean < 0, flip so that high score = bullish
        direction = 1.0 if (fac.ic_mean or 0) >= 0 else -1.0
        w = weights.get(fac.expression, 0.0)
        contribution = z * direction * w
        composite = contribution if composite is None else composite.add(contribution, fill_value=0.0)
    if composite is None:
        return pd.DataFrame()
    return composite


# ---------------------------------------------------------------------------
# portfolio selection & backtest
# ---------------------------------------------------------------------------


def _select_top_n(score: pd.DataFrame, top_n: int, freq: int) -> pd.DataFrame:
    """Pick top-N stocks by score every ``freq`` trading days.

    Returns a (date, asset) -> weight DataFrame (equal weight within a rebalance).
    """
    if score.empty:
        return pd.DataFrame()

    holdings = []
    rebalance_days = score.index[::freq]
    last_pick: pd.Series | None = None
    for date in score.index:
        if date in rebalance_days:
            row = score.loc[date].dropna()
            if len(row) >= top_n:
                last_pick = row.nlargest(top_n)
            elif len(row) > 0:
                last_pick = row.nlargest(min(top_n, len(row)))
            else:
                last_pick = None
        if last_pick is not None and len(last_pick) > 0:
            w = 1.0 / len(last_pick)
            for asset in last_pick.index:
                holdings.append({"date": date, "asset": asset, "weight": w})
    if not holdings:
        return pd.DataFrame()
    df = pd.DataFrame(holdings)
    return df.pivot_table(index="date", columns="asset", values="weight", fill_value=0.0)


def _portfolio_returns(holdings: pd.DataFrame, close_panel: pd.DataFrame) -> pd.Series:
    """Compute daily portfolio returns from holdings and close prices.

    ``close_panel`` is date × asset close prices.  Returns are computed as
    the weighted average of individual stock daily returns, with the weights
    fixed between rebalances (forward-looking holdings applied to same-day returns).
    """
    if holdings.empty or close_panel.empty:
        return pd.Series(dtype=float)

    # align dates
    common_dates = holdings.index.intersection(close_panel.index)
    if common_dates.empty:
        return pd.Series(dtype=float)
    holdings = holdings.loc[common_dates]
    close_panel = close_panel.loc[common_dates]

    daily_ret = close_panel.pct_change().fillna(0.0)
    # shift holdings by one day so that today's return is earned on yesterday's holdings
    holdings_shifted = holdings.shift(1).fillna(0.0)
    portfolio_ret = (holdings_shifted * daily_ret).sum(axis=1)
    # first day has no return (no prior holdings)
    if len(portfolio_ret) > 0:
        portfolio_ret.iloc[0] = 0.0
    return portfolio_ret


def _summarize(returns: pd.Series) -> dict[str, float]:
    """Compute annualized return / Sharpe / max drawdown from daily returns."""
    if returns.empty:
        return {}
    n_days = len(returns)
    total_return = float((1 + returns).prod() - 1)
    ann_return = float((1 + total_return) ** (252 / n_days) - 1) if n_days > 0 else 0.0
    ann_vol = float(returns.std() * np.sqrt(252)) if n_days > 1 else 0.0
    sharpe = float(ann_return / ann_vol) if ann_vol > 0 else 0.0
    cumulative = (1 + returns).cumprod()
    peak = cumulative.cummax()
    drawdown = (cumulative - peak) / peak
    max_dd = float(drawdown.min()) if len(drawdown) > 0 else 0.0
    win_rate = float((returns > 0).sum() / n_days) if n_days > 0 else 0.0
    return {
        "total_return": total_return,
        "ann_return": ann_return,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "win_rate": win_rate,
        "n_days": n_days,
    }


# ---------------------------------------------------------------------------
# public entry
# ---------------------------------------------------------------------------


def compose_factors(config: ComposeConfig | None = None) -> ComposeResult:
    """Run the multi-factor composition pipeline."""
    config = config or ComposeConfig()

    # default evaluation window
    if not config.start_date or not config.end_date:
        from quantify.factor.evaluator import evaluation_window_default

        ds, de = evaluation_window_default()
        config.start_date = config.start_date or ds
        config.end_date = config.end_date or de

    result = ComposeResult()

    # 1. load & select factors (all statuses — no gate during mining)
    all_factors = list_factors()
    if not all_factors:
        log.warning("因子库为空，无法构建组合。请先运行 `quantify factor mine`。")
        return result

    selected = _select_factors(all_factors, config.max_factors, config.min_icir)
    if not selected:
        log.warning(f"没有因子满足 |ICIR|>={config.min_icir}，无法构建组合。")
        return result
    log.info(f"选定 {len(selected)} 个因子参与合成：")
    for f in selected:
        log.info(f"  {f.name}  IC={f.ic_mean:.4f} IR={f.icir:.4f}  {f.expression}")

    # 2. compute factor panels
    panels: dict[str, pd.DataFrame] = {}
    for f in selected:
        log.info(f"  计算因子面板: {f.expression}")
        panel = compute_factor_panel(f.expression, config.universe, config.start_date, config.end_date)
        if panel.empty:
            log.warning(f"  因子面板为空，跳过: {f.expression}")
            continue
        panels[f.expression] = panel

    selected = [f for f in selected if f.expression in panels]
    if not selected:
        log.warning("所有因子面板均为空，无法构建组合。")
        return result

    # 3. decorrelation filter
    if config.max_corr < 1.0 and len(selected) > 1:
        before = len(selected)
        selected = _decorrelate(panels, selected, config.max_corr)
        if len(selected) < before:
            log.info(f"去相关过滤：{before} -> {len(selected)} 个因子")

    # 4. weights
    weights = _compute_weights(selected, config.weight)
    result.selected = selected
    result.weights = weights
    log.info(f"合成方式={config.weight}  权重: " + ", ".join(f"{k[:30]}={v:.3f}" for k, v in weights.items()))

    # 5. composite score
    score = _compose_score(panels, selected, weights)
    if score.empty:
        log.warning("合成分数为空。")
        return result

    # 6. select stocks
    holdings = _select_top_n(score, config.top_n, config.rebalance_freq)
    if holdings.empty:
        log.warning("选股结果为空。")
        return result
    result.holdings = holdings
    log.info(
        f"选股完成：{holdings.shape[0]} 个交易日，每期最多 {config.top_n} 只，调仓频率 {config.rebalance_freq} 日"
    )

    # 7. portfolio returns — need close prices
    from qlib.data import D

    from quantify.factor.evaluator import _resolve_universe
    from quantify.factor.qlib_data import init_qlib

    init_qlib()
    instruments = _resolve_universe(config.universe, config.start_date, config.end_date)
    close_raw = D.features(instruments, ["$close"], start_time=config.start_date, end_time=config.end_date)
    if close_raw is not None and not close_raw.empty:
        close_s = close_raw.iloc[:, 0]
        close_s.index = close_s.index.set_names(["asset", "date"])
        close_s = close_s.reorder_levels(["date", "asset"]).sort_index()
        close_panel = close_s.unstack("asset")
    else:
        close_panel = pd.DataFrame()

    returns = _portfolio_returns(holdings, close_panel)
    if returns.empty:
        log.warning("组合收益序列为空。")
        return result

    result.daily_returns = returns
    result.cumulative = (1 + returns).cumprod()
    result.n_obs = len(returns)
    result.metrics = _summarize(returns)

    m = result.metrics
    log.info(
        f"组合回测：总收益={m.get('total_return', 0):.2%}  年化={m.get('ann_return', 0):.2%}  "
        f"夏普={m.get('sharpe', 0):.3f}  最大回撤={m.get('max_drawdown', 0):.2%}  "
        f"日胜率={m.get('win_rate', 0):.2%}"
    )
    return result
