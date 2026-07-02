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

import re
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
    # Transaction costs (per-side rates, applied to turnover)
    commission_rate: float = 0.0005  # 0.05% per side
    slippage_rate: float = 0.001  # 0.1% per side
    stamp_duty_rate: float = 0.0005  # 0.05% sell-side only
    save_to_db: bool = True  # save composite factor + strategy to DB


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
    """Pick top factors by |ICIR|, respecting the min_icir floor.

    Only ``factor_type='single'`` factors are eligible — composed factors have
    placeholder expressions (``COMPOSED(...)``) that cannot be evaluated by Qlib.
    """
    eligible = [
        f
        for f in factors
        if f.icir is not None
        and abs(f.icir) >= min_icir
        and (f.factor_type is None or f.factor_type == "single")
    ]
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


def _portfolio_returns(
    holdings: pd.DataFrame,
    close_panel: pd.DataFrame,
    commission_rate: float = 0.0,
    slippage_rate: float = 0.0,
    stamp_duty_rate: float = 0.0,
) -> pd.Series:
    """Compute daily portfolio returns from holdings and close prices.

    ``holdings`` is date × asset weights, already shifted for T+1 (signal at t
    → trade at t+1).  Transaction costs are applied to turnover (weight changes)
    at each rebalance: commission+slippage on both buys and sells, stamp duty on
    sells only.
    """
    if holdings.empty or close_panel.empty:
        return pd.Series(dtype=float)

    # align dates
    common_dates = holdings.index.intersection(close_panel.index)
    if common_dates.empty:
        return pd.Series(dtype=float)
    holdings = holdings.loc[common_dates]
    close_panel = close_panel.loc[common_dates]

    daily_ret = close_panel.pct_change(fill_method=None).fillna(0.0)
    portfolio_ret = (holdings * daily_ret).sum(axis=1)

    # Transaction costs: turnover = |weight change|, costs drag returns
    if commission_rate > 0 or slippage_rate > 0 or stamp_duty_rate > 0:
        holdings_diff = holdings.diff().fillna(holdings)  # first day = full position
        turnover = holdings_diff.abs().sum(axis=1)
        sells = (-holdings_diff.clip(upper=0)).sum(axis=1)
        cost = turnover * (commission_rate + slippage_rate) + sells * stamp_duty_rate
        portfolio_ret = portfolio_ret - cost

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
    # Shift holdings by 1 day for T+1: signal at date t → trade at date t+1
    # Exported holdings reflect actual trading dates, not signal dates
    holdings = holdings.shift(1).fillna(0.0)
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

    returns = _portfolio_returns(
        holdings,
        close_panel,
        commission_rate=config.commission_rate,
        slippage_rate=config.slippage_rate,
        stamp_duty_rate=config.stamp_duty_rate,
    )
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

    # 8. Save composite factor + generate strategy (if --save)
    if getattr(config, "save_to_db", False):
        _save_compose_result(selected, weights, config, result, score)

    return result


def _save_compose_result(
    selected: list[FactorRecord],
    weights: dict[str, float],
    config: ComposeConfig,
    result: ComposeResult,
    score: pd.DataFrame,
) -> None:
    """Save the composite factor to factor_library and a strategy to strategy table."""
    import hashlib
    import json as _json

    from quantify.database.factor_store import FactorRecord as FR, save_factor, update_backtest_metrics
    from quantify.database.strategy_store import save_strategy
    from quantify.factor.expr_codegen import CodegenError

    # Evaluate the composite factor IC/ICIR from the score panel
    eval_metrics = _evaluate_composite_panel(
        score, config.universe, config.start_date, config.end_date
    )

    parent_ids = ",".join(str(f.id) for f in selected if f.id is not None)
    comp_name = f"composed_stats_{hashlib.sha1(parent_ids.encode()).hexdigest()[:6]}"
    universe_str = config.universe if isinstance(config.universe, str) else "custom"

    # Save composite factor
    comp_record = FR(
        name=comp_name,
        expression=f"COMPOSED({parent_ids}, {config.weight})",
        hypothesis=f"Statistical composition of {len(selected)} single factors via {config.weight} weighting",
        category="composed",
        universe=universe_str,
        start_date=config.start_date,
        end_date=config.end_date,
        periods="1,5,10",
        ic_mean=eval_metrics.get("ic_mean"),
        ic_std=eval_metrics.get("ic_std"),
        icir=eval_metrics.get("icir"),
        rank_ic_mean=eval_metrics.get("rank_ic_mean"),
        rank_icir=eval_metrics.get("rank_icir"),
        coverage=None,
        status="composed",
        factor_type="composed",
        parent_factor_ids=parent_ids,
        metrics_json=_json.dumps(eval_metrics, default=str),
    )
    saved_factor = save_factor(comp_record)
    log.info(f"  ✓ 合成因子入库: [{saved_factor.id}] {comp_name}  IC={eval_metrics.get('ic_mean', 'NA')}  IR={eval_metrics.get('icir', 'NA')}")

    # Generate strategy source
    try:
        strategy_source = _generate_compose_strategy_source(selected, weights, config)
    except CodegenError as e:
        log.warning(f"  策略生成失败（表达式转换不支持）: {e}")
        return

    # Run backtest with the real engine
    from quantify.backtest.engine import BacktestEngine
    from quantify.backtest.universe import index_constituents_union

    universe_str_cfg = config.universe if isinstance(config.universe, str) else "000300.SH"
    bt_codes = index_constituents_union(universe_str_cfg, config.start_date, config.end_date)

    bt_metrics: dict = {}
    try:
        eng = BacktestEngine(
            strategy_source=strategy_source,
            ts_codes=bt_codes,
            start_date=config.start_date,
            end_date=config.end_date,
            initial_cash=1_000_000,
            benchmark_code=universe_str_cfg,
        )
        bt_result = eng.run()
        bt_metrics = {
            "total_return_pct": bt_result.metrics.total_return_pct,
            "annual_return_pct": bt_result.metrics.annual_return_pct,
            "max_drawdown_pct": bt_result.metrics.max_drawdown_pct,
            "sharpe_ratio": getattr(bt_result.metrics, "sharpe_ratio", 0),
            "trade_count": bt_result.metrics.trade_count,
        }
        log.info(
            f"  ✓ 策略回测: 总收益={bt_metrics['total_return_pct']:.2f}%  "
            f"年化={bt_metrics['annual_return_pct']:.2f}%  回撤={bt_metrics['max_drawdown_pct']:.2f}%"
        )
    except Exception as e:
        log.warning(f"  策略回测失败: {e}")

    # Save strategy
    factor_summary = ", ".join(f"{f.name}(w={weights.get(f.expression, 0):.3f})" for f in selected)
    saved_strategy = save_strategy(
        name=f"factor_compose_{comp_name}",
        source=strategy_source,
        description=f"factor compose 统计合成策略：{len(selected)}因子{config.weight}加权，"
        f"选top-{config.top_n}，{config.rebalance_freq}日调仓。因子: {factor_summary}",
    )
    log.info(f"  ✓ 策略入库: #{saved_strategy.id} factor_compose_{comp_name}")

    # Link strategy back to factor
    update_backtest_metrics(saved_factor.id, saved_strategy.id, _json.dumps(bt_metrics, default=str))


# ---------------------------------------------------------------------------
# LLM-driven composition (used by the mining pipeline)
# ---------------------------------------------------------------------------


@dataclass
class ComposePlan:
    """LLM's plan for composing a composite factor."""

    name: str
    factor_ids: list[int]
    weight_method: WeightMethod
    hypothesis: str
    top_n: int = 20
    rebalance_days: int = 5


def _factor_library_summary(factors: list[FactorRecord], max_show: int = 30) -> str:
    """Build a text summary of the factor library for the LLM."""
    lines = []
    for f in factors[:max_show]:
        ic = f"{f.ic_mean:.4f}" if f.ic_mean is not None else "NA"
        ir = f"{f.icir:.4f}" if f.icir is not None else "NA"
        lines.append(f"  [id={f.id}] {f.name}  IC={ic} IR={ir}  type={f.factor_type}  {f.expression[:80]}")
    if len(factors) > max_show:
        lines.append(f"  ... 共 {len(factors)} 个因子")
    return "\n".join(lines)


def compose_factors_llm(
    *,
    universe: str | list[str] | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    feedback: str | None = None,
    extra_instruction: str | None = None,
) -> tuple[ComposePlan, pd.DataFrame, dict, list[FactorRecord], dict[str, float]]:
    """LLM-driven composition: LLM picks factors + weights, we compute the
    composite factor panel and evaluate it via Alphalens.

    Returns (plan, composite_score_panel, evaluation_metrics, selected_factors, weights).

    Raises on any failure — empty library, invalid LLM plan, empty panels, etc.
    """
    from quantify.factor.evaluator import evaluation_window_default
    from quantify.factor.llm import LLMClient

    if not start_date or not end_date:
        ds, de = evaluation_window_default()
        start_date = start_date or ds
        end_date = end_date or de

    all_factors = list_factors()
    if not all_factors:
        raise RuntimeError("因子库为空，无法合成。请先运行 `quantify factor mine`。")

    # 合成因子只能从单因子中选取——合成因子的 expression 是占位符（COMPOSED(...)），
    # 不是合法的 Qlib 表达式，无法用于计算面板。
    single_factors = [f for f in all_factors if (f.factor_type or "single") == "single"]
    if not single_factors:
        raise RuntimeError("因子库中没有单因子，无法合成。")

    summary = _factor_library_summary(single_factors)
    llm = LLMClient()
    plan_raw = llm.generate_compose_plan(
        factor_library_summary=summary,
        feedback=feedback,
        extra_instruction=extra_instruction,
    )
    if not plan_raw or not plan_raw.get("factor_ids"):
        raise RuntimeError(f"LLM 未给出有效合成计划: {plan_raw}")

    # Resolve factor IDs to records (only single factors are eligible)
    id_map = {f.id: f for f in single_factors if f.id is not None}
    missing_ids = [fid for fid in plan_raw["factor_ids"] if fid not in id_map]
    selected = [id_map[fid] for fid in plan_raw["factor_ids"] if fid in id_map]
    if not selected:
        raise RuntimeError(
            f"LLM 选择的因子ID {plan_raw['factor_ids']} 在单因子库中不存在"
            f"{'（含不存在的ID: ' + str(missing_ids) + '）' if missing_ids else ''}。"
        )

    plan = ComposePlan(
        name=str(plan_raw.get("name", "composed")),
        factor_ids=[f.id for f in selected],
        weight_method=plan_raw.get("weight_method", "icir"),
        hypothesis=str(plan_raw.get("hypothesis", "")),
        top_n=int(plan_raw.get("top_n", 20)),
        rebalance_days=int(plan_raw.get("rebalance_days", 5)),
    )
    log.info(f"LLM 合成计划: {plan.name}, {len(selected)} 个因子, 权重={plan.weight_method}")
    for f in selected:
        log.info(f"  [id={f.id}] {f.expression[:80]}")
    if missing_ids:
        log.warning(f"  跳过不存在的因子ID: {missing_ids}")

    # Compute factor panels
    panels: dict[str, pd.DataFrame] = {}
    for f in selected:
        panel = compute_factor_panel(f.expression, universe, start_date, end_date)
        if not panel.empty:
            panels[f.expression] = panel
    selected = [f for f in selected if f.expression in panels]
    if not selected:
        raise RuntimeError("所有因子面板为空，无法合成。")

    # Weights
    weights = _compute_weights(selected, plan.weight_method)

    # Composite score
    composite = _compose_score(panels, selected, weights)
    if composite.empty:
        raise RuntimeError("合成分数为空。")

    # Evaluate the composite as a single factor
    eval_metrics = _evaluate_composite_panel(composite, universe, start_date, end_date)
    if not eval_metrics:
        raise RuntimeError("合成因子评估失败，无法计算 IC/IR。")
    log.info(
        f"合成因子评估: IC={eval_metrics.get('ic_mean', 'NA')} "
        f"IR={eval_metrics.get('icir', 'NA')} "
        f"RankIC={eval_metrics.get('rank_ic_mean', 'NA')}"
    )
    return plan, composite, eval_metrics, selected, weights


def _evaluate_composite_panel(
    score: pd.DataFrame,
    universe: str | list[str] | None,
    start_date: str,
    end_date: str,
) -> dict:
    """Evaluate the composite score panel: compute IC/RankIC/IR using close prices."""
    from qlib.data import D

    from quantify.factor.evaluator import _resolve_universe
    from quantify.factor.qlib_data import init_qlib

    init_qlib()
    instruments = _resolve_universe(universe, start_date, end_date)
    close_raw = D.features(instruments, ["$close"], start_time=start_date, end_time=end_date)
    if close_raw is None or close_raw.empty:
        return {}

    close_s = close_raw.iloc[:, 0]
    close_s.index = close_s.index.set_names(["asset", "date"])
    close_s = close_s.reorder_levels(["date", "asset"]).sort_index()
    close_panel = close_s.unstack("asset")

    # Align dates
    common_dates = score.index.intersection(close_panel.index)
    score = score.loc[common_dates]
    close_panel = close_panel.loc[common_dates]

    # Forward returns
    fwd_5 = close_panel.pct_change(5, fill_method=None).shift(-5)

    # Cross-sectional IC per day
    import numpy as np

    ic_series = []
    rank_ic_series = []
    for dt in score.index:
        s = score.loc[dt].dropna()
        r = fwd_5.loc[dt] if dt in fwd_5.index else None
        if r is None:
            continue
        r = r.dropna()
        common = s.index.intersection(r.index)
        if len(common) < 10:
            continue
        s_aligned = s.loc[common]
        r_aligned = r.loc[common]
        ic_series.append(s_aligned.corr(r_aligned))
        rank_ic_series.append(s_aligned.corr(r_aligned, method="spearman"))

    if not ic_series:
        return {}

    ic_arr = np.array(ic_series, dtype=float)
    rank_arr = np.array(rank_ic_series, dtype=float)
    ic_mean = float(np.nanmean(ic_arr))
    ic_std = float(np.nanstd(ic_arr, ddof=1)) if len(ic_arr) > 1 else 0.0
    icir = float(ic_mean / ic_std) if ic_std != 0 else 0.0
    rank_ic_mean = float(np.nanmean(rank_arr))
    rank_std = float(np.nanstd(rank_arr, ddof=1)) if len(rank_arr) > 1 else 0.0
    rank_icir = float(rank_ic_mean / rank_std) if rank_std != 0 else 0.0

    return {
        "ic_mean": ic_mean,
        "ic_std": ic_std,
        "icir": icir,
        "rank_ic_mean": rank_ic_mean,
        "rank_icir": rank_icir,
        "n_days": len(ic_arr),
    }


# ---------------------------------------------------------------------------
# Strategy source generation (no LLM — pure codegen from factor expressions)
# ---------------------------------------------------------------------------


def _generate_compose_strategy_source(
    selected: list[FactorRecord],
    weights: dict[str, float],
    config: ComposeConfig,
) -> str:
    """Generate a JoinQuant-format strategy source string from the selected factors.

    Each factor's Qlib expression is translated to Python (numpy) code via
    :mod:`quantify.factor.expr_codegen`. The strategy computes all factor
    values per stock, cross-sectional z-scores them, aligns direction with
    ``sign(ic_mean)``, weight-combines, and picks top-N stocks.
    """
    from quantify.factor.expr_codegen import HELPER_FUNCS, CodegenError, expression_to_python

    universe_str = config.universe if isinstance(config.universe, str) else "000300.SH"
    jq_index = universe_str.replace(".SH", ".XSHG").replace(".SZ", ".XSHE")

    # Qlib field name → Python variable name in the strategy
    # ($open → "open" in codegen, but "open" is a Python builtin → use "open_")
    FIELD_TO_VAR = {
        "close": "close",
        "open": "open_",
        "high": "high",
        "low": "low",
        "volume": "volume",
        "amount": "amount",
        "turnover_rate": "turnover_rate",
        "turn": "turnover_rate",   # Qlib uses $turn, our DB uses turnover_rate
        "vwap": "vwap",
    }

    # Determine which fields are needed
    needed_fields: set[str] = set()
    factor_snippets: list[str] = []
    factor_weights: list[float] = []
    factor_directions: list[bool] = []

    for i, fac in enumerate(selected):
        expr = fac.expression
        try:
            py_expr = expression_to_python(expr)
        except CodegenError as e:
            log.warning(f"  因子 {fac.name} 无法转换: {e}，跳过")
            continue

        # Replace field names: Qlib uses $turn → codegen produces "turn",
        # but our variable is "turnover_rate". Also "open" → "open_".
        for qlib_field, var_name in FIELD_TO_VAR.items():
            if qlib_field != var_name:
                # Word-boundary replace to avoid partial matches (e.g. "open" in "open_")
                py_expr = re.sub(rf'\b{re.escape(qlib_field)}\b', var_name, py_expr)

        # Detect fields needed for attribute_history
        for fld in FIELD_TO_VAR.values():
            if fld in py_expr:
                needed_fields.add(fld)

        w = weights.get(fac.expression, 0.0)
        direction = (fac.ic_mean or 0) >= 0  # True = keep, False = flip

        # The expression produces a numpy array (rolling window per bar).
        # We only need the latest value (last element) for today's signal.
        factor_snippets.append(f'            f{i}_arr = {py_expr}\n            f{i} = f{i}_arr[-1] if f{i}_arr is not None and len(f{i}_arr) > 0 and not np.isnan(f{i}_arr[-1]) else np.nan')
        factor_weights.append(w)
        factor_directions.append(direction)

    if not factor_snippets:
        raise CodegenError("No factor could be translated to Python code")

    # Build field list for attribute_history — map variable names back to field names
    # For attribute_history, field names must match the engine's field names
    ATTR_FIELD_MAP = {
        "close": "close",
        "open_": "open",
        "high": "high",
        "low": "low",
        "volume": "volume",
        "amount": "amount",
        "turnover_rate": "turnover_rate",
        "vwap": "vwap",
    }
    fields_list = sorted(needed_fields, key=lambda f: list(ATTR_FIELD_MAP.keys()).index(f))
    fields_str = ", ".join(f'"{ATTR_FIELD_MAP[f]}"' for f in fields_list)

    # Determine lookback: max window in expressions + buffer
    # Default to 30 which covers most rolling windows
    lookback = 30

    weights_list_str = repr(factor_weights)
    directions_list_str = repr(factor_directions)
    n_factors = len(factor_snippets)

    strategy = f'''from jqdata import *
import builtins
sum = builtins.sum
max = builtins.max
min = builtins.min
abs = builtins.abs
round = builtins.round
import numpy as np
import pandas as pd
{HELPER_FUNCS}


def initialize(context):
    set_option("use_real_price", True)
    set_option("avoid_future_data", True)
    set_benchmark("{jq_index}")
    set_order_cost(OrderCost(
        open_tax=0, close_tax=0,
        open_commission=0.0003, close_commission=0.0003,
        min_commission=0.5,
    ), type="stock")
    set_slippage(PriceRelatedSlippage(0.002))

    context.top_n = {config.top_n}
    context.rebalance_days = {config.rebalance_freq}
    context.day_count = 0
    context.weights = {weights_list_str}
    context.directions = {directions_list_str}
    context.n_factors = {n_factors}
    context.lookback = {lookback}

    run_daily(rebalance, time="open")


def _zscore(x):
    mean = np.mean(x)
    std = np.std(x, ddof=1)
    if std == 0:
        return np.zeros_like(x)
    return (x - mean) / std


def rebalance(context):
    context.day_count += 1
    if context.day_count % context.rebalance_days != 0:
        return

    stocks = get_index_stocks("{jq_index}")
    if not stocks:
        log.warning(f"{{context.current_dt.date()}} 股票池为空")
        return

    fields = [{fields_str}]
    factor_data = [dict() for _ in range(context.n_factors)]

    for code in stocks:
        try:
            attr = attribute_history(code, context.lookback, "1d", fields)
            if len(attr) < context.lookback:
                continue

            close = attr["close"].values.astype(float) if "close" in fields else None
            open_ = attr["open"].values.astype(float) if "open" in fields else None
            high = attr["high"].values.astype(float) if "high" in fields else None
            low = attr["low"].values.astype(float) if "low" in fields else None
            volume = attr["volume"].values.astype(float) if "volume" in fields else None
            amount = attr["amount"].values.astype(float) if "amount" in fields else None
            turnover_rate = attr["turnover_rate"].values.astype(float) if "turnover_rate" in fields else None
            vwap = attr["vwap"].values.astype(float) if "vwap" in fields else None

{chr(10).join(factor_snippets)}

            vals = [{", ".join(f"f{i}" for i in range(n_factors))}]
            for i, val in enumerate(vals):
                if val is not None and not np.isnan(val) and not np.isinf(val):
                    factor_data[i][code] = val

        except Exception:
            continue

    # Intersection of stocks with all factor values
    valid = set(factor_data[0].keys())
    for i in range(1, context.n_factors):
        valid &= set(factor_data[i].keys())
    if len(valid) < context.top_n:
        log.warning(f"{{context.current_dt.date()}} 有效股票不足: {{len(valid)}}")
        return

    codes_list = list(valid)
    n = len(codes_list)

    # Cross-sectional z-score + direction align + weight-combine
    composite = np.zeros(n)
    for i in range(context.n_factors):
        arr = np.array([factor_data[i][c] for c in codes_list])
        z = _zscore(arr)
        if not context.directions[i]:
            z = -z
        composite += context.weights[i] * z

    # Select top-N
    sorted_idx = np.argsort(-composite)
    target = set()
    for i in range(min(context.top_n, n)):
        target.add(codes_list[sorted_idx[i]])

    log.info(f"{{context.current_dt.date()}} 有效={{n}} 持仓={{len(target)}}")

    # Rebalance: sell first, then buy
    for code in list(context.portfolio.positions.keys()):
        if code not in target:
            order_target_value(code, 0)
    if target:
        weight = context.portfolio.total_value / len(target) * 0.95
        for code in target:
            order_target_value(code, weight)
'''
    return strategy

