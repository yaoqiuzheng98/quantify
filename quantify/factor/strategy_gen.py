"""Generate, backtest and persist factor-based strategies.

Given a factor (single or composite) and its evaluation metrics, ask the LLM
to produce a JoinQuant-format strategy script, run it through the event-driven
``BacktestEngine``, save the strategy into the ``strategy`` table and link it
back to the factor via ``factor_library.strategy_id``.
"""

from __future__ import annotations

from dataclasses import dataclass

from quantify.database.factor_store import FactorRecord, update_strategy_id
from quantify.database.strategy_store import save_strategy
from quantify.utils.logger import log


@dataclass
class StrategyBacktestResult:
    strategy_id: int | None
    source: str
    metrics: dict | None
    error: str | None = None


def generate_and_backtest_strategy(
    factor: FactorRecord,
    *,
    universe: str,
    start_date: str,
    end_date: str,
    top_n: int = 20,
    rebalance_days: int = 5,
    initial_cash: float = 1_000_000,
    feedback: str | None = None,
) -> StrategyBacktestResult:
    """Full cycle: LLM → strategy code → backtest → persist → link to factor.

    Parameters
    ----------
    factor : FactorRecord
        The factor to build a strategy for (uses ``expression`` + evaluation metrics).
    universe : str
        Index code (e.g. ``"000300.SH"``) or ``"all"``.
    feedback : str, optional
        Previous backtest feedback for the LLM to improve upon.
    """
    from quantify.factor.llm import LLMClient

    llm = LLMClient()
    factor_metrics = _factor_metrics_text(factor)

    log.info(f"  生成策略代码: {factor.expression[:60]}...")
    source = llm.generate_strategy(
        factor_expression=factor.expression,
        factor_metrics=factor_metrics,
        universe=universe,
        start_date=start_date,
        end_date=end_date,
        top_n=top_n,
        rebalance_days=rebalance_days,
        feedback=feedback,
    )
    if not source:
        return StrategyBacktestResult(strategy_id=None, source="", metrics=None, error="LLM 未生成策略代码")

    # Run backtest
    result = _run_backtest(source, universe, start_date, end_date, initial_cash)
    if result.get("error"):
        return StrategyBacktestResult(strategy_id=None, source=source, metrics=None, error=result["error"])

    metrics = result["metrics"]

    # Persist strategy
    strategy_name = f"factor_{factor.id or 'x'}_{factor.name[:30]}"
    description = (
        f"因子: {factor.expression}\n"
        f"IC={factor.ic_mean}, IR={factor.icir}\n"
        f"回测: 总收益={metrics.get('total_return_pct', 0):.2f}%, "
        f"夏普={metrics.get('sharpe_ratio', 0):.2f}, "
        f"回撤={metrics.get('max_drawdown_pct', 0):.2f}%"
    )
    saved = save_strategy(name=strategy_name, source=source, description=description)

    # Link factor → strategy
    if factor.id is not None:
        update_strategy_id(factor.id, saved.id)

    log.info(
        f"  策略入库 #{saved.id}: 总收益={metrics.get('total_return_pct', 0):.2f}% "
        f"夏普={metrics.get('sharpe_ratio', 0):.2f} 回撤={metrics.get('max_drawdown_pct', 0):.2f}%"
    )
    return StrategyBacktestResult(strategy_id=saved.id, source=source, metrics=metrics)


def _run_backtest(
    source: str,
    universe: str,
    start_date: str,
    end_date: str,
    initial_cash: float,
) -> dict:
    """Run the backtest engine and return metrics dict or error."""
    from quantify.backtest.engine import BacktestEngine

    # Resolve universe to a concrete code list for the engine
    ts_codes = _resolve_ts_codes(universe, start_date, end_date)
    if not ts_codes:
        return {"error": f"股票池 {universe} 解析为空"}

    benchmark = _to_jq_code(universe) if universe not in {"all", ""} else "000300.SH"

    try:
        engine = BacktestEngine(
            strategy_source=source,
            ts_codes=ts_codes,
            start_date=start_date,
            end_date=end_date,
            initial_cash=initial_cash,
            benchmark_code=benchmark,
            commission_rate=0.0005,
            commission_min=0.5,
            slippage_rate=0.002,
        )
        result = engine.run()
        return {"metrics": result.metrics.to_dict(), "error": None}
    except Exception as exc:  # noqa: BLE001
        return {"error": f"回测失败: {exc}"}


def _resolve_ts_codes(universe: str, start_date: str, end_date: str) -> list[str]:
    """Resolve universe spec to a list of Tushare codes for the engine."""
    if universe in {"all", ""}:
        from quantify.factor.qlib_data import list_instruments

        from quantify.factor.qlib_data import qlib_to_ts_code

        return [qlib_to_ts_code(c) for c in list_instruments()]
    # index code → constituent union
    from quantify.backtest.universe import index_constituents_union

    return index_constituents_union(universe, start_date, end_date)


def _to_jq_code(ts_code: str) -> str:
    """Convert Tushare code to JoinQuant format."""
    if ts_code.endswith(".SH"):
        return ts_code.replace(".SH", ".XSHG")
    if ts_code.endswith(".SZ"):
        return ts_code.replace(".SZ", ".XSHE")
    return ts_code


def _factor_metrics_text(factor: FactorRecord) -> str:
    """Compact text summary of the factor's evaluation metrics for the LLM."""
    lines = [
        f"因子表达式: {factor.expression}",
        f"IC均值: {factor.ic_mean}",
        f"IC标准差: {factor.ic_std}",
        f"IC_IR: {factor.icir}",
        f"Rank_IC: {factor.rank_ic_mean}",
        f"Rank_IC_IR: {factor.rank_icir}",
        f"多空分层收益差: {factor.quantile_spread}",
        f"顶层换手率: {factor.turnover}",
        f"覆盖率: {factor.coverage}",
        f"状态: {factor.status}",
    ]
    if factor.hypothesis:
        lines.append(f"因子逻辑: {factor.hypothesis}")
    return "\n".join(str(x) for x in lines)
