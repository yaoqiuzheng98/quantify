"""Two-stage backtest: vectorized screening → event-driven validation.

The ML/DL modules produce daily stock scores via vectorized backtest (no
trading friction).  This module takes the **holdings** from a vectorized
backtest and generates a JoinQuant-compatible strategy that replays those
exact holdings, then runs it through the full event-driven ``BacktestEngine``
with commission, slippage, T+1, and price limits.

This gives a realistic performance estimate for ML/DL-selected portfolios.

Usage::

    from quantify.ml.two_stage import TwoStageBacktest, TwoStageConfig

    # After running an ML/DL model that produced a VectorBacktestResult:
    result = TwoStageBacktest(TwoStageConfig(
        universe="000300.SH",
        start_date="2020-06-16",
        end_date="2026-06-16",
    )).validate(vector_bt_result.holdings)
    print(result.summary())
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from quantify.utils.logger import log

from .backtest import VectorBacktestResult


@dataclass
class TwoStageConfig:
    """Configuration for two-stage backtest validation."""

    universe: str = "000300.SH"
    start_date: str = "2020-06-16"
    end_date: str = "2026-06-16"
    initial_cash: float = 1_000_000
    commission_rate: float = 0.0005
    commission_min: float = 0.5
    slippage_rate: float = 0.002
    # How to handle stocks in holdings that aren't in the preloaded universe:
    # "skip" = ignore them, "include" = add to ts_codes
    handle_missing: str = "include"


@dataclass
class EventDrivenResult:
    """Result of event-driven validation."""

    metrics: dict
    strategy_logs: list[str]
    source: str
    vectorized_metrics: dict  # for comparison

    def summary(self) -> str:
        m = self.metrics
        vm = self.vectorized_metrics
        lines = [
            "=== Two-Stage Backtest: Vectorized vs Event-Driven ===",
            "",
            "--- Vectorized (no friction) ---",
            f"  Total return: {vm.get('total_return_pct', 0):.2f}%",
            f"  Sharpe: {vm.get('sharpe_ratio', 0):.2f}",
            f"  Max DD: {vm.get('max_drawdown_pct', 0):.2f}%",
            "",
            "--- Event-Driven (with friction) ---",
            f"  Total return: {m.get('total_return_pct', 0):.2f}%",
            f"  Annual return: {m.get('annual_return_pct', 0):.2f}%",
            f"  Sharpe: {m.get('sharpe_ratio', 0):.2f}",
            f"  Max DD: {m.get('max_drawdown_pct', 0):.2f}%",
            f"  Volatility: {m.get('volatility_pct', 0):.2f}%",
            f"  Win rate: {m.get('win_rate_pct', 0):.2f}%",
            f"  Profit factor: {m.get('profit_factor', 0):.2f}",
            f"  Trade count: {m.get('trade_count', 0)}",
            f"  Total commission: {m.get('total_commission', 0):.2f}",
            f"  Total slippage: {m.get('total_slippage', 0):.2f}",
            f"  Total tax: {m.get('total_tax', 0):.2f}",
            "",
            "--- Friction Cost ---",
            f"  Return drag: {vm.get('total_return_pct', 0) - m.get('total_return_pct', 0):.2f}%",
        ]
        return "\n".join(lines)


class TwoStageBacktest:
    """Validate vectorized backtest results in the event-driven engine."""

    def __init__(self, config: TwoStageConfig | None = None) -> None:
        self.config = config or TwoStageConfig()

    def _generate_strategy_source(
        self,
        holdings: pd.DataFrame,
        rebalance_days: int,
    ) -> str:
        """Generate a JoinQuant-compatible strategy that replays the holdings.

        The strategy embeds the holdings as a dict: {date_str: {stock: weight}}.
        On each trading day, it checks if there's a target portfolio for that
        date and rebalances to match it.

        Note: holdings from vectorized backtest are already shifted by 1 day
        (scores.shift(1) in backtest.py), so the T+1 constraint is already
        baked into the holdings dates. No additional shift is needed here.
        """
        # Build holdings dict: only non-zero weights, grouped by date
        holdings_dict: dict[str, dict[str, float]] = {}
        for dt in holdings.index:
            row = holdings.loc[dt]
            nonzero = row[row > 0.001]  # filter dust
            if len(nonzero) > 0:
                # Normalize weights to sum to 1.0 (in case of rounding drift)
                w_sum = nonzero.sum()
                if w_sum > 0:
                    if abs(w_sum - 1.0) > 0.01:
                        log.info(f"权重归一化: {dt} sum={w_sum:.3f} → 1.0")
                    nonzero = nonzero / w_sum
                holdings_dict[str(dt)] = {str(stock): float(w) for stock, w in nonzero.items()}

        # Serialize as Python literal (compact)
        import json

        holdings_json = json.dumps(holdings_dict, separators=(",", ":"))

        # Determine the benchmark code
        from quantify.factor.llm import _to_jq_code

        # Build strategy source by parts to avoid f-string escaping issues
        header = (
            """from jqdata import *
import builtins
sum = builtins.sum
max = builtins.max
min = builtins.min
abs = builtins.abs
round = builtins.round
import json

# Pre-computed holdings from ML/DL model: {date_str: {stock: weight}}
_HOLDINGS = json.loads("""
            + repr(holdings_json)
            + """)

def initialize(context):
    set_option("use_real_price", True)
    set_option("avoid_future_data", True)
    set_benchmark("""
            + f'"{_to_jq_code(self.config.universe)}"'
            + """)
    set_order_cost(OrderCost(
        open_tax=0, close_tax=0,
        open_commission="""
            + str(self.config.commission_rate)
            + """, close_commission="""
            + str(self.config.commission_rate)
            + """,
        min_commission="""
            + str(self.config.commission_min)
            + """,
    ), type="stock")
    set_slippage(PriceRelatedSlippage("""
            + str(self.config.slippage_rate)
            + """))
    context.day_count = 0
    run_daily(rebalance, time="open")

def rebalance(context):
    context.day_count += 1
    dt_str = str(context.current_dt.date())

    if dt_str not in _HOLDINGS:
        return

    target = _HOLDINGS[dt_str]
    if not target:
        log.warning(f"{dt_str} 目标持仓为空")
        return

    log.info(f"{dt_str} 调仓: {len(target)} 只股票")

    # Sell positions not in target
    current_positions = list(context.portfolio.positions.keys())
    for code in current_positions:
        if code not in target:
            order_target_value(code, 0)

    # Buy / adjust target positions (use 100% of total value, no idle cash)
    total_value = context.portfolio.total_value
    for code, weight in target.items():
        try:
            order_target_value(code, total_value * weight)
        except Exception as e:
            log.warning(f"下单失败 {code}: {e}")
"""
        )
        return header

    def _resolve_ts_codes(self, holdings: pd.DataFrame) -> list[str]:
        """Get all stock codes that appear in holdings + universe index."""
        from quantify.backtest.universe import index_constituents_union

        # Start with the universe index constituents
        ts_codes: list[str] = []
        if self.config.universe not in {"all", ""}:
            ts_codes = index_constituents_union(
                self.config.universe, self.config.start_date, self.config.end_date
            )

        # Add all stocks from holdings (already in Tushare format from vectorized backtest)
        holding_ts = [str(col) for col in holdings.columns]

        if self.config.handle_missing == "include":
            # Merge: universe + holding stocks
            all_codes = list(set(ts_codes) | set(holding_ts))
        else:
            # Only universe stocks
            all_codes = ts_codes

        return all_codes if all_codes else holding_ts

    def validate(
        self,
        holdings: pd.DataFrame,
        vectorized_metrics: dict | None = None,
        rebalance_days: int = 5,
    ) -> EventDrivenResult:
        """Validate vectorized holdings in the event-driven engine.

        Parameters
        ----------
        holdings : pd.DataFrame
            (date × asset) DataFrame of portfolio weights from vectorized backtest.
        vectorized_metrics : dict, optional
            Metrics from the vectorized backtest for comparison.
        rebalance_days : int
            Rebalance frequency (used for strategy generation).
        """
        from quantify.backtest.engine import BacktestEngine

        cfg = self.config

        # 1. Generate strategy source
        log.info("生成事件驱动策略代码（回放 ML/DL 持仓）...")
        source = self._generate_strategy_source(holdings, rebalance_days)
        log.info(f"策略代码: {len(source)} chars, {len(holdings.index)} 个调仓日")

        # 2. Resolve ts_codes (universe + all stocks in holdings)
        ts_codes = self._resolve_ts_codes(holdings)
        if not ts_codes:
            raise RuntimeError("无法解析股票池")
        log.info(f"事件驱动回测: {len(ts_codes)} 只股票")

        # 3. Run event-driven backtest
        benchmark = cfg.universe if cfg.universe not in {"all", ""} else "000300.SH"

        engine = BacktestEngine(
            strategy_source=source,
            ts_codes=ts_codes,
            start_date=cfg.start_date,
            end_date=cfg.end_date,
            initial_cash=cfg.initial_cash,
            benchmark_code=benchmark,
            commission_rate=cfg.commission_rate,
            commission_min=cfg.commission_min,
            slippage_rate=cfg.slippage_rate,
        )
        result = engine.run()

        return EventDrivenResult(
            metrics=result.metrics.to_dict(),
            strategy_logs=result.strategy_logs,
            source=source,
            vectorized_metrics=vectorized_metrics or {},
        )

    def validate_vector_result(
        self,
        vector_result: VectorBacktestResult,
        rebalance_days: int = 5,
    ) -> EventDrivenResult:
        """Convenience: validate a VectorBacktestResult directly."""
        return self.validate(
            holdings=vector_result.holdings,
            vectorized_metrics=vector_result.metrics,
            rebalance_days=rebalance_days,
        )
