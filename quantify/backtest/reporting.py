"""Report helpers for backtest metrics and interactive views."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _format_pct(value: float | None, digits: int = 2) -> str:
    if value is None or not np.isfinite(value):
        return "--"
    return f"{value:.{digits}f}%"


def _format_float(value: float | None, digits: int = 3) -> str:
    if value is None or not np.isfinite(value):
        return "--"
    return f"{value:.{digits}f}"


def _format_int(value: int | None) -> str:
    if value is None:
        return "--"
    return str(value)


def _daily_returns(values: np.ndarray) -> np.ndarray:
    if len(values) < 2:
        return np.array([], dtype=float)
    return np.diff(values) / values[:-1]


def _annualized_return(total_return: float, trading_days: int) -> float:
    if trading_days <= 0 or total_return <= -1:
        return 0.0
    return ((1 + total_return) ** (250 / trading_days) - 1) * 100


def _annualized_volatility(daily_returns: np.ndarray) -> float | None:
    if len(daily_returns) < 2:
        return None
    return float(np.std(daily_returns) * np.sqrt(250))


def _max_drawdown_info(curve: np.ndarray, dates: pd.DatetimeIndex) -> tuple[float, str]:
    if len(curve) == 0:
        return 0.0, ""

    peak = np.maximum.accumulate(curve)
    drawdown = (peak - curve) / peak
    trough_idx = int(np.argmax(drawdown))
    peak_idx = int(np.argmax(curve[: trough_idx + 1]))
    period = f"{dates[peak_idx].strftime('%Y/%m/%d')},{dates[trough_idx].strftime('%Y/%m/%d')}"
    return float(drawdown[trough_idx] * 100), period


def _sharpe_ratio(
    annual_return_pct: float, annual_volatility: float | None, risk_free_rate: float = 0.03
) -> float | None:
    if annual_volatility is None or annual_volatility <= 0:
        return None
    return (annual_return_pct / 100 - risk_free_rate) / annual_volatility


def _sortino_ratio(
    daily_returns: np.ndarray, annual_return_pct: float, risk_free_rate: float = 0.03
) -> float | None:
    if len(daily_returns) == 0:
        return None
    daily_risk_free = risk_free_rate / 250
    downside_returns = daily_returns[daily_returns < daily_risk_free] - daily_risk_free
    if len(downside_returns) == 0:
        return None
    downside_volatility = float(np.sqrt(np.mean(np.square(downside_returns))) * np.sqrt(250))
    if downside_volatility <= 0:
        return None
    return (annual_return_pct / 100 - risk_free_rate) / downside_volatility


def _realized_trade_stats(trades: list[Any] | None) -> tuple[int, int, float | None, float | None]:
    if not trades:
        return 0, 0, None, None

    positions: dict[str, tuple[int, float]] = {}
    profits: list[float] = []
    losses: list[float] = []

    for trade in trades:
        code = getattr(trade, "ts_code", "")
        amount = int(getattr(trade, "amount", 0) or 0)
        price = getattr(trade, "filled_price", None)
        if not code or amount == 0 or price is None:
            continue

        commission = float(getattr(trade, "commission", 0.0) or 0.0)
        slippage = float(getattr(trade, "slippage", 0.0) or 0.0)
        current_amount, avg_cost = positions.get(code, (0, 0.0))

        if amount > 0:
            total_cost = amount * float(price) + commission + slippage
            new_amount = current_amount + amount
            new_avg_cost = (current_amount * avg_cost + total_cost) / new_amount if new_amount > 0 else 0.0
            positions[code] = (new_amount, new_avg_cost)
            continue

        sell_amount = min(current_amount, abs(amount))
        if sell_amount <= 0:
            continue

        proceeds = sell_amount * float(price) - commission - slippage
        pnl = proceeds - sell_amount * avg_cost
        if pnl > 0:
            profits.append(pnl)
        elif pnl < 0:
            losses.append(pnl)

        positions[code] = (current_amount - sell_amount, avg_cost if current_amount > sell_amount else 0.0)

    profit_count = len(profits)
    loss_count = len(losses)
    total_closed = profit_count + loss_count
    win_rate = profit_count / total_closed if total_closed > 0 else None
    profit_loss_ratio = float(np.mean(profits) / abs(np.mean(losses))) if profits and losses else None
    return profit_count, loss_count, win_rate, profit_loss_ratio


def benchmark_return_series(
    dates: pd.DatetimeIndex,
    benchmark_df: pd.DataFrame | None,
) -> pd.Series | None:
    """Return benchmark cumulative return aligned to the supplied dates."""
    if benchmark_df is None or benchmark_df.empty:
        return None

    benchmark = pd.Series(
        benchmark_df["value"].astype(float).values,
        index=pd.to_datetime(benchmark_df["date"]),
    ).sort_index()
    benchmark = benchmark.reindex(dates).ffill().dropna()
    if benchmark.empty or benchmark.iloc[0] <= 0:
        return None
    return benchmark / benchmark.iloc[0] - 1


def _compute_alpha_beta(
    strategy_returns: np.ndarray,
    benchmark_returns: np.ndarray,
    annual_return_pct: float,
    benchmark_annual_pct: float,
    risk_free_rate: float = 0.03,
) -> tuple[float | None, float | None]:
    if len(strategy_returns) < 2 or len(benchmark_returns) < 2:
        return None, None

    count = min(len(strategy_returns), len(benchmark_returns))
    strategy_returns = strategy_returns[-count:]
    benchmark_returns = benchmark_returns[-count:]
    variance = float(np.var(benchmark_returns))
    if variance <= 0:
        return None, None

    beta = float(np.cov(strategy_returns, benchmark_returns)[0, 1] / variance)
    alpha = (
        annual_return_pct / 100 - risk_free_rate - beta * (benchmark_annual_pct / 100 - risk_free_rate)
    ) * 100
    return alpha, beta


def build_report_items(
    equity_df: pd.DataFrame,
    benchmark_df: pd.DataFrame | None = None,
    metrics: Any | None = None,
    trades: list[Any] | None = None,
) -> list[tuple[str, str, float | None]]:
    """Return JoinQuant-style metrics for interactive report views."""
    del metrics
    if equity_df.empty:
        return []

    values = equity_df["value"].astype(float).to_numpy()
    strategy_daily = _daily_returns(values)
    trading_days = len(values)
    dates = pd.DatetimeIndex(pd.to_datetime(equity_df["date"]))

    strategy_return = values[-1] / values[0] - 1
    annual_return = _annualized_return(strategy_return, trading_days)
    strategy_volatility = _annualized_volatility(strategy_daily)
    sharpe = _sharpe_ratio(annual_return, strategy_volatility)
    max_drawdown, max_drawdown_period = _max_drawdown_info(values / values[0], dates)
    sortino = _sortino_ratio(strategy_daily, annual_return)
    profit_count, loss_count, trade_win_rate, profit_loss_ratio = _realized_trade_stats(trades)

    benchmark_total_return = None
    benchmark_annual = None
    alpha = None
    beta = None
    benchmark_volatility = None
    excess_return = None
    excess_mean = None
    excess_max_drawdown = None
    excess_sharpe = None
    information_ratio = None
    benchmark_curve = benchmark_return_series(dates, benchmark_df)
    if benchmark_curve is not None:
        benchmark_total_return = float(benchmark_curve.iloc[-1])
        benchmark_annual = _annualized_return(benchmark_total_return, trading_days)
        benchmark_daily = (1 + benchmark_curve).pct_change().dropna().to_numpy()
        benchmark_volatility = _annualized_volatility(benchmark_daily)
        alpha, beta = _compute_alpha_beta(strategy_daily, benchmark_daily, annual_return, benchmark_annual)
        benchmark_wealth = (1 + benchmark_curve).to_numpy(dtype=float)
        strategy_wealth = values / values[0]
        excess_curve = strategy_wealth / benchmark_wealth
        excess_return = float(excess_curve[-1] - 1)
        excess_daily = (
            strategy_daily[-len(benchmark_daily) :] - benchmark_daily
            if len(benchmark_daily) > 0
            else np.array([])
        )
        excess_mean = float(np.mean(excess_daily) * 100) if len(excess_daily) > 0 else None
        excess_max_drawdown, _ = _max_drawdown_info(excess_curve, dates)
        excess_volatility = _annualized_volatility(excess_daily)
        excess_annual = _annualized_return(excess_return, trading_days)
        excess_sharpe = _sharpe_ratio(excess_annual, excess_volatility, risk_free_rate=0.0)
        information_ratio = excess_sharpe

    daily_win_rate = float(np.mean(strategy_daily > 0)) if len(strategy_daily) > 0 else None

    return [
        ("策略收益", _format_pct(strategy_return * 100), strategy_return),
        ("年化收益", _format_pct(annual_return), annual_return),
        ("超额收益", _format_pct(excess_return * 100 if excess_return is not None else None), excess_return),
        (
            "基准收益",
            _format_pct(benchmark_total_return * 100 if benchmark_total_return is not None else None),
            benchmark_total_return,
        ),
        ("阿尔法", _format_float(alpha), alpha),
        ("贝塔", _format_float(beta), None),
        ("夏普比率", _format_float(sharpe), None),
        ("胜率", _format_float(trade_win_rate), None),
        ("盈亏比", _format_float(profit_loss_ratio), None),
        ("最大回撤", _format_pct(max_drawdown), None),
        ("索提诺比率", _format_float(sortino), None),
        ("日均超额收益", _format_pct(excess_mean), excess_mean),
        ("超额收益最大回撤", _format_pct(excess_max_drawdown), None),
        ("超额收益夏普比率", _format_float(excess_sharpe), None),
        ("日胜率", _format_float(daily_win_rate), None),
        ("盈利次数", _format_int(profit_count), None),
        ("亏损次数", _format_int(loss_count), None),
        ("信息比率", _format_float(information_ratio), None),
        ("策略波动率", _format_float(strategy_volatility), None),
        ("基准波动率", _format_float(benchmark_volatility), None),
        ("最大回撤区间", max_drawdown_period, None),
    ]


def trade_turnover_series(equity_dates: pd.DatetimeIndex, trades: list[Any] | None) -> pd.Series:
    """Return signed daily traded value for filled orders."""
    turnover = pd.Series(0.0, index=equity_dates)
    if not trades:
        return turnover

    for trade in trades:
        filled_date = getattr(trade, "filled_date", None)
        filled_price = getattr(trade, "filled_price", None)
        amount = getattr(trade, "amount", 0)
        if filled_date is None or filled_price is None or amount == 0:
            continue
        trade_date = pd.Timestamp(filled_date)
        if trade_date in turnover.index:
            turnover.loc[trade_date] += amount * float(filled_price)
    return turnover
