"""Report helpers for backtest metrics and interactive views."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .metrics import realized_trade_stats


def _clean_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _clean_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clean_value(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    return value


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    number = float(value)
    return number if np.isfinite(number) else None


def _as_date(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    return pd.Timestamp(value).strftime("%Y-%m-%d")


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


def _daily_returns(values: np.ndarray, base_value: float | None = None) -> np.ndarray:
    if len(values) == 0:
        return np.array([], dtype=float)
    base = float(values[0] if base_value is None else base_value)
    previous = np.concatenate(([base], values[:-1]))
    return (values - previous) / previous


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
    annual_return_pct: float, annual_volatility: float | None, risk_free_rate: float = 0.04
) -> float | None:
    if annual_volatility is None or annual_volatility <= 0:
        return None
    return (annual_return_pct / 100 - risk_free_rate) / annual_volatility


def _sortino_ratio(
    daily_returns: np.ndarray, annual_return_pct: float, risk_free_rate: float = 0.04
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
    return (annual_return_pct / 100) / downside_volatility


def _downside_volatility(values: np.ndarray) -> float | None:
    downside = np.minimum(values, 0.0)
    if len(downside) == 0:
        return None
    volatility = float(np.sqrt(np.mean(np.square(downside))) * np.sqrt(250))
    return volatility if volatility > 0 else None


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
    base_value = (
        float(benchmark_df.attrs.get("base_value", benchmark.iloc[0])) if not benchmark.empty else 0.0
    )
    if benchmark.empty or base_value <= 0:
        return None
    return benchmark / base_value - 1


def benchmark_daily_return_series(
    dates: pd.DatetimeIndex,
    benchmark_df: pd.DataFrame | None,
) -> np.ndarray | None:
    if benchmark_df is None or benchmark_df.empty:
        return None

    value_col = "daily_value" if "daily_value" in benchmark_df.columns else "value"
    benchmark = pd.Series(
        benchmark_df[value_col].astype(float).values,
        index=pd.to_datetime(benchmark_df["date"]),
    ).sort_index()
    benchmark = benchmark.reindex(dates).ffill().dropna()
    base_key = "daily_base_value" if value_col == "daily_value" else "base_value"
    base_value = float(benchmark_df.attrs.get(base_key, benchmark.iloc[0])) if not benchmark.empty else 0.0
    if benchmark.empty or base_value <= 0:
        return None
    return _daily_returns(benchmark.to_numpy(dtype=float), base_value=base_value)


def _compute_alpha_beta(
    strategy_returns: np.ndarray,
    benchmark_returns: np.ndarray,
    annual_return_pct: float,
    benchmark_annual_pct: float,
    risk_free_rate: float = 0.04,
) -> tuple[float | None, float | None]:
    if len(strategy_returns) < 2 or len(benchmark_returns) < 2:
        return None, None

    count = min(len(strategy_returns), len(benchmark_returns))
    strategy_returns = strategy_returns[-count:]
    benchmark_returns = benchmark_returns[-count:]
    variance = float(np.var(benchmark_returns, ddof=1))
    if variance <= 0:
        return None, None

    beta = float(np.cov(strategy_returns, benchmark_returns)[0, 1] / variance)
    alpha = annual_return_pct / 100 - risk_free_rate - beta * (benchmark_annual_pct / 100 - risk_free_rate)
    return alpha, beta


def build_report_items(
    equity_df: pd.DataFrame,
    benchmark_df: pd.DataFrame | None = None,
    metrics: Any | None = None,
    trades: list[Any] | None = None,
    dividends: list[Any] | None = None,
) -> list[tuple[str, str, float | None]]:
    """Return JoinQuant-style metrics for interactive report views."""
    del metrics
    if equity_df.empty:
        return []

    values = equity_df["value"].astype(float).to_numpy()
    strategy_daily = _daily_returns(values, base_value=values[0])
    trading_days = len(values)
    dates = pd.DatetimeIndex(pd.to_datetime(equity_df["date"]))

    strategy_return = values[-1] / values[0] - 1
    annual_return = _annualized_return(strategy_return, trading_days)
    strategy_volatility = _annualized_volatility(strategy_daily)
    sharpe = _sharpe_ratio(annual_return, strategy_volatility)
    max_drawdown, max_drawdown_period = _max_drawdown_info(values / values[0], dates)
    sortino = _sortino_ratio(strategy_daily, annual_return)
    _trade_stats = realized_trade_stats(trades)
    profit_count = _trade_stats.profit_count
    loss_count = _trade_stats.loss_count
    trade_win_rate = _trade_stats.win_rate
    profit_loss_ratio = _trade_stats.profit_loss_ratio

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
    benchmark_daily = None
    benchmark_curve = benchmark_return_series(dates, benchmark_df)
    if benchmark_curve is not None:
        benchmark_total_return = float(benchmark_curve.iloc[-1])
        benchmark_annual = _annualized_return(benchmark_total_return, trading_days)
        benchmark_daily = benchmark_daily_return_series(dates, benchmark_df)
        benchmark_volatility = (
            _annualized_volatility(benchmark_daily) if benchmark_daily is not None else None
        )
        if benchmark_daily is not None:
            alpha, beta = _compute_alpha_beta(
                strategy_daily, benchmark_daily, annual_return, benchmark_annual
            )
        benchmark_wealth = (1 + benchmark_curve).to_numpy(dtype=float)
        strategy_wealth = values / values[0]
        excess_curve = strategy_wealth / benchmark_wealth
        excess_return = float(excess_curve[-1] - 1)
        excess_daily = (
            strategy_daily[-len(benchmark_daily) :] - benchmark_daily
            if benchmark_daily is not None and len(benchmark_daily) > 0
            else np.array([])
        )
        excess_mean = float(np.mean(excess_daily) * 100) if len(excess_daily) > 0 else None
        excess_max_drawdown, _ = _max_drawdown_info(excess_curve, dates)
        excess_volatility = _annualized_volatility(excess_daily)
        excess_annual = _annualized_return(excess_return, trading_days)
        excess_downside_volatility = _downside_volatility(excess_daily)
        excess_sharpe = (
            excess_annual / 100 / excess_downside_volatility
            if excess_downside_volatility is not None
            else None
        )
        information_ratio = (
            ((annual_return - benchmark_annual) / 100) / excess_volatility
            if excess_volatility is not None and excess_volatility > 0
            else None
        )

    daily_win_rate = None
    if len(strategy_daily) > 0:
        if benchmark_curve is not None and benchmark_daily is not None and len(benchmark_daily) > 0:
            count = min(len(strategy_daily), len(benchmark_daily))
            daily_win_rate = float(np.mean(strategy_daily[-count:] > benchmark_daily[-count:]))
        else:
            daily_win_rate = float(np.mean(strategy_daily > 0))

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


def _trade_records(trades: list[Any] | None) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    # 按平均成本法跟踪每个标的的持仓,卖出时结算平仓盈亏(扣除买入费用分摊
    # 与本次卖出费用,对齐 _realized_trade_stats 的净盈亏口径)。
    positions: dict[str, dict[str, float]] = {}
    for trade in trades or []:
        amount = int(getattr(trade, "filled_amount", 0) or getattr(trade, "amount", 0) or 0)
        price = _as_float(getattr(trade, "filled_price", None))
        trade_value = abs(amount) * price if price is not None else None
        commission = _as_float(getattr(trade, "commission", 0.0))
        slippage = _as_float(getattr(trade, "slippage", 0.0))
        tax = _as_float(getattr(trade, "tax", 0.0))
        code = getattr(trade, "ts_code", "")
        trade_fees = (commission or 0.0) + (slippage or 0.0) + (tax or 0.0)

        realized_pnl: float | None = None
        position = positions.setdefault(code, {"amount": 0.0, "cost": 0.0, "fees": 0.0})
        current_amount = int(position["amount"])

        if price is not None and amount > 0:
            position["amount"] = current_amount + amount
            position["cost"] += amount * price
            position["fees"] += trade_fees
        elif price is not None and amount < 0 and current_amount > 0:
            sell_amount = min(current_amount, abs(amount))
            ratio = sell_amount / current_amount
            cost = position["cost"] * ratio
            buy_fees = position["fees"] * ratio
            proceeds = sell_amount * price
            realized_pnl = proceeds - cost - buy_fees - trade_fees
            position["amount"] = current_amount - sell_amount
            position["cost"] -= cost
            position["fees"] -= buy_fees

        records.append(
            {
                "date": _as_date(getattr(trade, "filled_date", None)),
                "ts_code": code,
                "direction": "buy" if amount > 0 else "sell" if amount < 0 else "flat",
                "amount": amount,
                "price": price,
                "value": trade_value,
                "commission": commission,
                "slippage": slippage,
                "tax": tax,
                "realized_pnl": _as_float(realized_pnl),
            }
        )
    return records


def build_report_payload(
    equity_df: pd.DataFrame,
    benchmark_df: pd.DataFrame | None = None,
    metrics: Any | None = None,
    trades: list[Any] | None = None,
    dividends: list[Any] | None = None,
) -> dict[str, Any]:
    """Build the canonical report payload shared by Web and LLM outputs."""
    metric_values = _clean_value(metrics.to_dict()) if metrics is not None else {}
    report_items = [
        {"label": label, "value": value, "numeric_value": _clean_value(numeric_value)}
        for label, value, numeric_value in build_report_items(
            equity_df,
            benchmark_df,
            metrics,
            trades,
            dividends,
        )
    ]
    trade_records = _trade_records(trades)

    if equity_df.empty:
        return {
            "metrics": metric_values,
            "report_items": report_items,
            "curves": [],
            "trades": trade_records,
        }

    dates = pd.DatetimeIndex(pd.to_datetime(equity_df["date"]))
    values = equity_df["value"].astype(float).to_numpy()
    initial_value = float(values[0])
    strategy_return = values / initial_value - 1
    daily_pnl = np.diff(values, prepend=values[0])
    wealth = values / initial_value
    drawdown = wealth / np.maximum.accumulate(wealth) - 1
    turnover = trade_turnover_series(dates, trades)

    # 每日持仓市值占总资产比例(持仓比例),总资产为 0 时记为 0。
    if "position_value" in equity_df.columns:
        position_value = equity_df["position_value"].astype(float).to_numpy()
    else:
        position_value = np.zeros_like(values)
    with np.errstate(divide="ignore", invalid="ignore"):
        position_ratio = np.where(values != 0, position_value / values, 0.0)

    # 每日各标的持仓市值(用于堆叠展示各标的占总资产比例)。
    per_code_values: list[dict[str, float]] = (
        list(equity_df["position_values"])
        if "position_values" in equity_df.columns
        else [{} for _ in range(len(values))]
    )

    benchmark_return = pd.Series(np.nan, index=dates)
    benchmark_curve = benchmark_return_series(dates, benchmark_df)
    if benchmark_curve is not None:
        benchmark_return.loc[benchmark_curve.index] = benchmark_curve.to_numpy(dtype=float)

    curves: list[dict[str, Any]] = []
    for index, curve_date in enumerate(dates):
        benchmark_value = _as_float(benchmark_return.iloc[index])
        excess_return = strategy_return[index] - benchmark_value if benchmark_value is not None else None
        benchmark_equity = initial_value * (1 + benchmark_value) if benchmark_value is not None else None
        total_value = values[index]
        code_values = per_code_values[index] if index < len(per_code_values) else {}
        position_ratios = {
            code: _as_float((market_value / total_value * 100) if total_value else 0.0)
            for code, market_value in (code_values or {}).items()
        }
        curves.append(
            {
                "date": curve_date.strftime("%Y-%m-%d"),
                "equity": _as_float(values[index]),
                "strategy_return": _as_float(strategy_return[index]),
                "strategy_return_pct": _as_float(strategy_return[index] * 100),
                "benchmark_equity": _as_float(benchmark_equity),
                "benchmark_return": benchmark_value,
                "benchmark_return_pct": _as_float(
                    benchmark_value * 100 if benchmark_value is not None else None
                ),
                "excess_return": _as_float(excess_return),
                "excess_return_pct": _as_float(excess_return * 100 if excess_return is not None else None),
                "drawdown": _as_float(drawdown[index]),
                "drawdown_pct": _as_float(drawdown[index] * 100),
                "daily_pnl": _as_float(daily_pnl[index]),
                "turnover": _as_float(turnover.iloc[index]),
                "position_value": _as_float(position_value[index]),
                "position_ratio": _as_float(position_ratio[index]),
                "position_ratio_pct": _as_float(position_ratio[index] * 100),
                "position_ratios_pct": position_ratios,
            }
        )

    return {
        "metrics": metric_values,
        "report_items": report_items,
        "curves": curves,
        "trades": trade_records,
    }
