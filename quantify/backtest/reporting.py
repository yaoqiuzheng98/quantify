"""Report helpers for backtest metrics and interactive views."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


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


def _annualized_return(total_return: float, calendar_days: int) -> float:
    """年化收益率(聚宽口径): (1 + 总收益率)^(365/回测自然天数) - 1"""
    if calendar_days <= 0 or total_return <= -1:
        return 0.0
    return ((1 + total_return) ** (365 / calendar_days) - 1) * 100


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


def _sortino_ratio(
    daily_returns: np.ndarray, annual_return_pct: float, risk_free_rate: float = 0.04
) -> float | None:
    # 聚宽口径:索提诺 = (年化收益 − 无风险利率) / 下行波动率。
    # 下行波动率 σ_d = sqrt( (1/N) · Σ min(R_i − R_target, 0)^2 ) · sqrt(250)，
    # 其中 R_target = 日无风险利率，**分母除以全部 N 天**(不是仅下行天数)，年化用 250。
    if len(daily_returns) == 0:
        return None
    daily_risk_free = risk_free_rate / 250
    downside = np.minimum(daily_returns - daily_risk_free, 0.0)
    downside_volatility = float(np.sqrt(np.mean(np.square(downside))) * np.sqrt(250))
    if downside_volatility <= 0:
        return None
    return (annual_return_pct / 100 - risk_free_rate) / downside_volatility


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
    """Return JoinQuant-style metrics for interactive report views.

    策略侧指标(收益/年化/夏普/波动率/回撤/胜率/盈亏比等)一律从传入的 ``metrics``
    对象取值,与 ``compute_metrics`` / ``to_llm_prompt`` **同源**,确保 Web 与 LLM
    两端数值完全一致。基准相关指标(超额收益/alpha/beta/信息比率等)和最大回撤区间、
    索提诺比率、日胜率等 Web 独有项在此函数内补充计算。
    """
    del trades, dividends
    if equity_df.empty or metrics is None:
        return []

    values = equity_df["value"].astype(float).to_numpy()
    dates = pd.DatetimeIndex(pd.to_datetime(equity_df["date"]))
    strategy_daily = _daily_returns(values, base_value=values[0])

    max_drawdown_pct = metrics.max_drawdown_pct
    _, max_drawdown_period = _max_drawdown_info(values / values[0], dates)
    sortino = _sortino_ratio(strategy_daily, metrics.annual_return_pct)

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
        benchmark_annual = _annualized_return(benchmark_total_return, metrics.calendar_days)
        benchmark_daily = benchmark_daily_return_series(dates, benchmark_df)
        benchmark_volatility = (
            _annualized_volatility(benchmark_daily) if benchmark_daily is not None else None
        )
        if benchmark_daily is not None:
            alpha, beta = _compute_alpha_beta(
                strategy_daily, benchmark_daily, metrics.annual_return_pct, benchmark_annual
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
        excess_annual = _annualized_return(excess_return, metrics.calendar_days)
        # 超额收益夏普 = (超额年化收益 − 无风险利率) / 超额收益年化波动率(全波动，非下行)，
        # 对齐聚宽口径。无风险利率 4%，年化波动率用 250 交易日(见 _annualized_volatility)。
        excess_sharpe = (
            (excess_annual / 100 - 0.04) / excess_volatility
            if excess_volatility is not None and excess_volatility > 0
            else None
        )
        information_ratio = (
            ((metrics.annual_return_pct - benchmark_annual) / 100) / excess_volatility
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
        ("策略收益", _format_pct(metrics.total_return_pct), metrics.total_return_pct),
        ("年化收益", _format_pct(metrics.annual_return_pct), metrics.annual_return_pct),
        ("超额收益", _format_pct(excess_return * 100 if excess_return is not None else None), excess_return),
        (
            "基准收益",
            _format_pct(benchmark_total_return * 100 if benchmark_total_return is not None else None),
            benchmark_total_return,
        ),
        ("阿尔法", _format_float(alpha), alpha),
        ("贝塔", _format_float(beta), beta),
        ("夏普比率", _format_float(metrics.sharpe_ratio), metrics.sharpe_ratio),
        ("胜率", _format_pct(metrics.win_rate_pct), metrics.win_rate_pct),
        ("盈亏比", _format_float(metrics.profit_factor), metrics.profit_factor),
        ("最大回撤", _format_pct(max_drawdown_pct), max_drawdown_pct),
        ("索提诺比率", _format_float(sortino), sortino),
        ("日均超额收益", _format_pct(excess_mean), excess_mean),
        ("超额收益最大回撤", _format_pct(excess_max_drawdown), excess_max_drawdown),
        ("超额收益夏普比率", _format_float(excess_sharpe), excess_sharpe),
        (
            "日胜率",
            _format_pct(daily_win_rate * 100 if daily_win_rate is not None else None),
            daily_win_rate * 100 if daily_win_rate is not None else None,
        ),
        ("盈利次数", _format_int(metrics.profit_count), metrics.profit_count),
        ("亏损次数", _format_int(metrics.loss_count), metrics.loss_count),
        ("信息比率", _format_float(information_ratio), information_ratio),
        ("策略波动率", _format_pct(metrics.volatility_pct), metrics.volatility_pct),
        (
            "基准波动率",
            _format_pct(benchmark_volatility * 100 if benchmark_volatility is not None else None),
            benchmark_volatility * 100 if benchmark_volatility is not None else None,
        ),
        ("最大回撤区间", max_drawdown_period, None),
    ]


def _trade_records(
    trades: list[Any] | None,
    dividends: list[Any] | None = None,
    splits: list[Any] | None = None,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    # 按平均成本法跟踪每个标的的持仓,卖出时结算平仓盈亏(扣除买入费用分摊
    # 与本次卖出费用,对齐 realized_trade_stats 的净盈亏口径)。
    # 必须做 split/dividend 调整,与 realized_trade_stats 保持一致,
    # 否则 broker 已做 split 调整的卖出股数与原始买入股数不匹配,盈亏失真。

    # Build (ts_code) -> [(ex_date, div_cash)] map for cost-basis adjustment.
    div_by_code: dict[str, list[tuple[Any, float]]] = {}
    for d in dividends or []:
        code = getattr(d, "ts_code", None)
        ex_date = getattr(d, "ex_date", None) or getattr(d, "record_date", None)
        div_cash = getattr(d, "div_cash", None)
        if code and ex_date and div_cash:
            div_by_code.setdefault(code, []).append((ex_date, float(div_cash)))

    # Build (ts_code) -> [(ex_date, ratio)] map for share-split adjustment.
    split_by_code: dict[str, list[tuple[Any, float]]] = {}
    for s in splits or []:
        code = getattr(s, "ts_code", None)
        ex_date = getattr(s, "ex_date", None)
        ratio = getattr(s, "ratio", None)
        if code and ex_date and ratio and abs(ratio - 1.0) > 1e-6:
            split_by_code.setdefault(code, []).append((ex_date, float(ratio)))

    def _sort_key(t: Any) -> Any:
        d = getattr(t, "filled_date", None)
        return pd.Timestamp(d) if d is not None else pd.Timestamp.min

    positions: dict[str, dict[str, float]] = {}
    for trade in sorted(trades or [], key=_sort_key):
        amount = int(getattr(trade, "filled_amount", 0) or getattr(trade, "amount", 0) or 0)
        price = _as_float(getattr(trade, "filled_price", None))
        trade_value = abs(amount) * price if price is not None else None
        commission = _as_float(getattr(trade, "commission", 0.0))
        slippage = _as_float(getattr(trade, "slippage", 0.0))
        tax = _as_float(getattr(trade, "tax", 0.0))
        code = getattr(trade, "ts_code", "")
        trade_fees = (commission or 0.0) + (slippage or 0.0) + (tax or 0.0)
        trade_date = getattr(trade, "filled_date", None)

        realized_pnl: float | None = None
        position = positions.setdefault(code, {"amount": 0.0, "cost": 0.0, "fees": 0.0})

        # Apply pending share-split adjustments (multiply share count, keep total cost).
        if code in split_by_code and trade_date is not None:
            remaining_splits: list[tuple[Any, float]] = []
            for ex_date, ratio in split_by_code[code]:
                if pd.Timestamp(trade_date) >= pd.Timestamp(ex_date):
                    position["amount"] = int(round(position["amount"] * ratio))
                else:
                    remaining_splits.append((ex_date, ratio))
            split_by_code[code] = remaining_splits

        # Apply pending dividend cost-basis adjustments.
        if code in div_by_code and position["amount"] > 0 and trade_date is not None:
            remaining_divs: list[tuple[Any, float]] = []
            for ex_date, div_cash in div_by_code[code]:
                if pd.Timestamp(trade_date) >= pd.Timestamp(ex_date):
                    position["cost"] -= div_cash * position["amount"]
                else:
                    remaining_divs.append((ex_date, div_cash))
            div_by_code[code] = remaining_divs

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
    splits: list[Any] | None = None,
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
    trade_records = _trade_records(trades, dividends, splits)

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

    benchmark_return = pd.Series(np.nan, index=dates)
    benchmark_curve = benchmark_return_series(dates, benchmark_df)
    if benchmark_curve is not None:
        benchmark_return.loc[benchmark_curve.index] = benchmark_curve.to_numpy(dtype=float)

    curves: list[dict[str, Any]] = []
    for index, curve_date in enumerate(dates):
        benchmark_value = _as_float(benchmark_return.iloc[index])
        excess_return = strategy_return[index] - benchmark_value if benchmark_value is not None else None
        benchmark_equity = initial_value * (1 + benchmark_value) if benchmark_value is not None else None
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
            }
        )

    return {
        "metrics": metric_values,
        "report_items": report_items,
        "curves": curves,
        "trades": trade_records,
    }
