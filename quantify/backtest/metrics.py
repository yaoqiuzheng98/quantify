"""Performance metrics: Sharpe, max-drawdown, annual-return, win-rate, etc."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TradeStats:
    """已平仓交易统计(round-trip，聚宽口径)。

    胜负与盈亏比均按**毛盈亏**统计，聚宽口径：分红除息日调低 avg_cost
    (减去每股分红)，卖出时用调整后的成本算盈亏。
    """

    profit_count: int
    loss_count: int
    win_rate: float | None  # 0~1，盈利交易占已平仓交易的比例
    profit_loss_ratio: float | None  # 总毛盈利 / |总毛亏损|
    avg_win: float | None  # 平均每笔毛盈利(元)
    avg_loss: float | None  # 平均每笔毛亏损(元，负数)
    avg_win_pct: float | None  # 平均每笔盈利交易的毛收益率(%)
    avg_loss_pct: float | None  # 平均每笔亏损交易的毛收益率(%，负数)


def realized_trade_stats(
    trades: list[Any] | None,
    dividends: list[Any] | None = None,
    splits: list[Any] | None = None,
) -> TradeStats:
    """按平仓交易(round-trip)统计盈亏，对齐聚宽口径。

    用平均成本法跟踪每只标的持仓，卖出时按比例结算一笔平仓盈亏。聚宽在
    分红除息日将 ``avg_cost`` 调低 ``div_cash``（每股分红），这会降低后续
    卖出的成本基、提高已实现盈亏。本函数在遍历交易时，遇到除息日即对持仓
    成本做同等调整，确保盈亏比与聚宽一致。**份额折算**（split）时聚宽按
    ``ratio`` 调整持仓股数（总成本不变、每股成本降低），本函数同样处理。
    命令行(``metrics``)与 Web 报表(``reporting``)共用此函数，确保胜率/
    盈亏比/盈利次数两处同源同值。
    """
    if not trades:
        return TradeStats(0, 0, None, None, None, None, None, None)

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

    positions: dict[str, dict[str, float]] = {}
    profits: list[float] = []
    losses: list[float] = []
    profit_pcts: list[float] = []
    loss_pcts: list[float] = []

    def _sort_key(t: Any) -> Any:
        d = getattr(t, "filled_date", None)
        return pd.Timestamp(d) if d is not None else pd.Timestamp.min

    for trade in sorted(trades, key=_sort_key):
        code = getattr(trade, "ts_code", "")
        amount = int(getattr(trade, "amount", 0) or 0)
        price = getattr(trade, "filled_price", None)
        if not code or amount == 0 or price is None:
            continue
        pos = positions.setdefault(code, {"amount": 0.0, "cost": 0.0})
        trade_date = getattr(trade, "filled_date", None)

        # Apply pending share-split adjustments (multiply share count, keep total cost).
        if code in split_by_code and trade_date is not None:
            remaining_splits: list[tuple[Any, float]] = []
            for ex_date, ratio in split_by_code[code]:
                if pd.Timestamp(trade_date) >= pd.Timestamp(ex_date):
                    pos["amount"] = int(round(pos["amount"] * ratio))
                else:
                    remaining_splits.append((ex_date, ratio))
            split_by_code[code] = remaining_splits

        # Apply pending dividend cost-basis adjustments (JQ adjusts avg_cost on ex_date).
        if code in div_by_code and pos["amount"] > 0 and trade_date is not None:
            remaining_divs: list[tuple[Any, float]] = []
            for ex_date, div_cash in div_by_code[code]:
                if pd.Timestamp(trade_date) >= pd.Timestamp(ex_date):
                    pos["cost"] -= div_cash * pos["amount"]
                else:
                    remaining_divs.append((ex_date, div_cash))
            div_by_code[code] = remaining_divs

        current = int(pos["amount"])
        if amount > 0:
            pos["amount"] = current + amount
            pos["cost"] += amount * float(price)
            continue
        sell = min(current, abs(amount))
        if sell <= 0:
            continue
        ratio = sell / current
        cost = pos["cost"] * ratio
        gross_pnl = sell * float(price) - cost
        pnl_pct = (gross_pnl / cost * 100) if cost > 0 else 0.0
        eps = 1e-6
        if gross_pnl > eps:
            profits.append(gross_pnl)
            profit_pcts.append(pnl_pct)
        elif gross_pnl < -eps:
            losses.append(gross_pnl)
            loss_pcts.append(pnl_pct)
        pos["amount"] = current - sell
        pos["cost"] -= cost

    profit_count = len(profits)
    loss_count = len(losses)
    total_closed = profit_count + loss_count
    win_rate = profit_count / total_closed if total_closed > 0 else None
    profit_loss_ratio = float(sum(profits) / abs(sum(losses))) if profits and losses else None
    avg_win = float(np.mean(profits)) if profits else None
    avg_loss = float(np.mean(losses)) if losses else None
    avg_win_pct = float(np.mean(profit_pcts)) if profit_pcts else None
    avg_loss_pct = float(np.mean(loss_pcts)) if loss_pcts else None
    return TradeStats(
        profit_count, loss_count, win_rate, profit_loss_ratio, avg_win, avg_loss, avg_win_pct, avg_loss_pct
    )


@dataclass
class BacktestMetrics:
    """Summary statistics computed from an equity curve."""

    start_date: str
    end_date: str
    trading_days: int
    calendar_days: int
    initial_cash: float
    final_value: float
    total_return_pct: float
    annual_return_pct: float
    max_drawdown_pct: float
    max_drawdown_duration: int
    sharpe_ratio: float
    calmar_ratio: float
    volatility_pct: float
    win_rate_pct: float
    avg_win_pct: float
    avg_loss_pct: float
    profit_factor: float
    profit_count: int
    loss_count: int
    total_commission: float
    total_slippage: float
    total_tax: float
    trade_count: int

    def to_dict(self) -> dict:
        return {
            "start_date": self.start_date,
            "end_date": self.end_date,
            "trading_days": self.trading_days,
            "calendar_days": self.calendar_days,
            "initial_cash": round(self.initial_cash, 2),
            "final_value": round(self.final_value, 2),
            "total_return_pct": round(self.total_return_pct, 2),
            "annual_return_pct": round(self.annual_return_pct, 2),
            "max_drawdown_pct": round(self.max_drawdown_pct, 2),
            "max_drawdown_duration": self.max_drawdown_duration,
            "sharpe_ratio": round(self.sharpe_ratio, 2),
            "calmar_ratio": round(self.calmar_ratio, 2),
            "volatility_pct": round(self.volatility_pct, 2),
            "win_rate_pct": round(self.win_rate_pct, 2),
            "avg_win_pct": round(self.avg_win_pct, 4),
            "avg_loss_pct": round(self.avg_loss_pct, 4),
            "profit_factor": round(self.profit_factor, 2),
            "profit_count": self.profit_count,
            "loss_count": self.loss_count,
            "total_commission": round(self.total_commission, 2),
            "total_slippage": round(self.total_slippage, 2),
            "total_tax": round(self.total_tax, 2),
            "trade_count": self.trade_count,
        }

    def to_llm_prompt(self) -> str:
        """Return a human-readable summary suitable for LLM analysis."""
        d = self.to_dict()
        lines = [
            "=== Backtest Results ===",
            f"Period: {d['start_date']} → {d['end_date']} ({d['trading_days']} trading days, {d['calendar_days']} calendar days)",
            f"Initial capital: ¥{d['initial_cash']:,.2f}",
            f"Final value:    ¥{d['final_value']:,.2f}",
            "",
            "--- Returns ---",
            f"Total return:   {d['total_return_pct']:+.2f}%",
            f"Annual return:  {d['annual_return_pct']:+.2f}%",
            "",
            "--- Risk ---",
            f"Max drawdown:   {d['max_drawdown_pct']:.2f}% (duration: {d['max_drawdown_duration']} days)",
            f"Volatility:     {d['volatility_pct']:.2f}%",
            "",
            "--- Risk-Adjusted ---",
            f"Sharpe ratio:   {d['sharpe_ratio']:.2f}",
            f"Calmar ratio:   {d['calmar_ratio']:.2f}",
            "",
            "--- Trading ---",
            f"Trade count:    {d['trade_count']}",
            f"Win rate:       {d['win_rate_pct']:.2f}% ({d['profit_count']}W/{d['loss_count']}L round-trips)",
            f"Avg win:        {d['avg_win_pct']:.4f}% per trade",
            f"Avg loss:       {d['avg_loss_pct']:.4f}% per trade",
            f"Profit factor:  {d['profit_factor']:.2f} (total gross profit / loss)",
            "",
            "--- Costs ---",
            f"Commission:     ¥{d['total_commission']:,.2f}",
            f"Slippage:       ¥{d['total_slippage']:,.2f}",
            f"Stamp duty:     ¥{d['total_tax']:,.2f}",
        ]
        return "\n".join(lines)


def compute_metrics(
    equity_df: pd.DataFrame,
    initial_cash: float,
    commission: float = 0,
    slippage: float = 0,
    tax: float = 0,
    trade_count: int = 0,
    risk_free_rate: float = 0.04,
    trades: list[Any] | None = None,
    dividends: list[Any] | None = None,
    splits: list[Any] | None = None,
) -> BacktestMetrics:
    """Derive a full suite of performance statistics from a daily equity curve.

    Parameters
    ----------
    equity_df:
        Must contain columns ``date`` (datetime), ``value`` (float).
    initial_cash:
        Starting capital, used to compute initial-return baselines.
    commission, slippage, trade_count:
        Cost & trade counters reported directly in the output.
    risk_free_rate:
        Annual risk-free rate (default 3 %).
    """
    if equity_df.empty or len(equity_df) < 2:
        return _empty_metrics(initial_cash, commission, slippage, trade_count)

    values = equity_df["value"].values
    dates = equity_df["date"].values

    start_date = pd.Timestamp(dates[0]).strftime("%Y-%m-%d")
    end_date = pd.Timestamp(dates[-1]).strftime("%Y-%m-%d")
    trading_days = len(values)
    calendar_days = (pd.Timestamp(dates[-1]) - pd.Timestamp(dates[0])).days

    final_value = float(values[-1])
    total_return_pct = (final_value / initial_cash - 1) * 100

    # daily returns
    daily_returns = np.diff(values) / values[:-1]

    # 年化收益率(聚宽口径): (1 + 总收益率)^(365/回测自然天数) - 1
    if calendar_days > 0 and final_value > 0:
        annual_return_pct = ((final_value / initial_cash) ** (365 / calendar_days) - 1) * 100
    else:
        annual_return_pct = 0.0

    # max drawdown
    peak = np.maximum.accumulate(values)
    drawdowns = (peak - values) / peak
    max_dd_pct = float(np.max(drawdowns) * 100)

    # drawdown duration
    dd_duration = 0
    current_duration = 0
    for dd in drawdowns:
        if dd > 0:
            current_duration += 1
            dd_duration = max(dd_duration, current_duration)
        else:
            current_duration = 0

    # volatility (annualised with 250 trading days)
    vol_pct = float(np.std(daily_returns) * 100 * math.sqrt(250)) if len(daily_returns) > 0 else 0.0

    # Sharpe (聚宽口径): (年化收益率 - 无风险利率) / 年化波动率
    # 年化收益率用 365 自然日复利,年化波动率用 250 交易日,无风险利率 4%。
    annual_vol = np.std(daily_returns) * math.sqrt(250)
    sharpe = float((annual_return_pct / 100 - risk_free_rate) / annual_vol) if annual_vol > 0 else 0.0

    # Calmar
    calmar = annual_return_pct / max_dd_pct if max_dd_pct > 0 else 0.0

    # win / loss —— 按平仓交易(round-trip)统计，对齐聚宽口径，与 Web 报表同源。
    # 聚宽在除息日调低 avg_cost (减去每股分红)，影响已实现盈亏，需传入分红事件。
    # 份额折算时聚宽按 ratio 调整持仓股数(成本不变)，需传入折算事件。
    stats = realized_trade_stats(trades, dividends, splits)
    win_rate = (stats.win_rate * 100) if stats.win_rate is not None else 0.0
    avg_win = stats.avg_win_pct if stats.avg_win_pct is not None else 0.0
    avg_loss = stats.avg_loss_pct if stats.avg_loss_pct is not None else 0.0
    # profit_factor 即聚宽"盈亏比" = 总毛盈利 / |总毛亏损|。
    profit_factor = stats.profit_loss_ratio if stats.profit_loss_ratio is not None else float("inf")
    profit_count = stats.profit_count
    loss_count = stats.loss_count

    return BacktestMetrics(
        start_date=start_date,
        end_date=end_date,
        trading_days=trading_days,
        calendar_days=calendar_days,
        initial_cash=initial_cash,
        final_value=final_value,
        total_return_pct=total_return_pct,
        annual_return_pct=annual_return_pct,
        max_drawdown_pct=max_dd_pct,
        max_drawdown_duration=dd_duration,
        sharpe_ratio=sharpe,
        calmar_ratio=calmar,
        volatility_pct=vol_pct,
        win_rate_pct=win_rate,
        avg_win_pct=avg_win,
        avg_loss_pct=avg_loss,
        profit_factor=profit_factor,
        profit_count=profit_count,
        loss_count=loss_count,
        total_commission=commission,
        total_slippage=slippage,
        total_tax=tax,
        trade_count=trade_count,
    )


def _empty_metrics(
    initial_cash: float, commission: float, slippage: float, trades: int, tax: float = 0.0
) -> BacktestMetrics:
    return BacktestMetrics(
        start_date="",
        end_date="",
        trading_days=0,
        calendar_days=0,
        initial_cash=initial_cash,
        final_value=initial_cash,
        total_return_pct=0.0,
        annual_return_pct=0.0,
        max_drawdown_pct=0.0,
        max_drawdown_duration=0,
        sharpe_ratio=0.0,
        calmar_ratio=0.0,
        volatility_pct=0.0,
        win_rate_pct=0.0,
        avg_win_pct=0.0,
        avg_loss_pct=0.0,
        profit_factor=0.0,
        profit_count=0,
        loss_count=0,
        total_commission=commission,
        total_slippage=slippage,
        total_tax=tax,
        trade_count=trades,
    )
