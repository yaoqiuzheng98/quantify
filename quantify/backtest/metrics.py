"""Performance metrics: Sharpe, max-drawdown, annual-return, win-rate, etc."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class BacktestMetrics:
    """Summary statistics computed from an equity curve."""

    start_date: str
    end_date: str
    trading_days: int
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
    total_commission: float
    total_slippage: float
    total_tax: float
    trade_count: int

    def to_dict(self) -> dict:
        return {
            "start_date": self.start_date,
            "end_date": self.end_date,
            "trading_days": self.trading_days,
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
            f"Period: {d['start_date']} → {d['end_date']} ({d['trading_days']} trading days)",
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
            f"Win rate:       {d['win_rate_pct']:.2f}%",
            f"Avg win:        {d['avg_win_pct']:.4f}%",
            f"Avg loss:       {d['avg_loss_pct']:.4f}%",
            f"Profit factor:  {d['profit_factor']:.2f}",
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

    final_value = float(values[-1])
    total_return_pct = (final_value / initial_cash - 1) * 100

    # daily returns
    daily_returns = np.diff(values) / values[:-1]

    # annual (compounded) — JoinQuant 用 250 个交易日年化
    years = trading_days / 250
    if years > 0 and final_value > 0:
        annual_return_pct = ((final_value / initial_cash) ** (1 / years) - 1) * 100
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

    # volatility (annualised with 250 trading days, JoinQuant 口径)
    vol_pct = float(np.std(daily_returns) * 100 * math.sqrt(250)) if len(daily_returns) > 0 else 0.0

    # Sharpe — 对齐聚宽：(年化收益率 - 无风险利率) / 年化波动率
    # 分子用复利年化收益(非日收益算术均值)，年化波动率用 250 个交易日，无风险利率 4%。
    annual_vol = np.std(daily_returns) * math.sqrt(250)
    sharpe = float((annual_return_pct / 100 - risk_free_rate) / annual_vol) if annual_vol > 0 else 0.0

    # Calmar
    calmar = annual_return_pct / max_dd_pct if max_dd_pct > 0 else 0.0

    # win / loss
    wins = daily_returns[daily_returns > 0]
    losses = daily_returns[daily_returns < 0]
    win_rate = (len(wins) / len(daily_returns) * 100) if len(daily_returns) > 0 else 0.0
    avg_win = float(np.mean(wins) * 100) if len(wins) > 0 else 0.0
    avg_loss = float(np.mean(losses) * 100) if len(losses) > 0 else 0.0
    profit_factor = abs(wins.sum() / losses.sum()) if len(losses) > 0 and losses.sum() != 0 else float("inf")

    return BacktestMetrics(
        start_date=start_date,
        end_date=end_date,
        trading_days=trading_days,
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
        total_commission=commission,
        total_slippage=slippage,
        total_tax=tax,
        trade_count=trades,
    )
