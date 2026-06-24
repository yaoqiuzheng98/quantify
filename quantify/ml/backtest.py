"""Vectorized backtest for ML/DL models.

Lightweight portfolio simulation: given daily stock scores, select top-N,
equal-weight (or score-weighted), compute portfolio returns and metrics.

Does NOT go through the event-driven BacktestEngine — no trading friction,
no T+1, no price limits.  Designed for fast iteration before validating
winners in the full engine.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class VectorBacktestResult:
    """Result of a vectorized backtest."""

    daily_returns: pd.Series
    cumulative: pd.Series
    holdings: pd.DataFrame  # (date, asset) → weight
    metrics: dict

    @property
    def total_return(self) -> float:
        return self.metrics.get("total_return_pct", 0.0)

    @property
    def sharpe(self) -> float:
        return self.metrics.get("sharpe_ratio", 0.0)

    @property
    def max_drawdown(self) -> float:
        return self.metrics.get("max_drawdown_pct", 0.0)


def vectorized_backtest(
    scores: pd.DataFrame,
    close_prices: pd.DataFrame,
    top_n: int = 20,
    rebalance_days: int = 5,
    weight_method: str = "equal",
) -> VectorBacktestResult:
    """Run a vectorized backtest from daily stock scores.

    Parameters
    ----------
    scores : pd.DataFrame
        (date × asset) DataFrame of model scores.  Higher = better.
    close_prices : pd.DataFrame
        (date × asset) DataFrame of close prices, same index/columns as scores.
    top_n : int
        Number of stocks to hold.
    rebalance_days : int
        Rebalance every N trading days.
    weight_method : str
        "equal" or "score" (proportional to score).
    """
    # Align
    common_dates = sorted(set(scores.index) & set(close_prices.index))
    common_assets = sorted(set(scores.columns) & set(close_prices.columns))
    scores = scores.loc[common_dates, common_assets]
    close_prices = close_prices.loc[common_dates, common_assets]

    holdings = pd.DataFrame(0.0, index=scores.index, columns=scores.columns)
    current_holdings: pd.Series | None = None

    for i, dt in enumerate(common_dates):
        if i % rebalance_days == 0:
            # Select top-N stocks by score
            day_scores = scores.loc[dt].dropna()
            if len(day_scores) == 0:
                current_holdings = None
                continue

            top_stocks = day_scores.nlargest(min(top_n, len(day_scores)))

            if weight_method == "score":
                # Proportional to positive part of score
                weights = top_stocks.clip(lower=0)
                total = weights.sum()
                if total > 0:
                    weights = weights / total
                else:
                    weights = pd.Series(1.0 / len(top_stocks), index=top_stocks.index)
            else:
                # Equal weight
                weights = pd.Series(1.0 / len(top_stocks), index=top_stocks.index)

            current_holdings = weights

        if current_holdings is not None:
            for stock, w in current_holdings.items():
                holdings.loc[dt, stock] = w

    # Compute daily portfolio returns
    daily_ret = close_prices.pct_change(fill_method=None).fillna(0.0)
    # Shift holdings by 1 day: today's return earned on yesterday's holdings
    holdings_shifted = holdings.shift(1).fillna(0.0)
    portfolio_ret = (holdings_shifted * daily_ret).sum(axis=1)
    # First day has no return
    portfolio_ret.iloc[0] = 0.0

    cumulative = (1 + portfolio_ret).cumprod()
    metrics = _compute_metrics(portfolio_ret, cumulative)

    return VectorBacktestResult(
        daily_returns=portfolio_ret,
        cumulative=cumulative,
        holdings=holdings,
        metrics=metrics,
    )


def _compute_metrics(returns: pd.Series, cumulative: pd.Series) -> dict:
    """Compute standard backtest metrics from daily returns."""
    total_return = float(cumulative.iloc[-1] - 1) if len(cumulative) else 0.0
    n_days = len(returns)
    n_years = n_days / 250

    annual_return = float((cumulative.iloc[-1] ** (1 / n_years) - 1)) if n_years > 0 else 0.0

    volatility = float(returns.std() * np.sqrt(250)) if n_days > 1 else 0.0
    sharpe = float((annual_return - 0.04) / volatility) if volatility > 0 else 0.0

    # Max drawdown
    peak = cumulative.expanding().max()
    drawdown = (cumulative - peak) / peak
    max_dd = float(drawdown.min()) if len(drawdown) else 0.0

    # Win rate (daily)
    win_rate = float((returns > 0).sum() / max((returns != 0).sum(), 1))

    return {
        "total_return_pct": total_return * 100,
        "annual_return_pct": annual_return * 100,
        "volatility_pct": volatility * 100,
        "sharpe_ratio": sharpe,
        "max_drawdown_pct": max_dd * 100,
        "win_rate_pct": win_rate * 100,
        "n_days": n_days,
    }


def compute_ic(
    scores: pd.DataFrame,
    forward_returns: pd.DataFrame,
) -> dict:
    """Compute daily IC and Rank IC between scores and forward returns.

    Returns dict with ic_mean, ic_std, icir, rank_ic_mean, rank_icir.
    """
    common_dates = sorted(set(scores.index) & set(forward_returns.index))
    common_assets = sorted(set(scores.columns) & set(forward_returns.columns))

    ic_series = []
    rank_ic_series = []
    for dt in common_dates:
        s = scores.loc[dt, common_assets]
        r = forward_returns.loc[dt, common_assets]
        valid = s.notna() & r.notna()
        if valid.sum() < 10:
            continue
        s_valid = s.loc[valid]
        r_valid = r.loc[valid]
        ic = float(s_valid.corr(r_valid))  # Pearson IC
        rank_ic = float(s_valid.corr(r_valid, method="spearman"))  # Rank IC
        if np.isfinite(ic):
            ic_series.append(ic)
        if np.isfinite(rank_ic):
            rank_ic_series.append(rank_ic)

    if not ic_series:
        return {"ic_mean": 0, "ic_std": 0, "icir": 0, "rank_ic_mean": 0, "rank_icir": 0, "n_days": 0}

    ic_arr = np.array(ic_series)
    rank_arr = np.array(rank_ic_series)
    ic_mean = float(ic_arr.mean())
    ic_std = float(ic_arr.std())
    icir = float(ic_mean / ic_std) if ic_std > 0 else 0.0
    rank_mean = float(rank_arr.mean()) if len(rank_arr) else 0.0
    rank_std = float(rank_arr.std()) if len(rank_arr) > 1 else 0.0
    rank_icir = float(rank_mean / rank_std) if rank_std > 0 else 0.0

    return {
        "ic_mean": ic_mean,
        "ic_std": ic_std,
        "icir": icir,
        "rank_ic_mean": rank_mean,
        "rank_icir": rank_icir,
        "n_days": len(ic_series),
    }
