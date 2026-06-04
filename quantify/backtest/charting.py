"""Chart generation for backtest results, powered by matplotlib."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import dates as mdates
from matplotlib import ticker as mticker


def _cn_font_setup() -> None:
    """Attempt to configure a CJK-capable font so Chinese labels render correctly."""
    import platform

    from matplotlib import font_manager

    font_names = [
        "Microsoft YaHei",
        "SimHei",
        "SimSun",
        "Noto Sans CJK SC",
        "WenQuanYi Micro Hei",
        "WenQuanYi Zen Hei",
        "PingFang SC",
        "Heiti SC",
        "Arial Unicode MS",
    ]
    font_paths = []
    if platform.system() == "Linux":
        font_paths.extend(
            [
                Path("/mnt/c/Windows/Fonts/msyh.ttc"),
                Path("/mnt/c/Windows/Fonts/simhei.ttf"),
                Path("/mnt/c/Windows/Fonts/simsun.ttc"),
            ]
        )

    for path in font_paths:
        if path.exists():
            font_manager.fontManager.addfont(str(path))
            font = font_manager.FontProperties(fname=str(path))
            plt.rcParams["font.sans-serif"] = [font.get_name(), "DejaVu Sans"]
            plt.rcParams["axes.unicode_minus"] = False
            return

    for name in font_names:
        try:
            font_manager.findfont(name, fallback_to_default=False)
        except ValueError:
            continue
        plt.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False
        return

    plt.rcParams["axes.unicode_minus"] = False


_cn_font_setup()

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _write_png(fig, save_path: str | None) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    if save_path:
        with open(save_path, "wb") as output:
            output.write(buf.getvalue())

    return buf.getvalue()


def _format_pct(value: float | None, digits: int = 2, *, signed: bool = False) -> str:
    if value is None or not np.isfinite(value):
        return "--"
    sign = "+" if signed and value > 0 else ""
    return f"{sign}{value:.{digits}f}%"


def _format_float(value: float | None, digits: int = 3) -> str:
    if value is None or not np.isfinite(value):
        return "--"
    return f"{value:.{digits}f}"


def _format_int(value: int | None) -> str:
    if value is None:
        return "--"
    return str(value)


def _format_money(value: float | None) -> str:
    if value is None or not np.isfinite(value):
        return "--"
    abs_value = abs(value)
    sign = "-" if value < 0 else ""
    if abs_value >= 10000:
        return f"{sign}{abs_value / 10000:.1f}万"
    return f"{sign}{abs_value:.0f}"


def _value_color(value: float | None) -> str:
    if value is None or not np.isfinite(value):
        return "#111111"
    if value > 0:
        return "#d62728"
    if value < 0:
        return "#2ca02c"
    return "#111111"


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


def _max_drawdown_info(curve: np.ndarray, dates: pd.Series | pd.DatetimeIndex) -> tuple[float, str]:
    if len(curve) == 0:
        return 0.0, ""

    peak = np.maximum.accumulate(curve)
    drawdown = (peak - curve) / peak
    trough_idx = int(np.argmax(drawdown))
    peak_idx = int(np.argmax(curve[: trough_idx + 1]))
    period = f"{pd.Timestamp(dates.iloc[peak_idx]).strftime('%Y/%m/%d')},{pd.Timestamp(dates.iloc[trough_idx]).strftime('%Y/%m/%d')}"
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
    profit_loss_ratio = None
    if profits and losses:
        profit_loss_ratio = float(np.mean(profits) / abs(np.mean(losses)))
    return profit_count, loss_count, win_rate, profit_loss_ratio


def _benchmark_return_series(
    dates: pd.DatetimeIndex,
    benchmark_df: pd.DataFrame | None,
) -> pd.Series | None:
    if benchmark_df is None or benchmark_df.empty:
        return None

    benchmark = pd.Series(
        benchmark_df["value"].astype(float).values,
        index=pd.to_datetime(benchmark_df["date"]),
    ).sort_index()
    benchmark = benchmark.reindex(dates).ffill()
    benchmark = benchmark.dropna()
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


def _report_metrics(
    equity_df: pd.DataFrame,
    benchmark_df: pd.DataFrame | None,
    metrics: Any | None,
    trades: list[Any] | None,
) -> list[tuple[str, str, float | None]]:
    del metrics

    values = equity_df["value"].astype(float).values
    strategy_daily = _daily_returns(values)
    trading_days = len(values)

    strategy_return = values[-1] / values[0] - 1
    annual_return = _annualized_return(strategy_return, trading_days)
    strategy_volatility = _annualized_volatility(strategy_daily)
    sharpe = _sharpe_ratio(annual_return, strategy_volatility)
    max_drawdown, max_drawdown_period = _max_drawdown_info(
        values / values[0], pd.to_datetime(equity_df["date"])
    )
    sortino = _sortino_ratio(strategy_daily, annual_return)
    profit_count, loss_count, trade_win_rate, profit_loss_ratio = _realized_trade_stats(trades)

    benchmark_total_return = None
    benchmark_annual = None
    alpha = None
    beta = None
    benchmark_daily = np.array([], dtype=float)
    benchmark_volatility = None
    excess_return = None
    excess_daily = np.array([], dtype=float)
    excess_mean = None
    excess_max_drawdown = None
    excess_sharpe = None
    information_ratio = None
    dates = pd.to_datetime(equity_df["date"])
    benchmark_curve = _benchmark_return_series(dates, benchmark_df)
    if benchmark_curve is not None:
        benchmark_total_return = float(benchmark_curve.iloc[-1])
        benchmark_annual = _annualized_return(benchmark_total_return, trading_days)
        benchmark_daily = (1 + benchmark_curve).pct_change().dropna().values
        benchmark_volatility = _annualized_volatility(benchmark_daily)
        alpha, beta = _compute_alpha_beta(strategy_daily, benchmark_daily, annual_return, benchmark_annual)
        benchmark_wealth = (1 + benchmark_curve).values
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


def build_report_items(
    equity_df: pd.DataFrame,
    benchmark_df: pd.DataFrame | None = None,
    metrics: Any | None = None,
    trades: list[Any] | None = None,
) -> list[tuple[str, str, float | None]]:
    """Return JoinQuant-style metrics used by static charts and web views."""
    return _report_metrics(equity_df, benchmark_df, metrics, trades)


def benchmark_return_series(
    dates: pd.DatetimeIndex,
    benchmark_df: pd.DataFrame | None,
) -> pd.Series | None:
    """Return benchmark cumulative return aligned to the supplied dates."""
    return _benchmark_return_series(dates, benchmark_df)


def trade_turnover_series(equity_dates: pd.DatetimeIndex, trades: list[Any] | None) -> pd.Series:
    """Return signed daily traded value for filled orders."""
    return _trade_turnover_series(equity_dates, trades)


def _plot_metric_panel(ax, report_items: list[tuple[str, str, float | None]]) -> None:
    ax.set_axis_off()
    ax.axhline(0.02, color="#e6e6e6", linewidth=1)

    cols = 11
    for index, (label, value, numeric_value) in enumerate(report_items):
        row = index // cols
        col = index % cols
        x_pos = col / cols + 0.012
        label_y = 0.82 if row == 0 else 0.38
        value_y = 0.61 if row == 0 else 0.17
        ax.text(x_pos, label_y, label, color="#777777", fontsize=9, transform=ax.transAxes)
        ax.text(
            x_pos,
            value_y,
            value,
            color=_value_color(numeric_value),
            fontsize=12,
            fontweight="bold",
            transform=ax.transAxes,
        )


def _style_time_axis(ax) -> None:
    ax.grid(True, color="#d0d0d0", linewidth=0.6, alpha=0.85)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=6, maxticks=12))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%y-%m-%d"))


def _trade_turnover_series(equity_dates: pd.DatetimeIndex, trades: list[Any] | None) -> pd.Series:
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


def plot_equity_curve(
    equity_df: pd.DataFrame,
    benchmark_df: pd.DataFrame | None = None,
    metrics: Any | None = None,
    trades: list[Any] | None = None,
    title: str = "回测报告",
    figsize: tuple[int, int] = (18, 10),
    save_path: str | None = None,
) -> bytes:
    """Plot a JoinQuant-style single report chart.

    Parameters
    ----------
    equity_df:
        Columns: ``date``, ``value``.  Value is the total portfolio value.
    benchmark_df:
        Optional.  Same format as *equity_df*.  If provided, the plot shows
        the relative performance vs. benchmark.
    title:
        Chart title.
    figsize:
        Matplotlib figure size in inches.
    save_path:
        If given, save the figure to disk at this path.

    Returns
    -------
    bytes
        PNG-encoded image.
    """
    fig = plt.figure(figsize=figsize, facecolor="white")
    grid = fig.add_gridspec(
        nrows=4,
        ncols=1,
        height_ratios=[1.25, 3.2, 1.45, 1.35],
        hspace=0.16,
    )
    ax_metrics = fig.add_subplot(grid[0])
    ax_equity = fig.add_subplot(grid[1])
    ax_pnl = fig.add_subplot(grid[2], sharex=ax_equity)
    ax_turnover = fig.add_subplot(grid[3], sharex=ax_equity)

    _plot_metric_panel(ax_metrics, build_report_items(equity_df, benchmark_df, metrics, trades))

    dates = pd.to_datetime(equity_df["date"])
    values = equity_df["value"].astype(float).values
    initial = values[0]
    strategy_return = values / initial - 1
    benchmark_return = _benchmark_return_series(dates, benchmark_df)

    ax_equity.plot(dates, strategy_return * 100, color="#2f6fab", linewidth=1.5, label="策略收益")
    ax_equity.fill_between(dates, 0, strategy_return * 100, color="#2f6fab", alpha=0.12)

    if benchmark_return is not None:
        benchmark_return = benchmark_return.reindex(dates).ffill()
        excess_return = strategy_return - benchmark_return.values
        ax_equity.plot(
            dates,
            benchmark_return.values * 100,
            color="#c84035",
            linewidth=1.2,
            label="基准收益",
        )
        ax_equity.plot(
            dates,
            excess_return * 100,
            color="#f28e2b",
            linewidth=1.0,
            alpha=0.75,
            label="超额收益",
        )

    max_idx = int(np.argmax(strategy_return))
    min_idx = int(np.argmin(strategy_return))
    ax_equity.scatter(
        [dates.iloc[max_idx], dates.iloc[min_idx]],
        [strategy_return[max_idx] * 100, strategy_return[min_idx] * 100],
        color="#16a000",
        s=42,
        zorder=5,
    )
    ax_equity.axhline(0, color="#333333", linewidth=0.8)
    ax_equity.set_title(title, fontsize=13, pad=8)
    ax_equity.set_ylabel("收益率")
    ax_equity.yaxis.set_major_formatter(mticker.PercentFormatter(decimals=1))
    ax_equity.legend(loc="upper left", ncol=3, frameon=False)
    _style_time_axis(ax_equity)

    daily_pnl = np.diff(values, prepend=values[0])
    pnl_colors = np.where(daily_pnl >= 0, "#8aa851", "#7e6aa0")
    ax_pnl.bar(dates, daily_pnl, color=pnl_colors, width=1.0, alpha=0.9)
    ax_pnl.axhline(0, color="#333333", linewidth=0.8)
    ax_pnl.set_ylabel("每日盈亏")
    ax_pnl.yaxis.set_major_formatter(mticker.FuncFormatter(lambda value, _pos: _format_money(value)))
    _style_time_axis(ax_pnl)

    turnover = _trade_turnover_series(dates, trades)
    turnover_colors = np.where(turnover.values >= 0, "#00a6d6", "#ff7f0e")
    ax_turnover.bar(dates, turnover.values, color=turnover_colors, width=1.0, alpha=0.9)
    ax_turnover.axhline(0, color="#333333", linewidth=0.8)
    ax_turnover.set_ylabel("每日成交")
    ax_turnover.yaxis.set_major_formatter(mticker.FuncFormatter(lambda value, _pos: _format_money(value)))
    _style_time_axis(ax_turnover)

    plt.setp(ax_equity.get_xticklabels(), visible=False)
    plt.setp(ax_pnl.get_xticklabels(), visible=False)
    fig.autofmt_xdate(rotation=0)
    fig.subplots_adjust(top=0.96, bottom=0.07, left=0.05, right=0.98)

    return _write_png(fig, save_path)


def plot_return_drawdown(
    equity_df: pd.DataFrame,
    benchmark_df: pd.DataFrame | None = None,
    title: str = "收益曲线",
    figsize: tuple[int, int] = (16, 8),
    save_path: str | None = None,
) -> bytes:
    """Plot the compact equity + drawdown chart."""
    fig, (ax_equity, ax_drawdown) = plt.subplots(
        nrows=2,
        figsize=figsize,
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.05},
    )

    dates = pd.to_datetime(equity_df["date"])
    values = equity_df["value"].values
    initial = values[0]

    ax_equity.plot(dates, values / initial, color="#1f77b4", linewidth=1.5, label="策略收益")
    ax_equity.set_ylabel("累计收益")

    if benchmark_df is not None and not benchmark_df.empty:
        bench_dates = pd.to_datetime(benchmark_df["date"])
        bench_vals = benchmark_df["value"].values
        bench_init = bench_vals[0]
        if bench_init > 0:
            ax_equity.plot(
                bench_dates,
                bench_vals / bench_init,
                color="#d62728",
                linewidth=1.2,
                linestyle="--",
                label="基准收益",
            )

    ax_equity.axhline(y=1, color="gray", linestyle=":", alpha=0.5)
    ax_equity.legend(loc="upper left")
    ax_equity.set_title(title)
    ax_equity.grid(True, alpha=0.3)

    peak = np.maximum.accumulate(values)
    drawdown = (values - peak) / peak * 100
    ax_drawdown.fill_between(dates, 0, drawdown, color="#ff7f0e", alpha=0.35)
    ax_drawdown.plot(dates, drawdown, color="#ff7f0e", linewidth=0.8)
    ax_drawdown.set_ylabel("回撤")
    min_drawdown = float(np.min(drawdown))
    ax_drawdown.set_ylim(min_drawdown * 1.2 if min_drawdown < 0 else -1, 0)
    ax_drawdown.grid(True, alpha=0.3)

    fig.subplots_adjust(top=0.92, bottom=0.08)

    return _write_png(fig, save_path)


def plot_returns_histogram(
    equity_df: pd.DataFrame,
    bins: int = 50,
    title: str = "每日收益分布",
    figsize: tuple[int, int] = (10, 4),
    save_path: str | None = None,
) -> bytes:
    """Histogram + density plot of daily returns."""
    values = equity_df["value"].values
    daily_ret = np.diff(values) / values[:-1] * 100

    fig, ax = plt.subplots(figsize=figsize)
    ax.hist(daily_ret, bins=bins, density=True, alpha=0.6, color="#2ca02c", edgecolor="white")
    ax.axvline(0, color="gray", linestyle="--", alpha=0.5)

    # normal overlay
    mu, sigma = np.mean(daily_ret), np.std(daily_ret)
    if sigma > 0:
        sample_points = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 200)
        ax.plot(
            sample_points,
            1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-((sample_points - mu) ** 2) / (2 * sigma**2)),
            color="#1f77b4",
            linewidth=1.5,
            label=f"正态分布(μ={mu:.2f}%, σ={sigma:.2f}%)",
        )

    ax.set_title(title)
    ax.set_xlabel("每日收益率")
    ax.set_ylabel("密度")
    ax.legend()
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    plt.close(fig)

    if save_path:
        with open(save_path, "wb") as output:
            output.write(buf.getvalue())

    return buf.getvalue()


def plot_rolling_sharpe(
    equity_df: pd.DataFrame,
    window: int = 60,
    title: str = "60日滚动夏普比率",
    figsize: tuple[int, int] = (16, 4),
    save_path: str | None = None,
) -> bytes:
    """Plot the rolling Sharpe ratio over a specified window."""
    values = equity_df["value"].values
    daily_ret = np.diff(values) / values[:-1]
    dates = pd.to_datetime(equity_df["date"].values[1:])

    rolling = (
        pd.Series(daily_ret, index=dates)
        .rolling(window=window)
        .apply(
            lambda window_values: np.mean(window_values) / (np.std(window_values) + 1e-12) * np.sqrt(252),
            raw=True,
        )
    )

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(rolling.index, rolling.values, color="#9467bd", linewidth=1.0)
    ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
    ax.set_ylabel("夏普")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    plt.close(fig)

    if save_path:
        with open(save_path, "wb") as output:
            output.write(buf.getvalue())

    return buf.getvalue()


def generate_report_charts(
    equity_df: pd.DataFrame,
    benchmark_df: pd.DataFrame | None = None,
    metrics: Any | None = None,
    trades: list[Any] | None = None,
    save_dir: str | None = None,
) -> dict[str, bytes]:
    """Generate the default single-chart backtest report.

    Returns a dict mapping chart names to PNG bytes.
    """
    charts: dict[str, bytes] = {}
    if save_dir:
        import os

        os.makedirs(save_dir, exist_ok=True)

    charts["equity_curve"] = plot_equity_curve(
        equity_df,
        benchmark_df=benchmark_df,
        metrics=metrics,
        trades=trades,
        save_path=f"{save_dir}/equity_curve.png" if save_dir else None,
    )
    return charts
