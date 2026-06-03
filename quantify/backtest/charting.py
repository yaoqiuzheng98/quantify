"""Chart generation for backtest results, powered by matplotlib."""

from __future__ import annotations

import io

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")

import matplotlib.pyplot as plt


def _cn_font_setup() -> None:
    """Attempt to configure a CJK-capable font so Chinese labels render correctly."""
    import platform

    if platform.system() == "Linux":
        candidates = [
            "WenQuanYi Micro Hei",
            "WenQuanYi Zen Hei",
            "Noto Sans CJK SC",
            "SimHei",
        ]
        from matplotlib.font_manager import FontProperties

        for name in candidates:
            try:
                font = FontProperties(fname=name)
                plt.rcParams["font.sans-serif"] = [font.get_name(), "DejaVu Sans"]
                plt.rcParams["axes.unicode_minus"] = False
                return
            except Exception:  # noqa: PERF203
                continue
    if platform.system() == "Darwin":  # macOS
        plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "Heiti SC"]
    elif platform.system() == "Windows":
        plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]

    plt.rcParams["axes.unicode_minus"] = False


_cn_font_setup()

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def plot_equity_curve(
    equity_df: pd.DataFrame,
    benchmark_df: pd.DataFrame | None = None,
    title: str = "Equity Curve",
    figsize: tuple[int, int] = (16, 8),
    save_path: str | None = None,
) -> bytes:
    """Plot a JoinQuant-style single report chart with equity and drawdown.

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
    fig, (ax_equity, ax_drawdown) = plt.subplots(
        nrows=2,
        figsize=figsize,
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.05},
    )

    dates = pd.to_datetime(equity_df["date"])
    values = equity_df["value"].values
    initial = values[0]

    ax_equity.plot(dates, values / initial, color="#1f77b4", linewidth=1.5, label="Portfolio")
    ax_equity.set_ylabel("Cumulative Return")

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
                label="Benchmark",
            )

    ax_equity.axhline(y=1, color="gray", linestyle=":", alpha=0.5)
    ax_equity.legend(loc="upper left")
    ax_equity.set_title(title)
    ax_equity.grid(True, alpha=0.3)

    peak = np.maximum.accumulate(values)
    drawdown = (values - peak) / peak * 100
    ax_drawdown.fill_between(dates, 0, drawdown, color="#ff7f0e", alpha=0.35)
    ax_drawdown.plot(dates, drawdown, color="#ff7f0e", linewidth=0.8)
    ax_drawdown.set_ylabel("Drawdown %")
    min_drawdown = float(np.min(drawdown))
    ax_drawdown.set_ylim(min_drawdown * 1.2 if min_drawdown < 0 else -1, 0)
    ax_drawdown.grid(True, alpha=0.3)

    fig.subplots_adjust(top=0.92, bottom=0.08)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    plt.close(fig)

    if save_path:
        with open(save_path, "wb") as f:
            f.write(buf.getvalue())

    return buf.getvalue()


def plot_returns_histogram(
    equity_df: pd.DataFrame,
    bins: int = 50,
    title: str = "Daily Returns Distribution",
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
        x = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 200)
        ax.plot(
            x,
            1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-((x - mu) ** 2) / (2 * sigma**2)),
            color="#1f77b4",
            linewidth=1.5,
            label=f"Normal(μ={mu:.2f}%, σ={sigma:.2f}%)",
        )

    ax.set_title(title)
    ax.set_xlabel("Daily Return %")
    ax.set_ylabel("Density")
    ax.legend()
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    plt.close(fig)

    if save_path:
        with open(save_path, "wb") as f:
            f.write(buf.getvalue())

    return buf.getvalue()


def plot_rolling_sharpe(
    equity_df: pd.DataFrame,
    window: int = 60,
    title: str = "Rolling Sharpe Ratio (60-day)",
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
        .apply(lambda x: np.mean(x) / (np.std(x) + 1e-12) * np.sqrt(252), raw=True)
    )

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(rolling.index, rolling.values, color="#9467bd", linewidth=1.0)
    ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
    ax.set_ylabel("Sharpe")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    plt.close(fig)

    if save_path:
        with open(save_path, "wb") as f:
            f.write(buf.getvalue())

    return buf.getvalue()


def generate_report_charts(
    equity_df: pd.DataFrame,
    benchmark_df: pd.DataFrame | None = None,
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
        save_path=f"{save_dir}/equity_curve.png" if save_dir else None,
    )
    return charts
