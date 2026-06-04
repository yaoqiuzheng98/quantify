"""Interactive Streamlit dashboard for strategy backtests."""

from __future__ import annotations

from datetime import date
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from quantify.backtest import BacktestEngine, BacktestResult
from quantify.backtest.reporting import build_report_items, benchmark_return_series, trade_turnover_series

try:
    from streamlit_ace import st_ace
except ModuleNotFoundError:  # pragma: no cover - optional UI enhancement
    st_ace = None


DEFAULT_STRATEGY = """def initialize(context):
    context.short_window = 5
    context.long_window = 20


def handle_data(context):
    code = "510300.SH"
    closes = context.data.history(code, count=context.long_window + 1, field="close")
    if len(closes) < context.long_window + 1:
        return

    short_ma = sum(closes[-context.short_window:]) / context.short_window
    long_ma = sum(closes[-context.long_window:]) / context.long_window

    if short_ma > long_ma:
        context.order_target_percent(code, 0.95)
    else:
        context.order_target_percent(code, 0)
"""


def _parse_codes(raw_codes: str) -> list[str]:
    return [code.strip().upper() for code in raw_codes.split(",") if code.strip()]


def _run_backtest(
    strategy_source: str,
    ts_codes: list[str],
    start_date: date,
    end_date: date,
    initial_cash: float,
    benchmark_code: str | None,
    commission_rate: float,
    commission_min: float,
    slippage_rate: float,
) -> BacktestResult:
    load_codes = list(dict.fromkeys([*ts_codes, benchmark_code] if benchmark_code else ts_codes))
    engine = BacktestEngine(
        strategy_source=strategy_source,
        ts_codes=load_codes,
        start_date=start_date.isoformat(),
        end_date=end_date.isoformat(),
        initial_cash=initial_cash,
        benchmark_code=benchmark_code,
        commission_rate=commission_rate,
        commission_min=commission_min,
        slippage_rate=slippage_rate,
    )
    return engine.run()


def _returns_frame(result: BacktestResult) -> pd.DataFrame:
    equity_df = result.equity_df.copy()
    if equity_df.empty:
        return pd.DataFrame(columns=["date", "strategy_return"])

    dates = pd.to_datetime(equity_df["date"])
    values = equity_df["value"].astype(float)
    strategy_return = values / values.iloc[0] - 1
    frame = pd.DataFrame(
        {
            "date": dates,
            "strategy_return": strategy_return * 100,
            "equity": values,
        }
    )

    benchmark_return = benchmark_return_series(pd.DatetimeIndex(dates), result.benchmark_df)
    if benchmark_return is not None:
        benchmark_return = benchmark_return.reindex(dates).ffill()
        frame["benchmark_return"] = benchmark_return.to_numpy(dtype=float) * 100
        frame["excess_return"] = frame["strategy_return"] - frame["benchmark_return"]
    return frame


def _drawdown_frame(result: BacktestResult) -> pd.DataFrame:
    equity_df = result.equity_df.copy()
    if equity_df.empty:
        return pd.DataFrame(columns=["date", "drawdown"])

    values = equity_df["value"].astype(float)
    wealth = values / values.iloc[0]
    drawdown = wealth / wealth.cummax() - 1
    return pd.DataFrame({"date": pd.to_datetime(equity_df["date"]), "drawdown": drawdown * 100})


def _daily_pnl_frame(result: BacktestResult) -> pd.DataFrame:
    equity_df = result.equity_df.copy()
    if equity_df.empty:
        return pd.DataFrame(columns=["date", "daily_pnl"])

    values = equity_df["value"].astype(float)
    return pd.DataFrame(
        {
            "date": pd.to_datetime(equity_df["date"]),
            "daily_pnl": values.diff().fillna(0.0),
        }
    )


def _turnover_frame(result: BacktestResult) -> pd.DataFrame:
    if result.equity_df.empty:
        return pd.DataFrame(columns=["date", "turnover"])
    dates = pd.DatetimeIndex(pd.to_datetime(result.equity_df["date"]))
    turnover = trade_turnover_series(dates, result.trades)
    return pd.DataFrame({"date": turnover.index, "turnover": turnover.values})


def _trades_frame(trades: list[Any]) -> pd.DataFrame:
    records = []
    for trade in trades:
        amount = int(getattr(trade, "filled_amount", 0) or getattr(trade, "amount", 0) or 0)
        price = getattr(trade, "filled_price", None)
        records.append(
            {
                "成交日": getattr(trade, "filled_date", None),
                "代码": getattr(trade, "ts_code", ""),
                "方向": "买入" if amount > 0 else "卖出",
                "数量": amount,
                "价格": float(price) if price is not None else np.nan,
                "成交额": abs(amount) * float(price) if price is not None else np.nan,
                "佣金": float(getattr(trade, "commission", 0.0) or 0.0),
                "滑点": float(getattr(trade, "slippage", 0.0) or 0.0),
            }
        )
    return pd.DataFrame(records)


def _line_chart(frame: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=frame["date"],
            y=frame["strategy_return"],
            name="策略收益",
            mode="lines",
            line={"color": "#2f6fab", "width": 2},
            hovertemplate="%{x|%Y-%m-%d}<br>策略收益 %{y:.2f}%<extra></extra>",
        )
    )
    if "benchmark_return" in frame:
        fig.add_trace(
            go.Scatter(
                x=frame["date"],
                y=frame["benchmark_return"],
                name="基准收益",
                mode="lines",
                line={"color": "#c84035", "width": 1.6},
                hovertemplate="%{x|%Y-%m-%d}<br>基准收益 %{y:.2f}%<extra></extra>",
            )
        )
    if "excess_return" in frame:
        fig.add_trace(
            go.Scatter(
                x=frame["date"],
                y=frame["excess_return"],
                name="超额收益",
                mode="lines",
                line={"color": "#f28e2b", "width": 1.4},
                hovertemplate="%{x|%Y-%m-%d}<br>超额收益 %{y:.2f}%<extra></extra>",
            )
        )
    fig.add_hline(y=0, line_color="#444", line_width=1)
    fig.update_layout(
        title="收益曲线",
        hovermode="x unified",
        template="plotly_white",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "left", "x": 0},
        margin={"l": 10, "r": 10, "t": 55, "b": 10},
        yaxis_title="收益率",
    )
    return fig


def _bar_chart(frame: pd.DataFrame, value_col: str, title: str, yaxis_title: str) -> go.Figure:
    colors = np.where(frame[value_col].astype(float) >= 0, "#8aa851", "#7e6aa0")
    fig = go.Figure(
        go.Bar(
            x=frame["date"],
            y=frame[value_col],
            marker_color=colors,
            hovertemplate="%{x|%Y-%m-%d}<br>%{y:,.2f}<extra></extra>",
        )
    )
    fig.add_hline(y=0, line_color="#444", line_width=1)
    fig.update_layout(
        title=title,
        hovermode="x unified",
        template="plotly_white",
        margin={"l": 10, "r": 10, "t": 45, "b": 10},
        yaxis_title=yaxis_title,
    )
    return fig


def _drawdown_chart(frame: pd.DataFrame) -> go.Figure:
    fig = go.Figure(
        go.Scatter(
            x=frame["date"],
            y=frame["drawdown"],
            name="回撤",
            mode="lines",
            fill="tozeroy",
            line={"color": "#ff7f0e", "width": 1.2},
            hovertemplate="%{x|%Y-%m-%d}<br>回撤 %{y:.2f}%<extra></extra>",
        )
    )
    fig.update_layout(
        title="回撤曲线",
        hovermode="x unified",
        template="plotly_white",
        margin={"l": 10, "r": 10, "t": 45, "b": 10},
        yaxis_title="回撤",
    )
    return fig


def _render_metrics(result: BacktestResult) -> None:
    report_items = build_report_items(result.equity_df, result.benchmark_df, result.metrics, result.trades)
    for start in range(0, len(report_items), 5):
        columns = st.columns(5)
        for column, (label, value, _numeric_value) in zip(columns, report_items[start : start + 5]):
            column.metric(label, value)


def _render_result(result: BacktestResult) -> None:
    if result.equity_df.empty:
        st.warning("没有读取到行情数据，请先确认 `etf_daily` 已入库且日期范围有效。")
        return

    _render_metrics(result)

    returns = _returns_frame(result)
    st.plotly_chart(_line_chart(returns), use_container_width=True)

    st.plotly_chart(
        _bar_chart(_daily_pnl_frame(result), "daily_pnl", "每日盈亏", "金额"), use_container_width=True
    )
    st.plotly_chart(
        _bar_chart(_turnover_frame(result), "turnover", "每日成交", "成交额"), use_container_width=True
    )
    st.plotly_chart(_drawdown_chart(_drawdown_frame(result)), use_container_width=True)

    trades_df = _trades_frame(result.trades)
    st.subheader("交易明细")
    if trades_df.empty:
        st.info("本次回测没有成交记录。")
    else:
        st.dataframe(trades_df, use_container_width=True, hide_index=True)


def _strategy_editor(default_source: str) -> str:
    if st_ace is None:
        return st.text_area("策略代码", value=default_source, height=520)
    source = st_ace(
        value=default_source,
        language="python",
        theme="chrome",
        keybinding="vscode",
        min_lines=24,
        max_lines=36,
        tab_size=4,
        show_gutter=True,
        auto_update=True,
        key="strategy_editor",
    )
    return source or default_source


def main() -> None:
    st.set_page_config(page_title="Quantify 回测工作台", layout="wide")
    st.title("Quantify 回测工作台")
    st.caption("编辑 JoinQuant 风格策略，运行 ETF 日线回测，并用交互图表查看收益、回撤、盈亏和成交。")

    with st.sidebar:
        st.header("回测参数")
        raw_codes = st.text_input("标的代码", value="510300.SH", help="多个代码用英文逗号分隔")
        benchmark_code = st.text_input("基准代码", value="510300.SH", help="仅用于收益对比，会自动加载行情")
        start_date = st.date_input("开始日期", value=date(2022, 1, 1))
        end_date = st.date_input("结束日期", value=date.today())
        initial_cash = st.number_input("初始资金", min_value=1_000.0, value=100_000.0, step=10_000.0)

        st.header("交易成本")
        commission_rate = st.number_input("佣金费率", min_value=0.0, value=0.0005, step=0.0001, format="%.6f")
        commission_min = st.number_input("最低佣金", min_value=0.0, value=0.5, step=0.5)
        slippage_rate = st.number_input("滑点比例", min_value=0.0, value=0.002, step=0.0005, format="%.6f")
        run_clicked = st.button("运行回测", type="primary", use_container_width=True)

    strategy_source = _strategy_editor(DEFAULT_STRATEGY)
    st.info("策略源码会在当前 Python 进程中执行，请只运行可信代码。")

    if run_clicked:
        ts_codes = _parse_codes(raw_codes)
        if not ts_codes:
            st.error("请至少输入一个标的代码。")
            return
        if start_date >= end_date:
            st.error("开始日期必须早于结束日期。")
            return

        with st.spinner("正在运行回测..."):
            try:
                st.session_state["backtest_result"] = _run_backtest(
                    strategy_source=strategy_source,
                    ts_codes=ts_codes,
                    start_date=start_date,
                    end_date=end_date,
                    initial_cash=float(initial_cash),
                    benchmark_code=benchmark_code.strip().upper() or None,
                    commission_rate=float(commission_rate),
                    commission_min=float(commission_min),
                    slippage_rate=float(slippage_rate),
                )
            except Exception as exc:  # noqa: BLE001
                st.exception(exc)
                return

    result = st.session_state.get("backtest_result")
    if result is None:
        st.write("调整参数后点击侧边栏的 `运行回测`。")
        return
    _render_result(result)


if __name__ == "__main__":
    main()
