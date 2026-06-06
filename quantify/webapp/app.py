"""Interactive Streamlit dashboard for strategy backtests."""

from __future__ import annotations

from datetime import date, datetime
from html import escape
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from quantify.backtest import BacktestEngine, BacktestResult
from quantify.database.strategy_store import StrategyRecord, list_strategies, save_strategy

try:
    from streamlit_ace import st_ace
except ModuleNotFoundError:  # pragma: no cover - optional UI enhancement
    st_ace = None


NEW_STRATEGY_TEMPLATE = """from jqdata import *


def initialize(context):
    set_benchmark("510300.XSHG")
    run_daily(rebalance, time="open")


def rebalance(context):
    code = "510300.XSHG"
    # 在这里编写你的交易逻辑
    pass
"""


def _parse_codes(raw_codes: str) -> list[str]:
    return [code.strip().upper() for code in raw_codes.split(",") if code.strip()]


def _init_strategy_state() -> None:
    if "strategy_source" not in st.session_state:
        st.session_state["strategy_id"] = None
        st.session_state["strategy_name"] = "未命名策略"
        st.session_state["strategy_description"] = ""
        st.session_state["strategy_source"] = NEW_STRATEGY_TEMPLATE
        st.session_state["strategy_editor_revision"] = 0
        st.session_state["strategy_view"] = "list"
    st.session_state.setdefault("strategy_view", "list")
    st.session_state.setdefault("strategy_editor_revision", 0)


def _new_strategy() -> None:
    st.session_state["strategy_id"] = None
    st.session_state["strategy_name"] = f"新建策略 {datetime.now():%Y%m%d %H%M}"
    st.session_state["strategy_description"] = ""
    st.session_state["strategy_source"] = NEW_STRATEGY_TEMPLATE
    st.session_state["strategy_editor_revision"] = st.session_state.get("strategy_editor_revision", 0) + 1
    st.session_state["strategy_view"] = "editor"
    st.session_state.pop("backtest_result", None)


def _load_strategy(record: StrategyRecord | None) -> None:
    st.session_state["strategy_id"] = record.id if record else None
    st.session_state["strategy_name"] = record.name if record else "未命名策略"
    st.session_state["strategy_description"] = (record.description or "") if record else ""
    st.session_state["strategy_source"] = record.source if record else NEW_STRATEGY_TEMPLATE
    st.session_state["strategy_editor_revision"] = st.session_state.get("strategy_editor_revision", 0) + 1
    st.session_state["strategy_view"] = "editor"
    st.session_state.pop("backtest_result", None)


def _load_strategy_records() -> list[StrategyRecord]:
    try:
        return list_strategies()
    except Exception as exc:  # noqa: BLE001
        st.warning(f"策略库暂不可用：{exc}")
        return []


def _render_strategy_list(records: list[StrategyRecord]) -> None:
    title_col, action_col = st.columns([4, 1])
    with title_col:
        st.subheader("策略列表")
        st.caption("选择一个已保存策略进入编辑和回测，或新建一个基础框架策略。")
    with action_col:
        if st.button("新建策略", type="primary", width="stretch"):
            _new_strategy()
            st.rerun()

    if not records:
        st.info("暂无已保存策略。点击 `新建策略` 开始。")
        return

    header_cols = st.columns([5, 2, 1])
    header_cols[0].markdown("**策略名称**")
    header_cols[1].markdown("**更新时间**")
    header_cols[2].markdown("**操作**")
    for record in records:
        row_cols = st.columns([5, 2, 1])
        row_cols[0].markdown(f"**{record.name}**")
        if record.description:
            row_cols[0].caption(record.description)
        row_cols[1].write(record.updated_at.strftime("%Y-%m-%d %H:%M") if record.updated_at else "--")
        if row_cols[2].button("编辑", key=f"edit_strategy_{record.id}", width="stretch"):
            _load_strategy(record)
            st.rerun()


def _render_toast(message: str) -> None:
    safe_message = escape(message)
    st.markdown(
        f"""
        <style>
        @keyframes q-toast-fade {{
            0% {{ opacity: 0; transform: translate(-50%, -10px); }}
            12% {{ opacity: 1; transform: translate(-50%, 0); }}
            82% {{ opacity: 1; transform: translate(-50%, 0); }}
            100% {{ opacity: 0; transform: translate(-50%, -10px); }}
        }}
        .q-toast {{
            position: fixed;
            top: 72px;
            left: 50%;
            z-index: 999999;
            padding: 10px 16px;
            border: 1px solid #b7dbff;
            border-radius: 10px;
            background: #e8f3ff;
            color: #14538a;
            box-shadow: 0 8px 24px rgba(20, 83, 138, 0.16);
            font-size: 14px;
            font-weight: 500;
            pointer-events: none;
            animation: q-toast-fade 3.2s ease-in-out forwards;
        }}
        </style>
        <div class="q-toast">{safe_message}</div>
        """,
        unsafe_allow_html=True,
    )


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
        override_strategy_costs=True,
    )
    return engine.run()


def _payload_frame(report: dict[str, Any]) -> pd.DataFrame:
    frame = pd.DataFrame(report.get("curves", []))
    if not frame.empty:
        frame["date"] = pd.to_datetime(frame["date"])
    return frame


def _returns_frame(report: dict[str, Any]) -> pd.DataFrame:
    frame = _payload_frame(report)
    if frame.empty:
        return pd.DataFrame(columns=["date", "strategy_return"])

    returns = pd.DataFrame(
        {
            "date": frame["date"],
            "equity": frame["equity"],
            "strategy_return": frame["strategy_return_pct"],
        }
    )
    if "benchmark_return_pct" in frame.columns:
        returns["benchmark_return"] = frame["benchmark_return_pct"]
    if "excess_return_pct" in frame.columns:
        returns["excess_return"] = frame["excess_return_pct"]
    return returns


def _drawdown_frame(report: dict[str, Any]) -> pd.DataFrame:
    frame = _payload_frame(report)
    if frame.empty:
        return pd.DataFrame(columns=["date", "drawdown"])
    return pd.DataFrame({"date": frame["date"], "drawdown": frame["drawdown_pct"]})


def _daily_pnl_frame(report: dict[str, Any]) -> pd.DataFrame:
    frame = _payload_frame(report)
    if frame.empty:
        return pd.DataFrame(columns=["date", "daily_pnl"])
    return frame.loc[:, ["date", "daily_pnl"]].copy()


def _turnover_frame(report: dict[str, Any]) -> pd.DataFrame:
    frame = _payload_frame(report)
    if frame.empty:
        return pd.DataFrame(columns=["date", "turnover"])
    return frame.loc[:, ["date", "turnover"]].copy()


def _trades_frame(report: dict[str, Any]) -> pd.DataFrame:
    records = [
        {
            "成交日": trade["date"],
            "代码": trade["ts_code"],
            "方向": "买入" if trade["direction"] == "buy" else "卖出",
            "数量": trade["amount"],
            "价格": trade["price"],
            "成交额": trade["value"],
            "佣金": trade["commission"],
            "滑点": trade["slippage"],
        }
        for trade in report.get("trades", [])
    ]
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
    if "benchmark_return" in frame.columns and bool(frame["benchmark_return"].notna().any()):
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
    if "excess_return" in frame.columns and bool(frame["excess_return"].notna().any()):
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


def _metric_value_class(value: Any) -> str:
    if isinstance(value, int | float | np.integer | np.floating) and np.isfinite(value):
        if value > 0:
            return "positive"
        if value < 0:
            return "negative"
    return "neutral"


def _render_metrics(report: dict[str, Any]) -> None:
    report_items = report.get("report_items", [])
    cards = []
    for item in report_items:
        label = escape(str(item.get("label", "")))
        value = escape(str(item.get("value", "--")))
        value_class = _metric_value_class(item.get("numeric_value"))
        wide_class = " wide" if item.get("label") == "最大回撤区间" else ""
        cards.append(
            f'<div class="q-metric-card{wide_class}"><div class="q-metric-label">{label}</div>'
            f'<div class="q-metric-value {value_class}" title="{value}">{value}</div></div>'
        )

    st.markdown(
        """
        <style>
        .q-metric-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(118px, 1fr));
            gap: 8px 10px;
            margin: 0.25rem 0 1.1rem 0;
        }
        .q-metric-card {
            min-height: 48px;
            min-width: 0;
            padding: 7px 8px;
            border: 1px solid #e6eaf0;
            border-radius: 8px;
            background: #ffffff;
        }
        .q-metric-card.wide {
            grid-column: span 2;
        }
        .q-metric-label {
            color: #6b7280;
            font-size: 11px;
            line-height: 1.2;
            margin-bottom: 4px;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        .q-metric-value {
            color: #111827;
            font-size: 14px;
            font-weight: 600;
            line-height: 1.25;
            overflow-wrap: anywhere;
            white-space: normal;
        }
        .q-metric-value.positive { color: #d62728; }
        .q-metric-value.negative { color: #2ca02c; }
        @media (max-width: 1200px) {
            .q-metric-grid { grid-template-columns: repeat(auto-fit, minmax(132px, 1fr)); }
        }
        @media (max-width: 760px) {
            .q-metric-grid { grid-template-columns: repeat(2, minmax(120px, 1fr)); }
        }
        </style>
        """
        + f'<div class="q-metric-grid">{"".join(cards)}</div>',
        unsafe_allow_html=True,
    )


def _render_result(result: BacktestResult) -> None:
    report = result.to_report_dict()
    if not report["curves"]:
        st.warning("没有读取到行情数据，请先确认 `etf_daily` 已入库且日期范围有效。")
        return

    _render_metrics(report)

    returns = _returns_frame(report)
    st.plotly_chart(_line_chart(returns), width="stretch")

    st.plotly_chart(_bar_chart(_daily_pnl_frame(report), "daily_pnl", "每日盈亏", "金额"), width="stretch")
    st.plotly_chart(_bar_chart(_turnover_frame(report), "turnover", "每日成交", "成交额"), width="stretch")
    st.plotly_chart(_drawdown_chart(_drawdown_frame(report)), width="stretch")

    trades_df = _trades_frame(report)
    st.subheader("交易明细")
    if trades_df.empty:
        st.info("本次回测没有成交记录。")
    else:
        st.dataframe(trades_df, width="stretch", hide_index=True)


def _strategy_editor(default_source: str) -> str:
    editor_key = f"strategy_editor_{st.session_state.get('strategy_editor_revision', 0)}"
    if st_ace is None:
        return st.text_area("策略代码", value=default_source, height=520, key=editor_key)
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
        key=editor_key,
    )
    return source or default_source


def _save_current_strategy(strategy_source: str) -> None:
    try:
        saved = save_strategy(
            strategy_id=st.session_state.get("strategy_id"),
            name=st.session_state.get("strategy_name", ""),
            description=st.session_state.get("strategy_description", ""),
            source=strategy_source,
        )
    except Exception as exc:  # noqa: BLE001
        st.error(f"保存失败：{exc}")
        return
    st.session_state["strategy_id"] = saved.id
    st.session_state["strategy_description"] = saved.description or ""
    st.session_state["strategy_source"] = saved.source
    st.session_state["strategy_saved_message"] = f"已保存策略：{saved.name}"
    st.rerun()


def main() -> None:
    st.set_page_config(page_title="Quantify 回测工作台", layout="wide")
    _init_strategy_state()
    if saved_message := st.session_state.pop("strategy_saved_message", None):
        _render_toast(saved_message)

    records = _load_strategy_records()
    if st.session_state.get("strategy_view") == "list":
        st.title("Quantify 回测工作台")
        st.caption("编辑 JoinQuant 风格策略，运行 ETF 日线回测，并用交互图表查看收益、回撤、盈亏和成交。")
        _render_strategy_list(records)
        return

    header_cols = st.columns([6, 1, 1])
    with header_cols[0]:
        st.title("Quantify 回测工作台")
    with header_cols[1]:
        return_clicked = st.button("返回列表", width="stretch")
    with header_cols[2]:
        save_clicked = st.button("保存策略", type="primary", width="stretch")
    st.caption("编辑 JoinQuant 风格策略，运行 ETF 日线回测，并用交互图表查看收益、回撤、盈亏和成交。")
    if return_clicked:
        st.session_state["strategy_view"] = "list"
        st.rerun()

    with st.sidebar:
        st.header("回测参数")
        raw_codes = st.text_input("标的代码", value="510300.SH", help="多个代码用英文逗号分隔")
        benchmark_code = st.text_input("基准代码", value="510300.SH", help="仅用于收益对比，会自动加载行情")
        start_date = st.date_input("开始日期", value=date(2019, 1, 1))
        end_date = st.date_input("结束日期", value=date(2019, 6, 30))
        initial_cash = st.number_input("初始资金", min_value=1_000.0, value=100_000.0, step=10_000.0)

        st.header("交易成本")
        commission_rate = st.number_input("佣金费率", min_value=0.0, value=0.0005, step=0.0001, format="%.6f")
        commission_min = st.number_input("最低佣金", min_value=0.0, value=0.5, step=0.5)
        slippage_rate = st.number_input("滑点比例", min_value=0.0, value=0.002, step=0.0005, format="%.6f")
        run_clicked = st.button("运行回测", type="primary", width="stretch")

    st.text_input("策略名称", key="strategy_name")
    strategy_source = _strategy_editor(st.session_state["strategy_source"])
    st.session_state["strategy_source"] = strategy_source
    st.info("策略源码会在当前 Python 进程中执行，请只运行可信代码。")
    if save_clicked:
        _save_current_strategy(strategy_source)

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
