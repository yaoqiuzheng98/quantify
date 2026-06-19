"""Interactive Streamlit dashboard for strategy backtests."""

from __future__ import annotations

import re
from datetime import date, datetime
from html import escape
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from sqlalchemy import select

from quantify.backtest import BacktestEngine, BacktestResult
from quantify.backtest.codes import classify_asset, normalize_codes
from quantify.backtest.universe import index_constituents_union
from quantify.database.engine import session_scope
from quantify.database.models import EtfDaily
from quantify.database.strategy_store import (
    StrategyRecord,
    delete_strategy,
    list_strategies,
    save_strategy,
)


@st.cache_data(show_spinner=False)
def _all_etf_codes() -> list[str]:
    """数据库中所有有日线行情的 ETF 代码（用于'全部加载'压测开关）。"""
    with session_scope() as sess:
        rows = sess.execute(select(EtfDaily.ts_code).distinct()).scalars().all()
    return sorted(rows)


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


# 匹配 6 位数字 + 交易所后缀的证券代码（ETF / 个股 / 指数）：
# 聚宽格式 510300.XSHG / 159915.XSHE，或 Tushare 格式 600519.SH / 000001.SZ / 830799.BJ。
_CODE_PATTERN = re.compile(r"\b(\d{6})\.(XSHG|XSHE|SH|SZ|BJ)\b", re.IGNORECASE)


def _extract_codes_from_source(strategy_source: str) -> list[str]:
    """从策略源码中提取所有标的代码（去重、保序、统一大写）。

    回测引擎只加载策略真正引用的标的，因此无需手动输入标的列表，
    避免"策略写了 7 个资产、输入框只填 1 个"导致的静默漏加载。
    """
    seen: dict[str, None] = {}
    for num, suffix in _CODE_PATTERN.findall(strategy_source):
        seen[f"{num}.{suffix.upper()}"] = None
    # 归一化为 Tushare 格式并去重：.XSHG/.XSHE 与 .SH/.SZ 视为同一标的，
    # 避免源码注释里同时出现两种写法导致重复加载。
    return normalize_codes(list(seen))


def _resolve_universe(strategy_source: str, start_date: date, end_date: date) -> list[str]:
    """解析策略股票池；若策略用了 ``get_index_stocks``，把指数展开为区间内成分股并集。

    本地引擎只加载传入的 ts_codes，而 ``get_index_stocks`` 在运行时才动态选股，
    因此需先把指数(如 000300.XSHG)在 [start, end] 内的成分**并集**预加载进来；
    策略内部仍按调仓日做点到点选股，选股口径不受影响。未用 ``get_index_stocks``
    的策略行为完全不变。
    """
    codes = _extract_codes_from_source(strategy_source)
    if "get_index_stocks" not in strategy_source:
        return codes
    resolved: list[str] = []
    for code in codes:
        if classify_asset(code) == "index":
            members = index_constituents_union(code, start_date.isoformat(), end_date.isoformat())
            resolved.extend(members or [code])
        else:
            resolved.append(code)
    return normalize_codes(resolved)


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


@st.dialog("确认删除策略")
def _confirm_delete_strategy(record: StrategyRecord) -> None:
    st.warning(f"确定要删除策略 **{record.name}**（ID: {record.id}）吗？此操作不可撤销。")
    confirm_col, cancel_col = st.columns(2)
    if confirm_col.button("确认删除", type="primary", width="stretch", key="confirm_delete_yes"):
        try:
            deleted = delete_strategy(record.id)
        except Exception as exc:  # noqa: BLE001
            st.error(f"删除失败：{exc}")
            return
        if deleted and st.session_state.get("strategy_id") == record.id:
            st.session_state.pop("backtest_result", None)
        st.session_state["strategy_saved_message"] = (
            f"已删除策略：{record.name}" if deleted else f"策略不存在：{record.name}"
        )
        st.session_state.pop("pending_delete_id", None)
        st.rerun()
    if cancel_col.button("取消", width="stretch", key="confirm_delete_no"):
        st.session_state.pop("pending_delete_id", None)
        st.rerun()


def _render_strategy_list(records: list[StrategyRecord]) -> None:
    # 按钮配色：新建=蓝色、删除=红色。Streamlit 会给带 key 的组件容器加上
    # ``st-key-<key>`` class，借此精确着色（覆盖默认主题色，故用 !important）。
    st.markdown(
        """
        <style>
        .st-key-btn_new_strategy button {
            background-color: #1f6feb !important;
            border-color: #1f6feb !important;
            color: #ffffff !important;
        }
        .st-key-btn_new_strategy button:hover,
        .st-key-btn_new_strategy button:focus,
        .st-key-btn_new_strategy button:active {
            background-color: #1a5fd0 !important;
            border-color: #1a5fd0 !important;
            color: #ffffff !important;
        }
        [class*="st-key-delete_strategy_"] button {
            background-color: #d62728 !important;
            border-color: #d62728 !important;
            color: #ffffff !important;
        }
        [class*="st-key-delete_strategy_"] button:hover,
        [class*="st-key-delete_strategy_"] button:focus,
        [class*="st-key-delete_strategy_"] button:active {
            background-color: #b71f20 !important;
            border-color: #b71f20 !important;
            color: #ffffff !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    title_col, action_col = st.columns([4, 1])
    with title_col:
        st.subheader("策略列表")
        st.caption("选择一个已保存策略进入编辑和回测，或新建一个基础框架策略。")
    with action_col:
        if st.button("新建策略", type="primary", width="stretch", key="btn_new_strategy"):
            _new_strategy()
            st.rerun()

    if not records:
        st.info("暂无已保存策略。点击 `新建策略` 开始。")
        return

    header_cols = st.columns([1, 5, 2, 1, 1])
    header_cols[0].markdown("**ID**")
    header_cols[1].markdown("**策略名称**")
    header_cols[2].markdown("**更新时间**")
    header_cols[3].markdown("**操作**")
    header_cols[4].markdown("**删除**")
    for record in records:
        row_cols = st.columns([1, 5, 2, 1, 1])
        row_cols[0].write(str(record.id) if record.id is not None else "--")
        row_cols[1].markdown(f"**{record.name}**")
        if record.description:
            row_cols[1].caption(record.description)
        row_cols[2].write(record.updated_at.strftime("%Y-%m-%d %H:%M") if record.updated_at else "--")
        if row_cols[3].button("编辑", key=f"edit_strategy_{record.id}", width="stretch"):
            _load_strategy(record)
            st.rerun()
        if row_cols[4].button("删除", key=f"delete_strategy_{record.id}", width="stretch"):
            st.session_state["pending_delete_id"] = record.id
            st.rerun()

    pending_id = st.session_state.get("pending_delete_id")
    if pending_id is not None:
        target = next((r for r in records if r.id == pending_id), None)
        if target is None:
            st.session_state.pop("pending_delete_id", None)
        else:
            _confirm_delete_strategy(target)


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


# 持仓占比图最多单列显示的标的数；超出则按峰值占比取前 N、其余并入「其他」。
# 持有数百只股票的策略（如沪深300 截面选股）若每只画一条堆叠 bar，会产生上千条
# Plotly trace，叠加 hovermode="x unified" 会直接卡死浏览器，故在此封顶。
_MAX_POSITION_SERIES = 15


def _cap_position_series(frame: pd.DataFrame, code_cols: list[str]) -> pd.DataFrame:
    """标的过多时只保留峰值占比最高的前 N 只，其余汇总成一列「其他」。"""
    if len(code_cols) <= _MAX_POSITION_SERIES:
        return frame
    peak = frame[code_cols].max().sort_values(ascending=False)
    keep = list(peak.index[:_MAX_POSITION_SERIES])
    others = [col for col in code_cols if col not in keep]
    capped = frame[["date", *keep]].copy()
    if others:
        capped["其他"] = frame[others].sum(axis=1)
    return capped


def _position_ratio_frame(report: dict[str, Any]) -> pd.DataFrame:
    """每日各标的持仓占总资产比例(宽表:每列一个标的代码)。

    标的数超过 ``_MAX_POSITION_SERIES`` 时只单列展示峰值占比最高的前 N 只，其余
    并入「其他」——避免成百上千条堆叠 bar trace 拖垮前端（回测跑完后卡死的根因）。
    """
    curves = report.get("curves", [])
    if not curves:
        return pd.DataFrame(columns=["date"])
    rows: list[dict[str, Any]] = []
    for point in curves:
        row: dict[str, Any] = {"date": point["date"]}
        row.update(point.get("position_ratios_pct") or {})
        rows.append(row)
    frame = pd.DataFrame(rows)
    frame["date"] = pd.to_datetime(frame["date"])
    code_cols = [col for col in frame.columns if col != "date"]
    if code_cols:
        frame[code_cols] = frame[code_cols].fillna(0.0)
        frame = _cap_position_series(frame, code_cols)
    return frame


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
            "印花税": trade.get("tax", 0.0),
            "平仓盈亏": trade.get("realized_pnl"),
        }
        for trade in report.get("trades", [])
    ]
    return pd.DataFrame(records)


def _time_xaxis(show_rangeslider: bool = True) -> dict[str, Any]:
    """统一的时间轴配置:快捷区间按钮 + 底部缩放滑块。

    长周期(如 2019-2026)下日频柱子会细到看不见,通过区间选择器和滑块
    让用户拖拽缩放到任意时间窗口,悬浮查看单日数据。
    """
    axis: dict[str, Any] = {
        "rangeselector": {
            "buttons": [
                {"count": 1, "label": "1月", "step": "month", "stepmode": "backward"},
                {"count": 3, "label": "3月", "step": "month", "stepmode": "backward"},
                {"count": 6, "label": "6月", "step": "month", "stepmode": "backward"},
                {"count": 1, "label": "1年", "step": "year", "stepmode": "backward"},
                {"count": 1, "label": "今年", "step": "year", "stepmode": "todate"},
                {"step": "all", "label": "全部"},
            ],
            "x": 0,
            "y": 1.0,
            "xanchor": "left",
            "yanchor": "bottom",
            "font": {"size": 11},
            "bgcolor": "#f1f3f5",
            "activecolor": "#d0d7de",
        },
        "type": "date",
    }
    if show_rangeslider:
        axis["rangeslider"] = {"visible": True, "thickness": 0.08}
    return axis


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
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
        margin={"l": 10, "r": 10, "t": 70, "b": 10},
        yaxis_title="收益率",
        xaxis=_time_xaxis(),
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
        margin={"l": 10, "r": 10, "t": 70, "b": 10},
        yaxis_title=yaxis_title,
        xaxis=_time_xaxis(),
    )
    return fig


def _position_ratio_chart(frame: pd.DataFrame) -> go.Figure:
    code_cols = [col for col in frame.columns if col != "date"]
    fig = go.Figure()
    for code in code_cols:
        fig.add_trace(
            go.Bar(
                x=frame["date"],
                y=frame[code],
                name=code,
                hovertemplate="%{x|%Y-%m-%d}<br>" + escape(code) + " %{y:.2f}%<extra></extra>",
            )
        )
    fig.update_layout(
        title="每日持仓比例",
        barmode="stack",
        hovermode="x unified",
        template="plotly_white",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
        margin={"l": 10, "r": 10, "t": 70, "b": 10},
        yaxis_title="持仓比例",
        yaxis={"ticksuffix": "%"},
        xaxis=_time_xaxis(),
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
        margin={"l": 10, "r": 10, "t": 70, "b": 10},
        yaxis_title="回撤",
        xaxis=_time_xaxis(),
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
        st.warning(
            "没有读取到行情数据，请先确认日线表（ETF 用 `fund_daily`，个股用 `daily`）已入库且日期范围有效。"
        )
        return

    _render_metrics(report)

    returns = _returns_frame(report)
    st.plotly_chart(_line_chart(returns), width="stretch")

    st.plotly_chart(_bar_chart(_daily_pnl_frame(report), "daily_pnl", "每日盈亏", "金额"), width="stretch")
    st.plotly_chart(_bar_chart(_turnover_frame(report), "turnover", "每日成交", "成交额"), width="stretch")
    st.plotly_chart(_position_ratio_chart(_position_ratio_frame(report)), width="stretch")
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
        st.caption("编辑 JoinQuant 风格策略，运行 ETF/个股日线回测，并用交互图表查看收益、回撤、盈亏和成交。")
        _render_strategy_list(records)
        return

    header_cols = st.columns([6, 1, 1])
    with header_cols[0]:
        st.title("Quantify 回测工作台")
    with header_cols[1]:
        return_clicked = st.button("返回列表", width="stretch")
    with header_cols[2]:
        save_clicked = st.button("保存策略", type="primary", width="stretch")
    st.caption("编辑 JoinQuant 风格策略，运行 ETF/个股日线回测，并用交互图表查看收益、回撤、盈亏和成交。")
    if return_clicked:
        st.session_state["strategy_view"] = "list"
        st.rerun()

    with st.sidebar:
        st.header("回测参数")
        st.caption("标的代码自动从策略源码解析，无需手动填写。")
        benchmark_code = st.text_input("基准代码", value="510300.SH", help="仅用于收益对比，会自动加载行情")
        start_date = st.date_input("开始日期", value=date(2019, 1, 1))
        end_date = st.date_input("结束日期", value=date(2019, 6, 30))
        initial_cash = st.number_input("初始资金", min_value=1_000.0, value=100_000.0, step=10_000.0)

        st.header("交易成本")
        commission_rate = st.number_input("佣金费率", min_value=0.0, value=0.0005, step=0.0001, format="%.6f")
        commission_min = st.number_input("最低佣金", min_value=0.0, value=0.5, step=0.5)
        slippage_rate = st.number_input("滑点比例", min_value=0.0, value=0.002, step=0.0005, format="%.6f")

        st.header("调试")
        load_all = st.checkbox(
            "加载全部 ETF（压测用）",
            value=False,
            help="忽略源码解析，加载库中全部 ETF 行情。仅用于体验加载耗时，正常回测请关闭。",
        )

        run_clicked = st.button("运行回测", type="primary", width="stretch")

    st.text_input("策略名称", key="strategy_name")
    strategy_source = _strategy_editor(st.session_state["strategy_source"])
    st.session_state["strategy_source"] = strategy_source

    preview_codes = _extract_codes_from_source(strategy_source)
    if "get_index_stocks" in strategy_source:
        idx = [c for c in preview_codes if classify_asset(c) == "index"]
        idx_str = " ".join(f"`{c}`" for c in idx) if idx else "指数"
        st.caption(
            f"策略用 get_index_stocks 动态选股：运行时会把 {idx_str} 展开为回测区间内的成分股并集后加载。"
        )
    elif preview_codes:
        st.caption(f"将从源码加载 {len(preview_codes)} 个标的：" + " ".join(f"`{c}`" for c in preview_codes))
    else:
        st.caption("未在源码中解析到标的代码（形如 510300.XSHG / 159915.SZ）。")

    st.info("策略源码会在当前 Python 进程中执行，请只运行可信代码。")
    if save_clicked:
        _save_current_strategy(strategy_source)

    if run_clicked:
        if load_all:
            # 压测模式：加载库中全部 ETF，忽略源码解析。
            ts_codes = _all_etf_codes()
            if not ts_codes:
                st.error("数据库中没有任何 ETF 日线行情，无法压测加载。")
                return
        else:
            ts_codes = _resolve_universe(strategy_source, start_date, end_date)
            if not ts_codes:
                st.error("未能从策略源码中解析到任何标的代码（形如 510300.XSHG / 159915.SZ）。")
                return
        if start_date >= end_date:
            st.error("开始日期必须早于结束日期。")
            return

        st.session_state["loaded_codes"] = ts_codes
        spinner_msg = f"正在运行回测（加载 {len(ts_codes)} 个标的）..."
        with st.spinner(spinner_msg):
            try:
                elapsed_start = datetime.now()
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
                st.session_state["last_elapsed"] = (datetime.now() - elapsed_start).total_seconds()
            except Exception as exc:  # noqa: BLE001
                st.exception(exc)
                return

    loaded_codes = st.session_state.get("loaded_codes")
    if loaded_codes:
        elapsed = st.session_state.get("last_elapsed")
        elapsed_txt = f"，耗时 {elapsed:.2f}s" if elapsed is not None else ""
        if len(loaded_codes) <= 50:
            codes_txt = " ".join(f"`{c}`" for c in loaded_codes)
            st.markdown(f"**加载的标的（{len(loaded_codes)} 个{elapsed_txt}）**：{codes_txt}")
        else:
            st.markdown(f"**加载的标的（{len(loaded_codes)} 个{elapsed_txt}）** —— 数量过多已折叠")
            with st.expander("展开查看全部标的代码"):
                st.write(", ".join(loaded_codes))

    result = st.session_state.get("backtest_result")
    if result is None:
        st.write("调整参数后点击侧边栏的 `运行回测`。")
        return
    _render_result(result)


if __name__ == "__main__":
    main()
