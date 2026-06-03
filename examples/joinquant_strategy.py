"""聚宽版双均线策略 — 与 quantify 回测引擎对比验证.

使用方式：复制全部代码到聚宽策略编辑器（https://www.joinquant.com）。
回测区间：2022-01-01 → 2025-12-31，初始资金 100000。

注：
1. 聚宽基金代码格式为 '510300.XSHG'（自研引擎使用 '510300.SH'）。
2. 聚宽日线模式下 order_target_percent 在当日收盘价成交；自研引擎在**下一根 bar
   收盘价**成交（引擎事件循环是先执行昨日挂单再调用 handle_data 生成今日订单）。
   这一天的时序差对缓慢移动的均线策略影响有限，但仍会造成指标不完全一致。
3. 聚宽默认使用 250 个交易日年化；自研引擎图表已改为 250。
"""

from jqdata import *


def initialize(context):
    set_benchmark("510300.XSHG")

    set_order_cost(
        OrderCost(
            open_tax=0,
            close_tax=0,
            open_commission=0.0005,
            close_commission=0.0005,
            min_commission=0.5,
        ),
        type="fund",
    )

    set_slippage(FixedSlippage(0))

    context.short_window = 5
    context.long_window = 20

    run_daily(handle_data, time="close")
    run_daily(_record, time="after_close")


def handle_data(context):
    code = "510300.XSHG"

    closes = attribute_history(code, context.long_window + 1, "1d", ["close"])["close"]
    if len(closes) < context.long_window + 1:
        return

    short_ma = closes[-context.short_window:].mean()
    long_ma = closes[-context.long_window:].mean()

    if short_ma > long_ma:
        order_target_percent(code, 0.95)
    else:
        order_target_percent(code, 0)


def _record(context):
    """记录每日净值供比对."""
    g.daily_values.append(context.portfolio.total_value)


g.daily_values = []
