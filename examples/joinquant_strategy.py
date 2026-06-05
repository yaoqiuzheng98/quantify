"""聚宽版双均线策略 — 与 quantify 回测引擎对比验证.

使用方式：复制全部代码到聚宽策略编辑器（https://www.joinquant.com）。
回测区间：2022-01-01 → 2025-12-31，初始资金 100000。

注：
1. 聚宽基金代码格式为 '510300.XSHG'；自研引擎会自动转换为 Tushare 格式。
2. 本策略在每日开盘运行：用上一交易日及以前的 close 计算信号，当日开盘成交。
3. 聚宽默认一手 100 份，order_target_percent 会按交易单位取整；自研引擎同样按
   100 份取整。
4. 聚宽默认使用 250 个交易日年化；自研引擎 Web 报表同样使用 250。
"""

from jqdata import *


def initialize(context):
    set_option("use_real_price", True)
    set_option("avoid_future_data", True)
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

    set_slippage(PriceRelatedSlippage(0.002))

    context.short_window = 5
    context.long_window = 20

    run_daily(handle_data, time="open")


def handle_data(context):
    code = "510300.XSHG"

    # 开盘时 attribute_history 返回上一交易日及以前的完整日线数据，
    # 与自研引擎 history() 不包含当天收盘价的规则一致。
    closes = attribute_history(code, context.long_window + 1, "1d", ["close"])["close"]
    if len(closes) < context.long_window + 1:
        return

    short_ma = closes[-context.short_window:].mean()
    long_ma = closes[-context.long_window:].mean()

    if short_ma > long_ma:
        order_target_percent(code, 0.95)
    else:
        order_target_percent(code, 0)
