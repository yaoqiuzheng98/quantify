"""Built-in example strategy sources."""

from __future__ import annotations


DEFAULT_STRATEGY_SOURCE = """from jqdata import *


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
    closes = attribute_history(code, context.long_window + 1, "1d", ["close"])["close"]
    if len(closes) < context.long_window + 1:
        return

    short_ma = closes[-context.short_window:].mean()
    long_ma = closes[-context.long_window:].mean()

    # 金叉买入，死叉卖出
    if short_ma > long_ma:
        order_target_percent(code, 0.95)
    else:
        order_target_percent(code, 0)
"""
