"""Built-in example strategy sources."""

from __future__ import annotations


DEFAULT_STRATEGY_SOURCE = """from jqdata import *
import builtins
sum = builtins.sum
max = builtins.max
min = builtins.min
abs = builtins.abs
round = builtins.round


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

    run_daily(rebalance, time="open")

def rebalance(context):
    code = "510300.XSHG"
    closes = attribute_history(code, context.long_window + 1, "1d", ["close"])["close"]
    if len(closes) < context.long_window + 1:
        return

    short_ma = closes[-context.short_window:].mean()
    long_ma = closes[-context.long_window:].mean()
    position_amount = (
        context.portfolio.positions[code].total_amount if code in context.portfolio.positions else 0
    )

    # 金叉买入，死叉卖出
    if short_ma > long_ma:
        if position_amount == 0:
            order_target_value(code, context.portfolio.total_value * 0.95)
    elif position_amount > 0:
        order_target_value(code, 0)
"""
