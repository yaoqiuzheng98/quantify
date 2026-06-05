"""Built-in example strategy sources."""

from __future__ import annotations


DEFAULT_STRATEGY_SOURCE = """def initialize(context):
    context.set_benchmark("510300.SH")
    context.short_window = 5
    context.long_window = 20

def handle_data(context):
    code = "510300.SH"
    closes = context.data.history(code, count=context.long_window + 1, field="close")
    if len(closes) < context.long_window + 1:
        return

    short_ma = sum(closes[-context.short_window:]) / context.short_window
    long_ma = sum(closes[-context.long_window:]) / context.long_window

    # 金叉买入，死叉卖出
    if short_ma > long_ma:
        context.order_target_percent(code, 0.95)
    else:
        context.order_target_percent(code, 0)
"""
