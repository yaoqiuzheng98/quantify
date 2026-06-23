# 本地回测引擎使用手册（聚宽兼容层）

本地引擎实现了聚宽策略 API 的子集。策略源码格式与聚宽完全一致（`from jqdata import *` + `initialize` + `handle_data`/`run_daily`），但**本地引擎行为与聚宽有细微差异**，必须遵守以下规范。

## 可用 API（本地兼容层注入的全部函数/对象）

| 函数/对象 | 说明 |
|---|---|
| `initialize(context)` | 策略初始化，设置基准/佣金/滑点/调仓 |
| `handle_data(context, data)` | 逐 bar 回调（一般不用，用 run_daily 代替） |
| `run_daily(func, time="open")` | 每个交易日开盘执行 func（**只支持 time="open"**） |
| `run_weekly(func, weekday=1, time="open")` | 每周第 N 个交易日执行 |
| `run_monthly(func, monthday=1, time="open")` | 每月第 N 个交易日执行 |
| `set_option(key, value)` | 如 `set_option("use_real_price", True)` |
| `set_benchmark(code)` | 设置基准（聚宽格式如 `000300.XSHG`） |
| `set_order_cost(OrderCost(...), type="stock")` | 设置佣金（type 只支持 "stock"/"fund"） |
| `set_slippage(PriceRelatedSlippage(0.002))` | 设置滑点 |
| `attribute_history(code, count, "1d", [fields])` | 获取前复权历史数据，返回 DataFrame（**只支持 1d**） |
| `get_index_stocks(index_code, date=None)` | 获取指数成分股（聚宽格式代码，点到点选股） |
| `get_industry(security)` | 获取股票所属行业 |
| `order(code, amount)` | 按股数下单 |
| `order_value(code, value)` | 按市值下单 |
| `order_target_value(code, target_value)` | 调仓至目标市值（**推荐用这个**） |
| `order_target_percent(code, pct)` | 调仓至目标比例（等价于 order_target_value(code, total_value*pct)） |
| `log.info(msg)` / `log.warning(msg)` | 日志输出 |
| `OrderCost(open_tax, close_tax, open_commission, close_commission, min_commission)` | 佣金/税配置 |
| `PriceRelatedSlippage(rate)` | 滑点配置 |

## context 对象

- `context.portfolio.total_value`: 当前总资产（float）
- `context.portfolio.positions`: 持仓字典，`{code: Position}`，用 `code in context.portfolio.positions` 判断是否持仓
- `context.portfolio.positions[code].total_amount`: 持仓股数
- `context.portfolio.positions[code].avg_cost`: 持仓均价
- `context.portfolio.positions[code].value`: 持仓市值
- `context.portfolio.positions[code].pnl`: 持仓盈亏
- `context.current_dt`: 当前日期（datetime）
- `context.start_date` / `context.end_date`: 回测区间

## ⚠️ 必须遵守的规则（违反会报错或结果错误）

1. **代码格式一律用聚宽格式** `.XSHG`（上交所）/`.XSHE`（深交所），如 `000300.XSHG`、`600000.XSHG`、`000001.XSHE`
2. **下单一律用 `order_target_value(code, value)`**，不要用 `order_target_percent`（虽然也支持，但统一用前者避免歧义）
3. **`from jqdata import *` 后必须绑回 builtins**：`from jqdata import *` 会注入 numpy 同名函数遮蔽内建 `sum`/`max`/`min`/`abs`/`round`，导致 `sum(dict.values())` 不求和。**必须在 `from jqdata import *` 后加**：
   ```python
   import builtins
   sum = builtins.sum
   max = builtins.max
   min = builtins.min
   abs = builtins.abs
   round = builtins.round
   ```
4. **`attribute_history(...)["close"]` 返回日期索引的 Series**，取值用 `.iloc[-1]`/`.iloc[0]`（按位置），**不能用 `[-1]`**（按标签会抛 `KeyError: -1`）
5. 因子计算在策略内用 `attribute_history` 取历史数据手动计算（不依赖 Qlib），用 pandas/numpy 做滚动窗口计算
6. 选股逻辑：按因子值排序，取 top-N，等权配置
7. 调仓时先卖后买（先 `order_target_value(code, 0)` 清仓不在目标中的，再买入目标）
8. **`run_daily` 只支持 `time="open"`**，不支持 `time="9:30"` 等具体时间
9. **`attribute_history` 只支持 `unit="1d"`**，不支持分钟数据
10. 可用字段：`["open", "high", "low", "close", "volume", "money", "factor"]`（factor 为复权因子）

## A 股摩擦（引擎自动处理，策略无需关心）

- 印花税：卖出单边 0.05%
- T+1：当日买入次日才能卖出
- 涨跌停：±10%（主板），开盘涨停拒买、跌停拒卖
- 买入按 100 股整数倍取整（引擎自动向下取整）
- 停牌：停牌日无行情数据，无法成交

## 完整示例（top-N 选股策略）

```python
from jqdata import *
import builtins
sum = builtins.sum
max = builtins.max
min = builtins.min
abs = builtins.abs
round = builtins.round

def initialize(context):
    set_option("use_real_price", True)
    set_option("avoid_future_data", True)
    set_benchmark("000300.XSHG")
    set_order_cost(OrderCost(
        open_tax=0, close_tax=0,
        open_commission=0.0005, close_commission=0.0005,
        min_commission=0.5,
    ), type="stock")
    set_slippage(PriceRelatedSlippage(0.002))
    context.top_n = 20
    context.rebalance_days = 5
    context.day_count = 0
    run_daily(rebalance, time="open")

def rebalance(context):
    context.day_count += 1
    if context.day_count % context.rebalance_days != 0:
        return

    stocks = get_index_stocks("000300.XSHG")
    scores = {}
    for code in stocks:
        try:
            closes = attribute_history(code, 20, "1d", ["close"])["close"]
            if len(closes) < 20:
                continue
            # 示例因子：20日反转
            score = -1 * ((closes.iloc[-1] / closes.iloc[0]) - 1)
            scores[code] = score
        except Exception:
            continue

    if not scores:
        return

    # 按因子值排序，取 top-N
    sorted_stocks = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    target_stocks = set(code for code, _ in sorted_stocks[:context.top_n])

    # 先卖后买
    for code in list(context.portfolio.positions.keys()):
        if code not in target_stocks:
            order_target_value(code, 0)

    # 等权买入
    weight = context.portfolio.total_value / context.top_n * 0.95
    for code in target_stocks:
        order_target_value(code, weight)
```
