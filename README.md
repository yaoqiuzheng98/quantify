# Quantify

> 基于 Python 的个人量化策略研究平台 · 事件驱动回测引擎 · Tushare 全量数据接入

[![Python](https://img.shields.io/badge/Python-3.11-3776AB.svg)](https://www.python.org/)
[![MySQL](https://img.shields.io/badge/MySQL-8.0+-4479A1.svg)](https://www.mysql.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Quantify 是一个 **Python** 量化研究框架。它从 **Tushare Pro** 拉取 A 股/ETF/指数/期货/行业/宏观全量日频数据，持久化到 **MySQL**，并提供**事件驱动逐 bar 回测引擎**（兼容聚宽策略 API）、**Streamlit 回测工作台**，以及**基于 LLM + Qlib + Alphalens 的自动化因子挖掘流水线**。

---

## 核心特性

- **全量数据接入**：A 股日线/周月线、复权因子、每日指标、三大财报、财务指标、分红送股、沪深港通、融资融券、技术指标、ETF、指数成分/权重、行业分类、宏观经济、期货——**50+ 张数据表**，覆盖量化研究全场景
- **幂等增量同步**：所有写入走 `INSERT ... ON DUPLICATE KEY UPDATE`，重复运行安全，断点续跑无需额外操作；时间序列阶段自动查库内最大日期仅拉增量
- **事件驱动回测**：逐 bar 模拟，聚宽 `initialize`/`handle_data` 策略 API 兼容，支持 ETF/个股/指数多资产，前复权历史价格、真实开盘价撮合、佣金/滑点/分红/送转、A 股印花税/T+1/涨跌停全建模，并提供聚宽同款 `get_index_stocks` 指数成分点到点选股
- **Streamlit 工作台**：代码编辑器 + 策略持久化 + 参数面板 + 交互式收益/回撤/持仓图表 + 20+ 指标卡片
- **LLM 因子挖掘**：DeepSeek 生成 Qlib 因子表达式 → 语法校验 → Alphalens IC/分层回测评估 → 评估反馈回灌 LLM 的**闭环迭代**，**无门槛全部入库** `factor_library`（`status` 区分 passed/evaluated），保留给后面正交组合使用
- **Tushare 客户端**：直连镜像站 HTTP 接口（不走 SDK）、滑动窗口限流、指数退避重试、并发硬上限 2 路的线程池安全调用

---

## 项目架构

```
┌──────────────────────────────────────────────────────────┐
│                   数据采集层 (Fetcher)                     │
│     Tushare Pro API · 6 个 Fetcher · 并发 + 限流 + 重试    │
│     ETF / Stock / Index / Industry / Futures / Macro       │
└─────────────────────┬────────────────────────────────────┘
                       ↓
┌──────────────────────────────────────────────────────────┐
│                   存储层 (MySQL 8)                         │
│     50+ 张表 · 元数据 + 时序 · INSERT ON DUPLICATE KEY    │
└─────────────────────┬────────────────────────────────────┘
                       ↓
┌──────────────────────────────────────────────────────────┐
│               回测层 (Backtest Engine)                     │
│   逐 bar 事件驱动 · 佣金/滑点/分红/拆股 · 聚宽 API 兼容      │
└─────────────────────┬────────────────────────────────────┘
                       ↓
┌──────────────────────────────────────────────────────────┐
│              Web 工作台 (Streamlit Dashboard)              │
│   策略编辑器 · 参数配置 · 交互式图表 · 指标卡片 · 策略持久化   │
└─────────────────────┬────────────────────────────────────┘
                       ↓
┌──────────────────────────────────────────────────────────┐
│            因子挖掘层 (LLM Factor Mining)                  │
│  DeepSeek 生成 · Qlib 求值 · Alphalens 评估 · 闭环反馈入库   │
└──────────────────────────────────────────────────────────┘
```

---

## 快速开始

### 环境要求

- **Python 3.11**（`pyproject.toml` 锁定）
- **MySQL 8.0+**（utf8mb4）
- **Tushare Pro 账号**（ETF 行情 ≥ 2000 积分；财务/行业 ≥ 5000 积分）
- 推荐：Linux / macOS，16GB+ 内存

### 1. 安装

```bash
git clone <repo-url> && cd quantify
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"          # 核心依赖
pip install -e ".[web]"          # Streamlit 工作台（可选）
pip install -e ".[mining]"       # LLM 因子挖掘：Qlib + Alphalens + OpenAI SDK（可选）
```

### 2. 配置 `.env`

```bash
cp .env.example .env
```

编辑 `.env`：

```ini
TUSHARE_TOKEN=你的_token
TUSHARE_RATE_PER_MIN=480
MYSQL_HOST=127.0.0.1
MYSQL_PORT=3306
MYSQL_USER=quantify
MYSQL_PASSWORD=your_password
MYSQL_DATABASE=quantify

# 因子挖掘（可选，仅 `quantify factor` 用到）
LLM_API_KEY=你的_deepseek_key
LLM_BASE_URL=https://api.deepseek.com
LLM_MODEL=deepseek-chat
QLIB_PROVIDER_URI=          # 留空则用 <项目根>/qlib_data/cn_data
QLIB_REGION=cn
```

### 3. 初始化数据库

```bash
quantify db init
```

创建所有 50+ 张表。重建：`quantify db drop --yes && quantify db init`

### 4. 拉取数据

```bash
# === 基础：交易日历 ===
quantify fetch industry trade-cal

# === ETF（必须先拉 basic） ===
quantify fetch etf basic
quantify fetch etf all

# === A 股个股（必须先拉 basic） ===
quantify fetch stock basic
quantify fetch stock all                # 默认增量，跳过周月线/融资明细/金股
quantify fetch stock all --full         # 全量回填

# === 行业分类 + 行情 ===
quantify fetch industry all --provider all

# === 指数 ===
quantify fetch index all

# === 宏观/跨资产 ===
quantify fetch macro all

# === 期货 ===
quantify fetch futures all

# === 公募基金公司 ===
quantify fetch fund all

# === 一键全部 ===
quantify fetch all                      # 按依赖顺序：日历→ETF→个股→行业→指数→宏观→期货→基金
quantify fetch all --skip futures,macro # 跳过指定组
```

### 5. 日常增量更新

```bash
quantify fetch all     # 默认增量模式
```

或按需只更新特定组：

```bash
quantify fetch stock daily --ts-code 600000.SH    # 单只股票最新日线
quantify fetch etf all                             # 仅更新 ETF
```

---

## CLI 命令参考

### 数据库管理

```bash
quantify db init         # 创建库 + 全部表
quantify db drop --yes   # 删除全部表
```

### 数据采集

| 命令 | 阶段 | 关键参数 |
|------|------|---------|
| `quantify fetch etf` | `basic\|daily\|nav\|adj\|dividend\|share\|share-size\|portfolio\|manager` | `--ts-code`, `--incremental/--full`, `--skip` |
| `quantify fetch stock` | `basic\|daily\|adj-factor\|daily-basic\|weekly\|monthly\|suspend\|namechange\|income\|balancesheet\|cashflow\|fina-indicator\|forecast\|express\|dividend\|moneyflow-hsgt\|margin\|margin-detail\|stk-factor\|broker-recommend` | 同上 |
| `quantify fetch industry` | `trade-cal\|sw-classify\|sw-member\|sw-daily\|ci-member\|ci-daily` | `--provider sw\|ci\|all`, `--sw-src` |
| `quantify fetch index` | `index-basic\|index-daily\|index-dailybasic\|index-weight\|moneyflow-ind-dc` | `--market`, `--all-index` |
| `quantify fetch macro` | `yc-cb\|index-global\|us-tycr\|us-trycr` | `--ts-code`, `--start-date/--end-date` |
| `quantify fetch futures` | `fut-basic\|fut-daily\|fut-holding\|fut-wsr\|fut-settle` | `--incremental/--full` |
| `quantify fetch fund` | `company` | — |
| `quantify fetch all` | 全部数据组依赖顺序执行 | `--skip trade_cal\|etf\|stock\|industry\|index\|macro\|futures\|fund` |

### 回测工作台

```bash
quantify dashboard              # 默认 8501 端口
quantify dashboard --port 8502  # 指定端口
```

### 因子挖掘

```bash
quantify factor dump-data                          # 把 MySQL 个股日线(前复权)导出为 Qlib .bin
quantify factor dump-data --ts-code 600000.SH,000001.SZ  # 仅导出指定标的(快速验证)
quantify factor mine --universe 000300.SH --rounds 3 --per-round 5   # 运行 LLM 闭环挖掘
quantify factor mine --min-ic 0.03 --min-icir 0.5  # 自定义 status=passed 标记门槛（不影响入库）
quantify factor eval "Mean($close,5)/Mean($close,20)" --universe 000300.SH  # 评估单个表达式
quantify factor list                               # 列出已入库因子
```

---

## 数据表清单

全部 51 张表，分为 **Tushare 同步表（49 张）** 和 **自建本地表（2 张）**。Tushare 同步表由 `quantify fetch` 从 Tushare Pro API 拉取，表名与接口名一一对应；自建表由项目内部逻辑写入，非 Tushare 接口。所有写入均为幂等 upsert。

### Tushare 同步表（49 张）

#### ETF（10 表）

| 表名 | 内容 |
|------|------|
| `fund_basic` | ETF 基础信息（market='E'） |
| `etf_basic` | ETF→跟踪指数映射（index_code/index_name） |
| `fund_daily` | ETF 日线 OHLCV |
| `fund_nav` | 单位净值/累计净值/复权净值 |
| `fund_adj` | 复权因子 |
| `fund_div` | 分红记录 |
| `fund_share` | 份额变动 |
| `etf_share_size` | 份额+规模+AUM+净值+收盘价 |
| `fund_portfolio` | 季报披露持仓 |
| `fund_manager` | 基金经理信息 |

#### A 股个股（20 表）

| 表名 | 内容 |
|------|------|
| `stock_basic` | A 股基础列表（含上市/退市状态、行业、地域） |
| `daily` | 日线 OHLCV |
| `adj_factor` | 复权因子 |
| `daily_basic` | 每日指标：PE/PB/PS/换手率/总市值/流通市值 |
| `weekly` / `monthly` | 周线/月线 OHLCV |
| `suspend_d` | 停复牌信息 |
| `namechange` | 历史名称变更 |
| `income` | 利润表（94 列） |
| `balancesheet` | 资产负债表（158 列） |
| `cashflow` | 现金流量表（97 列） |
| `fina_indicator` | 财务指标：ROE/ROA/毛利率/同比增长率等（167 列） |
| `forecast` | 业绩预告 |
| `express` | 业绩快报 |
| `dividend` | 分红送股 |
| `moneyflow_hsgt` | 沪深港通资金流向 |
| `margin` | 融资融券交易汇总（SSE/SZSE） |
| `margin_detail` | 融资融券交易明细（per stock per day） |
| `stk_factor` | 每日技术指标：MACD/KDJ/RSI/BOLL/CCI |
| `broker_recommend` | 券商月度金股 |

#### 行业（6 表）

| 表名 | 内容 |
|------|------|
| `trade_cal` | 交易所交易日历 |
| `index_classify` | 申万行业分类（SW2021） |
| `index_member_all` | 申万行业成分股 |
| `sw_daily` | 申万行业指数日线 |
| `ci_index_member` | 中信行业成分股 |
| `ci_daily` | 中信行业指数日线 |

#### 指数（5 表）

| 表名 | 内容 |
|------|------|
| `index_basic` | 指数基本信息 |
| `index_daily` | 指数日线行情 |
| `index_dailybasic` | 指数每日指标（仅主要宽基） |
| `index_weight` | 指数成分权重（月度） |
| `moneyflow_ind_dc` | 东方财富行业/概念资金流 |

#### 宏观/跨资产（4 表）

| 表名 | 内容 |
|------|------|
| `yc_cb` | 中债国债收益率曲线（即期/到期） |
| `index_global` | 国际主要指数日线（SPX/DJI/HSI 等 22 个） |
| `us_tycr` | 美国国债名义收益率曲线（1M–30Y） |
| `us_trycr` | 美国国债实际收益率曲线（5Y–30Y） |

#### 期货（5 表）

| 表名 | 内容 |
|------|------|
| `fut_basic` | 合约列表（6 个交易所） |
| `fut_daily` | 期货日线 OHLCV + 持仓量 |
| `fut_holding` | 每日成交持仓排名 |
| `fut_wsr` | 仓单日报 |
| `fut_settle` | 结算参数 |

#### 公募基金（1 表）

| 表名 | 内容 |
|------|------|
| `fund_company` | 公募基金公司信息 |

### 自建本地表（2 张）

非 Tushare 接口，由项目内部逻辑写入。

| 表名 | 内容 | 写入来源 |
|------|------|---------|
| `strategy` | 回测策略持久化（源码 + 名称 + 说明） | Dashboard「保存策略」/ `save_strategy()` |
| `factor_library` | LLM 挖掘通过门槛的因子（表达式 + IC/IR/分层指标 + 假说） | `quantify factor mine` 闭环入库 |

---

## 回测引擎

事件驱动逐 bar 模拟，策略 API 对齐聚宽 JoinQuant。

### 策略示例

```python
from jqdata import *

def initialize(context):
    set_benchmark("510300.XSHG")
    set_order_cost(OrderCost(open_commission=0.0005, close_commission=0.0005, min_commission=0.5), type="fund")
    set_slippage(PriceRelatedSlippage(0.002))
    context.short_window = 5
    context.long_window = 20
    run_daily(rebalance, time="open")

def rebalance(context):
    code = "510300.XSHG"
    closes = attribute_history(code, 21, "1d", ["close"])["close"]
    if len(closes) < 21:
        return
    short_ma = closes[-5:].mean()
    long_ma = closes[-20:].mean()
    if short_ma > long_ma:
        order_target_value(code, context.portfolio.total_value * 0.95)
    else:
        order_target_value(code, 0)
```

### 引擎调用

```python
from quantify.backtest import BacktestEngine

engine = BacktestEngine(
    strategy_source=open("my_strategy.py").read(),
    ts_codes=["510300.SH"],
    start_date="2022-01-01", end_date="2025-12-31",
    initial_cash=100000, benchmark_code="510300.SH",
    commission_rate=0.0005, commission_min=0.5, slippage_rate=0.002,
)
result = engine.run()
print(result.metrics.to_llm_prompt())
```

### Context API

| 方法 | 说明 |
|------|------|
| `context.order(ts_code, amount)` | 按股数下单 |
| `context.order_value(ts_code, value)` | 按金额下单 |
| `context.order_target_value(ts_code, target)` | 调仓至目标市值 |
| `context.order_target_percent(ts_code, pct)` | 调仓至目标仓位比例 |
| `context.data.current(ts_code)` | 当前 bar |
| `context.data.history(ts_code, count, field)` | 前复权历史序列 |
| `context.portfolio.cash / .total_value` | 现金/总资产 |
| `context.portfolio.positions[code]` | 持仓（amount, avg_cost, value, pnl） |

### 选股 API（指数成分，聚宽同款）

| 函数 | 说明 |
|------|------|
| `get_index_stocks(index, date=None)` | 按 `date`（缺省=当前回测日）返回指数**点到点**成分股（聚宽格式代码），数据来自 `index_weight` 表的最近一期月度快照。沪深300 = `000300.XSHG` |

> ⚠️ 本地引擎只加载传入引擎的 `ts_codes`，而 `get_index_stocks` 运行时才动态选股。跑指数成分策略需把**成分并集**预加载为 `ts_codes`：直接调引擎时用 `quantify.backtest.universe.index_constituents_union(index, start, end)` 取并集传入；Streamlit 工作台会自动识别源码里的 `get_index_stocks` 并把指数展开为成分并集后加载。完整示例已存入 `strategy` 表（沪深300 截面滚动回归选股）。

### 关键行为

- **前复权**：`attribute_history()` 返回的价格按复权因子前复权，避免分红除息跳空污染计算
- **真实成交价**：订单按当天真实开盘价 + 滑点撮合
- **整手取整**：100 份一手，不足一手忽略
- **多资产路由**：按代码自动识别 ETF / 个股 / 指数，分别走对应数据表，一次回测可混合 ETF 与个股（指数仅作基准）
- **指数成分股票池**：`get_index_stocks("000300.XSHG", date)` 读 `index_weight` 表取**点到点**成分，按调仓日动态选股，避免幸存者偏差（详见上文「选股 API」）
- **ETF 分红/拆股**：登记日锁定持仓、派息日现金入账；份额折算基于 accum_nav/unit_nav 比率自动检测并调整持仓
- **个股送转/分红**：送转比例取 `dividend.stk_div`（纯送转，比 adj_factor 跳变更准），现金分红取税后 `cash_div`
- **A 股摩擦（仅个股生效，ETF/指数不受影响）**：卖出印花税 0.05% 单边、T+1（当日买入次日才可卖）、涨跌停（开盘涨停拒买/跌停拒卖，±10% 主板口径）
- **佣金/滑点/印花税**：直接扣减现金并计入指标

### 指标输出

总收益率、年化收益率、最大回撤（含持续天数）、年化波动率、Sharpe/Sortino/Calmar/信息比率、胜率、盈亏比、Profit Factor、累计佣金、累计滑点、累计印花税、Alpha/Beta。

### 编写策略的坑（务必避开）

> **`from jqdata import *` 会用 numpy 同名函数遮蔽 Python 内建的 `sum`/`max`/`min`/`abs`/`round`。**
>
> 这是聚宽线上环境特有的坑：`np.sum(dict.values())` 不会求和，而是把 `dict_values` 包成 0 维 object 数组原样返回，于是 `s = sum(d.values()); x / s` 会抛
> `TypeError: unsupported operand type(s) for /: 'float' and 'dict_values'`。
> 本地引擎用原生 builtins，**本地能跑通、传到聚宽才报错**，极易漏掉。
>
> 凡是对 `dict.values()` / 推导式 / 生成器做 `sum/max/min` 聚合的策略，在 `from jqdata import *` 之后显式绑回内建实现：
>
> ```python
> from jqdata import *
> import builtins
> sum = builtins.sum
> max = builtins.max
> min = builtins.min
> abs = builtins.abs
> round = builtins.round
> ```

> **下单一律用 `order_target_value`，不要用 `order_target_percent`。**
>
> `order_*` 系列下单函数不来自 `jqdata`，而是由运行时注入到策略全局命名空间——本地兼容层和聚宽线上注入的集合不完全一致，`order_target_percent` 等按比例下单的变体可能在某一边抛 `NameError: name 'order_target_percent' is not defined`。
> 统一改用按目标市值下单，两边都稳，且结果等价：
>
> ```python
> # 不要：order_target_percent(code, weight)
> order_target_value(code, context.portfolio.total_value * weight)
> ```

> **代码一律写聚宽格式 `.XSHG` / `.XSHE`（含基准和指数）。**
>
> 聚宽只认 `.XSHG`（上交所）/`.XSHE`（深交所），沪深300 指数是 `000300.XSHG` 而非 `000300.SH`；写成 Tushare 的 `.SH/.SZ` 会让聚宽报 `InvalidParam: 标的'xxx'不存在`。本地引擎两种格式都兼容，所以以聚宽格式为准即可两边通跑。
> 另：`attribute_history(...)["close"]` 返回日期索引的 Series，取值用 `.iloc[-1]` / `.iloc[0]`（按位置），用 `closes[-1]` 会按标签查找而抛 `KeyError: -1`。

---

## Streamlit 回测工作台

```bash
quantify dashboard --port 8501
```

功能：策略代码编辑器（ACE）、策略库 CRUD（MySQL `strategy` 表）、回测参数面板（基准/日期/现金/佣金/滑点）、交互式收益曲线/日盈亏/回撤/持仓占比图、20+ 指标卡片、交易明细表。

> 标的自动从源码解析加载。若策略用了 `get_index_stocks`（指数成分选股），工作台会把源码里的指数代码展开为回测区间内的**成分并集**后加载（如沪深300 约 300–500 只）；此类多股策略首次运行预加载较多、稍慢，且初始资金需设得足够大（Top-N 等权下每只至少买得起一手）。

> **策略落库约定**：策略统一以 `strategy` 表为权威来源——通过 Dashboard 的「保存策略」或 `quantify.database.strategy_store.save_strategy(name=, source=, description=)` 入库，由该表读取并运行。生成策略时如在本地临时创建了 `.py` 文件（仅用于编写/跑通验证），**入库后应删除**该临时文件，避免仓库与库内版本不一致。

---

## LLM 因子挖掘

基于 **LLM（DeepSeek）+ Qlib + Alphalens** 的自动化因子挖掘闭环：LLM 生成 Qlib 因子表达式 → 语法校验 → Alphalens IC/分层回测评估 → 评估结果回灌 LLM 进行下一轮迭代，**所有评估完成的因子无门槛直接入库** `factor_library`（`status` 区分 passed/evaluated），保留给后面正交组合使用。

### 流水线

```
LLM 生成因子表达式 ──► 语法校验 ──► 统计质量门槛 ──► Qlib 求值 + Alphalens 评估
      ▲                  (validator)   (覆盖率/常数)        (IC/RankIC/IR/分层/换手)
      │                                                              │
      └──────────────── 评估反馈(通过+未通过) ◄───────────────────────┘
                              每轮回灌，迭代优化
                                     │
                        全部入库 factor_library（无门槛）
                        status=passed（满足门槛）/ evaluated（不满足）
```

### 前置步骤

1. **安装依赖**：`pip install -e ".[mining]"`（Qlib + Alphalens + OpenAI SDK）
2. **配置 `.env`**：`LLM_API_KEY`（DeepSeek 控制台获取）、`LLM_BASE_URL`、`LLM_MODEL`
3. **导出 Qlib 数据**：`quantify factor dump-data` —— 从 MySQL `daily`/`adj_factor`/`daily_basic` 读个股日线，**前复权**后写为 Qlib `.bin`（需先 `quantify fetch stock all` 入库个股数据）

### 因子表达式

LLM 产出标准 Qlib 表达式，可用字段（均为前复权价/常用指标）：

```
$open $high $low $close $volume $amount $vwap $factor
$turn(换手率) $pe $pb $ps $total_mv $circ_mv
```

常用算子：`Ref/Mean/Sum/Std/Var/Max/Min/Delta/Slope/Rsquare/Resi/WMA/EMA/Rank/Quantile/Corr/Cov/Abs/Log/Sign/Greater/Less/CSRank/CSZScore` 等。示例：

```
Mean($close, 5) / Mean($close, 20)                       # 均线比值(动量)
-1 * (($close - Ref($close, 20)) / Ref($close, 20))      # 20 日反转
Corr(Rank($close, 5), Rank($volume, 5), 10)              # 量价背离
($close - Mean($close, 20)) / Std($close, 20)            # 价格 zscore
```

### 评估指标与入库策略

- 用 Alphalens 计算各前瞻周期（默认 1/5/10 日）的 **IC（Pearson）/ Rank-IC（Spearman）/ IC_IR / t 值 / 多空分层收益差 / 顶层换手率**
- **无门槛入库**：所有评估完成的因子连同完整指标 JSON、假说、类别写入 `factor_library` 表（本地表，按因子名唯一 upsert）
- `status` 字段区分质量：满足 `|IC| ≥ 0.02`、`|IC_IR| ≥ 0.3`、`|RankIC| ≥ 0.02`、覆盖率 ≥ 0.6（可用 `--min-ic`/`--min-icir` 调整）的标记为 `passed`，其余为 `evaluated`——两者都入库，供下游 `factor compose` 按 `--min-icir` 筛选优质因子做正交组合

### 命令

```bash
quantify factor dump-data [--ts-code ...] [--start-date ...] [--end-date ...]
quantify factor mine --universe 000300.SH --rounds 3 --per-round 5 [--instruction "侧重低换手"]
quantify factor eval "<表达式>" --universe 000300.SH [--save]
quantify factor list [--status passed]
```

> `--universe` 可填 `all`（全部已导出标的）或指数代码（如 `000300.SH` 沪深300，运行时展开为区间内成分并集）。重依赖（qlib/alphalens/openai）全部**惰性导入**，不装 `[mining]` 也不影响其他功能。

---

## 技术栈

| 层级 | 选型 |
|------|------|
| 语言 | Python 3.11 |
| CLI | Typer |
| 配置 | Pydantic Settings |
| ORM | SQLAlchemy 2.0 |
| 数据库 | MySQL 8.0 (PyMySQL) |
| 数据源 | Tushare Pro |
| 重试 | tenacity |
| 日志 | loguru |
| 回测 | 事件驱动逐 bar（自研引擎） |
| Web | Streamlit + Plotly |
| 因子挖掘 | Qlib（数据层/表达式求值）+ Alphalens（IC/分层评估）+ DeepSeek（LLM 生成） |
| 代码检查 | Ruff（行宽 110） |

---

## 项目目录

```
quantify/
├── quantify/
│   ├── cli.py                    # Typer CLI（db + fetch + factor + dashboard）
│   ├── config.py                 # Pydantic Settings（Tushare/MySQL/Log/LLM/Qlib）
│   ├── database/
│   │   ├── models.py             # 51 个 SQLAlchemy ORM 模型
│   │   ├── engine.py             # MySQL 连接池 + session
│   │   ├── init_db.py            # 建库建表
│   │   ├── upsert.py             # upsert_dataframe() 幂等写入
│   │   ├── strategy_store.py     # 策略 CRUD
│   │   └── factor_store.py       # 因子库 CRUD（factor_library 表）
│   ├── fetcher/
│   │   ├── etf.py                # EtfFetcher（10 阶段）
│   │   ├── stock.py              # StockFetcher（20 阶段）
│   │   ├── industry.py           # IndustryFetcher（SW + CITIC）
│   │   ├── index.py              # IndexFetcher
│   │   ├── macro.py              # MacroFetcher
│   │   └── future.py             # FuturesFetcher
│   ├── tushare_client/
│   │   ├── client.py             # TushareClient（直连 HTTP + 限流 + 重试）
│   │   └── rate_limiter.py       # 滑动窗口限流器
│   ├── backtest/
│   │   ├── engine.py             # 核心引擎
│   │   ├── context.py            # Context / Portfolio / DataProxy
│   │   ├── broker.py             # 订单执行 / 佣金 / 滑点
│   │   ├── joinquant.py          # 聚宽兼容层（jqdata）
│   │   ├── metrics.py            # 绩效指标计算
│   │   ├── reporting.py          # 报表生成
│   │   ├── codes.py              # 代码格式转换
│   │   ├── examples.py           # 内置示例策略
│   │   └── universe.py           # 指数成分查询（get_index_stocks 后端，读 index_weight）
│   ├── factor/                   # LLM 因子挖掘
│   │   ├── qlib_data.py          # MySQL → Qlib .bin 导出 + init_qlib
│   │   ├── validator.py          # Qlib 表达式语法校验
│   │   ├── evaluator.py          # Qlib 求值 + 质量门槛 + Alphalens 评估
│   │   ├── llm.py                # DeepSeek 客户端 + 生成/迭代 prompt
│   │   └── pipeline.py           # 生成→校验→评估→反馈闭环 + 入库
│   ├── webapp/
│   │   └── app.py                # Streamlit 工作台
│   └── utils/
│       └── logger.py             # Loguru 配置
├── tests/                        # pytest 单测（factor 无重依赖部分）
├── logs/                         # 日志输出
├── pyproject.toml
└── README.md
```

---

## 路线图

- [x] **数据层**：Tushare 客户端 + 6 个 Fetcher + 51 张表全量入库
- [x] **回测层**：事件驱动引擎 + 聚宽 API 兼容 + 佣金/滑点/分红/拆股
- [x] **Web 工作台**：Streamlit 策略编辑器 + 交互式图表 + 策略持久化
- [x] **因子层**（`factor/`）：LLM 因子挖掘闭环（Qlib 求值 + Alphalens IC/分层评估 + 自动入库）
- [ ] **Agent 层**（`agent/`）：LLM 策略生成 + 报告解读
- [ ] **分析层**（`analysis/`）：行业稳健性诊断 + 因子组合/中性化

---

## 重要声明

- 本项目仅供个人学习与研究，不构成任何投资建议。
- Tushare 数据使用请遵守其服务条款与积分规则。
- 历史回测结果不代表未来表现，量化策略存在失效风险。

---

## License

MIT License
