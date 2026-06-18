# Quantify

> 基于 Python 的个人量化策略研究平台 · 事件驱动回测引擎 · Tushare 全量数据接入

[![Python](https://img.shields.io/badge/Python-3.11-3776AB.svg)](https://www.python.org/)
[![MySQL](https://img.shields.io/badge/MySQL-8.0+-4479A1.svg)](https://www.mysql.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Quantify 是一个 **Python** 量化研究框架。它从 **Tushare Pro** 拉取 A 股/ETF/指数/期货/行业/宏观全量日频数据，持久化到 **MySQL**，并提供**事件驱动逐 bar 回测引擎**（兼容聚宽策略 API）及 **Streamlit 回测工作台**。

---

## 核心特性

- **全量数据接入**：A 股日线/周月线、复权因子、每日指标、三大财报、财务指标、分红送股、沪深港通、融资融券、技术指标、ETF、指数成分/权重、行业分类、宏观经济、期货——**50+ 张数据表**，覆盖量化研究全场景
- **幂等增量同步**：所有写入走 `INSERT ... ON DUPLICATE KEY UPDATE`，重复运行安全，断点续跑无需额外操作；时间序列阶段自动查库内最大日期仅拉增量
- **事件驱动回测**：逐 bar 模拟，聚宽 `initialize`/`handle_data` 策略 API 兼容，支持 ETF/个股/指数多资产，前复权历史价格、真实开盘价撮合、佣金/滑点/分红/送转、A 股印花税/T+1/涨跌停全建模
- **Streamlit 工作台**：代码编辑器 + 策略持久化 + 参数面板 + 交互式收益/回撤/持仓图表 + 20+ 指标卡片
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

---

## 数据表清单

全部 51 张表，表名与 Tushare 接口名一一对应。所有写入均为幂等 upsert。

### ETF（10 表）

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

### A 股个股（20 表）

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

### 行业（7 表）

| 表名 | 内容 |
|------|------|
| `trade_cal` | 交易所交易日历 |
| `index_classify` | 申万行业分类（SW2021） |
| `index_member_all` | 申万行业成分股 |
| `sw_daily` | 申万行业指数日线 |
| `ci_index_member` | 中信行业成分股 |
| `ci_daily` | 中信行业指数日线 |

### 指数（5 表）

| 表名 | 内容 |
|------|------|
| `index_basic` | 指数基本信息 |
| `index_daily` | 指数日线行情 |
| `index_dailybasic` | 指数每日指标（仅主要宽基） |
| `index_weight` | 指数成分权重（月度） |
| `moneyflow_ind_dc` | 东方财富行业/概念资金流 |

### 宏观/跨资产（4 表）

| 表名 | 内容 |
|------|------|
| `yc_cb` | 中债国债收益率曲线（即期/到期） |
| `index_global` | 国际主要指数日线（SPX/DJI/HSI 等 22 个） |
| `us_tycr` | 美国国债名义收益率曲线（1M–30Y） |
| `us_trycr` | 美国国债实际收益率曲线（5Y–30Y） |

### 期货（5 表）

| 表名 | 内容 |
|------|------|
| `fut_basic` | 合约列表（6 个交易所） |
| `fut_daily` | 期货日线 OHLCV + 持仓量 |
| `fut_holding` | 每日成交持仓排名 |
| `fut_wsr` | 仓单日报 |
| `fut_settle` | 结算参数 |

### 公募基金（1 表）

| 表名 | 内容 |
|------|------|
| `fund_company` | 公募基金公司信息 |

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

### 关键行为

- **前复权**：`attribute_history()` 返回的价格按复权因子前复权，避免分红除息跳空污染计算
- **真实成交价**：订单按当天真实开盘价 + 滑点撮合
- **整手取整**：100 份一手，不足一手忽略
- **多资产路由**：按代码自动识别 ETF / 个股 / 指数，分别走对应数据表，一次回测可混合 ETF 与个股（指数仅作基准）
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
| 代码检查 | Ruff（行宽 110） |

---

## 项目目录

```
quantify/
├── quantify/
│   ├── cli.py                    # Typer CLI（db + fetch + dashboard）
│   ├── config.py                 # Pydantic Settings（Tushare/MySQL/Log）
│   ├── database/
│   │   ├── models.py             # 51 个 SQLAlchemy ORM 模型
│   │   ├── engine.py             # MySQL 连接池 + session
│   │   ├── init_db.py            # 建库建表
│   │   ├── upsert.py             # upsert_dataframe() 幂等写入
│   │   └── strategy_store.py     # 策略 CRUD
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
│   │   └── examples.py           # 内置示例策略
│   ├── webapp/
│   │   └── app.py                # Streamlit 工作台
│   └── utils/
│       └── logger.py             # Loguru 配置
├── logs/                         # 日志输出
├── pyproject.toml
└── README.md
```

---

## 路线图

- [x] **数据层**：Tushare 客户端 + 6 个 Fetcher + 51 张表全量入库
- [x] **回测层**：事件驱动引擎 + 聚宽 API 兼容 + 佣金/滑点/分红/拆股
- [x] **Web 工作台**：Streamlit 策略编辑器 + 交互式图表 + 策略持久化
- [ ] **因子层**（`factor/`）：经典因子计算 + 行业/市值中性化
- [ ] **Agent 层**（`agent/`）：LLM 策略生成 + 报告解读
- [ ] **分析层**（`analysis/`）：行业稳健性诊断 + IC 分析

---

## 重要声明

- 本项目仅供个人学习与研究，不构成任何投资建议。
- Tushare 数据使用请遵守其服务条款与积分规则。
- 历史回测结果不代表未来表现，量化策略存在失效风险。

---

## License

MIT License
