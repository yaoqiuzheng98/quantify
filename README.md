# Quantify 📈

> 基于 Python 的个人量化策略研究平台 · LLM Agent 辅助因子组合与回测验证

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB.svg)](https://www.python.org/)
[![MySQL](https://img.shields.io/badge/MySQL-8.0+-4479A1.svg)](https://www.mysql.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-WIP-orange.svg)]()

Quantify 是一个使用 **Python** 构建的中低频量化研究框架。它从 **Tushare Pro** 拉取股票、基金、指数的日频行情与财务数据，持久化到 **MySQL**，再结合 **LLM Agent** 自动生成因子组合假设，由确定性回测引擎执行验证，最终通过 **Walk-forward 时间切分 + 行业稳健性诊断** 锤炼出可落地的投资策略。

---

## ✨ 核心特性

- 🐍 **Python 全栈**：拥抱成熟的量化生态（pandas / qlib / vectorbt），开发与迭代极快
- 📊 **Tushare 单数据源**：日频行情 / 财务 / 指数 / 行业分类 / 基金净值，覆盖个人研究全场景
- 🗄️ **MySQL 单库架构**：元数据 + 时序面板统一存储，运维成本最低
- 🧠 **LLM 辅助研究**：Agent 提假设、引擎做计算、LLM 读报告，分工清晰
- ⏱️ **Walk-forward 验证**：滚动时间窗口训练 + 样本外验证，避免风格漂移过拟合
- 🏭 **行业稳健性诊断**：行业中性化 + 分行业拆解，验证跨行业鲁棒性
- 📈 **完整评估体系**：IC / ICIR / Sharpe / 最大回撤 / 换手率 一应俱全

---

## 🏗️ 项目架构

```
┌─────────────────────────────────────────────────────────────┐
│                     数据采集层 (Fetcher)                      │
│         Tushare Pro API · 异步并发 · 增量更新 · 限流控制       │
└──────────────────────────┬──────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│                     存储层 (MySQL 8)                          │
│   元数据：交易日历 / 股票池 / 行业分类 / 因子定义              │
│   时序数据：日频行情 / 财务报表 / 因子面板 (按月分区)          │
└──────────────────────────┬──────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│                     因子层 (Factor Engine)                    │
│   Alpha101/191 · Barra 风格因子 · 自定义因子                  │
│   行业中性化 · 市值中性化 · 因子正交化                         │
└──────────────────────────┬──────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│                  策略生成层 (LLM Agent)                       │
│   读取因子描述 + IC 报告 + 相关性矩阵                          │
│        ↓                                                     │
│   输出结构化策略 JSON {factors, weights, rebalance, ...}     │
└──────────────────────────┬──────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│                  回测层 (Backtest Engine)                    │
│   逐 bar 事件驱动 · 佣金/滑点模型 · 指标 + LLM 报告           │
└──────────────────────────┬──────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│                    分析层 (Analyzer)                          │
│   LLM 解读结构化报告 · 分行业稳健性诊断 · 策略池管理            │
└─────────────────────────────────────────────────────────────┘
```

---

## 🚀 快速开始

### 环境要求

- **Python 3.11+**
- **MySQL 8.0+**（推荐启用 InnoDB + utf8mb4）
- **Tushare Pro 账号**（积分 ≥ 5000 推荐）
- 操作系统：Linux / macOS / Windows
- 推荐内存：**16GB+**（全市场因子计算时）

### 0. Ubuntu 安装 pyenv 与 Python（可选）

如果你的 Ubuntu 系统没有合适的 Python 版本，推荐通过 **pyenv** 管理多版本 Python。

#### 0.1 安装系统依赖

```bash
sudo apt update && sudo apt install -y \
    make build-essential libssl-dev zlib1g-dev \
    libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm \
    libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev \
    libffi-dev liblzma-dev git
```

#### 0.2 安装 pyenv

```bash
curl https://pyenv.run | bash
```

#### 0.3 配置 Shell

将以下内容追加到 `~/.bashrc`（使用 zsh 则改为 `~/.zshrc`）：

```bash
export PYENV_ROOT="$HOME/.pyenv"
[[ -d $PYENV_ROOT/bin ]] && export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"
```

使配置立即生效：

```bash
source ~/.bashrc
```

#### 0.4 安装 Python 3.11

```bash
pyenv install 3.11.9
```

> 安装过程会从源码编译，通常需要 3–5 分钟。运行 `pyenv install --list | grep "3\.11"` 可查看可用的 3.11.x 版本，选最新的即可。

#### 0.5 设置项目 Python 版本

```bash
# 进入项目目录后，设置本目录专用的 Python 版本
pyenv local 3.11.9

# 验证
python --version   # 应输出 Python 3.11.9
```

> 也可以用 `pyenv global 3.11.9` 全局生效，但推荐按项目隔离。

---

### 1. 克隆与安装

```bash
# 克隆仓库
git clone https://github.com/yourname/quantify.git
cd quantify

# 创建虚拟环境
python -m venv .venv
# Windows (PowerShell)
.venv\Scripts\Activate.ps1
# Linux / macOS
source .venv/bin/activate

# 升级 pip 并安装项目（开发模式）
python -m pip install --upgrade pip
pip install -e ".[dev]"

# 如需使用 Streamlit 回测工作台，再安装 Web 依赖
pip install -e ".[web]"
```

安装完成后，命令行入口 `quantify` 会自动注册到当前虚拟环境，运行 `quantify --help` 即可看到全部子命令。

### 2. 准备 MySQL

本地或远程都可以，推荐版本 **MySQL 8.0+**，字符集 **utf8mb4**。账号需有创建数据库与读写权限。无需手动建库——`quantify db init` 会自动 `CREATE DATABASE IF NOT EXISTS`。

```sql
-- 仅作示例：用 root 创建一个独立账号
CREATE USER 'quantify'@'%' IDENTIFIED BY 'your_password';
GRANT ALL PRIVILEGES ON quantify.* TO 'quantify'@'%';
FLUSH PRIVILEGES;
```

### 3. 配置 `.env`

```bash
cp .env.example .env       # Windows: copy .env.example .env
```

编辑 `.env`，至少填好两类配置：

```ini
# Tushare Pro
TUSHARE_TOKEN=你的_tushare_pro_token
TUSHARE_RATE_PER_MIN=480       # 按账号积分调整：低权限账号请下调到 100~200

# MySQL
MYSQL_HOST=127.0.0.1
MYSQL_PORT=3306
MYSQL_USER=quantify
MYSQL_PASSWORD=your_password
MYSQL_DATABASE=quantify
```

> 没有 token 时执行任何 `fetch` 命令都会立即报错；建议先到 <https://tushare.pro> 注册并获取 token。ETF 行情通常 2000 积分以上即可调用。

### 4. 初始化数据库

```bash
quantify db init
```

该命令会：

1. 用 `.env` 里的连接信息登录 MySQL；
2. 若库不存在则创建（utf8mb4 + `_unicode_ci`）；
3. 在库内创建本项目当前涉及的全部表：`etf_basic` / `etf_daily` / `etf_nav` / `etf_adj_factor` / `etf_dividend` / `etf_share` / `etf_portfolio` / `etf_manager`。

如需重建（**会清空数据**）：

```bash
quantify db drop --yes
quantify db init
```

### 5. 拉取 ETF 全量数据

首次拉全量约 10–30 分钟，主要受 Tushare 的限流约束。建议按下面的顺序执行：

```bash
# 5.1 必须先拉基础信息（决定后续要遍历的 ts_code 列表）
quantify fetch etf basic

# 5.2 一键拉全部子集（增量模式：仅拉本地已有数据之后的部分）
quantify fetch etf all

# 想强制全量回填（忽略本地已存在的最大日期）
# --full 是 --incremental/--full 布尔开关对中的一个，二者互斥
quantify fetch etf all --full
```

常用的细粒度命令：

```bash
# 仅拉日频行情
quantify fetch etf daily

# 只针对特定 ETF（逗号分隔）
quantify fetch etf nav --ts-code 510300.SH,159915.SZ

# 跳过持仓 / 经理人这种偏慢的子集
quantify fetch etf all --skip portfolio,manager

# 单独跑某个阶段：basic / daily / nav / adj / dividend / share / portfolio / manager
quantify fetch etf adj
quantify fetch etf dividend
```

所有写入均为 `INSERT ... ON DUPLICATE KEY UPDATE`，**重复运行安全**，断点续跑无需任何额外操作。

### 6. 验证数据是否入库

```bash
mysql -u quantify -p quantify
```

```sql
SELECT COUNT(*) AS n_etf FROM etf_basic;
SELECT COUNT(*) AS n_daily, MAX(trade_date) AS last_date FROM etf_daily;
SELECT * FROM etf_basic ORDER BY list_date DESC LIMIT 5;
SELECT * FROM etf_daily WHERE ts_code = '510300.SH' ORDER BY trade_date DESC LIMIT 10;
```

或者在 Python 里直接读：

```python
import pandas as pd
from quantify.database.engine import get_engine

df = pd.read_sql(
    "SELECT trade_date, close, vol FROM etf_daily "
    "WHERE ts_code = '510300.SH' ORDER BY trade_date",
    get_engine(),
)
print(df.tail())
```

### 7. 日常增量更新

建议每个交易日收盘后跑一次：

```bash
quantify fetch etf basic
quantify fetch etf all      # 默认就是增量
```

可结合系统计划任务：

- **Windows**：任务计划程序（Task Scheduler）每日 17:30 触发 `powershell -Command "cd D:\learning\quantify; .venv\Scripts\Activate.ps1; quantify fetch etf all"`。
- **Linux/macOS**：`crontab -e` 添加 `30 17 * * 1-5 cd /path/to/quantify && . .venv/bin/activate && quantify fetch etf all >> logs/cron.log 2>&1`。

### 8. 后续路线（暂未实现）

以下命令在路线图中，**当前版本尚未实现**，仅作为后续里程碑示意：

```bash
# 计算并入库因子（M2）
quantify factor compute --names pe_ttm,roe,momentum_60d

# 单因子 IC 测试（M3）
quantify backtest ic --factor pe_ttm --start 2018-01-01 --end 2025-12-31

# LLM 驱动的因子组合搜索（M4）
quantify agent search --industry "游戏" --rounds 5
```

### 常见问题

- **`TUSHARE_TOKEN is empty`**：`.env` 没找到或字段未填。确认在项目根目录运行命令，且 `.env` 与 `pyproject.toml` 同级。
- **`Access denied for user ...`**：MySQL 账号/密码或权限问题，先用 `mysql -u ...` 直连验证。
- **`抱歉，您每分钟最多访问该接口 N 次`**：把 `.env` 中的 `TUSHARE_RATE_PER_MIN` 调小一些（如 60）。
- **`pymysql.err.OperationalError: (2003, ...)`**：MySQL 没启动或 host/port 不对。
- **Windows 编码乱码**：建议在 PowerShell 里执行 `chcp 65001` 切到 UTF-8 后再跑命令。

---

## 📦 数据粒度与表设计

所有数据日频为主，统一存储于 MySQL，关键时序表按月分区以保证查询性能。

| **数据类型** | **粒度** | **表名** | **说明** |
|------------|---------|---------|---------|
| 行情 OHLCV | 日频 | `daily_quote` | 按 `trade_date` 月分区 |
| 复权因子 | 日频 | `adj_factor` | 前复权计算用 |
| 财务报表 | 季频 | `financial_report` | 含 PIT 字段避免未来函数 |
| 基金净值 | 日频 | `fund_nav` | 单位净值 + 累计净值 |
| 因子面板 | 日频横截面 | `factor_value` | 按 `trade_date` 月分区，长格式 |
| 交易日历 | — | `trade_calendar` | 元数据 |
| 股票池 | — | `stock_basic` | 含上市/退市状态 |
| 行业分类 | — | `industry_classify` | 申万 + 中信 |
| 因子定义 | — | `factor_meta` | 因子描述 / 公式 / 频率 |

### 关键索引设计

```sql
-- 时序查询主索引
CREATE INDEX idx_daily_code_date ON daily_quote(ts_code, trade_date);
CREATE INDEX idx_factor_date_factor ON factor_value(trade_date, factor_name);

-- 因子横截面查询（最常用）
CREATE INDEX idx_factor_factor_date_code ON factor_value(factor_name, trade_date, ts_code);
```

> 💡 **为什么 MySQL 够用？** 日频粒度下，A 股 5000+ 标的 × 30 年 ≈ 5000 万行，加上分区与合理索引，单因子全市场查询毫秒级。配合 `pandas.read_sql` 的 chunked 读取与连接池，性能完全够用。

---

## 📈 回测引擎

回测引擎采用**事件驱动逐 bar 模拟**，策略 API 对齐聚宽（JoinQuant）风格，熟悉 `initialize` / `handle_data` 模式的用户可以零学习成本上手。

### 策略写法

编写一个包含 `initialize(context)` 和 `handle_data(context)` 的 Python 代码段即可：

```python
def initialize(context):
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
```

### Context API 一览

| 方法 | 说明 |
|------|------|
| `context.set_benchmark(ts_code)` | 设置基准标的 |
| `context.order(ts_code, amount)` | 按股数下单（正=买，负=卖） |
| `context.order_value(ts_code, value)` | 按金额下单 |
| `context.order_target_value(ts_code, target)` | 调仓至目标市值 |
| `context.order_target_percent(ts_code, pct)` | 调仓至目标仓位比例 |
| `context.data.current(ts_code)` | 获取当前 bar（返回 Bar 对象） |
| `context.data.history(ts_code, count, field)` | 获取历史 N 根 bar 的字段序列 |
| `context.portfolio.cash` | 当前现金 |
| `context.portfolio.total_value` | 当前总资产 |
| `context.portfolio.positions[code]` | 持仓对象（`.amount`, `.avg_cost`, `.market_value`, `.pnl`） |

### 撮合与数据对齐

- 时间轴使用所有 `ts_codes` 的交易日期并集推进；某标的当天没有 bar 时，`context.data.current()` 返回 `None`，不会复用上一交易日价格，也不会读到未来价格。
- 策略在每日开盘时运行：`context.data.current()` 只暴露当天开盘可知信息，`history()` 只返回上一交易日及以前的完整历史数据，不包含当天收盘价。
- `handle_data()` 中提交的订单会在当天按开盘价撮合；如果当天无可交易 bar 则不会生成订单。
- 订单数量按一手 `100` 份取整；不足一手的买卖请求会被忽略，现金不足时也只会按可负担的整手数量部分成交。
- 买入现金不足时会按可负担数量部分成交；完全不可成交或无持仓卖出会标记为拒单，不计入 `trades`。
- 佣金与滑点都会实际扣减现金，并计入结果指标中的 `total_commission` 与 `total_slippage`。

### 在代码中调用引擎

```python
from quantify.backtest import BacktestEngine

engine = BacktestEngine(
    strategy_source=open("my_strategy.py").read(),   # 或直接传字符串
    ts_codes=["510300.SH", "510050.SH"],
    start_date="2022-01-01",
    end_date="2025-12-31",
    initial_cash=100000,
    benchmark_code="510300.SH",
    commission_rate=0.0005,    # 万五
    commission_min=0.5,        # 最低 0.5 元
    slippage_rate=0.002,       # 滑点比例 0.2%
)

result = engine.run()
```

### 结果输出

引擎提供**两种格式**的输出：

```python
# 1) 人类可读 — 格式化文本指标
print(result.metrics.to_llm_prompt())

# 2) LLM 分析 — 结构化字典（归一化净值曲线 + 交易记录）
lm_dict = result.to_llm_dict()
# lm_dict["metrics"]      → {sharpe_ratio, max_drawdown_pct, win_rate_pct, ...}
# lm_dict["equity_curve"] → [{date, value}, ...]
# lm_dict["benchmark"]    → [{date, value}, ...]   # 已归一化对齐
# lm_dict["trades"]       → [{ts_code, amount, filled_price, ...}]
```

### 指标清单

| 类别 | 指标 |
|------|------|
| 收益 | 总收益率、年化收益率 |
| 风险 | 最大回撤 (含持续天数)、年化波动率 |
| 风险调整 | Sharpe 比率、Calmar 比率 |
| 交易 | 胜率、平均盈亏比、Profit Factor、交易次数 |
| 成本 | 累计佣金、累计滑点 |

### 佣金与滑点

佣金通过 `commission_rate` + `commission_min` 参数自由配置，`commission_rate=0, commission_min=0` 即为零佣金：

```python
engine = BacktestEngine(
    ...,
    commission_rate=0.0005,    # 费率（如万五 = 0.05%）
    commission_min=0.5,        # 最低佣金（0 表示无下限）
    slippage_rate=0.002,       # 滑点比例（可选，默认 0）
)
```

也支持传入完全自定义的函数：`make_commission(rate, minimum)` / `make_slippage(rate)` 或自定义 `callable`。

> 💡 目前仅回测 ETF 日线数据。数据源来自 `etf_daily` 表（OHLCV），读取逻辑在 `engine.py` 的 `_load_data()` 中，可方便扩展至股票 → 期货等资产。

### Streamlit 回测工作台

安装 Web 依赖后，可以启动交互式回测工作台：

```bash
pip install -e ".[web]"
quantify dashboard
```

工作台提供策略代码编辑器、回测参数面板、聚宽风格指标卡片，以及支持鼠标悬浮查看明细的收益曲线、每日盈亏、每日成交和回撤图。默认使用 `etf_daily` 表数据，因此需要先执行 `quantify fetch etf basic` 和 `quantify fetch etf all` 完成 ETF 日线入库。

默认端口为 `8501`；如果端口已被占用，CLI 会自动尝试后续端口，也可以手动指定：`quantify dashboard --port 8502`。

---

## 🤖 LLM Agent 工作流

Agent 与回测引擎遵循"**LLM 提假设 + 引擎做计算 + LLM 读报告**"的黄金分工：

```
1. LLM 输入：因子库元数据 + 历史 IC 报告 + 因子相关性矩阵
2. LLM 输出：结构化策略假设 (Pydantic Schema 强校验)
   {
     "factors": ["pe_ttm", "roe", "momentum_60d"],
     "weights": [-0.4, 0.4, 0.2],
     "rebalance": "monthly",
     "filter": {"market_cap_min": 5e9, "exclude_st": true},
     "neutralize": ["industry", "market_cap"]
   }
3. Python 回测引擎：向量化计算，输出标准化报告
4. LLM 分析：诊断问题（过拟合？暴露过高？换手太频？）
5. 迭代终止：连续 N 轮无提升 / 达到上限轮数 / token 预算耗尽
```

### 设计原则 🔑

- **LLM 永不直接计算数值**，只生成"配方"和读"报告"
- 策略输出由 **Pydantic Model** 强制校验，幻觉直接拒绝
- 报告同时输出**结构化版本（给 LLM）+ 自然语言版本（给人）**
- 每轮记录到 `agent_session` 表，研究过程可追溯、可复盘

### LLM 接入

通过 `LiteLLM` 统一接口对接，目前支持：

- OpenAI 兼容协议（GPT、DeepSeek、Qwen、Kimi 等）
- Anthropic Claude
- 本地 Ollama / vLLM

切换只需改 `config.yaml` 中的 `llm.model` 字段。

---

## 🎯 样本划分方法论

Quantify 采用 **时间切分为主、行业维度为辅** 的双轴验证体系。

### 主轴：Walk-forward 时间切分

```
训练 2015–2017 → 验证 2018
        训练 2016–2018 → 验证 2019
                训练 2017–2019 → 验证 2020
                        训练 2018–2020 → 验证 2021
                                ... 滚动到当前
```

只有在**多个滚动窗口验证期都稳定**的策略，才会进入策略池。

### 辅轴：行业稳健性诊断

行业**不用于切分样本**（避免横截面信息泄漏），而是用于：

- **行业中性化**：剔除行业 beta，提纯 alpha
- **行业内排序选股**：避免行业暴露集中
- **分行业拆解验证**：策略是否依赖少数几个行业？
- **行业特化策略**（可选）：对单一行业用时间切分训练专属权重

### 评估指标

| **类别** | **指标** |
|---------|---------|
| 选股有效性 | IC / Rank IC / ICIR / IC 衰减 |
| 组合收益 | 年化收益 / Sharpe / Sortino / Calmar |
| 风险控制 | 最大回撤 / 波动率 / 下行波动率 |
| 实操性 | 换手率 / 容量 / 滑点敏感性 |
| 稳健性 | 分组单调性 / 跨行业一致性 / 跨周期稳定性 |

---

## 📂 项目目录结构

```
quantify/
├── quantify/                     # 主包
│   ├── __init__.py
│   ├── cli.py                    # Typer CLI 入口
│   ├── config.py                 # Pydantic Settings
│   ├── database/                 # MySQL 连接 + Migration
│   │   ├── engine.py
│   │   ├── models.py             # SQLAlchemy ORM 模型
│   │   └── migrations/
│   ├── tushare/                  # Tushare 客户端 (限流/重试)
│   │   ├── client.py
│   │   └── rate_limiter.py
│   ├── fetcher/                  # 数据采集任务
│   │   ├── calendar.py
│   │   ├── daily.py
│   │   ├── financial.py
│   │   └── fund.py
│   ├── factor/                   # 因子计算引擎
│   │   ├── alpha101.py
│   │   ├── barra.py
│   │   ├── neutralize.py
│   │   └── registry.py
│   ├── backtest/                 # 回测引擎 (事件驱动)
│   │   ├── engine.py             # 核心引擎：加载 → 逐bar执行 → 输出
│   │   ├── context.py            # Context / Portfolio / DataProxy
│   │   ├── broker.py             # 订单执行 / 佣金 / 滑点
│   │   ├── metrics.py            # Sharpe / 最大回撤 / 年化 / 胜率
│   │   └── reporting.py          # 报表指标 / 基准收益 / 成交序列
│   ├── agent/                    # LLM Agent 编排
│   │   ├── proposer.py           # 策略生成
│   │   ├── analyzer.py           # 报告解读
│   │   ├── schemas.py            # Pydantic 输出约束
│   │   └── prompts/              # Prompt 模板 (Jinja2)
│   ├── analysis/                 # 评估与诊断
│   │   ├── ic.py
│   │   └── industry.py
│   └── utils/
├── config/
│   ├── config.example.yaml
│   └── factors/                  # 因子定义 YAML
├── tests/
├── notebooks/                    # 探索性研究 Notebook
├── docs/                         # 设计文档
├── pyproject.toml
├── Makefile
└── README.md
```

---

## 🛠️ 技术栈

| **层级** | **选型** | **说明** |
|---------|---------|---------|
| 语言 | **Python 3.11+** | 类型提示 + 异步 + 性能改进 |
| 包管理 | **pip** | 标准 Python 包管理器 |
| 数据源 | **Tushare Pro** | 唯一数据源，覆盖股/基/指数 |
| 数据库 | **MySQL 8** | 元数据 + 时序面板统一存储 |
| ORM | **SQLAlchemy 2.0** | 模型映射 + 类型安全 |
| 迁移 | **Alembic** | 数据库版本管理 |
| 数据处理 | **pandas / polars / numpy** | polars 用于大面板加速 |
| 可视化 | **Streamlit / Plotly** | 交互式回测工作台与悬浮图表 |
| CLI | **Typer** | 现代化命令行框架 |
| 配置 | **Pydantic Settings** | 类型安全的配置加载 |
| 日志 | **loguru** | 开箱即用的结构化日志 |
| 任务调度 | **APScheduler** | 定时数据更新 / Agent 迭代 |
| LLM 接入 | **LiteLLM** | 统一接口适配多家模型 |
| 测试 | **pytest** | 单元 + 集成测试 |

---

## 🗺️ 路线图

- [x] 项目骨架与配置系统
- [x] **M1 – 数据层**：Tushare 客户端 + ETF 全量数据入库 MySQL
- [ ] **M2 – 因子层**：实现 20 个经典因子 + 中性化工具
- [x] **M3 – 回测层**：事件驱动回测引擎 + 指标/LLM 报告
- [ ] **M4 – Agent 层**：LLM 策略生成器 + 报告解读器
- [ ] **M5 – 分析层**：行业稳健性诊断 + 策略池管理
- [ ] **M6 – 实盘联调**：QMT / Ptrade 模拟盘对接
- [ ] **M7 – Web Dashboard**：Streamlit 策略可视化与监控（回测工作台已实现）

---

## ⚠️ 重要声明

- 本项目**仅供个人学习与研究**，不构成任何投资建议。
- Tushare 数据使用请遵守其服务条款与积分规则。
- 历史回测结果不代表未来表现，量化策略存在失效风险，请独立判断。
- 实盘交易前请充分理解策略逻辑、风险敞口与极端市场情形下的表现。

---

## 📚 推荐阅读

- 《Advances in Financial Machine Learning》— López de Prado（Purged K-Fold 与多重检验）
- 《Active Portfolio Management》— Grinold & Kahn（IC、信息比率体系）
- 《101 Formulaic Alphas》— WorldQuant（经典 Alpha 因子）
- Tushare Pro 官方文档：<https://tushare.pro/document/2>
- Microsoft qlib 官方文档：<https://qlib.readthedocs.io/>

---

## 📄 License

MIT License © 2026 [Your Name]

---

## 🙋 鸣谢

感谢 Tushare、qlib、vectorbt、pandas 等开源项目与数据社区。这是站在巨人肩膀上的实验。
