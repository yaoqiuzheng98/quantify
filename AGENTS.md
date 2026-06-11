# AGENTS.md — Quantify

## 环境与安装

- **Python 3.11.9**（通过 pyenv 锁定在 `.python-version`）
- 虚拟环境：项目根目录的 `.venv/`
- 安装命令：`pip install -e ".[dev]"`；Web 依赖：`pip install -e ".[web]"`
- `.env` 文件**必须**放在项目根目录（已 gitignore），从 `.env.example` 复制

## 常用命令

```bash
ruff check . && ruff format --check .   # 代码检查 + 格式化（行宽=110，目标 py311）
quantify --help                           # CLI 入口
quantify db init                          # 创建数据库及所有表
quantify db drop --yes                    # 删除所有表（危险操作）
quantify fetch etf basic                  # 【必须首先执行】填充 ETF 基础信息
quantify fetch etf all                    # 拉取全部 ETF 数据（默认增量模式）
quantify fetch etf all --full             # 全量回填（忽略已有日期）
quantify fetch etf all --skip portfolio,manager        # 跳过耗时阶段
quantify fetch etf daily --ts-code 510300.SH,159915.SZ  # 单阶段，指定代码
quantify fetch industry all --provider sw               # 申万行业分类+成分+日线
quantify fetch industry all --provider all              # 申万+中信
quantify fetch index all                                # 指数（日线/权重/资金流）
quantify fetch macro all                                # 宏观/跨资产（国债/美债/全球指数）
quantify fetch all                                      # 一键拉全部数据组（顺序：日历→ETF→行业→指数→宏观）
quantify fetch all --skip industry,macro                # 跳过指定数据组
quantify dashboard                                      # 启动 Streamlit 回测工作台
quantify dashboard --port 8502                          # 指定端口
quantify version                                        # 打印版本号
```

## 架构

这是一个**单包 Python 项目**（`quantify/`）。入口点：`quantify cli:app`（Typer）。

**当前已实现：**
- 数据采集层：Tushare Pro 客户端（限流+重试）→ MySQL，覆盖 ETF、行业（SW/CITIC）、指数、宏观/跨资产
- 回测引擎：事件驱动逐 bar 模拟，兼容 JoinQuant 策略 API（`quantify/backtest/`）
- Streamlit Web 回测工作台（`quantify/webapp/app.py`）
- 策略持久化：`quantify/database/strategy_store.py` → `strategy` 表

**尚未实现**：
- `factor/`、`agent/`、`analysis/` 子包 —— 这些目录**还不存在**
- 个股日线、财务报告 —— 仅接入了 ETF/指数/行业/宏观

## 数据库

- MySQL 8.0+，连接信息在 `.env` 中配置（前缀 `MYSQL_`）
- `quantify db init` 先调用 `ensure_database_exists()`（CREATE DATABASE IF NOT EXISTS），再调用 `Base.metadata.create_all()`
- **表名与 Tushare 接口名一一对应**：例如 `fund_basic` 表 ← `fund_basic` 接口，`fund_daily` 表 ← `fund_daily` 接口。例外：`strategy` 表是本地表（策略存储），`etf_share_size` 表来自 `etf_share_size` 接口（仅 ETF 份额规模，有别于 `fund_share` 的基金份额）。全部定义见 `quantify/database/models.py` 顶部注释。
- 所有写入使用 `INSERT ... ON DUPLICATE KEY UPDATE`（幂等，可重复执行）
- 标准写入路径为 `quantify/database/upsert.py` 中的 `upsert_dataframe()`
- `EtfManager` 使用自增 `BigInteger` 主键 + 独立唯一索引 `(ts_code, name, begin_date)` —— 与其他表直接使用联合主键不同

## 回测引擎

- 入口：`quantify/backtest/engine.py` → `BacktestEngine`
- 策略 API 对齐 JoinQuant（`initialize(context)` / `handle_data(context)`），兼容层在 `quantify/backtest/joinquant.py`
- 示例策略源码在 `quantify/backtest/examples.py` 的 `DEFAULT_STRATEGY_SOURCE`
- 数据源：`fund_daily` 表（OHLCV），复权因子来自 `fund_adj`
- 成交价走真实开盘价，历史价格走前复权（对齐聚宽 `use_real_price=True`）
- 默认预加载回测开始日前 365 天历史供信号计算
- 策略源码保存到 MySQL `strategy` 表（`SavedStrategy` 模型），Dashboard 读取/写入该表

## 并发

- ETF 采集器使用 `ThreadPoolExecutor(max_workers=5)` 按代码并发调用 API
- 限流器：滑动窗口、线程安全（`tushare_client/rate_limiter.py` 中的 `RateLimiter`）
- Tushare 客户端失败重试 5 次，指数退避（tenacity）

## 配置

- 全部配置通过 Pydantic Settings 加载，环境变量前缀：`TUSHARE_`、`MYSQL_`、`LOG_`
- `get_settings()` 是 `@lru_cache(maxsize=1)` 缓存的单例
- `TUSHARE_HTTP_URL` 默认指向**镜像站**（`http://jiaoch.site`），而非 Tushare 官方 API
- `PROJECT_ROOT` = `quantify/config.py` 的父目录（即仓库根目录）

## 代码风格

- **每个** `.py` 文件都要写 `from __future__ import annotations`
- 日志：从 `quantify.utils.logger` 导入 `log`（Loguru），不要使用 stdlib 的 `logging`
- 单例模式使用 `@lru_cache(maxsize=1)`（engine、client、settings）
- Tushare 返回的日期列是 `YYYYMMDD` 字符串；`_normalize_dates()` 负责转换为 Python `date` 类型
- 访问私有成员时用 `# noqa: SLF001` 抑制 lint 报错

## CLI 注意事项

`--incremental/--full` 是 **Typer 布尔开关对**（不是两个独立的 flag），二者互斥：
- `quantify fetch etf all` → 增量模式（默认）
- `quantify fetch etf all --full` → 全量回填

并非所有阶段都受此开关影响：日线/净值/复权/份额/规模走增量；分红/持仓/基金经理/basic 表始终全量（接口本身无增量语义，靠 upsert 去重）。

## 数据顺序依赖

1. `quantify fetch etf basic` **必须先执行**。它会填充 `fund_basic` 表，其他 ETF 采集器从该表读取标的列表（排除已退市：`status != "D"`）。未先拉取则其他阶段标的列表为空。
2. `quantify fetch all` 自动按依赖顺序：交易日历 → ETF（basic 优先）→ 行业 → 指数 → 宏观
3. 行业/指数/宏观部分接口需 5000+ 积分；积分不足时用 `--skip` 跳过对应组

## 测试

暂无测试。`ruff check . && ruff format --check .` 是当前唯一的代码质量检查。Ruff 配置在 `pyproject.toml` 中。
