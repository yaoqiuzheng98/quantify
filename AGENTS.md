# AGENTS.md — Quantify

## 环境与安装

- **Python 3.11.9**（通过 pyenv 锁定在 `.python-version`）
- 虚拟环境：项目根目录的 `.venv/`
- 安装命令：`pip install -e ".[dev]"`
- `.env` 文件**必须**放在项目根目录（已 gitignore），从 `.env.example` 复制。

## 常用命令

```bash
ruff check . && ruff format --check .   # 代码检查 + 格式化（行宽=110，目标 py311）
pytest                                    # 运行全部测试（暂无测试）
quantify --help                           # CLI 入口
quantify db init                          # 创建数据库及所有表
quantify db drop --yes                    # 删除所有表（危险操作）
quantify fetch etf basic                  # 【必须首先执行】填充 ETF 基础信息
quantify fetch etf all                    # 拉取全部 ETF 数据（默认增量模式）
quantify fetch etf all --full             # 全量回填（忽略已有日期）
quantify fetch etf daily --ts-code 510300.SH,159915.SZ  # 单个阶段，指定代码
quantify fetch etf all --skip portfolio,manager        # 跳过耗时阶段
```

## 架构

这是一个**单包 Python 项目**（`quantify/`）。入口点：`quantify cli:app`（Typer）。

**当前已实现（M1 — 数据层）：**
- Tushare Pro 客户端，带限流和重试（`quantify/tushare_client/`）
- 基于 SQLAlchemy 2.0 ORM 的 MySQL 表结构（`quantify/database/models.py`）
- ETF 数据采集器：8 个 Tushare 接口 → 8 张 MySQL 表（`quantify/fetcher/etf.py`）

**尚未实现**（README 描述的是规划架构）：
- `factor/`、`backtest/`、`agent/`、`analysis/` 子包 —— 这些目录**还不存在**
- 股票/个股日线、财务报告、指数数据 —— 仅接入了 ETF

## 数据库

- MySQL 8.0+，连接信息在 `.env` 中配置（前缀 `MYSQL_`）
- `quantify db init` 先调用 `ensure_database_exists()`（CREATE DATABASE IF NOT EXISTS），再调用 `Base.metadata.create_all()`
- 所有写入使用 `INSERT ... ON DUPLICATE KEY UPDATE`（幂等，可重复执行）
- 标准写入路径为 `quantify/database/upsert.py` 中的 `upsert_dataframe()`
- `EtfManager` 使用自增 `BigInteger` 主键 + 独立唯一索引 `(ts_code, name, begin_date)` —— 与其他表直接使用联合主键不同

## 并发

- ETF 采集器使用 `ThreadPoolExecutor(max_workers=5)` 按代码并发调用 API
- 限流器：滑动窗口、线程安全（`tushare_client/rate_limiter.py` 中的 `RateLimiter`）
- Tushare 客户端失败重试 5 次，指数退避（tenacity）

## 配置

- 全部配置通过 Pydantic Settings 加载，环境变量前缀：`TUSHARE_`、`MYSQL_`、`LOG_`
- `get_settings()` 是 `lru_cache` 缓存的单例
- `TUSHARE_HTTP_URL` 默认指向**镜像站**（`http://jiaoch.site`），而非 Tushare 官方 API —— 用户需在 `.env` 中设置
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

## 数据顺序依赖

`quantify fetch etf basic` **必须先执行**。它会填充 `fund_basic` 表，其他所有 ETF 采集器都从该表读取标的列表（排除已退市：`status != "D"`）。如果未先拉取基础数据，其他阶段会因为标的列表为空而不做任何操作。

## 测试

暂无测试。添加测试后运行 `pytest`。Ruff 配置在 `pyproject.toml` 中（行宽 110，目标 py311）。
