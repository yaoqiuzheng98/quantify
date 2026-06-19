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
quantify fetch stock basic                # 填充 A 股基础列表（依赖所有后续阶段）
quantify fetch stock all                  # 拉取全部个股数据(日线/财务/资金流等)
quantify fetch stock all --full           # 全量回填
quantify fetch stock all --include-skipped             # 拉全部个股数据（含默认跳过的周线/月线/因子/融券明细/券商推荐）
quantify fetch futures all --include-skipped          # 拉全部期货数据（含默认跳过的持仓/仓单/结算）
quantify fetch all --include-skipped                  # 一键拉全部（含各数据组默认跳过的阶段）
quantify fetch skipped                                # 只拉各数据组默认跳过的阶段
quantify fetch stock daily --ts-code 600000.SH          # 单只股票日线
quantify fetch futures all                # 拉取期货数据（合约列表+日线）
quantify fetch fund all                   # 拉取公募基金公司信息
quantify fetch industry all --provider sw               # 申万行业分类+成分+日线
quantify fetch industry all --provider all              # 申万+中信
quantify fetch index all                                # 指数（日线/权重/资金流）
quantify fetch macro all                                # 宏观/跨资产（国债/美债/全球指数）
quantify fetch all                                      # 一键拉全部数据组（顺序：日历→ETF→个股→行业→指数→宏观→期货→基金）
quantify fetch all --skip stock,industry,macro          # 跳过指定数据组
quantify dashboard                                      # 启动 Streamlit 回测工作台
quantify dashboard --port 8502                          # 指定端口
quantify version                                        # 打印版本号
```

## 架构

这是一个**单包 Python 项目**（`quantify/`）。入口点：`quantify cli:app`（Typer）。

**当前已实现：**
- 数据采集层：Tushare Pro 客户端（限流+重试）→ MySQL，覆盖 ETF、个股（日线/财务/资金流）、行业（SW/CITIC）、指数、宏观/跨资产、期货、公募基金
- 回测引擎：事件驱动逐 bar 模拟，兼容 JoinQuant 策略 API（`quantify/backtest/`）
- Streamlit Web 回测工作台（`quantify/webapp/app.py`）
- 策略持久化：`quantify/database/strategy_store.py` → `strategy` 表

**尚未实现**：
- `factor/`、`agent/`、`analysis/` 子包 —— 这些目录**还不存在**

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
- **数据源抽象层**（`quantify/backtest/datasource.py`）：引擎不再硬编码 ETF 表，按代码自动路由：
  - `EtfDataSource` → `fund_daily`/`fund_adj`/`fund_nav`/`fund_div`（份额折算比例：用 `fund_div` 的 `ex_date`+`div_cash` 区分现金分红与份额折算——有分红时 `split_ratio = adj_ratio / (prev_close/(prev_close−div_cash))`，无分红时 `split_ratio = adj_ratio`（纯折算）；±0.5% 以内视为噪声）
  - `StockDataSource` → `daily`/`adj_factor`/`dividend`（送转比例直接取 `dividend.stk_div`，纯送转 `1+stk_div`，比 adj_factor 跳变更准；现金分红取税后 `cash_div`，仅 `div_proc='实施'`）
  - `IndexDataSource` → `index_daily`（无复权/分红，仅作基准）
  - `CompositeDataSource` 按 `classify_asset()`（codes.py，按代码前缀+后缀判 stock/etf/index）路由，**一次回测可混合 ETF 与个股**
- 成交价走真实开盘价，历史价格走前复权（对齐聚宽 `use_real_price=True`）
- **A 股摩擦建模（仅对个股生效，ETF/指数不受影响，由 broker 按 `classify_asset` 判定）**：
  - **印花税**：卖出单边 0.05%（`STOCK_STAMP_DUTY_RATE`）；聚宽 `OrderCost.close_tax` 经 `set_order_cost` 透传覆盖；计入 `portfolio.total_tax` → metrics `total_tax` → 报表"印花税"列
  - **T+1**：`Position.locked_amount`（当日买入锁定）/ `closeable_amount`（可卖=总持仓-锁定）；引擎每个交易日开盘前清零 locked，隔夜解锁；卖出受 `closeable_amount` 约束
  - **涨跌停**：以 `pre_close` 推 ±10% 限价（`STOCK_PRICE_LIMIT_PCT`，主板口径；创业板/科创板 20%、北交所 30% 暂用 10% 保守近似），开盘涨停拒买、跌停拒卖
  - 三项开关：`Broker(enforce_t_plus_1=, enforce_price_limit=, stamp_duty_rate=)`，默认全开
  - **停牌**：沿用"缺 bar"隐式机制——停牌日 `daily`/`fund_daily` 表本就无该行，当日无法成交
  - **买卖单位（lot 取整口径）**：A 股买入必须为 100 股整数倍（`_round_to_lot` 对买单向下取整）；**卖出允许零股**（送转后持仓常非整手，卖单不取整）。`order_target_value(code, 0)`（清仓）有特例：直接卖出全部持仓（含零股），不做 lot 取整——对齐聚宽文档"要卖出全部股票时, 可以使用 order_target_value(security, 0), 不需要考虑零股问题"。否则送转后 `order_target_value(code, 0)` 只卖整手、残留零股持仓，长期累积导致与聚宽结果分叉。
- 默认预加载回测开始日前 365 天历史供信号计算
- **指标口径统一对齐聚宽**：策略侧指标（收益/年化/夏普/波动率/回撤/胜率/盈亏比等）一律从 `compute_metrics()` 产出的 `BacktestMetrics` 对象取值，命令行/LLM（`metrics.py` 的 `to_llm_prompt`）与 Web 报表（`reporting.py` 的 `build_report_items`）**共用同源**，两处数值必然一致。聚宽口径：策略收益=(期末总资产−期初总资产)/期初总资产，期初=`initial_cash`；年化收益率=(1+总收益率)^(365/回测自然天数)−1；夏普=(年化收益率−无风险利率)/年化波动率，年化波动率用 250 交易日(`std×√250`)、无风险利率 **4%**；胜率=盈利平仓数/总平仓数（round-trip，非日胜率）；盈亏比=总毛盈利/总毛亏损（**毛盈亏**=卖出额−买入成本，不扣佣金/印花税/滑点）；**聚宽在分红除息日调低 `avg_cost`（减去每股 `div_cash`），降低后续卖出的成本基、提高已实现盈亏**，`realized_trade_stats` 接收 `dividends` 参数（含 `ex_date`）并在遍历交易时做同等调整；**份额折算（split）时聚宽按 `ratio` 调整持仓股数（总成本不变），`realized_trade_stats` 接收 `splits` 参数做同等股数调整**，否则卖出股数对不上买入股数、盈亏比严重失真。基准相关指标（超额收益/alpha/beta/信息比率等）和最大回撤区间、索提诺比率、日胜率等 Web 独有项在 `build_report_items` 内补充计算。改交易统计逻辑只改 `realized_trade_stats` 一处即可。
- 策略源码保存到 MySQL `strategy` 表（`SavedStrategy` 模型），Dashboard 读取/写入该表
- **⚠️ 策略落库约定（生成即入库、清理本地临时文件）**：生成/编写好的策略**直接存入 MySQL `strategy` 表**——用 `quantify.database.strategy_store.save_strategy(name=, source=, description=)`，或 Dashboard 的「保存策略」。`strategy` 表是策略的**唯一权威来源**，Dashboard 从该表读取并运行。为本地编写/验证而临时创建的策略 `.py`（如临时放在 `strategies/` 下用于跑通/回归）属于**一次性脚手架**，**入库后必须删除**，避免仓库文件与库内版本产生 drift。即标准流程：写临时 `.py` → 跑通验证 → `save_strategy()` 落库 → **删除临时 `.py`**。
- **指数成分股票池 `get_index_stocks`（聚宽同款）**：兼容层（`joinquant.py`）注入 `get_index_stocks(index, date=None)`，由 `quantify/backtest/universe.py` 读 `index_weight` 表做**点到点成分**（取 ≤date 的最近一期月度快照，date 缺省用 `context.current_dt`），返回聚宽格式代码。配套 `index_constituents_union(index, start, end)` 供「预加载并集」用。⚠️ 关键：**引擎只加载传入 `ts_codes` 的行情**，而 `get_index_stocks` 是运行时动态选股——所以跑指数成分策略必须先把成分**并集**作为 `ts_codes` 预加载：命令行直接调引擎时用 `universe.index_constituents_union(index, start, end)` 取并集传入，Web 端由 `webapp/app.py` 的 `_resolve_universe()` 在检测到源码含 `get_index_stocks` 时自动把指数代码展开为区间内成分并集再加载（未用 `get_index_stocks` 的老策略行为不变）；策略内部仍按调仓日做点到点选股，避免幸存者偏差。⚠️ **数据质量限制**：Tushare `index_weight` 接口对部分指数（如 000300.SH 沪深300）在 2019-03~12、2021-02~10 等月份无数据（API 返回空），导致这些月份用过期快照选股，与聚宽结果产生偏差。补充拉取命令：`quantify fetch index index-weight --ts-code 000300.SH --full --start-date 20190101 --end-date 20260601`。参考 `strategy` 表 id=39（沪深300 截面滚动回归，源码以库内为准）
- **⚠️ 写策略必避坑**：聚宽 `from jqdata import *` 会用 numpy 同名函数遮蔽内建 `sum`/`max`/`min`/`abs`/`round`。`np.sum(dict.values())` 不求和而是把 `dict_values` 包成 0 维 object 数组原样返回，导致 `s = sum(d.values()); x / s` 抛 `TypeError: unsupported operand type(s) for /: 'float' and 'dict_values'`。**本地引擎用原生 builtins（兼容层只注入下单/历史等少数函数，不含 numpy），所以本地能跑通、传到聚宽才报错**。凡对 `dict.values()`/推导式/生成器做聚合的策略，在 `from jqdata import *` 后显式 `import builtins` 并把 `sum = builtins.sum`（max/min/abs/round 同理）绑回。参考 `strategy` 表 id=26。
- **⚠️ 跨平台下单 API（本地 vs 聚宽）**：下单函数（`order` / `order_value` / `order_target_value` / `order_target_percent` 等）**不来自 `jqdata`**——它们在聚宽由运行时注入到策略全局命名空间，在本地由兼容层（`joinquant.py` 的 `namespace()`）注入。两边注入的集合**不完全一致**：本地兼容层目前只注入 `order`/`order_value`/`order_target_value`/`order_target_percent`，**没有** `order_target`/`order_percent`；聚宽则可能因运行时上下文导致某些 `order_*` 未注入而抛 `NameError`。**统一规范：下单一律只用 `order_target_value(code, context.portfolio.total_value * weight)`**（按目标市值下单），不要用 `order_target_percent` 等按比例的变体，避免两边注入差异。`order_target_percent(code, w)` 等价于 `order_target_value(code, total_value * w)`，回测结果完全一致（已验证）。参考 `strategy` 表 id=36。
- **⚠️ `order_target_value` 取整口径（对齐聚宽）**：聚宽 `order_target_value(code, value)` 的计算方式是**先算 diff 再向零取整到 lot**——`delta = int((target_value - current_value) / price)`（`int()` 向零截断），再 `delta = int(delta / lot) * lot`（同样向零截断到 lot）。**不能**先算 `target_shares = floor(target_value/price/lot)*lot` 再减持仓——对卖出时 `floor()` 向负无穷截断会多卖 100 股（如 delta=-1990, `int`→-1900, `floor`→-2000）。本地 `context.py` 的 `order_target_value` 已按聚宽口径实现。
- **⚠️ 标的/基准代码格式**：聚宽**只认聚宽格式** `.XSHG`（上交所）/`.XSHE`（深交所），指数也是（沪深300 = `000300.XSHG`，不是 `000300.SH`）；本地引擎两种格式通吃（`to_tushare_code` 会把 `.XSHG/.XSHE` 转回 `.SH/.SZ`）。**统一规范：策略里所有代码（`universe`/`set_benchmark`/下单）一律写聚宽格式 `.XSHG/.XSHE`**，这样本地、聚宽都能跑；写 `.SH/.SZ` 则只有本地能跑、传聚宽报 `InvalidParam: 标的'xxx'不存在`。另注意 `attribute_history(...)["close"]` 返回的是日期索引的 Series，取值用 `.iloc[-1]`/`.iloc[0]`（按位置），**不能**用 `closes[-1]`（按标签会抛 `KeyError: -1`）。

## 并发

- ETF 采集器使用 `ThreadPoolExecutor(max_workers=...)` 按代码并发调用 API，并发线程数由 `TUSHARE_MAX_WORKERS` 配置（默认 2）统一控制，5 个 fetcher 都读 `self.client.max_workers`，不再各自硬编码。镜像站 `jiaoch.site` 并发硬上限为 2（超过会触发"并发请求过多"）；官方 API 无并发墙，切到官方后可调大，但仍受每分钟频次限制约束
- 限流器：滑动窗口、线程安全（`tushare_client/rate_limiter.py` 中的 `RateLimiter`）
- Tushare 客户端**直接用 `requests.post` 调镜像站 HTTP 接口**，不走 `tushare` SDK。SDK 的 `DataApi.query` 会把非 2xx 响应静默吞成空 DataFrame，与"标的本就无数据"无法区分；本地实现用 `res.raise_for_status()` + `code != 0` 抛异常，让传输层/业务层错误都能触发重试，而真正的空结果（`items=[]` 或 `data=null`）才返回干净的空 DataFrame
- Tushare 客户端失败重试 5 次，指数退避（tenacity，`retry_if_exception_type(Exception)` 覆盖 HTTPError/ChunkedEncodingError/JSONDecodeError 等镜像站大响应截断的瞬时错误）
- `_fetch_concurrent()` 中无论单次请求有无数据，每行都打印进度（空数据 `"empty"`，有数据 `"+N rows"`），保持终端可见性。写新 fetcher 时要遵循相同模式。空 DataFrame 现在**只可能**是真没数据，故 `_run_one` 拿到空就直接跳过、不重试（重试已在 client 层用异常机制收口）

## 配置

- 全部配置通过 Pydantic Settings 加载，环境变量前缀：`TUSHARE_`、`MYSQL_`、`LOG_`
- `get_settings()` 是 `@lru_cache(maxsize=1)` 缓存的单例
- `TUSHARE_HTTP_URL` 默认指向**镜像站**（`http://jiaoch.site`），而非 Tushare 官方 API
- `TUSHARE_HTTP_TIMEOUT` 控制单次 HTTP 请求超时（默认 30 秒）
- `TUSHARE_MAX_WORKERS` 控制并发采集线程数（默认 2），全部 fetcher 共用此值
- `PROJECT_ROOT` = `quantify/config.py` 的父目录（即仓库根目录）

## 代码风格

- **每个** `.py` 文件都要写 `from __future__ import annotations`（**例外**：`strategies/` 下**临时**编写的聚宽策略脚本用 `from jqdata import *`、numpy/pandas 在 builtins 重绑后导入，`pyproject.toml` 已为 `strategies/*.py` 配 per-file-ignore 放行 E402/F403/F405；这些是落库前的一次性脚手架，不视为包内代码，入库后即删除，见上文「策略落库约定」）
- 日志：从 `quantify.utils.logger` 导入 `log`（Loguru），不要使用 stdlib 的 `logging`
- 单例模式使用 `@lru_cache(maxsize=1)`（engine、client、settings）
- Tushare 返回的日期列是 `YYYYMMDD` 字符串；`_normalize_dates()` 负责转换为 Python `date` 类型
- 访问私有成员时用 `# noqa: SLF001` 抑制 lint 报错

## CLI 注意事项

`--incremental/--full` 是 **Typer 布尔开关对**（不是两个独立的 flag），二者互斥：
- `quantify fetch etf all` → 增量模式（默认）
- `quantify fetch etf all --full` → 全量回填

并非所有阶段都受此开关影响。`fetch etf all` 实际是"混合模式"：

| 阶段 | 接口 | 默认行为 | `--full` 是否生效 |
|------|------|---------|------------------|
| `daily` | `fund_daily` | 增量（查 `max(trade_date)`，只拉之后） | ✅ |
| `nav` | `fund_nav` | 增量（按 `nav_date`） | ✅ |
| `adj` | `fund_adj` | 增量（按 `trade_date`） | ✅ |
| `share` | `fund_share` | 增量（按 `trade_date`） | ✅ |
| `share-size` | `etf_share_size` | 增量（按 `trade_date`） | ✅ |
| `dividend` | `fund_div` | 始终全量 | ❌ |
| `portfolio` | `fund_portfolio` | 始终全量（默认跳过） | ❌ |
| `manager` | `fund_manager` | 始终全量 | ❌ |
| `basic` / `etf-index-basic` | `fund_basic` / `etf_basic` | 始终全量刷新 | ❌ |

即：日线/净值/复权/份额/规模走增量；分红/持仓/基金经理/basic 每次全量重拉（接口本身无增量语义，靠 upsert 去重）。

## 数据顺序依赖

1. `quantify fetch etf basic` **必须先执行**。它会填充 `fund_basic` 表，其他 ETF 采集器从该表读取标的列表（排除已退市：`status != "D"`）。未先拉取则其他阶段标的列表为空。
2. `quantify fetch stock basic` **必须先执行**（在 stock 其他阶段之前）。填充 `stock_basic` 表，stock 的其他阶段（daily/adj_factor/财务等）从中读取标的列表。
3. `fetch stock all` 默认跳过 `weekly/monthly/stk_factor/margin_detail/broker_recommend`；`fetch futures all` 默认跳过 `fut_holding/fut_wsr/fut_settle`。加 `--include-skipped` 一次性全部拉取。
4. `quantify fetch all` 自动按依赖顺序：交易日历 → ETF（basic 优先）→ 个股 → 行业 → 指数 → 宏观 → 期货 → 基金
5. 行业/指数/宏观部分接口需 5000+ 积分；个股财务/资金流亦需较高积分。积分不足时用 `--skip` 跳过对应组

## 测试

暂无测试。`ruff check . && ruff format --check .` 是当前唯一的代码质量检查。Ruff 配置在 `pyproject.toml` 中。
