"""SQLAlchemy ORM models.

Table names mirror their Tushare Pro endpoint names one-to-one:
    - stock_basic     (A-share stock list)
    - trade_cal       (exchange trade calendar)
    - daily           (stock daily OHLCV)
    - adj_factor      (stock adjustment factor)
    - daily_basic     (stock daily indicators: PE/PB/换手率/市值)
    - weekly          (stock weekly OHLCV)
    - monthly         (stock monthly OHLCV)
    - suspend_d       (suspension / resumption)
    - namechange      (historical name changes)
    - income          (income statement)
    - balancesheet    (balance sheet)
    - cashflow        (cash flow statement)
    - fina_indicator  (financial indicators: ROE/ROA/gross_margin etc.)
    - forecast        (earnings forecast)
    - express         (earnings express)
    - dividend        (stock dividend / split)
    - moneyflow_hsgt  (north-bound capital flow)
    - margin          (margin trading summary)
    - margin_detail   (margin trading detail per stock)
    - stk_factor      (broker profit forecast / consensus)
    - broker_recommend(broker monthly gold-stock picks)
    - fund_basic      (ETF basic info, market='E')
    - fund_daily      (ETF daily quotes)
    - fund_nav        (ETF NAV)
    - fund_adj        (ETF adjustment factor)
    - fund_div        (ETF dividend)
    - fund_share      (ETF share)
    - fund_portfolio  (ETF portfolio)
    - fund_manager    (ETF manager)
    - fund_company    (public fund company info)
    - index_classify  (SW industry classification)
    - index_member_all(SW industry members)
    - sw_daily        (SW industry index daily)
    - ci_index_member (CITIC industry members)
    - ci_daily        (CITIC industry index daily)
    - index_basic     (index basic info)
    - index_daily     (index daily quotes)
    - index_dailybasic(index daily indicators)
    - index_weight    (index constituents & weights)
    - yc_cb           (CGB yield curve)
    - index_global    (global index daily)
    - us_tycr         (US treasury nominal yield curve)
    - us_trycr        (US treasury real yield curve)
    - fut_basic       (futures contract list)
    - fut_daily       (futures daily OHLCV)
    - fut_holding     (futures daily holding ranking)
    - fut_wsr         (futures warehouse receipts)
    - fut_settle      (futures settlement parameters)
    - strategy        (saved backtest strategies, local-only)
    - factor_library  (LLM-mined, alphalens-validated factors, local-only)

Primary keys are chosen so that re-running a fetch is idempotent via
INSERT ... ON DUPLICATE KEY UPDATE.
"""

from __future__ import annotations

from datetime import date, datetime

from sqlalchemy import (
    BigInteger,
    Date,
    DateTime,
    Float,
    Index,
    Integer,
    PrimaryKeyConstraint,
    String,
    Text,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """Project-wide declarative base."""


# ---------------------------------------------------------------------------
# Saved backtest strategies
# ---------------------------------------------------------------------------
class SavedStrategy(Base):
    __tablename__ = "strategy"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True, comment="自增主键")
    name: Mapped[str] = mapped_column(String(128), comment="策略名称")
    description: Mapped[str | None] = mapped_column(Text, comment="策略说明")
    source: Mapped[str] = mapped_column(Text, comment="策略源码")

    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), comment="创建时间")
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        server_default=func.now(),
        onupdate=func.now(),
        comment="更新时间",
    )

    __table_args__ = (Index("uq_strategy_name", "name", unique=True),)


# ---------------------------------------------------------------------------
# Factor library (LLM-mined factors that passed alphalens validation)
# Local-only table (not a Tushare endpoint). Populated by the factor-mining
# pipeline in quantify/factor/.
# ---------------------------------------------------------------------------
class FactorLibrary(Base):
    __tablename__ = "factor_library"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True, comment="自增主键")
    name: Mapped[str] = mapped_column(String(128), comment="因子名称")
    expression: Mapped[str] = mapped_column(Text, comment="Qlib 因子表达式")
    hypothesis: Mapped[str | None] = mapped_column(Text, comment="LLM 提出的因子逻辑/假说")
    category: Mapped[str | None] = mapped_column(String(64), comment="因子类别 momentum/value/volatility 等")

    # 评估口径
    universe: Mapped[str | None] = mapped_column(String(64), comment="评估股票池，如 csi300")
    start_date: Mapped[date | None] = mapped_column(Date, comment="评估开始日")
    end_date: Mapped[date | None] = mapped_column(Date, comment="评估结束日")
    periods: Mapped[str | None] = mapped_column(String(32), comment="前瞻收益周期，如 1,5,10")

    # Alphalens 核心指标（以主周期为准）
    ic_mean: Mapped[float | None] = mapped_column(Float, comment="IC 均值")
    ic_std: Mapped[float | None] = mapped_column(Float, comment="IC 标准差")
    icir: Mapped[float | None] = mapped_column(Float, comment="IC_IR = IC均值/IC标准差")
    ic_tstat: Mapped[float | None] = mapped_column(Float, comment="IC t 统计量")
    rank_ic_mean: Mapped[float | None] = mapped_column(Float, comment="Rank IC 均值")
    rank_icir: Mapped[float | None] = mapped_column(Float, comment="Rank IC_IR")
    quantile_spread: Mapped[float | None] = mapped_column(Float, comment="多空分层收益差(top-bottom)")
    turnover: Mapped[float | None] = mapped_column(Float, comment="顶层分位换手率")
    coverage: Mapped[float | None] = mapped_column(Float, comment="有效覆盖率(非空占比)")

    status: Mapped[str] = mapped_column(
        String(16), default="passed", comment="状态 passed/evaluated/composed"
    )
    iteration: Mapped[int | None] = mapped_column(Integer, comment="挖掘迭代轮次")
    metrics_json: Mapped[str | None] = mapped_column(Text, comment="完整评估指标 JSON")
    report_path: Mapped[str | None] = mapped_column(Text, comment="Alphalens 报告文件路径")

    # 策略关联：因子对应的回测策略 ID（指向 strategy 表），由 strategy_gen 模块回写
    strategy_id: Mapped[int | None] = mapped_column(BigInteger, comment="关联策略表ID（回测后回写）")
    # 回测结果快照（完整 BacktestMetrics.to_dict() 的 JSON），回测后回写
    backtest_metrics_json: Mapped[str | None] = mapped_column(Text, comment="回测结果指标快照 JSON")
    # 因子类型：single（单因子挖掘）/ composed（合成因子）
    factor_type: Mapped[str] = mapped_column(String(16), default="single", comment="因子类型 single/composed")
    # 合成因子的父因子 ID 列表（逗号分隔），仅 composed 因子有值
    parent_factor_ids: Mapped[str | None] = mapped_column(
        String(256), comment="合成因子的父因子ID列表(逗号分隔)"
    )

    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), comment="创建时间")
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        server_default=func.now(),
        onupdate=func.now(),
        comment="更新时间",
    )

    __table_args__ = (Index("uq_factor_name", "name", unique=True),)


# ---------------------------------------------------------------------------
# ETF basic info (fund_basic, market='E')
# ---------------------------------------------------------------------------
class EtfBasic(Base):
    __tablename__ = "fund_basic"

    ts_code: Mapped[str] = mapped_column(String(16), primary_key=True, comment="基金代码")
    name: Mapped[str | None] = mapped_column(String(128), comment="简称")
    management: Mapped[str | None] = mapped_column(String(128), comment="管理人")
    custodian: Mapped[str | None] = mapped_column(String(128), comment="托管人")
    fund_type: Mapped[str | None] = mapped_column(String(64), comment="投资类型")
    found_date: Mapped[date | None] = mapped_column(Date, comment="成立日期")
    due_date: Mapped[date | None] = mapped_column(Date, comment="到期日期")
    list_date: Mapped[date | None] = mapped_column(Date, comment="上市日期")
    issue_date: Mapped[date | None] = mapped_column(Date, comment="发行日期")
    delist_date: Mapped[date | None] = mapped_column(Date, comment="退市日期")
    issue_amount: Mapped[float | None] = mapped_column(Float, comment="发行份额(亿)")
    m_fee: Mapped[float | None] = mapped_column(Float, comment="管理费")
    c_fee: Mapped[float | None] = mapped_column(Float, comment="托管费")
    duration_year: Mapped[float | None] = mapped_column(Float, comment="存续期")
    p_value: Mapped[float | None] = mapped_column(Float, comment="面值")
    min_amount: Mapped[float | None] = mapped_column(Float, comment="最小申购金额")
    exp_return: Mapped[float | None] = mapped_column(Float, comment="预期收益率")
    benchmark: Mapped[str | None] = mapped_column(Text, comment="业绩比较基准")
    status: Mapped[str | None] = mapped_column(String(8), comment="存续状态 D摘牌 I发行 L上市")
    invest_type: Mapped[str | None] = mapped_column(String(64), comment="投资风格")
    type: Mapped[str | None] = mapped_column(String(64), comment="基金类型")
    trustee: Mapped[str | None] = mapped_column(String(128), comment="受托人")
    purc_startdate: Mapped[date | None] = mapped_column(Date, comment="日常申购起始日")
    redm_startdate: Mapped[date | None] = mapped_column(Date, comment="日常赎回起始日")
    market: Mapped[str | None] = mapped_column(String(4), comment="E场内 O场外")

    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), onupdate=func.now())


# ---------------------------------------------------------------------------
# ETF tracking-index info (etf_basic) - distinct from fund_basic above.
# Provides the ETF -> tracked index mapping (index_code/index_name).
# ---------------------------------------------------------------------------
class EtfIndexBasic(Base):
    __tablename__ = "etf_basic"

    ts_code: Mapped[str] = mapped_column(String(16), primary_key=True, comment="基金交易代码")
    csname: Mapped[str | None] = mapped_column(String(128), comment="ETF中文简称")
    extname: Mapped[str | None] = mapped_column(String(128), comment="ETF扩位简称")
    cname: Mapped[str | None] = mapped_column(String(256), comment="基金中文全称")
    index_code: Mapped[str | None] = mapped_column(String(32), comment="ETF基准指数代码")
    index_name: Mapped[str | None] = mapped_column(String(256), comment="ETF基准指数中文全称")
    setup_date: Mapped[date | None] = mapped_column(Date, comment="设立日期")
    list_date: Mapped[date | None] = mapped_column(Date, comment="上市日期")
    list_status: Mapped[str | None] = mapped_column(String(8), comment="存续状态 L上市 D退市 P待上市")
    exchange: Mapped[str | None] = mapped_column(String(8), comment="交易所 SH/SZ")
    mgr_name: Mapped[str | None] = mapped_column(String(128), comment="基金管理人简称")
    custod_name: Mapped[str | None] = mapped_column(String(128), comment="基金托管人名称")
    mgt_fee: Mapped[float | None] = mapped_column(Float, comment="管理费率")
    etf_type: Mapped[str | None] = mapped_column(String(16), comment="投资通道类型 境内/QDII")

    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), onupdate=func.now())

    __table_args__ = (Index("idx_etf_basic_index_code", "index_code"),)


# ---------------------------------------------------------------------------
# ETF daily quotes
# ---------------------------------------------------------------------------
class EtfDaily(Base):
    __tablename__ = "fund_daily"

    ts_code: Mapped[str] = mapped_column(String(16), comment="基金代码")
    trade_date: Mapped[date] = mapped_column(Date, comment="交易日期")
    pre_close: Mapped[float | None] = mapped_column(Float, comment="昨收盘价")
    open: Mapped[float | None] = mapped_column(Float, comment="开盘价")
    high: Mapped[float | None] = mapped_column(Float, comment="最高价")
    low: Mapped[float | None] = mapped_column(Float, comment="最低价")
    close: Mapped[float | None] = mapped_column(Float, comment="收盘价")
    change: Mapped[float | None] = mapped_column(Float, comment="涨跌额")
    pct_chg: Mapped[float | None] = mapped_column(Float, comment="涨跌幅(%)")
    vol: Mapped[float | None] = mapped_column(Float, comment="成交量(手)")
    amount: Mapped[float | None] = mapped_column(Float, comment="成交额(千元)")

    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), onupdate=func.now())

    __table_args__ = (
        PrimaryKeyConstraint("ts_code", "trade_date", name="pk_fund_daily"),
        Index("idx_fund_daily_date", "trade_date"),
        Index("idx_fund_daily_trade_code_amount", "trade_date", "ts_code", "amount"),
    )


# ---------------------------------------------------------------------------
# ETF NAV
# ---------------------------------------------------------------------------
class EtfNav(Base):
    __tablename__ = "fund_nav"

    ts_code: Mapped[str] = mapped_column(String(16), comment="基金代码")
    nav_date: Mapped[date] = mapped_column(Date, comment="净值日期")
    ann_date: Mapped[date | None] = mapped_column(Date, comment="公告日期")
    unit_nav: Mapped[float | None] = mapped_column(Float, comment="单位净值")
    accum_nav: Mapped[float | None] = mapped_column(Float, comment="累计净值")
    accum_div: Mapped[float | None] = mapped_column(Float, comment="累计分红")
    net_asset: Mapped[float | None] = mapped_column(Float, comment="资产净值")
    total_netasset: Mapped[float | None] = mapped_column(Float, comment="合计资产净值")
    adj_nav: Mapped[float | None] = mapped_column(Float, comment="复权净值")
    update_flag: Mapped[str | None] = mapped_column(String(4), comment="更新标识")

    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), onupdate=func.now())

    __table_args__ = (
        PrimaryKeyConstraint("ts_code", "nav_date", name="pk_fund_nav"),
        Index("idx_fund_nav_nav_date", "nav_date"),
    )


# ---------------------------------------------------------------------------
# ETF adjustment factor (复权因子)
# 来源: fund_adj
# 用途: 后复权价 = close × adj_factor
#       前复权价 = close × adj_factor / 最新adj_factor
# 注意: 每次分红/拆分后历史因子会追溯更新，因此全量回填时需覆盖历史记录
# ---------------------------------------------------------------------------
class EtfAdjFactor(Base):
    __tablename__ = "fund_adj"

    ts_code: Mapped[str] = mapped_column(String(16), comment="基金代码")
    trade_date: Mapped[date] = mapped_column(Date, comment="交易日期")
    adj_factor: Mapped[float | None] = mapped_column(Float, comment="复权因子")

    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), onupdate=func.now())

    __table_args__ = (
        PrimaryKeyConstraint("ts_code", "trade_date", name="pk_fund_adj"),
        Index("idx_fund_adj_trade_date", "trade_date"),
    )


# ---------------------------------------------------------------------------
# ETF dividend
# ---------------------------------------------------------------------------
class EtfDividend(Base):
    __tablename__ = "fund_div"

    ts_code: Mapped[str] = mapped_column(String(16), comment="基金代码")
    ann_date: Mapped[date | None] = mapped_column(Date, comment="公告日")
    imp_anndate: Mapped[date | None] = mapped_column(Date, comment="信息披露日")
    base_date: Mapped[date | None] = mapped_column(Date, comment="基准日")
    div_proc: Mapped[str | None] = mapped_column(String(32), comment="分红方案进度")
    record_date: Mapped[date | None] = mapped_column(Date, comment="登记日")
    ex_date: Mapped[date | None] = mapped_column(Date, comment="除息日")
    pay_date: Mapped[date | None] = mapped_column(Date, comment="派息日")
    earpay_date: Mapped[date | None] = mapped_column(Date, comment="收益支付日")
    net_ex_date: Mapped[date | None] = mapped_column(Date, comment="净值除权日")
    div_cash: Mapped[float | None] = mapped_column(Float, comment="每份派息(元)")
    base_unit: Mapped[float | None] = mapped_column(Float, comment="基准份额")
    ear_distr: Mapped[float | None] = mapped_column(Float, comment="收益分配金额")
    ear_amount: Mapped[float | None] = mapped_column(Float, comment="可分配收益")
    account_date: Mapped[date | None] = mapped_column(Date, comment="会计日期")
    base_year: Mapped[str | None] = mapped_column(String(8), comment="基准年份")

    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), onupdate=func.now())

    __table_args__ = (
        PrimaryKeyConstraint("ts_code", "base_date", name="pk_fund_div"),
        Index("idx_fund_div_ex_date", "ex_date"),
    )


# ---------------------------------------------------------------------------
# ETF share (规模/份额变动)
# ---------------------------------------------------------------------------
class EtfShare(Base):
    __tablename__ = "fund_share"

    ts_code: Mapped[str] = mapped_column(String(16), comment="基金代码")
    trade_date: Mapped[date] = mapped_column(Date, comment="变动日期")
    fd_share: Mapped[float | None] = mapped_column(Float, comment="基金份额(万份)")
    fund_type: Mapped[str | None] = mapped_column(String(64), comment="基金类型")
    market: Mapped[str | None] = mapped_column(String(4), comment="E场内 O场外")

    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), onupdate=func.now())

    __table_args__ = (
        PrimaryKeyConstraint("ts_code", "trade_date", name="pk_fund_share"),
        Index("idx_fund_share_trade_date", "trade_date"),
    )


# ---------------------------------------------------------------------------
# ETF share & size (etf_share_size) - daily share + AUM, ETF-theme endpoint
# ---------------------------------------------------------------------------
class EtfShareSize(Base):
    __tablename__ = "etf_share_size"

    trade_date: Mapped[date] = mapped_column(Date, comment="交易日期")
    ts_code: Mapped[str] = mapped_column(String(16), comment="ETF代码")
    etf_name: Mapped[str | None] = mapped_column(String(128), comment="基金名称")
    total_share: Mapped[float | None] = mapped_column(Float, comment="总份额(万份)")
    total_size: Mapped[float | None] = mapped_column(Float, comment="总规模(万元)")
    nav: Mapped[float | None] = mapped_column(Float, comment="基金份额净值(元)")
    close: Mapped[float | None] = mapped_column(Float, comment="收盘价(元)")
    exchange: Mapped[str | None] = mapped_column(String(8), comment="交易所 SSE/SZSE/BSE")

    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), onupdate=func.now())

    __table_args__ = (
        PrimaryKeyConstraint("ts_code", "trade_date", name="pk_etf_share_size"),
        Index("idx_etf_share_size_trade_date", "trade_date"),
    )


# ---------------------------------------------------------------------------
# ETF portfolio (披露持仓)
# ---------------------------------------------------------------------------
class EtfPortfolio(Base):
    __tablename__ = "fund_portfolio"

    ts_code: Mapped[str] = mapped_column(String(16), comment="基金代码")
    end_date: Mapped[date] = mapped_column(Date, comment="截止日期")
    symbol: Mapped[str] = mapped_column(String(16), comment="持仓标的代码")
    ann_date: Mapped[date | None] = mapped_column(Date, comment="公告日期")
    mkv: Mapped[float | None] = mapped_column(Float, comment="持仓市值")
    amount: Mapped[float | None] = mapped_column(Float, comment="持仓数量(股)")
    stk_mkv_ratio: Mapped[float | None] = mapped_column(Float, comment="占股票市值比")
    stk_float_ratio: Mapped[float | None] = mapped_column(Float, comment="占流通股本比")

    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), onupdate=func.now())

    __table_args__ = (
        PrimaryKeyConstraint("ts_code", "end_date", "symbol", name="pk_fund_portfolio"),
        Index("idx_fund_portfolio_symbol", "symbol", "end_date"),
        Index("idx_fund_portfolio_end_date", "end_date"),
    )


# ---------------------------------------------------------------------------
# ETF manager
# ---------------------------------------------------------------------------
class EtfManager(Base):
    __tablename__ = "fund_manager"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True, comment="自增主键")
    ts_code: Mapped[str] = mapped_column(String(16), index=True, comment="基金代码")
    ann_date: Mapped[date | None] = mapped_column(Date, comment="公告日期")
    name: Mapped[str | None] = mapped_column(String(64), comment="基金经理姓名")
    gender: Mapped[str | None] = mapped_column(String(4), comment="性别")
    birth_year: Mapped[str | None] = mapped_column(String(8), comment="出生年份")
    edu: Mapped[str | None] = mapped_column(String(32), comment="学历")
    nationality: Mapped[str | None] = mapped_column(String(32), comment="国籍")
    begin_date: Mapped[date | None] = mapped_column(Date, comment="任职日期")
    end_date: Mapped[date | None] = mapped_column(Date, comment="离任日期")
    resume: Mapped[str | None] = mapped_column(Text, comment="简历摘要")

    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), onupdate=func.now())

    __table_args__ = (Index("uq_fund_manager", "ts_code", "name", "begin_date", unique=True),)


# ---------------------------------------------------------------------------
# SW industry classification (index_classify, src='SW2021')
# ---------------------------------------------------------------------------
class SwIndustryClassify(Base):
    __tablename__ = "index_classify"

    index_code: Mapped[str] = mapped_column(String(16), comment="申万行业指数代码")
    src: Mapped[str] = mapped_column(String(16), comment="分类版本，如 SW2021")
    industry_name: Mapped[str | None] = mapped_column(String(128), comment="行业名称")
    parent_code: Mapped[str | None] = mapped_column(String(16), comment="父级行业代码")
    level: Mapped[str | None] = mapped_column(String(8), comment="行业级别 L1/L2/L3")
    industry_code: Mapped[str | None] = mapped_column(String(16), comment="行业代码")
    is_pub: Mapped[str | None] = mapped_column(String(4), comment="是否发布指数")

    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), onupdate=func.now())

    __table_args__ = (
        PrimaryKeyConstraint("src", "index_code", name="pk_index_classify"),
        Index("idx_index_classify_level", "src", "level", "is_pub"),
    )


# ---------------------------------------------------------------------------
# SW industry members (index_member_all)
# ---------------------------------------------------------------------------
class SwIndustryMember(Base):
    __tablename__ = "index_member_all"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True, comment="自增主键")
    l1_code: Mapped[str | None] = mapped_column(String(16), comment="一级行业代码")
    l1_name: Mapped[str | None] = mapped_column(String(128), comment="一级行业名称")
    l2_code: Mapped[str | None] = mapped_column(String(16), comment="二级行业代码")
    l2_name: Mapped[str | None] = mapped_column(String(128), comment="二级行业名称")
    l3_code: Mapped[str | None] = mapped_column(String(16), comment="三级行业代码")
    l3_name: Mapped[str | None] = mapped_column(String(128), comment="三级行业名称")
    ts_code: Mapped[str] = mapped_column(String(16), comment="成分股票代码")
    name: Mapped[str | None] = mapped_column(String(128), comment="成分股票名称")
    in_date: Mapped[date | None] = mapped_column(Date, comment="纳入日期")
    out_date: Mapped[date | None] = mapped_column(Date, comment="剔除日期")
    is_new: Mapped[str | None] = mapped_column(String(4), comment="是否最新")

    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), onupdate=func.now())

    __table_args__ = (
        Index("uq_index_member_all", "ts_code", "l3_code", "in_date", unique=True),
        Index("idx_index_member_all_l1", "l1_code", "is_new"),
        Index("idx_index_member_all_l3", "l3_code", "is_new"),
    )


# ---------------------------------------------------------------------------
# SW industry daily quotes (sw_daily)
# ---------------------------------------------------------------------------
class SwIndustryDaily(Base):
    __tablename__ = "sw_daily"

    ts_code: Mapped[str] = mapped_column(String(16), comment="申万行业指数代码")
    trade_date: Mapped[date] = mapped_column(Date, comment="交易日期")
    name: Mapped[str | None] = mapped_column(String(128), comment="指数名称")
    open: Mapped[float | None] = mapped_column(Float, comment="开盘点位")
    low: Mapped[float | None] = mapped_column(Float, comment="最低点位")
    high: Mapped[float | None] = mapped_column(Float, comment="最高点位")
    close: Mapped[float | None] = mapped_column(Float, comment="收盘点位")
    change: Mapped[float | None] = mapped_column(Float, comment="涨跌点位")
    pct_change: Mapped[float | None] = mapped_column(Float, comment="涨跌幅")
    vol: Mapped[float | None] = mapped_column(Float, comment="成交量(万股)")
    amount: Mapped[float | None] = mapped_column(Float, comment="成交额(万元)")
    pe: Mapped[float | None] = mapped_column(Float, comment="市盈率")
    pb: Mapped[float | None] = mapped_column(Float, comment="市净率")
    float_mv: Mapped[float | None] = mapped_column(Float, comment="流通市值(万元)")
    total_mv: Mapped[float | None] = mapped_column(Float, comment="总市值(万元)")

    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), onupdate=func.now())

    __table_args__ = (
        PrimaryKeyConstraint("ts_code", "trade_date", name="pk_sw_daily"),
        Index("idx_sw_daily_trade_date", "trade_date"),
    )


# ---------------------------------------------------------------------------
# CITIC industry members (ci_index_member)
# ---------------------------------------------------------------------------
class CiticIndustryMember(Base):
    __tablename__ = "ci_index_member"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True, comment="自增主键")
    l1_code: Mapped[str | None] = mapped_column(String(16), comment="一级行业代码")
    l1_name: Mapped[str | None] = mapped_column(String(128), comment="一级行业名称")
    l2_code: Mapped[str | None] = mapped_column(String(16), comment="二级行业代码")
    l2_name: Mapped[str | None] = mapped_column(String(128), comment="二级行业名称")
    l3_code: Mapped[str | None] = mapped_column(String(16), comment="三级行业代码")
    l3_name: Mapped[str | None] = mapped_column(String(128), comment="三级行业名称")
    ts_code: Mapped[str] = mapped_column(String(16), comment="成分股票代码")
    name: Mapped[str | None] = mapped_column(String(128), comment="成分股票名称")
    in_date: Mapped[date | None] = mapped_column(Date, comment="纳入日期")
    out_date: Mapped[date | None] = mapped_column(Date, comment="剔除日期")
    is_new: Mapped[str | None] = mapped_column(String(4), comment="是否最新")

    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), onupdate=func.now())

    __table_args__ = (
        Index("uq_ci_index_member", "ts_code", "l3_code", "in_date", unique=True),
        Index("idx_ci_index_member_l1", "l1_code", "is_new"),
        Index("idx_ci_index_member_l3", "l3_code", "is_new"),
    )


# ---------------------------------------------------------------------------
# CITIC industry daily quotes (ci_daily)
# ---------------------------------------------------------------------------
class CiticIndustryDaily(Base):
    __tablename__ = "ci_daily"

    ts_code: Mapped[str] = mapped_column(String(16), comment="中信行业指数代码")
    trade_date: Mapped[date] = mapped_column(Date, comment="交易日期")
    open: Mapped[float | None] = mapped_column(Float, comment="开盘点位")
    low: Mapped[float | None] = mapped_column(Float, comment="最低点位")
    high: Mapped[float | None] = mapped_column(Float, comment="最高点位")
    close: Mapped[float | None] = mapped_column(Float, comment="收盘点位")
    pre_close: Mapped[float | None] = mapped_column(Float, comment="昨日收盘点位")
    change: Mapped[float | None] = mapped_column(Float, comment="涨跌点位")
    pct_change: Mapped[float | None] = mapped_column(Float, comment="涨跌幅")
    vol: Mapped[float | None] = mapped_column(Float, comment="成交量(万股)")
    amount: Mapped[float | None] = mapped_column(Float, comment="成交额(万元)")

    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), onupdate=func.now())

    __table_args__ = (
        PrimaryKeyConstraint("ts_code", "trade_date", name="pk_ci_daily"),
        Index("idx_ci_daily_trade_date", "trade_date"),
    )


# ---------------------------------------------------------------------------
# Trade calendar (trade_cal) - authoritative exchange open/close dates
# ---------------------------------------------------------------------------
class TradeCalendar(Base):
    __tablename__ = "trade_cal"

    exchange: Mapped[str] = mapped_column(String(8), comment="交易所 SSE/SZSE 等")
    cal_date: Mapped[date] = mapped_column(Date, comment="日历日期")
    is_open: Mapped[int | None] = mapped_column(Integer, comment="是否交易 1开市 0休市")
    pretrade_date: Mapped[date | None] = mapped_column(Date, comment="上一交易日")

    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), onupdate=func.now())

    __table_args__ = (
        PrimaryKeyConstraint("exchange", "cal_date", name="pk_trade_cal"),
        Index("idx_trade_cal_open", "exchange", "is_open", "cal_date"),
    )


# ---------------------------------------------------------------------------
# Index basic info (index_basic)
# ---------------------------------------------------------------------------
class IndexBasic(Base):
    __tablename__ = "index_basic"

    ts_code: Mapped[str] = mapped_column(String(24), primary_key=True, comment="TS指数代码")
    name: Mapped[str | None] = mapped_column(String(128), comment="简称")
    fullname: Mapped[str | None] = mapped_column(String(256), comment="指数全称")
    market: Mapped[str | None] = mapped_column(String(16), comment="市场")
    publisher: Mapped[str | None] = mapped_column(String(128), comment="发布方")
    index_type: Mapped[str | None] = mapped_column(String(64), comment="指数风格")
    category: Mapped[str | None] = mapped_column(String(64), comment="指数类别")
    base_date: Mapped[date | None] = mapped_column(Date, comment="基期")
    base_point: Mapped[float | None] = mapped_column(Float, comment="基点")
    list_date: Mapped[date | None] = mapped_column(Date, comment="发布日期")
    weight_rule: Mapped[str | None] = mapped_column(String(128), comment="加权方式")
    desc: Mapped[str | None] = mapped_column("desc", Text, comment="描述")
    exp_date: Mapped[date | None] = mapped_column(Date, comment="终止日期")

    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), onupdate=func.now())

    __table_args__ = (Index("idx_index_basic_market", "market", "category"),)


# ---------------------------------------------------------------------------
# Index daily quotes (index_daily)
# ---------------------------------------------------------------------------
class IndexDaily(Base):
    __tablename__ = "index_daily"

    ts_code: Mapped[str] = mapped_column(String(24), comment="TS指数代码")
    trade_date: Mapped[date] = mapped_column(Date, comment="交易日")
    close: Mapped[float | None] = mapped_column(Float, comment="收盘点位")
    open: Mapped[float | None] = mapped_column(Float, comment="开盘点位")
    high: Mapped[float | None] = mapped_column(Float, comment="最高点位")
    low: Mapped[float | None] = mapped_column(Float, comment="最低点位")
    pre_close: Mapped[float | None] = mapped_column(Float, comment="昨日收盘点")
    change: Mapped[float | None] = mapped_column(Float, comment="涨跌点")
    pct_chg: Mapped[float | None] = mapped_column(Float, comment="涨跌幅(%)")
    vol: Mapped[float | None] = mapped_column(Float, comment="成交量(手)")
    amount: Mapped[float | None] = mapped_column(Float, comment="成交额(千元)")

    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), onupdate=func.now())

    __table_args__ = (
        PrimaryKeyConstraint("ts_code", "trade_date", name="pk_index_daily"),
        Index("idx_index_daily_trade_date", "trade_date"),
    )


# ---------------------------------------------------------------------------
# Index daily basic indicators (index_dailybasic) - major broad indices only
# ---------------------------------------------------------------------------
class IndexDailyBasic(Base):
    __tablename__ = "index_dailybasic"

    ts_code: Mapped[str] = mapped_column(String(24), comment="TS代码")
    trade_date: Mapped[date] = mapped_column(Date, comment="交易日期")
    total_mv: Mapped[float | None] = mapped_column(Float, comment="当日总市值(元)")
    float_mv: Mapped[float | None] = mapped_column(Float, comment="当日流通市值(元)")
    total_share: Mapped[float | None] = mapped_column(Float, comment="当日总股本(股)")
    float_share: Mapped[float | None] = mapped_column(Float, comment="当日流通股本(股)")
    free_share: Mapped[float | None] = mapped_column(Float, comment="当日自由流通股本(股)")
    turnover_rate: Mapped[float | None] = mapped_column(Float, comment="换手率")
    turnover_rate_f: Mapped[float | None] = mapped_column(Float, comment="换手率(自由流通股本)")
    pe: Mapped[float | None] = mapped_column(Float, comment="市盈率")
    pe_ttm: Mapped[float | None] = mapped_column(Float, comment="市盈率TTM")
    pb: Mapped[float | None] = mapped_column(Float, comment="市净率")

    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), onupdate=func.now())

    __table_args__ = (
        PrimaryKeyConstraint("ts_code", "trade_date", name="pk_index_dailybasic"),
        Index("idx_index_dailybasic_trade_date", "trade_date"),
    )


# ---------------------------------------------------------------------------
# Index constituents & weights (index_weight) - monthly
# ---------------------------------------------------------------------------
class IndexWeight(Base):
    __tablename__ = "index_weight"

    index_code: Mapped[str] = mapped_column(String(24), comment="指数代码")
    con_code: Mapped[str] = mapped_column(String(24), comment="成分代码")
    trade_date: Mapped[date] = mapped_column(Date, comment="交易日期")
    weight: Mapped[float | None] = mapped_column(Float, comment="权重")

    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), onupdate=func.now())

    __table_args__ = (
        PrimaryKeyConstraint("index_code", "con_code", "trade_date", name="pk_index_weight"),
        Index("idx_index_weight_trade_date", "trade_date"),
    )


# ---------------------------------------------------------------------------
# Sector money flow (moneyflow_ind_dc) - DC industry/concept board flows
# ---------------------------------------------------------------------------
class MoneyflowIndDc(Base):
    __tablename__ = "moneyflow_ind_dc"

    trade_date: Mapped[date] = mapped_column(Date, comment="交易日期")
    ts_code: Mapped[str] = mapped_column(String(24), comment="DC板块代码")
    content_type: Mapped[str] = mapped_column(String(32), comment="数据类型(行业/概念/地域)")
    name: Mapped[str | None] = mapped_column(String(64), comment="板块名称")
    pct_change: Mapped[float | None] = mapped_column(Float, comment="板块涨跌幅(%)")
    close: Mapped[float | None] = mapped_column(Float, comment="板块最新指数")
    net_amount: Mapped[float | None] = mapped_column(Float, comment="主力净流入额(元)")
    net_amount_rate: Mapped[float | None] = mapped_column(Float, comment="主力净流入占比(%)")
    buy_elg_amount: Mapped[float | None] = mapped_column(Float, comment="超大单净流入额(元)")
    buy_elg_amount_rate: Mapped[float | None] = mapped_column(Float, comment="超大单净流入占比(%)")
    buy_lg_amount: Mapped[float | None] = mapped_column(Float, comment="大单净流入额(元)")
    buy_lg_amount_rate: Mapped[float | None] = mapped_column(Float, comment="大单净流入占比(%)")
    buy_md_amount: Mapped[float | None] = mapped_column(Float, comment="中单净流入额(元)")
    buy_md_amount_rate: Mapped[float | None] = mapped_column(Float, comment="中单净流入占比(%)")
    buy_sm_amount: Mapped[float | None] = mapped_column(Float, comment="小单净流入额(元)")
    buy_sm_amount_rate: Mapped[float | None] = mapped_column(Float, comment="小单净流入占比(%)")
    buy_sm_amount_stock: Mapped[str | None] = mapped_column(String(64), comment="小单净流入最大股")
    rank: Mapped[int | None] = mapped_column(Integer, comment="排名")

    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), onupdate=func.now())

    __table_args__ = (
        PrimaryKeyConstraint("trade_date", "ts_code", "content_type", name="pk_moneyflow_ind_dc"),
        Index("idx_moneyflow_ind_dc_code", "ts_code", "trade_date"),
    )


# ---------------------------------------------------------------------------
# 中债国债收益率曲线 (yc_cb)
# ---------------------------------------------------------------------------
class YcCb(Base):
    __tablename__ = "yc_cb"

    trade_date: Mapped[date] = mapped_column(Date, comment="交易日期")
    ts_code: Mapped[str] = mapped_column(String(16), comment="曲线编码 如 1001.CB")
    curve_name: Mapped[str | None] = mapped_column(String(64), comment="曲线名称")
    curve_type: Mapped[str] = mapped_column(String(2), comment="曲线类型 0到期 1即期")
    curve_term: Mapped[float] = mapped_column(Float, comment="期限(年)")
    # yield 是 SQL/Python 保留字，属性名加后缀，列名仍用 yield。
    yield_pct: Mapped[float | None] = mapped_column("yield", Float, comment="收益率(%)")

    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), onupdate=func.now())

    __table_args__ = (
        PrimaryKeyConstraint("ts_code", "curve_type", "curve_term", "trade_date", name="pk_yc_cb"),
        Index("idx_yc_cb_trade_date", "trade_date"),
    )


# ---------------------------------------------------------------------------
# 国际主要指数 (index_global)
# ---------------------------------------------------------------------------
class IndexGlobal(Base):
    __tablename__ = "index_global"

    ts_code: Mapped[str] = mapped_column(String(16), comment="TS指数代码 如 SPX")
    trade_date: Mapped[date] = mapped_column(Date, comment="交易日期")
    open: Mapped[float | None] = mapped_column(Float, comment="开盘点位")
    close: Mapped[float | None] = mapped_column(Float, comment="收盘点位")
    high: Mapped[float | None] = mapped_column(Float, comment="最高点位")
    low: Mapped[float | None] = mapped_column(Float, comment="最低点位")
    pre_close: Mapped[float | None] = mapped_column(Float, comment="昨日收盘点")
    change: Mapped[float | None] = mapped_column(Float, comment="涨跌点位")
    pct_chg: Mapped[float | None] = mapped_column(Float, comment="涨跌幅")
    swing: Mapped[float | None] = mapped_column(Float, comment="振幅")
    vol: Mapped[float | None] = mapped_column(Float, comment="成交量(多数无)")

    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), onupdate=func.now())

    __table_args__ = (
        PrimaryKeyConstraint("ts_code", "trade_date", name="pk_index_global"),
        Index("idx_index_global_trade_date", "trade_date"),
    )


# ---------------------------------------------------------------------------
# 美国国债收益率曲线利率 (us_tycr)
# ---------------------------------------------------------------------------
class UsTycr(Base):
    __tablename__ = "us_tycr"

    # date 是 SQL 保留字，属性名用 trade_date，列名保持 date。
    trade_date: Mapped[date] = mapped_column("date", Date, comment="日期")
    m1: Mapped[float | None] = mapped_column(Float, comment="1月期")
    m2: Mapped[float | None] = mapped_column(Float, comment="2月期")
    m3: Mapped[float | None] = mapped_column(Float, comment="3月期")
    m6: Mapped[float | None] = mapped_column(Float, comment="6月期")
    y1: Mapped[float | None] = mapped_column(Float, comment="1年期")
    y2: Mapped[float | None] = mapped_column(Float, comment="2年期")
    y3: Mapped[float | None] = mapped_column(Float, comment="3年期")
    y5: Mapped[float | None] = mapped_column(Float, comment="5年期")
    y7: Mapped[float | None] = mapped_column(Float, comment="7年期")
    y10: Mapped[float | None] = mapped_column(Float, comment="10年期")
    y20: Mapped[float | None] = mapped_column(Float, comment="20年期")
    y30: Mapped[float | None] = mapped_column(Float, comment="30年期")

    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), onupdate=func.now())

    __table_args__ = (PrimaryKeyConstraint("date", name="pk_us_tycr"),)


# ---------------------------------------------------------------------------
# 美国国债实际收益率曲线利率 (us_trycr)
# ---------------------------------------------------------------------------
class UsTrycr(Base):
    __tablename__ = "us_trycr"

    trade_date: Mapped[date] = mapped_column("date", Date, comment="日期")
    y5: Mapped[float | None] = mapped_column(Float, comment="5年期实际利率")
    y7: Mapped[float | None] = mapped_column(Float, comment="7年期实际利率")
    y10: Mapped[float | None] = mapped_column(Float, comment="10年期实际利率")
    y20: Mapped[float | None] = mapped_column(Float, comment="20年期实际利率")
    y30: Mapped[float | None] = mapped_column(Float, comment="30年期实际利率")

    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), onupdate=func.now())

    __table_args__ = (PrimaryKeyConstraint("date", name="pk_us_trycr"),)


# ===========================================================================
# Stock data (A-shares)
# ===========================================================================


# ---------------------------------------------------------------------------
# Stock basic info (stock_basic)
# ---------------------------------------------------------------------------
class StockBasic(Base):
    __tablename__ = "stock_basic"

    ts_code: Mapped[str] = mapped_column(String(16), primary_key=True, comment="TS代码")
    symbol: Mapped[str | None] = mapped_column(String(16), comment="股票代码")
    name: Mapped[str | None] = mapped_column(String(128), comment="股票名称")
    area: Mapped[str | None] = mapped_column(String(64), comment="地域")
    industry: Mapped[str | None] = mapped_column(String(128), comment="所属行业")
    fullname: Mapped[str | None] = mapped_column(String(256), comment="股票全称")
    enname: Mapped[str | None] = mapped_column(String(256), comment="英文全称")
    cnspell: Mapped[str | None] = mapped_column(String(64), comment="拼音缩写")
    market: Mapped[str | None] = mapped_column(String(32), comment="市场类型")
    exchange: Mapped[str | None] = mapped_column(String(16), comment="交易所代码")
    curr_type: Mapped[str | None] = mapped_column(String(8), comment="交易货币")
    list_status: Mapped[str | None] = mapped_column(String(4), comment="上市状态")
    list_date: Mapped[date | None] = mapped_column(Date, comment="上市日期")
    delist_date: Mapped[date | None] = mapped_column(Date, comment="退市日期")
    is_hs: Mapped[str | None] = mapped_column(String(4), comment="沪深港通标的")
    act_name: Mapped[str | None] = mapped_column(String(128), comment="实控人名称")
    act_ent_type: Mapped[str | None] = mapped_column(String(64), comment="实控人企业性质")

    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), onupdate=func.now())

    __table_args__ = (Index("idx_stock_basic_status", "list_status"),)


# ---------------------------------------------------------------------------
# Stock daily OHLCV (daily)
# ---------------------------------------------------------------------------
class StockDaily(Base):
    __tablename__ = "daily"

    ts_code: Mapped[str] = mapped_column(String(16), comment="股票代码")
    trade_date: Mapped[date] = mapped_column(Date, comment="交易日期")
    open: Mapped[float | None] = mapped_column(Float, comment="开盘价")
    high: Mapped[float | None] = mapped_column(Float, comment="最高价")
    low: Mapped[float | None] = mapped_column(Float, comment="最低价")
    close: Mapped[float | None] = mapped_column(Float, comment="收盘价")
    pre_close: Mapped[float | None] = mapped_column(Float, comment="昨收价")
    change: Mapped[float | None] = mapped_column(Float, comment="涨跌额")
    pct_chg: Mapped[float | None] = mapped_column(Float, comment="涨跌幅(%)")
    vol: Mapped[float | None] = mapped_column(Float, comment="成交量(手)")
    amount: Mapped[float | None] = mapped_column(Float, comment="成交额(千元)")

    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), onupdate=func.now())

    __table_args__ = (
        PrimaryKeyConstraint("ts_code", "trade_date", name="pk_daily"),
        Index("idx_daily_date", "trade_date"),
    )


# ---------------------------------------------------------------------------
# Stock adjustment factor (adj_factor)
# ---------------------------------------------------------------------------
class AdjFactor(Base):
    __tablename__ = "adj_factor"

    ts_code: Mapped[str] = mapped_column(String(16), comment="股票代码")
    trade_date: Mapped[date] = mapped_column(Date, comment="交易日期")
    adj_factor: Mapped[float | None] = mapped_column(Float, comment="复权因子")

    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), onupdate=func.now())

    __table_args__ = (
        PrimaryKeyConstraint("ts_code", "trade_date", name="pk_adj_factor"),
        Index("idx_adj_factor_date", "trade_date"),
    )


# ---------------------------------------------------------------------------
# Stock daily basic indicators (daily_basic) - PE/PB/换手率/市值等
# ---------------------------------------------------------------------------
class DailyBasic(Base):
    __tablename__ = "daily_basic"

    ts_code: Mapped[str] = mapped_column(String(16), comment="TS股票代码")
    trade_date: Mapped[date] = mapped_column(Date, comment="交易日期")
    close: Mapped[float | None] = mapped_column(Float, comment="当日收盘价")
    turnover_rate: Mapped[float | None] = mapped_column(Float, comment="换手率(%)")
    turnover_rate_f: Mapped[float | None] = mapped_column(Float, comment="换手率(自由流通股)(%)")
    volume_ratio: Mapped[float | None] = mapped_column(Float, comment="量比")
    pe: Mapped[float | None] = mapped_column(Float, comment="市盈率(PE)")
    pe_ttm: Mapped[float | None] = mapped_column(Float, comment="市盈率(TTM)")
    pb: Mapped[float | None] = mapped_column(Float, comment="市净率(PB)")
    ps: Mapped[float | None] = mapped_column(Float, comment="市销率(PS)")
    ps_ttm: Mapped[float | None] = mapped_column(Float, comment="市销率(TTM)")
    dv_ratio: Mapped[float | None] = mapped_column(Float, comment="股息率(%)")
    dv_ttm: Mapped[float | None] = mapped_column(Float, comment="股息率(TTM)(%)")
    total_share: Mapped[float | None] = mapped_column(Float, comment="总股本(万股)")
    float_share: Mapped[float | None] = mapped_column(Float, comment="流通股本(万股)")
    free_share: Mapped[float | None] = mapped_column(Float, comment="自由流通股本(万股)")
    total_mv: Mapped[float | None] = mapped_column(Float, comment="总市值(万元)")
    circ_mv: Mapped[float | None] = mapped_column(Float, comment="流通市值(万元)")

    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), onupdate=func.now())

    __table_args__ = (
        PrimaryKeyConstraint("ts_code", "trade_date", name="pk_daily_basic"),
        Index("idx_daily_basic_date", "trade_date"),
    )


# ---------------------------------------------------------------------------
# Stock weekly OHLCV (weekly)
# ---------------------------------------------------------------------------
class StockWeekly(Base):
    __tablename__ = "weekly"

    ts_code: Mapped[str] = mapped_column(String(16), comment="股票代码")
    trade_date: Mapped[date] = mapped_column(Date, comment="交易日期(周线最后一日)")
    open: Mapped[float | None] = mapped_column(Float, comment="开盘价")
    high: Mapped[float | None] = mapped_column(Float, comment="最高价")
    low: Mapped[float | None] = mapped_column(Float, comment="最低价")
    close: Mapped[float | None] = mapped_column(Float, comment="收盘价")
    pre_close: Mapped[float | None] = mapped_column(Float, comment="上一周收盘价")
    change: Mapped[float | None] = mapped_column(Float, comment="涨跌额")
    pct_chg: Mapped[float | None] = mapped_column(Float, comment="涨跌幅(%)")
    vol: Mapped[float | None] = mapped_column(Float, comment="成交量(手)")
    amount: Mapped[float | None] = mapped_column(Float, comment="成交额(千元)")

    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), onupdate=func.now())

    __table_args__ = (
        PrimaryKeyConstraint("ts_code", "trade_date", name="pk_weekly"),
        Index("idx_weekly_date", "trade_date"),
    )


# ---------------------------------------------------------------------------
# Stock monthly OHLCV (monthly)
# ---------------------------------------------------------------------------
class StockMonthly(Base):
    __tablename__ = "monthly"

    ts_code: Mapped[str] = mapped_column(String(16), comment="股票代码")
    trade_date: Mapped[date] = mapped_column(Date, comment="交易日期(月线最后一日)")
    open: Mapped[float | None] = mapped_column(Float, comment="开盘价")
    high: Mapped[float | None] = mapped_column(Float, comment="最高价")
    low: Mapped[float | None] = mapped_column(Float, comment="最低价")
    close: Mapped[float | None] = mapped_column(Float, comment="收盘价")
    pre_close: Mapped[float | None] = mapped_column(Float, comment="上一月收盘价")
    change: Mapped[float | None] = mapped_column(Float, comment="涨跌额")
    pct_chg: Mapped[float | None] = mapped_column(Float, comment="涨跌幅(%)")
    vol: Mapped[float | None] = mapped_column(Float, comment="成交量(手)")
    amount: Mapped[float | None] = mapped_column(Float, comment="成交额(千元)")

    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), onupdate=func.now())

    __table_args__ = (
        PrimaryKeyConstraint("ts_code", "trade_date", name="pk_monthly"),
        Index("idx_monthly_date", "trade_date"),
    )


# ---------------------------------------------------------------------------
# Suspension / resumption info (suspend_d)
# ---------------------------------------------------------------------------
class SuspendD(Base):
    __tablename__ = "suspend_d"

    ts_code: Mapped[str] = mapped_column(String(16), comment="股票代码")
    trade_date: Mapped[date] = mapped_column(Date, comment="停复牌日期")
    suspend_timing: Mapped[str | None] = mapped_column(String(64), comment="日内停牌时间")
    suspend_type: Mapped[str | None] = mapped_column(String(4), comment="停复牌类型 S停牌 R复牌")

    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), onupdate=func.now())

    __table_args__ = (
        PrimaryKeyConstraint("ts_code", "trade_date", name="pk_suspend_d"),
        Index("idx_suspend_d_date", "trade_date"),
    )


# ---------------------------------------------------------------------------
# Historical name changes (namechange)
# ---------------------------------------------------------------------------
class NameChange(Base):
    __tablename__ = "namechange"

    ts_code: Mapped[str] = mapped_column(String(16), comment="TS代码")
    name: Mapped[str | None] = mapped_column(String(128), comment="证券名称")
    start_date: Mapped[date | None] = mapped_column(Date, comment="开始日期")
    end_date: Mapped[date | None] = mapped_column(Date, comment="结束日期")
    ann_date: Mapped[date | None] = mapped_column(Date, comment="公告日期")
    change_reason: Mapped[str | None] = mapped_column(String(256), comment="变更原因")

    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), onupdate=func.now())

    __table_args__ = (PrimaryKeyConstraint("ts_code", "start_date", name="pk_namechange"),)


# ===========================================================================
# Financial statements
# ===========================================================================


# ---------------------------------------------------------------------------
# Income statement (income)
# ---------------------------------------------------------------------------
class Income(Base):
    __tablename__ = "income"

    ts_code: Mapped[str] = mapped_column(String(16), comment="TS代码")
    ann_date: Mapped[date | None] = mapped_column(Date, comment="公告日期")
    f_ann_date: Mapped[date | None] = mapped_column(Date, comment="实际公告日期")
    end_date: Mapped[date | None] = mapped_column(Date, comment="报告期")
    report_type: Mapped[str | None] = mapped_column(String(4), comment="报表类型 1合并 2单季 3调整")
    comp_type: Mapped[str | None] = mapped_column(String(4), comment="公司类型 1一般工商业 2银行 3保险 4证券")
    end_type: Mapped[str | None] = mapped_column(String(4), comment="报告期类型")
    basic_eps: Mapped[float | None] = mapped_column(Float, comment="基本每股收益")
    diluted_eps: Mapped[float | None] = mapped_column(Float, comment="稀释每股收益")
    total_revenue: Mapped[float | None] = mapped_column(Float, comment="营业总收入(元)")
    revenue: Mapped[float | None] = mapped_column(Float, comment="营业收入(元)")
    int_income: Mapped[float | None] = mapped_column(Float, comment="利息收入")
    prem_earned: Mapped[float | None] = mapped_column(Float, comment="已赚保费")
    comm_income: Mapped[float | None] = mapped_column(Float, comment="手续费及佣金收入")
    n_commis_income: Mapped[float | None] = mapped_column(Float, comment="手续费及佣金净收入")
    n_oth_income: Mapped[float | None] = mapped_column(Float, comment="其他经营净收益")
    n_oth_b_income: Mapped[float | None] = mapped_column(Float, comment="加:其他业务净收益")
    prem_income: Mapped[float | None] = mapped_column(Float, comment="保险业务收入")
    out_prem: Mapped[float | None] = mapped_column(Float, comment="减:分出保费")
    une_prem_reser: Mapped[float | None] = mapped_column(Float, comment="提取未到期责任准备金")
    reins_income: Mapped[float | None] = mapped_column(Float, comment="其中:分保费收入")
    n_sec_tb_income: Mapped[float | None] = mapped_column(Float, comment="代理买卖证券业务净收入")
    n_sec_uw_income: Mapped[float | None] = mapped_column(Float, comment="证券承销业务净收入")
    n_asset_mg_income: Mapped[float | None] = mapped_column(Float, comment="受托客户资产管理业务净收入")
    oth_b_income: Mapped[float | None] = mapped_column(Float, comment="其他业务收入")
    fv_value_chg_gain: Mapped[float | None] = mapped_column(Float, comment="公允价值变动净收益")
    invest_income: Mapped[float | None] = mapped_column(Float, comment="投资净收益")
    ass_invest_income: Mapped[float | None] = mapped_column(
        Float, comment="其中:对联营企业和合营企业的投资收益"
    )
    forex_gain: Mapped[float | None] = mapped_column(Float, comment="汇兑净收益")
    total_cogs: Mapped[float | None] = mapped_column(Float, comment="营业总成本")
    oper_cost: Mapped[float | None] = mapped_column(Float, comment="营业成本")
    int_exp: Mapped[float | None] = mapped_column(Float, comment="利息支出")
    comm_exp: Mapped[float | None] = mapped_column(Float, comment="手续费及佣金支出")
    biz_tax_surchg: Mapped[float | None] = mapped_column(Float, comment="营业税金及附加")
    sell_exp: Mapped[float | None] = mapped_column(Float, comment="销售费用")
    admin_exp: Mapped[float | None] = mapped_column(Float, comment="管理费用")
    fin_exp: Mapped[float | None] = mapped_column(Float, comment="财务费用")
    assets_impair_loss: Mapped[float | None] = mapped_column(Float, comment="资产减值损失")
    prem_refund: Mapped[float | None] = mapped_column(Float, comment="退保金")
    compens_payout: Mapped[float | None] = mapped_column(Float, comment="赔付总支出")
    reser_insur_liab: Mapped[float | None] = mapped_column(Float, comment="提取保险责任准备金")
    div_payt: Mapped[float | None] = mapped_column(Float, comment="保户红利支出")
    reins_exp: Mapped[float | None] = mapped_column(Float, comment="分保费用")
    oper_exp: Mapped[float | None] = mapped_column(Float, comment="营业支出")
    compens_payout_refu: Mapped[float | None] = mapped_column(Float, comment="减:摊回赔付支出")
    insur_reser_refu: Mapped[float | None] = mapped_column(Float, comment="减:摊回保险责任准备金")
    reins_cost_refund: Mapped[float | None] = mapped_column(Float, comment="减:摊回分保费用")
    other_bus_cost: Mapped[float | None] = mapped_column(Float, comment="其他业务成本")
    operate_profit: Mapped[float | None] = mapped_column(Float, comment="营业利润")
    non_oper_income: Mapped[float | None] = mapped_column(Float, comment="加:营业外收入")
    non_oper_exp: Mapped[float | None] = mapped_column(Float, comment="减:营业外支出")
    nca_disploss: Mapped[float | None] = mapped_column(Float, comment="其中:非流动资产处置净损失")
    total_profit: Mapped[float | None] = mapped_column(Float, comment="利润总额")
    income_tax: Mapped[float | None] = mapped_column(Float, comment="所得税费用")
    n_income: Mapped[float | None] = mapped_column(Float, comment="净利润(含少数股东损益)")
    n_income_attr_p: Mapped[float | None] = mapped_column(Float, comment="净利润(不含少数股东损益)")
    minority_gain: Mapped[float | None] = mapped_column(Float, comment="少数股东损益")
    oth_compr_income: Mapped[float | None] = mapped_column(Float, comment="其他综合收益")
    t_compr_income: Mapped[float | None] = mapped_column(Float, comment="综合收益总额")
    compr_inc_attr_p: Mapped[float | None] = mapped_column(
        Float, comment="归属于母公司(或股东)的综合收益总额"
    )
    compr_inc_attr_m_s: Mapped[float | None] = mapped_column(Float, comment="归属于少数股东的综合收益总额")
    ebit: Mapped[float | None] = mapped_column(Float, comment="息税前利润")
    ebitda: Mapped[float | None] = mapped_column(Float, comment="息税折旧摊销前利润")
    insurance_exp: Mapped[float | None] = mapped_column(Float, comment="保险业务支出")
    undist_profit: Mapped[float | None] = mapped_column(Float, comment="年初未分配利润")
    distable_profit: Mapped[float | None] = mapped_column(Float, comment="可分配利润")
    rd_exp: Mapped[float | None] = mapped_column(Float, comment="研发费用")
    fin_exp_int_exp: Mapped[float | None] = mapped_column(Float, comment="财务费用:利息费用")
    fin_exp_int_inc: Mapped[float | None] = mapped_column(Float, comment="财务费用:利息收入")
    transfer_surplus_rese: Mapped[float | None] = mapped_column(Float, comment="盈余公积转入")
    transfer_housing_imprest: Mapped[float | None] = mapped_column(Float, comment="住房周转金转入")
    transfer_oth: Mapped[float | None] = mapped_column(Float, comment="其他转入")
    adj_lossgain: Mapped[float | None] = mapped_column(Float, comment="调整以前年度损益")
    withdra_legal_surplus: Mapped[float | None] = mapped_column(Float, comment="提取法定盈余公积")
    withdra_legal_pubfund: Mapped[float | None] = mapped_column(Float, comment="提取法定公益金")
    withdra_biz_devfund: Mapped[float | None] = mapped_column(Float, comment="提取企业发展基金")
    withdra_rese_fund: Mapped[float | None] = mapped_column(Float, comment="提取储备基金")
    withdra_oth_ersu: Mapped[float | None] = mapped_column(Float, comment="提取其他盈余公积")
    workers_welfare: Mapped[float | None] = mapped_column(Float, comment="职工奖励及福利基金")
    distr_profit_shrhder: Mapped[float | None] = mapped_column(Float, comment="分配股利/利润")
    prfshare_payable_dvd: Mapped[float | None] = mapped_column(Float, comment="应付优先股股利")
    comshare_payable_dvd: Mapped[float | None] = mapped_column(Float, comment="应付普通股股利")
    capit_comstock_div: Mapped[float | None] = mapped_column(Float, comment="转作股本的普通股股利")
    credit_impa_loss: Mapped[float | None] = mapped_column(Float, comment="信用减值损失")
    net_expo_hedging_benefits: Mapped[float | None] = mapped_column(Float, comment="净敞口套期收益")
    oth_impair_loss_assets: Mapped[float | None] = mapped_column(Float, comment="其他资产减值损失")
    total_opcost: Mapped[float | None] = mapped_column(Float, comment="营业总成本(新)")
    amodcost_fin_assets: Mapped[float | None] = mapped_column(
        Float, comment="以摊余成本计量的金融资产终止确认收益"
    )
    oth_income: Mapped[float | None] = mapped_column(Float, comment="其他收益")
    asset_disp_income: Mapped[float | None] = mapped_column(Float, comment="资产处置收益")
    continued_net_profit: Mapped[float | None] = mapped_column(Float, comment="持续经营净利润")
    end_net_profit: Mapped[float | None] = mapped_column(Float, comment="终止经营净利润")
    update_flag: Mapped[str | None] = mapped_column(String(4), comment="更新标识")

    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), onupdate=func.now())

    __table_args__ = (
        PrimaryKeyConstraint("ts_code", "end_date", "report_type", name="pk_income"),
        Index("idx_income_end_date", "end_date"),
        Index("idx_income_ann_date", "ann_date"),
    )


# ---------------------------------------------------------------------------
# Balance sheet (balancesheet)
# ---------------------------------------------------------------------------
class BalanceSheet(Base):
    __tablename__ = "balancesheet"

    ts_code: Mapped[str] = mapped_column(String(16), comment="TS代码")
    ann_date: Mapped[date | None] = mapped_column(Date, comment="公告日期")
    f_ann_date: Mapped[date | None] = mapped_column(Date, comment="实际公告日期")
    end_date: Mapped[date | None] = mapped_column(Date, comment="报告期")
    report_type: Mapped[str | None] = mapped_column(String(4), comment="报表类型")
    comp_type: Mapped[str | None] = mapped_column(String(4), comment="公司类型")
    end_type: Mapped[str | None] = mapped_column(String(4), comment="报告期类型")
    total_share: Mapped[float | None] = mapped_column(Float, comment="期末总股本")
    cap_rese: Mapped[float | None] = mapped_column(Float, comment="资本公积金")
    surplus_rese: Mapped[float | None] = mapped_column(Float, comment="盈余公积金")
    special_rese: Mapped[float | None] = mapped_column(Float, comment="专项储备")
    undistr_porfit: Mapped[float | None] = mapped_column(Float, comment="未分配利润")
    total_cur_assets: Mapped[float | None] = mapped_column(Float, comment="流动资产合计")
    total_nca: Mapped[float | None] = mapped_column(Float, comment="非流动资产合计")
    total_assets: Mapped[float | None] = mapped_column(Float, comment="资产总计")
    total_cur_liab: Mapped[float | None] = mapped_column(Float, comment="流动负债合计")
    total_ncl: Mapped[float | None] = mapped_column(Float, comment="非流动负债合计")
    total_liab: Mapped[float | None] = mapped_column(Float, comment="负债合计")
    total_hldr_eqy_exc_min_int: Mapped[float | None] = mapped_column(Float, comment="归属母公司股东权益")
    minority_int: Mapped[float | None] = mapped_column(Float, comment="少数股东权益")
    total_hldr_eqy_inc_min_int: Mapped[float | None] = mapped_column(Float, comment="股东权益合计")
    total_liab_hldr_eqy: Mapped[float | None] = mapped_column(Float, comment="负债与股东权益合计")
    update_flag: Mapped[str | None] = mapped_column(String(4), comment="更新标识")

    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), onupdate=func.now())

    __table_args__ = (
        PrimaryKeyConstraint("ts_code", "end_date", "report_type", name="pk_balancesheet"),
        Index("idx_balancesheet_end_date", "end_date"),
    )


# ---------------------------------------------------------------------------
# Cash flow statement (cashflow)
# ---------------------------------------------------------------------------
class CashFlow(Base):
    __tablename__ = "cashflow"

    ts_code: Mapped[str] = mapped_column(String(16), comment="TS代码")
    ann_date: Mapped[date | None] = mapped_column(Date, comment="公告日期")
    f_ann_date: Mapped[date | None] = mapped_column(Date, comment="实际公告日期")
    end_date: Mapped[date | None] = mapped_column(Date, comment="报告期")
    report_type: Mapped[str | None] = mapped_column(String(4), comment="报表类型")
    comp_type: Mapped[str | None] = mapped_column(String(4), comment="公司类型")
    end_type: Mapped[str | None] = mapped_column(String(4), comment="报告期类型")
    net_profit: Mapped[float | None] = mapped_column(Float, comment="净利润")
    finan_exp: Mapped[float | None] = mapped_column(Float, comment="财务费用")
    c_fr_sale_sg: Mapped[float | None] = mapped_column(Float, comment="销售商品、提供劳务收到的现金")
    recp_tax_rends: Mapped[float | None] = mapped_column(Float, comment="收到的税费返还")
    n_depos_incr_fi: Mapped[float | None] = mapped_column(Float, comment="客户存款和同业存放款项净增加额")
    n_incr_loans_cb: Mapped[float | None] = mapped_column(Float, comment="向中央银行借款净增加额")
    n_inc_borr_oth_fi: Mapped[float | None] = mapped_column(Float, comment="向其他金融机构拆入资金净增加额")
    prem_fr_orig_contr: Mapped[float | None] = mapped_column(Float, comment="收到原保险合同保费取得的现金")
    n_incr_insured_dep: Mapped[float | None] = mapped_column(Float, comment="保户储金净增加额")
    n_reinsur_prem: Mapped[float | None] = mapped_column(Float, comment="收到再保险业务现金净额")
    n_incr_disp_tfa: Mapped[float | None] = mapped_column(Float, comment="处置交易性金融资产净增加额")
    ifc_cash_incr: Mapped[float | None] = mapped_column(Float, comment="收取利息、手续费及佣金的现金")
    n_incr_disp_faas: Mapped[float | None] = mapped_column(Float, comment="处置可供出售金融资产净增加额")
    n_incr_loans_oth_bank: Mapped[float | None] = mapped_column(Float, comment="拆入资金净增加额")
    n_cap_incr_repur: Mapped[float | None] = mapped_column(Float, comment="回购业务资金净增加额")
    c_fr_oth_operate_a: Mapped[float | None] = mapped_column(Float, comment="收到其他与经营活动有关的现金")
    c_inf_fr_operate_a: Mapped[float | None] = mapped_column(Float, comment="经营活动现金流入小计")
    c_paid_goods_s: Mapped[float | None] = mapped_column(Float, comment="购买商品、接受劳务支付的现金")
    c_paid_to_for_empl: Mapped[float | None] = mapped_column(Float, comment="支付给职工以及为职工支付的现金")
    c_paid_for_taxes: Mapped[float | None] = mapped_column(Float, comment="支付的各项税费")
    n_incr_clt_loan_adv: Mapped[float | None] = mapped_column(Float, comment="客户贷款及垫款净增加额")
    n_incr_dep_cbob: Mapped[float | None] = mapped_column(Float, comment="存放央行和同业款项净增加额")
    c_pay_claims_orig_inco: Mapped[float | None] = mapped_column(
        Float, comment="支付原保险合同赔付款项的现金"
    )
    pay_handling_chrg: Mapped[float | None] = mapped_column(Float, comment="支付手续费的现金")
    pay_comm_insur_plcy: Mapped[float | None] = mapped_column(Float, comment="支付保单红利的现金")
    oth_cash_pay_oper_act: Mapped[float | None] = mapped_column(Float, comment="支付其他与经营活动有关的现金")
    st_cash_out_act: Mapped[float | None] = mapped_column(Float, comment="经营活动现金流出小计")
    n_cashflow_act: Mapped[float | None] = mapped_column(Float, comment="经营活动产生的现金流量净额")
    oth_recp_ral_inv_act: Mapped[float | None] = mapped_column(Float, comment="收到其他与投资活动有关的现金")
    c_disp_withdrwl_invest: Mapped[float | None] = mapped_column(Float, comment="收回投资收到的现金")
    c_recp_return_invest: Mapped[float | None] = mapped_column(Float, comment="取得投资收益收到的现金")
    n_recp_disp_fiolta: Mapped[float | None] = mapped_column(
        Float, comment="处置固定资产、无形资产和其他长期资产收回的现金净额"
    )
    n_recp_disp_sobu: Mapped[float | None] = mapped_column(
        Float, comment="处置子公司及其他营业单位收到的现金净额"
    )
    stot_inflows_inv_act: Mapped[float | None] = mapped_column(Float, comment="投资活动现金流入小计")
    c_pay_acq_const_fiolta: Mapped[float | None] = mapped_column(
        Float, comment="购建固定资产、无形资产和其他长期资产支付的现金"
    )
    c_paid_invest: Mapped[float | None] = mapped_column(Float, comment="投资支付的现金")
    n_disp_subs_oth_biz: Mapped[float | None] = mapped_column(
        Float, comment="取得子公司及其他营业单位支付的现金净额"
    )
    oth_pay_ral_inv_act: Mapped[float | None] = mapped_column(Float, comment="支付其他与投资活动有关的现金")
    stot_out_inv_act: Mapped[float | None] = mapped_column(Float, comment="投资活动现金流出小计")
    n_cashflow_inv_act: Mapped[float | None] = mapped_column(Float, comment="投资活动产生的现金流量净额")
    c_recp_borrow: Mapped[float | None] = mapped_column(Float, comment="取得借款收到的现金")
    proc_issue_bonds: Mapped[float | None] = mapped_column(Float, comment="发行债券收到的现金")
    oth_cash_recp_ral_fnc_act: Mapped[float | None] = mapped_column(
        Float, comment="收到其他与筹资活动有关的现金"
    )
    stot_cash_in_fnc_act: Mapped[float | None] = mapped_column(Float, comment="筹资活动现金流入小计")
    free_cashflow: Mapped[float | None] = mapped_column(Float, comment="企业自由现金流量")
    c_prepay_amt_borr: Mapped[float | None] = mapped_column(Float, comment="偿还债务支付的现金")
    c_pay_dist_dpcp_int_exp: Mapped[float | None] = mapped_column(
        Float, comment="分配股利、利润或偿付利息支付的现金"
    )
    incl_dvd_profit_paid_sc_ms: Mapped[float | None] = mapped_column(
        Float, comment="其中:子公司支付给少数股东的股利、利润"
    )
    oth_cashpay_ral_fnc_act: Mapped[float | None] = mapped_column(
        Float, comment="支付其他与筹资活动有关的现金"
    )
    stot_cashout_fnc_act: Mapped[float | None] = mapped_column(Float, comment="筹资活动现金流出小计")
    n_cash_flows_fnc_act: Mapped[float | None] = mapped_column(Float, comment="筹资活动产生的现金流量净额")
    eff_fx_flu_cash: Mapped[float | None] = mapped_column(Float, comment="汇率变动对现金的影响")
    n_incr_cash_cash_equ: Mapped[float | None] = mapped_column(Float, comment="现金及现金等价物净增加额")
    c_cash_equ_beg_period: Mapped[float | None] = mapped_column(Float, comment="期初现金及现金等价物余额")
    c_cash_equ_end_period: Mapped[float | None] = mapped_column(Float, comment="期末现金及现金等价物余额")
    c_recp_cap_contrib: Mapped[float | None] = mapped_column(Float, comment="吸收投资收到的现金")
    incl_cash_rec_saims: Mapped[float | None] = mapped_column(
        Float, comment="其中:子公司吸收少数股东投资收到的现金"
    )
    update_flag: Mapped[str | None] = mapped_column(String(4), comment="更新标识")

    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), onupdate=func.now())

    __table_args__ = (
        PrimaryKeyConstraint("ts_code", "end_date", "report_type", name="pk_cashflow"),
        Index("idx_cashflow_end_date", "end_date"),
    )


# ---------------------------------------------------------------------------
# Financial indicators (fina_indicator) - ROE/ROA/gross_margin etc.
# ---------------------------------------------------------------------------
class FinaIndicator(Base):
    __tablename__ = "fina_indicator"

    ts_code: Mapped[str] = mapped_column(String(16), comment="TS代码")
    ann_date: Mapped[date | None] = mapped_column(Date, comment="公告日期")
    end_date: Mapped[date | None] = mapped_column(Date, comment="报告期")
    eps: Mapped[float | None] = mapped_column(Float, comment="基本每股收益")
    dt_eps: Mapped[float | None] = mapped_column(Float, comment="稀释每股收益")
    total_revenue_ps: Mapped[float | None] = mapped_column(Float, comment="每股营业总收入")
    revenue_ps: Mapped[float | None] = mapped_column(Float, comment="每股营业收入")
    capital_rese_ps: Mapped[float | None] = mapped_column(Float, comment="每股资本公积")
    surplus_rese_ps: Mapped[float | None] = mapped_column(Float, comment="每股盈余公积")
    undist_profit_ps: Mapped[float | None] = mapped_column(Float, comment="每股未分配利润")
    extra_item: Mapped[float | None] = mapped_column(Float, comment="非经常性损益")
    profit_dedt: Mapped[float | None] = mapped_column(
        Float, comment="扣除非经常性损益后的净利润（扣非净利润）"
    )
    gross_margin: Mapped[float | None] = mapped_column(Float, comment="毛利")
    current_ratio: Mapped[float | None] = mapped_column(Float, comment="流动比率")
    quick_ratio: Mapped[float | None] = mapped_column(Float, comment="速动比率")
    assets_turn: Mapped[float | None] = mapped_column(Float, comment="总资产周转率")
    inv_turn: Mapped[float | None] = mapped_column(Float, comment="存货周转率")
    ar_turn: Mapped[float | None] = mapped_column(Float, comment="应收账款周转率")
    debt_to_assets: Mapped[float | None] = mapped_column(Float, comment="资产负债率")
    roe: Mapped[float | None] = mapped_column(Float, comment="净资产收益率")
    roe_dt: Mapped[float | None] = mapped_column(Float, comment="净资产收益率(扣除非经常损益)")
    roa: Mapped[float | None] = mapped_column(Float, comment="总资产报酬率")
    roa_dp: Mapped[float | None] = mapped_column(Float, comment="总资产净利润")
    roic: Mapped[float | None] = mapped_column(Float, comment="投入资本回报率")
    profit_to_op: Mapped[float | None] = mapped_column(Float, comment="营业利润率")
    profit_to_gr: Mapped[float | None] = mapped_column(Float, comment="总利润同比增长率")
    netprofit_margin: Mapped[float | None] = mapped_column(Float, comment="净利润/营业总收入")
    saleexp_to_gr: Mapped[float | None] = mapped_column(Float, comment="销售费用/营业总收入")
    adminexp_of_gr: Mapped[float | None] = mapped_column(Float, comment="管理费用/营业总收入")
    finaexp_of_gr: Mapped[float | None] = mapped_column(Float, comment="财务费用/营业总收入")
    impai_ttm: Mapped[float | None] = mapped_column(Float, comment="资产减值损失/营业总收入")
    gc_of_gr: Mapped[float | None] = mapped_column(Float, comment="营业总成本/营业总收入")
    op_of_gr: Mapped[float | None] = mapped_column(Float, comment="营业利润/营业总收入")
    ebit_of_gr: Mapped[float | None] = mapped_column(Float, comment="息税前利润/营业总收入")
    roe_yearly: Mapped[float | None] = mapped_column(Float, comment="净资产收益率(年化)")
    roa2_yearly: Mapped[float | None] = mapped_column(Float, comment="总资产报酬率(年化)")
    roe_avg: Mapped[float | None] = mapped_column(Float, comment="净资产收益率(平均)")
    opincome_of_ebt: Mapped[float | None] = mapped_column(Float, comment="经营活动净收益/利润总额")
    investincome_of_ebt: Mapped[float | None] = mapped_column(Float, comment="价值变动净收益/利润总额")
    n_op_profit_of_ebt: Mapped[float | None] = mapped_column(Float, comment="营业外收支净额/利润总额")
    tax_to_ebt: Mapped[float | None] = mapped_column(Float, comment="所得税/利润总额")
    dtprofit_to_profit: Mapped[float | None] = mapped_column(Float, comment="扣除非经常损益后的净利润/净利润")
    salescash_to_or: Mapped[float | None] = mapped_column(
        Float, comment="销售商品提供劳务收到的现金/营业收入"
    )
    ocf_to_or: Mapped[float | None] = mapped_column(Float, comment="经营活动产生的现金流量净额/营业收入")
    ocf_to_opincome: Mapped[float | None] = mapped_column(
        Float, comment="经营活动产生的现金流量净额/经营活动净收益"
    )
    capitalized_to_da: Mapped[float | None] = mapped_column(Float, comment="资本支出/折旧和摊销")
    debt_to_eqt: Mapped[float | None] = mapped_column(Float, comment="权益乘数")
    ocfps: Mapped[float | None] = mapped_column(Float, comment="每股经营活动产生的现金流量净额")
    basic_eps_yoy: Mapped[float | None] = mapped_column(Float, comment="基本每股收益同比增长率(%)")
    dt_eps_yoy: Mapped[float | None] = mapped_column(Float, comment="稀释每股收益同比增长率(%)")
    cfps_yoy: Mapped[float | None] = mapped_column(
        Float, comment="每股经营活动产生的现金流量净额同比增长率(%)"
    )
    op_yoy: Mapped[float | None] = mapped_column(Float, comment="营业利润同比增长率(%)")
    ebt_yoy: Mapped[float | None] = mapped_column(Float, comment="利润总额同比增长率(%)")
    netprofit_yoy: Mapped[float | None] = mapped_column(Float, comment="归属母公司股东的净利润同比增长率(%)")
    dt_netprofit_yoy: Mapped[float | None] = mapped_column(Float, comment="扣非净利润同比增长率(%)")
    or_yoy: Mapped[float | None] = mapped_column(Float, comment="营业总收入同比增长率(%)")
    equity_yoy: Mapped[float | None] = mapped_column(Float, comment="净资产同比增长率")
    tr_yoy: Mapped[float | None] = mapped_column(Float, comment="总资产同比增长率")
    q_gr_qoq: Mapped[float | None] = mapped_column(Float, comment="营业总收入环比增长率(%)")
    q_op_qoq: Mapped[float | None] = mapped_column(Float, comment="营业利润环比增长率(%)")
    update_flag: Mapped[str | None] = mapped_column(String(4), comment="更新标识")

    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), onupdate=func.now())

    __table_args__ = (
        PrimaryKeyConstraint("ts_code", "end_date", name="pk_fina_indicator"),
        Index("idx_fina_indicator_date", "end_date"),
    )


# ---------------------------------------------------------------------------
# Earnings forecast (forecast)
# ---------------------------------------------------------------------------
class Forecast(Base):
    __tablename__ = "forecast"

    ts_code: Mapped[str] = mapped_column(String(16), comment="TS股票代码")
    ann_date: Mapped[date | None] = mapped_column(Date, comment="公告日期")
    end_date: Mapped[date | None] = mapped_column(Date, comment="报告期")
    type: Mapped[str | None] = mapped_column(String(16), comment="预告类型")
    p_change_min: Mapped[float | None] = mapped_column(Float, comment="预告净利润变动幅度下限(%)")
    p_change_max: Mapped[float | None] = mapped_column(Float, comment="预告净利润变动幅度上限(%)")
    net_profit_min: Mapped[float | None] = mapped_column(Float, comment="预告净利润下限(万元)")
    net_profit_max: Mapped[float | None] = mapped_column(Float, comment="预告净利润上限(万元)")
    last_parent_net: Mapped[float | None] = mapped_column(Float, comment="上年同期归属母公司净利润")
    first_ann_date: Mapped[date | None] = mapped_column(Date, comment="首次公告日")
    summary: Mapped[str | None] = mapped_column(Text, comment="业绩预告摘要")
    change_reason: Mapped[str | None] = mapped_column(Text, comment="业绩变动原因")

    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), onupdate=func.now())

    __table_args__ = (
        PrimaryKeyConstraint("ts_code", "end_date", "ann_date", name="pk_forecast"),
        Index("idx_forecast_end_date", "end_date"),
    )


# ---------------------------------------------------------------------------
# Earnings express (express)
# ---------------------------------------------------------------------------
class Express(Base):
    __tablename__ = "express"

    ts_code: Mapped[str] = mapped_column(String(16), comment="TS股票代码")
    ann_date: Mapped[date | None] = mapped_column(Date, comment="公告日期")
    end_date: Mapped[date | None] = mapped_column(Date, comment="报告期")
    revenue: Mapped[float | None] = mapped_column(Float, comment="营业收入(元)")
    operate_profit: Mapped[float | None] = mapped_column(Float, comment="营业利润(元)")
    total_profit: Mapped[float | None] = mapped_column(Float, comment="利润总额(元)")
    n_income: Mapped[float | None] = mapped_column(Float, comment="净利润(元)")
    total_assets: Mapped[float | None] = mapped_column(Float, comment="总资产(元)")
    total_hldr_eqy_exc_min_int: Mapped[float | None] = mapped_column(
        Float, comment="归属于母公司股东权益合计(元)"
    )
    diluted_eps: Mapped[float | None] = mapped_column(Float, comment="稀释每股收益(元)")
    diluted_roe: Mapped[float | None] = mapped_column(Float, comment="净资产收益率(%)")
    yoy_net_profit: Mapped[float | None] = mapped_column(Float, comment="归属母公司股东净利润同比(%)")
    bps: Mapped[float | None] = mapped_column(Float, comment="每股净资产(元)")
    yoy_sales: Mapped[float | None] = mapped_column(Float, comment="营业总收入同比(%)")
    yoy_op: Mapped[float | None] = mapped_column(Float, comment="营业利润同比(%)")
    yoy_tp: Mapped[float | None] = mapped_column(Float, comment="利润总额同比(%)")
    yoy_dedu_np: Mapped[float | None] = mapped_column(Float, comment="归属母公司股东净利润环比(%)")
    yoy_eps: Mapped[float | None] = mapped_column(Float, comment="基本每股收益同比(%)")
    yoy_roe: Mapped[float | None] = mapped_column(Float, comment="净资产收益率同比(%)")
    growth_assets: Mapped[float | None] = mapped_column(Float, comment="总资产同比(%)")

    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), onupdate=func.now())

    __table_args__ = (
        PrimaryKeyConstraint("ts_code", "end_date", name="pk_express"),
        Index("idx_express_end_date", "end_date"),
    )


# ---------------------------------------------------------------------------
# Stock dividend & split (dividend)
# ---------------------------------------------------------------------------
class StockDividend(Base):
    __tablename__ = "dividend"

    ts_code: Mapped[str] = mapped_column(String(16), comment="TS代码")
    end_date: Mapped[date | None] = mapped_column(Date, comment="分红年度")
    ann_date: Mapped[date | None] = mapped_column(Date, comment="预案公告日")
    div_proc: Mapped[str | None] = mapped_column(String(32), comment="实施进度")
    stk_div: Mapped[float | None] = mapped_column(Float, comment="每股送转")
    stk_bo_rate: Mapped[float | None] = mapped_column(Float, comment="每股送股比例")
    stk_co_rate: Mapped[float | None] = mapped_column(Float, comment="每股转增比例")
    cash_div: Mapped[float | None] = mapped_column(Float, comment="每股分红(税后)")
    cash_div_tax: Mapped[float | None] = mapped_column(Float, comment="每股分红(税前)")
    record_date: Mapped[date | None] = mapped_column(Date, comment="股权登记日")
    ex_date: Mapped[date | None] = mapped_column(Date, comment="除权除息日")
    pay_date: Mapped[date | None] = mapped_column(Date, comment="派息日")
    div_listdate: Mapped[date | None] = mapped_column(Date, comment="红股上市日")
    imp_ann_date: Mapped[date | None] = mapped_column(Date, comment="实施公告日")
    base_date: Mapped[date | None] = mapped_column(Date, comment="基准日")
    base_share: Mapped[float | None] = mapped_column(Float, comment="基准股本(万股)")

    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), onupdate=func.now())

    __table_args__ = (
        PrimaryKeyConstraint("ts_code", "end_date", "div_proc", name="pk_stock_dividend"),
        Index("idx_stock_div_ex_date", "ex_date"),
    )


# ===========================================================================
# Market reference data
# ===========================================================================


# ---------------------------------------------------------------------------
# North-bound capital flow (moneyflow_hsgt) - 沪深港通资金流向
# ---------------------------------------------------------------------------
class MoneyflowHsgt(Base):
    __tablename__ = "moneyflow_hsgt"

    trade_date: Mapped[date] = mapped_column(Date, comment="交易日期")
    ggt_ss: Mapped[float | None] = mapped_column(Float, comment="港股通(沪)流入(亿)")
    ggt_sz: Mapped[float | None] = mapped_column(Float, comment="港股通(深)流入(亿)")
    hgt: Mapped[float | None] = mapped_column(Float, comment="沪股通流入(亿)")
    sgt: Mapped[float | None] = mapped_column(Float, comment="深股通流入(亿)")
    north_money: Mapped[float | None] = mapped_column(Float, comment="北向流入(亿)")
    south_money: Mapped[float | None] = mapped_column(Float, comment="南向流入(亿)")

    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), onupdate=func.now())

    __table_args__ = (PrimaryKeyConstraint("trade_date", name="pk_moneyflow_hsgt"),)


# ---------------------------------------------------------------------------
# Margin trading summary (margin)
# ---------------------------------------------------------------------------
class Margin(Base):
    __tablename__ = "margin"

    trade_date: Mapped[date] = mapped_column(Date, comment="交易日期")
    exchange_id: Mapped[str] = mapped_column(String(8), comment="交易所 SSE/SZSE")
    rzye: Mapped[float | None] = mapped_column(Float, comment="融资余额(元)")
    rzmre: Mapped[float | None] = mapped_column(Float, comment="融资买入额(元)")
    rzche: Mapped[float | None] = mapped_column(Float, comment="融资偿还额(元)")
    rqye: Mapped[float | None] = mapped_column(Float, comment="融券余额(元)")
    rqmcl: Mapped[float | None] = mapped_column(Float, comment="融券卖出量(股)")
    rzrqye: Mapped[float | None] = mapped_column(Float, comment="融资融券余额(元)")
    rqyl: Mapped[float | None] = mapped_column(Float, comment="融券余量(股)")

    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), onupdate=func.now())

    __table_args__ = (
        PrimaryKeyConstraint("trade_date", "exchange_id", name="pk_margin"),
        Index("idx_margin_date", "trade_date"),
    )


# ---------------------------------------------------------------------------
# Margin trading detail per stock (margin_detail)
# ---------------------------------------------------------------------------
class MarginDetail(Base):
    __tablename__ = "margin_detail"

    trade_date: Mapped[date] = mapped_column(Date, comment="交易日期")
    ts_code: Mapped[str] = mapped_column(String(16), comment="TS股票代码")
    name: Mapped[str | None] = mapped_column(String(128), comment="股票名称")
    rzye: Mapped[float | None] = mapped_column(Float, comment="融资余额(元)")
    rqye: Mapped[float | None] = mapped_column(Float, comment="融券余额(元)")
    rzmre: Mapped[float | None] = mapped_column(Float, comment="融资买入额(元)")
    rqyl: Mapped[float | None] = mapped_column(Float, comment="融券余量(股)")
    rzche: Mapped[float | None] = mapped_column(Float, comment="融资偿还额(元)")
    rqchl: Mapped[float | None] = mapped_column(Float, comment="融券偿还量(股)")
    rqmcl: Mapped[float | None] = mapped_column(Float, comment="融券卖出量(股)")
    rzrqye: Mapped[float | None] = mapped_column(Float, comment="融资融券余额(元)")

    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), onupdate=func.now())

    __table_args__ = (
        PrimaryKeyConstraint("trade_date", "ts_code", name="pk_margin_detail"),
        Index("idx_margin_detail_ts_code", "ts_code", "trade_date"),
    )


# ---------------------------------------------------------------------------
# Stock daily factor / technical indicators (stk_factor)
# ---------------------------------------------------------------------------
class StkFactor(Base):
    __tablename__ = "stk_factor"

    ts_code: Mapped[str] = mapped_column(String(16), comment="股票代码")
    trade_date: Mapped[date] = mapped_column(Date, comment="交易日期")
    close: Mapped[float | None] = mapped_column(Float, comment="收盘价")
    open: Mapped[float | None] = mapped_column(Float, comment="开盘价")
    high: Mapped[float | None] = mapped_column(Float, comment="最高价")
    low: Mapped[float | None] = mapped_column(Float, comment="最低价")
    pre_close: Mapped[float | None] = mapped_column(Float, comment="昨收价")
    change: Mapped[float | None] = mapped_column(Float, comment="涨跌额")
    pct_change: Mapped[float | None] = mapped_column(Float, comment="涨跌幅(%)")
    vol: Mapped[float | None] = mapped_column(Float, comment="成交量(手)")
    amount: Mapped[float | None] = mapped_column(Float, comment="成交额(千元)")
    adj_factor: Mapped[float | None] = mapped_column(Float, comment="复权因子")
    macd_dif: Mapped[float | None] = mapped_column(Float, comment="MACD DIF")
    macd_dea: Mapped[float | None] = mapped_column(Float, comment="MACD DEA")
    macd: Mapped[float | None] = mapped_column(Float, comment="MACD 柱")
    kdj_k: Mapped[float | None] = mapped_column(Float, comment="KDJ K值")
    kdj_d: Mapped[float | None] = mapped_column(Float, comment="KDJ D值")
    kdj_j: Mapped[float | None] = mapped_column(Float, comment="KDJ J值")
    rsi_6: Mapped[float | None] = mapped_column(Float, comment="RSI 6日")
    rsi_12: Mapped[float | None] = mapped_column(Float, comment="RSI 12日")
    rsi_24: Mapped[float | None] = mapped_column(Float, comment="RSI 24日")
    boll_upper: Mapped[float | None] = mapped_column(Float, comment="BOLL 上轨")
    boll_mid: Mapped[float | None] = mapped_column(Float, comment="BOLL 中轨")
    boll_lower: Mapped[float | None] = mapped_column(Float, comment="BOLL 下轨")
    cci: Mapped[float | None] = mapped_column(Float, comment="CCI 顺势指标")

    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), onupdate=func.now())

    __table_args__ = (
        PrimaryKeyConstraint("ts_code", "trade_date", name="pk_stk_factor"),
        Index("idx_stk_factor_date", "trade_date"),
    )


# ---------------------------------------------------------------------------
# Broker monthly gold-stock picks (broker_recommend)
# ---------------------------------------------------------------------------
class BrokerRecommend(Base):
    __tablename__ = "broker_recommend"

    month: Mapped[str] = mapped_column(String(8), comment="推荐月份 YYYYMM")
    broker: Mapped[str] = mapped_column(String(128), comment="券商/研究所名称")
    ts_code: Mapped[str] = mapped_column(String(16), comment="股票代码")
    name: Mapped[str | None] = mapped_column(String(128), comment="股票名称")

    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), onupdate=func.now())

    __table_args__ = (PrimaryKeyConstraint("month", "broker", "ts_code", name="pk_broker_recommend"),)


# ---------------------------------------------------------------------------
# Note: The Tushare ``stk_factor_pro`` endpoint returns daily technical
# indicators (261 columns) — NOT broker profit forecasts. Broker consensus
# data is not available via current Tushare endpoints. Use ``stk_factor``
# for the core technical indicator subset (34 columns).
# ---------------------------------------------------------------------------


# ===========================================================================
# Public fund
# ===========================================================================


# ---------------------------------------------------------------------------
# Fund company info (fund_company) - 公募基金公司
# ---------------------------------------------------------------------------
class FundCompany(Base):
    __tablename__ = "fund_company"

    name: Mapped[str] = mapped_column(String(128), comment="基金公司名称")
    shortname: Mapped[str | None] = mapped_column(String(64), comment="简称")
    province: Mapped[str | None] = mapped_column(String(32), comment="省份")
    city: Mapped[str | None] = mapped_column(String(32), comment="城市")
    address: Mapped[str | None] = mapped_column(String(256), comment="注册地址")
    phone: Mapped[str | None] = mapped_column(String(128), comment="电话")
    office: Mapped[str | None] = mapped_column(String(256), comment="办公地址")
    website: Mapped[str | None] = mapped_column(String(256), comment="公司网址")
    chairman: Mapped[str | None] = mapped_column(String(64), comment="法人代表")
    manager: Mapped[str | None] = mapped_column(String(64), comment="总经理")
    reg_capital: Mapped[float | None] = mapped_column(Float, comment="注册资本(万)")
    setup_date: Mapped[date | None] = mapped_column(Date, comment="成立日期")
    end_date: Mapped[date | None] = mapped_column(Date, comment="公司终止日期")
    employees: Mapped[float | None] = mapped_column(Float, comment="员工总数")
    main_business: Mapped[str | None] = mapped_column(Text, comment="主要产品及业务")
    org_code: Mapped[str | None] = mapped_column(String(32), comment="组织机构代码")
    credit_code: Mapped[str | None] = mapped_column(String(32), comment="统一社会信用代码")

    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), onupdate=func.now())

    __table_args__ = (PrimaryKeyConstraint("name", name="pk_fund_company"),)


# ===========================================================================
# Futures data
# ===========================================================================


# ---------------------------------------------------------------------------
# Futures contract list (fut_basic)
# ---------------------------------------------------------------------------
class FutBasic(Base):
    __tablename__ = "fut_basic"

    ts_code: Mapped[str] = mapped_column(String(16), primary_key=True, comment="合约代码")
    symbol: Mapped[str | None] = mapped_column(String(16), comment="交易标识")
    exchange: Mapped[str | None] = mapped_column(String(16), comment="交易所")
    name: Mapped[str | None] = mapped_column(String(64), comment="合约名称")
    fut_code: Mapped[str | None] = mapped_column(String(16), comment="合约产品标识")
    multiplier: Mapped[float | None] = mapped_column(Float, comment="合约乘数")
    trade_unit: Mapped[str | None] = mapped_column(String(32), comment="交易计量单位")
    per_unit: Mapped[float | None] = mapped_column(Float, comment="交易单位")
    quote_unit: Mapped[str | None] = mapped_column(String(32), comment="报价单位")
    quote_unit_desc: Mapped[str | None] = mapped_column(String(128), comment="最小报价单位说明")
    d_mode_desc: Mapped[str | None] = mapped_column(String(64), comment="交割方式说明")
    list_date: Mapped[date | None] = mapped_column(Date, comment="上市日期")
    delist_date: Mapped[date | None] = mapped_column(Date, comment="退市日期")
    d_month: Mapped[str | None] = mapped_column(String(8), comment="连续合约交割月份")
    last_ddate: Mapped[date | None] = mapped_column(Date, comment="最后交易日")

    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), onupdate=func.now())

    __table_args__ = (Index("idx_fut_basic_exchange", "exchange"),)


# ---------------------------------------------------------------------------
# Futures daily OHLCV (fut_daily)
# ---------------------------------------------------------------------------
class FutDaily(Base):
    __tablename__ = "fut_daily"

    ts_code: Mapped[str] = mapped_column(String(16), comment="合约代码")
    trade_date: Mapped[date] = mapped_column(Date, comment="交易日期")
    pre_close: Mapped[float | None] = mapped_column(Float, comment="昨收价")
    pre_settle: Mapped[float | None] = mapped_column(Float, comment="昨结价")
    open: Mapped[float | None] = mapped_column(Float, comment="开盘价")
    high: Mapped[float | None] = mapped_column(Float, comment="最高价")
    low: Mapped[float | None] = mapped_column(Float, comment="最低价")
    close: Mapped[float | None] = mapped_column(Float, comment="收盘价")
    settle: Mapped[float | None] = mapped_column(Float, comment="结算价")
    change1: Mapped[float | None] = mapped_column(Float, comment="涨跌1 收盘-昨收")
    change2: Mapped[float | None] = mapped_column(Float, comment="涨跌2 结算-昨结")
    vol: Mapped[float | None] = mapped_column(Float, comment="成交量(手)")
    amount: Mapped[float | None] = mapped_column(Float, comment="成交额(万元)")
    oi: Mapped[float | None] = mapped_column(Float, comment="持仓量(手)")
    oi_chg: Mapped[float | None] = mapped_column(Float, comment="持仓量变化")
    delv_settle: Mapped[float | None] = mapped_column(Float, comment="交割结算价")

    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), onupdate=func.now())

    __table_args__ = (
        PrimaryKeyConstraint("ts_code", "trade_date", name="pk_fut_daily"),
        Index("idx_fut_daily_date", "trade_date"),
    )


# ---------------------------------------------------------------------------
# Futures daily holding ranking (fut_holding)
# ---------------------------------------------------------------------------
class FutHolding(Base):
    __tablename__ = "fut_holding"

    trade_date: Mapped[date] = mapped_column(Date, comment="交易日期")
    symbol: Mapped[str] = mapped_column(String(16), comment="品种代码")
    broker: Mapped[str] = mapped_column(String(128), comment="期货公司会员简称")
    vol: Mapped[int | None] = mapped_column(Integer, comment="成交量")
    vol_chg: Mapped[int | None] = mapped_column(Integer, comment="成交量变化")
    long_hld: Mapped[int | None] = mapped_column(Integer, comment="持买仓量")
    long_chg: Mapped[int | None] = mapped_column(Integer, comment="持买仓量变化")
    short_hld: Mapped[int | None] = mapped_column(Integer, comment="持卖仓量")
    short_chg: Mapped[int | None] = mapped_column(Integer, comment="持卖仓量变化")
    exchange: Mapped[str | None] = mapped_column(String(8), comment="交易所")

    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), onupdate=func.now())

    __table_args__ = (
        PrimaryKeyConstraint("trade_date", "symbol", "broker", name="pk_fut_holding"),
        Index("idx_fut_holding_symbol", "symbol", "trade_date"),
    )


# ---------------------------------------------------------------------------
# Futures warehouse receipts (fut_wsr) - 仓单日报
# ---------------------------------------------------------------------------
class FutWsr(Base):
    __tablename__ = "fut_wsr"

    trade_date: Mapped[date] = mapped_column(Date, comment="交易日期")
    symbol: Mapped[str] = mapped_column(String(16), comment="产品代码")
    fut_name: Mapped[str | None] = mapped_column(String(64), comment="合约名称")
    warehouse: Mapped[str] = mapped_column(String(128), comment="仓库名称")
    pre_vol: Mapped[int | None] = mapped_column(Integer, comment="上期仓单量")
    vol: Mapped[int | None] = mapped_column(Integer, comment="仓单数量")
    vol_chg: Mapped[int | None] = mapped_column(Integer, comment="仓单变化量")
    unit: Mapped[str | None] = mapped_column(String(32), comment="单位")

    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), onupdate=func.now())

    __table_args__ = (
        PrimaryKeyConstraint("trade_date", "symbol", "fut_name", "warehouse", name="pk_fut_wsr"),
        Index("idx_fut_wsr_symbol", "symbol", "trade_date"),
    )


# ---------------------------------------------------------------------------
# Futures settlement parameters (fut_settle)
# ---------------------------------------------------------------------------
class FutSettle(Base):
    __tablename__ = "fut_settle"

    ts_code: Mapped[str] = mapped_column(String(16), comment="合约代码")
    trade_date: Mapped[date] = mapped_column(Date, comment="交易日期")
    settle: Mapped[float | None] = mapped_column(Float, comment="结算价")
    trading_fee: Mapped[float | None] = mapped_column(Float, comment="交易手续费率(‱)")
    delivery_fee: Mapped[float | None] = mapped_column(Float, comment="交割手续费(元/手)")
    b_hedging: Mapped[str | None] = mapped_column(String(4), comment="买套保交易保证金率")
    s_hedging: Mapped[str | None] = mapped_column(String(4), comment="卖套保交易保证金率")
    long_td: Mapped[str | None] = mapped_column(String(4), comment="买投机交易保证金率")
    short_td: Mapped[str | None] = mapped_column(String(4), comment="卖投机交易保证金率")
    td_hedging: Mapped[str | None] = mapped_column(String(4), comment="套保交易保证金率")
    exchange: Mapped[str | None] = mapped_column(String(8), comment="交易所")

    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), onupdate=func.now())

    __table_args__ = (
        PrimaryKeyConstraint("ts_code", "trade_date", name="pk_fut_settle"),
        Index("idx_fut_settle_date", "trade_date"),
    )
