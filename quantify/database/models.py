"""SQLAlchemy ORM models.

Currently focuses on ETF and industry tables sourced from Tushare Pro:
    - fund_basic   -> etf_basic
    - fund_daily   -> etf_daily
    - fund_nav     -> etf_nav
    - fund_adj     -> etf_adj_factor
    - fund_div     -> etf_dividend
    - fund_share   -> etf_share
    - fund_portfolio -> etf_portfolio
    - fund_manager -> etf_manager
    - index_classify -> sw_industry_classify
    - index_member_all -> sw_industry_member
    - sw_daily -> sw_industry_daily
    - ci_index_member -> citic_industry_member
    - ci_daily -> citic_industry_daily
    - saved strategies -> strategy

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
# ETF basic info (fund_basic, market='E')
# ---------------------------------------------------------------------------
class EtfBasic(Base):
    __tablename__ = "etf_basic"

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
# ETF daily quotes
# ---------------------------------------------------------------------------
class EtfDaily(Base):
    __tablename__ = "etf_daily"

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
        PrimaryKeyConstraint("ts_code", "trade_date", name="pk_etf_daily"),
        Index("idx_etf_daily_date", "trade_date"),
        Index("idx_etf_daily_trade_code_amount", "trade_date", "ts_code", "amount"),
    )


# ---------------------------------------------------------------------------
# ETF NAV
# ---------------------------------------------------------------------------
class EtfNav(Base):
    __tablename__ = "etf_nav"

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
        PrimaryKeyConstraint("ts_code", "nav_date", name="pk_etf_nav"),
        Index("idx_etf_nav_nav_date", "nav_date"),
    )


# ---------------------------------------------------------------------------
# ETF adjustment factor (复权因子)
# 来源: fund_adj
# 用途: 后复权价 = close × adj_factor
#       前复权价 = close × adj_factor / 最新adj_factor
# 注意: 每次分红/拆分后历史因子会追溯更新，因此全量回填时需覆盖历史记录
# ---------------------------------------------------------------------------
class EtfAdjFactor(Base):
    __tablename__ = "etf_adj_factor"

    ts_code: Mapped[str] = mapped_column(String(16), comment="基金代码")
    trade_date: Mapped[date] = mapped_column(Date, comment="交易日期")
    adj_factor: Mapped[float | None] = mapped_column(Float, comment="复权因子")

    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), onupdate=func.now())

    __table_args__ = (
        PrimaryKeyConstraint("ts_code", "trade_date", name="pk_etf_adj"),
        Index("idx_etf_adj_trade_date", "trade_date"),
    )


# ---------------------------------------------------------------------------
# ETF dividend
# ---------------------------------------------------------------------------
class EtfDividend(Base):
    __tablename__ = "etf_dividend"

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
        PrimaryKeyConstraint("ts_code", "base_date", name="pk_etf_div"),
        Index("idx_etf_div_ex_date", "ex_date"),
    )


# ---------------------------------------------------------------------------
# ETF share (规模/份额变动)
# ---------------------------------------------------------------------------
class EtfShare(Base):
    __tablename__ = "etf_share"

    ts_code: Mapped[str] = mapped_column(String(16), comment="基金代码")
    trade_date: Mapped[date] = mapped_column(Date, comment="变动日期")
    fd_share: Mapped[float | None] = mapped_column(Float, comment="基金份额(万份)")
    fund_type: Mapped[str | None] = mapped_column(String(64), comment="基金类型")
    market: Mapped[str | None] = mapped_column(String(4), comment="E场内 O场外")

    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), onupdate=func.now())

    __table_args__ = (
        PrimaryKeyConstraint("ts_code", "trade_date", name="pk_etf_share"),
        Index("idx_etf_share_trade_date", "trade_date"),
    )


# ---------------------------------------------------------------------------
# ETF portfolio (披露持仓)
# ---------------------------------------------------------------------------
class EtfPortfolio(Base):
    __tablename__ = "etf_portfolio"

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
        PrimaryKeyConstraint("ts_code", "end_date", "symbol", name="pk_etf_portfolio"),
        Index("idx_etf_portfolio_symbol", "symbol", "end_date"),
        Index("idx_etf_portfolio_end_date", "end_date"),
    )


# ---------------------------------------------------------------------------
# ETF manager
# ---------------------------------------------------------------------------
class EtfManager(Base):
    __tablename__ = "etf_manager"

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

    __table_args__ = (Index("uq_etf_manager", "ts_code", "name", "begin_date", unique=True),)


# ---------------------------------------------------------------------------
# SW industry classification (index_classify, src='SW2021')
# ---------------------------------------------------------------------------
class SwIndustryClassify(Base):
    __tablename__ = "sw_industry_classify"

    index_code: Mapped[str] = mapped_column(String(16), comment="申万行业指数代码")
    src: Mapped[str] = mapped_column(String(16), comment="分类版本，如 SW2021")
    industry_name: Mapped[str | None] = mapped_column(String(128), comment="行业名称")
    parent_code: Mapped[str | None] = mapped_column(String(16), comment="父级行业代码")
    level: Mapped[str | None] = mapped_column(String(8), comment="行业级别 L1/L2/L3")
    industry_code: Mapped[str | None] = mapped_column(String(16), comment="行业代码")
    is_pub: Mapped[str | None] = mapped_column(String(4), comment="是否发布指数")

    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), onupdate=func.now())

    __table_args__ = (
        PrimaryKeyConstraint("src", "index_code", name="pk_sw_industry_classify"),
        Index("idx_sw_industry_classify_level", "src", "level", "is_pub"),
    )


# ---------------------------------------------------------------------------
# SW industry members (index_member_all)
# ---------------------------------------------------------------------------
class SwIndustryMember(Base):
    __tablename__ = "sw_industry_member"

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
        Index("uq_sw_industry_member", "ts_code", "l3_code", "in_date", unique=True),
        Index("idx_sw_industry_member_l1", "l1_code", "is_new"),
        Index("idx_sw_industry_member_l3", "l3_code", "is_new"),
    )


# ---------------------------------------------------------------------------
# SW industry daily quotes (sw_daily)
# ---------------------------------------------------------------------------
class SwIndustryDaily(Base):
    __tablename__ = "sw_industry_daily"

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
        PrimaryKeyConstraint("ts_code", "trade_date", name="pk_sw_industry_daily"),
        Index("idx_sw_industry_daily_trade_date", "trade_date"),
    )


# ---------------------------------------------------------------------------
# CITIC industry members (ci_index_member)
# ---------------------------------------------------------------------------
class CiticIndustryMember(Base):
    __tablename__ = "citic_industry_member"

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
        Index("uq_citic_industry_member", "ts_code", "l3_code", "in_date", unique=True),
        Index("idx_citic_industry_member_l1", "l1_code", "is_new"),
        Index("idx_citic_industry_member_l3", "l3_code", "is_new"),
    )


# ---------------------------------------------------------------------------
# CITIC industry daily quotes (ci_daily)
# ---------------------------------------------------------------------------
class CiticIndustryDaily(Base):
    __tablename__ = "citic_industry_daily"

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
        PrimaryKeyConstraint("ts_code", "trade_date", name="pk_citic_industry_daily"),
        Index("idx_citic_industry_daily_trade_date", "trade_date"),
    )


# ---------------------------------------------------------------------------
# Trade calendar (trade_cal) - authoritative exchange open/close dates
# ---------------------------------------------------------------------------
class TradeCalendar(Base):
    __tablename__ = "trade_calendar"

    exchange: Mapped[str] = mapped_column(String(8), comment="交易所 SSE/SZSE 等")
    cal_date: Mapped[date] = mapped_column(Date, comment="日历日期")
    is_open: Mapped[int | None] = mapped_column(Integer, comment="是否交易 1开市 0休市")
    pretrade_date: Mapped[date | None] = mapped_column(Date, comment="上一交易日")

    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), onupdate=func.now())

    __table_args__ = (
        PrimaryKeyConstraint("exchange", "cal_date", name="pk_trade_calendar"),
        Index("idx_trade_calendar_open", "exchange", "is_open", "cal_date"),
    )
