"""SQLAlchemy ORM models.

Currently focuses on ETF (fund market='E') tables sourced from Tushare Pro:
    - fund_basic   -> etf_basic
    - fund_daily   -> etf_daily
    - fund_nav     -> etf_nav
    - fund_adj     -> etf_adj_factor
    - fund_div     -> etf_dividend
    - fund_share   -> etf_share
    - fund_portfolio -> etf_portfolio
    - fund_manager -> etf_manager

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
    PrimaryKeyConstraint,
    String,
    Text,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """Project-wide declarative base."""


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

    updated_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now()
    )


# ---------------------------------------------------------------------------
# ETF daily quotes
# ---------------------------------------------------------------------------
class EtfDaily(Base):
    __tablename__ = "etf_daily"

    ts_code: Mapped[str] = mapped_column(String(16))
    trade_date: Mapped[date] = mapped_column(Date)
    pre_close: Mapped[float | None] = mapped_column(Float)
    open: Mapped[float | None] = mapped_column(Float)
    high: Mapped[float | None] = mapped_column(Float)
    low: Mapped[float | None] = mapped_column(Float)
    close: Mapped[float | None] = mapped_column(Float)
    change: Mapped[float | None] = mapped_column(Float)
    pct_chg: Mapped[float | None] = mapped_column(Float)
    vol: Mapped[float | None] = mapped_column(Float, comment="成交量(手)")
    amount: Mapped[float | None] = mapped_column(Float, comment="成交额(千元)")

    updated_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now()
    )

    __table_args__ = (
        PrimaryKeyConstraint("ts_code", "trade_date", name="pk_etf_daily"),
        Index("idx_etf_daily_date", "trade_date"),
    )


# ---------------------------------------------------------------------------
# ETF NAV
# ---------------------------------------------------------------------------
class EtfNav(Base):
    __tablename__ = "etf_nav"

    ts_code: Mapped[str] = mapped_column(String(16))
    nav_date: Mapped[date] = mapped_column(Date, comment="净值日期")
    ann_date: Mapped[date | None] = mapped_column(Date, comment="公告日期")
    unit_nav: Mapped[float | None] = mapped_column(Float, comment="单位净值")
    accum_nav: Mapped[float | None] = mapped_column(Float, comment="累计净值")
    accum_div: Mapped[float | None] = mapped_column(Float, comment="累计分红")
    net_asset: Mapped[float | None] = mapped_column(Float, comment="资产净值")
    total_netasset: Mapped[float | None] = mapped_column(Float, comment="合计资产净值")
    adj_nav: Mapped[float | None] = mapped_column(Float, comment="复权净值")
    update_flag: Mapped[str | None] = mapped_column(String(4))

    updated_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now()
    )

    __table_args__ = (
        PrimaryKeyConstraint("ts_code", "nav_date", name="pk_etf_nav"),
        Index("idx_etf_nav_nav_date", "nav_date"),
    )


# ---------------------------------------------------------------------------
# ETF adjustment factor
# ---------------------------------------------------------------------------
class EtfAdjFactor(Base):
    __tablename__ = "etf_adj_factor"

    ts_code: Mapped[str] = mapped_column(String(16))
    trade_date: Mapped[date] = mapped_column(Date)
    adj_factor: Mapped[float | None] = mapped_column(Float)

    updated_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now()
    )

    __table_args__ = (
        PrimaryKeyConstraint("ts_code", "trade_date", name="pk_etf_adj"),
    )


# ---------------------------------------------------------------------------
# ETF dividend
# ---------------------------------------------------------------------------
class EtfDividend(Base):
    __tablename__ = "etf_dividend"

    ts_code: Mapped[str] = mapped_column(String(16))
    ann_date: Mapped[date | None] = mapped_column(Date, comment="公告日")
    ex_date: Mapped[date | None] = mapped_column(Date, comment="除息日")
    pay_date: Mapped[date | None] = mapped_column(Date, comment="派息日")
    record_date: Mapped[date | None] = mapped_column(Date, comment="登记日")
    base_date: Mapped[date | None] = mapped_column(Date, comment="基准日")
    div_proc: Mapped[str | None] = mapped_column(String(32), comment="分红方案进度")
    base_share: Mapped[float | None] = mapped_column(Float, comment="基准份额(万份)")
    net_asset: Mapped[float | None] = mapped_column(Float)
    total_netasset: Mapped[float | None] = mapped_column(Float)
    div_cash: Mapped[float | None] = mapped_column(Float, comment="每股派息")

    updated_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now()
    )

    __table_args__ = (
        # ann_date / ex_date may be NULL in source data -> use COALESCE-like
        # composite key with safe fallbacks. ex_date is the most reliable.
        PrimaryKeyConstraint("ts_code", "ex_date", "base_date", name="pk_etf_div"),
        Index("idx_etf_div_ann", "ts_code", "ann_date"),
    )


# ---------------------------------------------------------------------------
# ETF share (规模/份额变动)
# ---------------------------------------------------------------------------
class EtfShare(Base):
    __tablename__ = "etf_share"

    ts_code: Mapped[str] = mapped_column(String(16))
    trade_date: Mapped[date] = mapped_column(Date)
    fd_share: Mapped[float | None] = mapped_column(Float, comment="基金份额(万份)")

    updated_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now()
    )

    __table_args__ = (PrimaryKeyConstraint("ts_code", "trade_date", name="pk_etf_share"),)


# ---------------------------------------------------------------------------
# ETF portfolio (披露持仓)
# ---------------------------------------------------------------------------
class EtfPortfolio(Base):
    __tablename__ = "etf_portfolio"

    ts_code: Mapped[str] = mapped_column(String(16))
    end_date: Mapped[date] = mapped_column(Date, comment="截止日期")
    symbol: Mapped[str] = mapped_column(String(16), comment="持仓标的代码")
    ann_date: Mapped[date | None] = mapped_column(Date)
    mkv: Mapped[float | None] = mapped_column(Float, comment="持仓市值")
    amount: Mapped[float | None] = mapped_column(Float, comment="持仓数量(股)")
    stk_mkv_ratio: Mapped[float | None] = mapped_column(Float, comment="占股票市值比")
    stk_float_ratio: Mapped[float | None] = mapped_column(Float, comment="占流通股本比")

    updated_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now()
    )

    __table_args__ = (
        PrimaryKeyConstraint("ts_code", "end_date", "symbol", name="pk_etf_portfolio"),
        Index("idx_etf_portfolio_symbol", "symbol", "end_date"),
    )


# ---------------------------------------------------------------------------
# ETF manager
# ---------------------------------------------------------------------------
class EtfManager(Base):
    __tablename__ = "etf_manager"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    ts_code: Mapped[str] = mapped_column(String(16), index=True)
    ann_date: Mapped[date | None] = mapped_column(Date)
    name: Mapped[str | None] = mapped_column(String(64))
    gender: Mapped[str | None] = mapped_column(String(4))
    birth_year: Mapped[str | None] = mapped_column(String(8))
    edu: Mapped[str | None] = mapped_column(String(32))
    nationality: Mapped[str | None] = mapped_column(String(32))
    begin_date: Mapped[date | None] = mapped_column(Date)
    end_date: Mapped[date | None] = mapped_column(Date)
    resume: Mapped[str | None] = mapped_column(Text)

    updated_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now()
    )

    __table_args__ = (
        Index("uq_etf_manager", "ts_code", "name", "begin_date", unique=True),
    )
