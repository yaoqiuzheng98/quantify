"""Pluggable market-data sources for the backtest engine.

The engine consumes a single canonical OHLCV DataFrame schema regardless of the
underlying instrument type. Each :class:`MarketDataSource` knows how to load one
asset class (ETF, stock, ...) from its own tables and emit that schema, including
a per-bar ``split_ratio`` (份额折算/送转股 share-count multiplier on the day it
takes effect, 1.0 otherwise).

Canonical OHLCV columns (one row per code per trading day)::

    ts_code, date, open, high, low, close, volume, amount,
    pre_close, pct_chg, adj_factor, split_ratio

A :class:`CompositeDataSource` routes each requested code to the right concrete
source by :func:`quantify.backtest.codes.classify_asset`, so a single backtest
can freely mix ETFs and stocks.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date

import pandas as pd
from sqlalchemy import select

from quantify.database.engine import session_scope
from quantify.database.models import (
    AdjFactor,
    EtfAdjFactor,
    EtfDaily,
    EtfDividend,
    EtfNav,
    IndexDaily,
    StockDaily,
    StockDividend,
)

from .codes import classify_asset

OHLCV_COLUMNS = [
    "ts_code",
    "date",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "amount",
    "pre_close",
    "pct_chg",
    "adj_factor",
    "split_ratio",
]


@dataclass(frozen=True)
class DividendEvent:
    """A cash dividend entitlement (per-share cash paid on ``pay_date``)."""

    ts_code: str
    record_date: date
    pay_date: date
    div_cash: float


def _empty_ohlcv() -> pd.DataFrame:
    return pd.DataFrame(columns=OHLCV_COLUMNS)


def _finalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce dtypes and sort a freshly loaded OHLCV frame to the canonical shape."""
    if df.empty:
        return _empty_ohlcv()
    df["date"] = pd.to_datetime(df["date"])
    df["adj_factor"] = pd.to_numeric(df["adj_factor"], errors="coerce").fillna(1.0)
    if "split_ratio" not in df.columns:
        df["split_ratio"] = 1.0
    df["split_ratio"] = pd.to_numeric(df["split_ratio"], errors="coerce").fillna(1.0)
    df = df.sort_values(["ts_code", "date"]).reset_index(drop=True)
    return df[OHLCV_COLUMNS]


class MarketDataSource(ABC):
    """Loads one asset class into the engine's canonical schema."""

    @abstractmethod
    def load_ohlcv(self, ts_codes: list[str], start_date: date, end_date: date) -> pd.DataFrame:
        """Return canonical OHLCV rows (incl. ``adj_factor``/``split_ratio``)."""

    @abstractmethod
    def load_dividends(self, ts_codes: list[str], start_date: date, end_date: date) -> list[DividendEvent]:
        """Return cash-dividend entitlement events within the window."""

    @abstractmethod
    def load_benchmark(self, ts_code: str, start_date: date, end_date: date) -> pd.DataFrame | None:
        """Return a benchmark close*adj series; carries ``attrs['base_value']``."""


def _compute_split_ratios_from_nav(group: pd.DataFrame) -> list[float]:
    """Per-bar ETF share-split ratios for one code (1.0 on non-split days).

    A split (份额折算) is detected on the day ``adj_factor`` jumps relative to the
    previous trading day. The magnitude is taken from the jump in
    ``accum_nav/unit_nav`` (记为 au), which reflects pure share folding and excludes
    the cash-dividend contamination present in ``adj_factor``. Because ``fund_nav``
    and ``fund_daily`` can be misaligned by one day, the ratio compares the au value
    on the jump day against a stable au value two trading days earlier. Falls back to
    the ``adj_factor`` jump ratio when net-asset-value data is missing.
    """
    n = len(group)
    ratios = [1.0] * n
    adj = group["adj_factor"].tolist()
    unit = group["unit_nav"].tolist()
    accum = group["accum_nav"].tolist()

    def _au(i: int) -> float | None:
        try:
            u = float(unit[i])
            a = float(accum[i])
        except (TypeError, ValueError):
            return None
        if not (u > 0) or not (a > 0):
            return None
        return a / u

    for i in range(1, n):
        prev_adj = float(adj[i - 1]) if adj[i - 1] else 1.0
        cur_adj = float(adj[i]) if adj[i] else 1.0
        if prev_adj <= 0 or abs(cur_adj - prev_adj) < 1e-9:
            continue
        au_now = _au(i)
        au_ref = _au(max(0, i - 2))
        if au_now is not None and au_ref is not None and au_ref > 0:
            ratio = au_now / au_ref
        else:
            ratio = cur_adj / prev_adj
        if ratio > 0 and abs(ratio - 1.0) > 1e-6:
            ratios[i] = ratio
    return ratios


class EtfDataSource(MarketDataSource):
    """ETF/fund data from ``fund_daily`` / ``fund_adj`` / ``fund_nav`` / ``fund_div``."""

    def load_ohlcv(self, ts_codes: list[str], start_date: date, end_date: date) -> pd.DataFrame:
        if not ts_codes:
            return _empty_ohlcv()
        start_str = start_date.strftime("%Y%m%d")
        end_str = end_date.strftime("%Y%m%d")
        with session_scope() as sess:
            rows = sess.execute(
                select(
                    EtfDaily.ts_code,
                    EtfDaily.trade_date,
                    EtfDaily.open,
                    EtfDaily.high,
                    EtfDaily.low,
                    EtfDaily.close,
                    EtfDaily.vol,
                    EtfDaily.amount,
                    EtfDaily.pre_close,
                    EtfDaily.pct_chg,
                    EtfAdjFactor.adj_factor,
                    EtfNav.unit_nav,
                    EtfNav.accum_nav,
                )
                .outerjoin(
                    EtfAdjFactor,
                    (EtfAdjFactor.ts_code == EtfDaily.ts_code)
                    & (EtfAdjFactor.trade_date == EtfDaily.trade_date),
                )
                .outerjoin(
                    EtfNav,
                    (EtfNav.ts_code == EtfDaily.ts_code) & (EtfNav.nav_date == EtfDaily.trade_date),
                )
                .where(EtfDaily.ts_code.in_(ts_codes))
                .where(EtfDaily.trade_date >= start_str)
                .where(EtfDaily.trade_date <= end_str)
                .order_by(EtfDaily.ts_code, EtfDaily.trade_date)
            ).all()
        if not rows:
            return _empty_ohlcv()
        df = pd.DataFrame(
            rows,
            columns=[
                "ts_code",
                "date",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "amount",
                "pre_close",
                "pct_chg",
                "adj_factor",
                "unit_nav",
                "accum_nav",
            ],
        )
        df["date"] = pd.to_datetime(df["date"])
        df["adj_factor"] = pd.to_numeric(df["adj_factor"], errors="coerce").fillna(1.0)
        df["unit_nav"] = pd.to_numeric(df["unit_nav"], errors="coerce")
        df["accum_nav"] = pd.to_numeric(df["accum_nav"], errors="coerce")
        df = df.sort_values(["ts_code", "date"]).reset_index(drop=True)
        parts = []
        for _code, group in df.groupby("ts_code"):
            group = group.copy()
            group["split_ratio"] = _compute_split_ratios_from_nav(group)
            parts.append(group)
        df = pd.concat(parts, ignore_index=True)
        return _finalize_ohlcv(df)

    def load_dividends(self, ts_codes: list[str], start_date: date, end_date: date) -> list[DividendEvent]:
        if not ts_codes:
            return []
        with session_scope() as sess:
            rows = sess.execute(
                select(
                    EtfDividend.ts_code,
                    EtfDividend.record_date,
                    EtfDividend.pay_date,
                    EtfDividend.div_cash,
                )
                .where(EtfDividend.ts_code.in_(ts_codes))
                .where(EtfDividend.record_date >= start_date)
                .where(EtfDividend.record_date <= end_date)
                .where(EtfDividend.pay_date >= start_date)
                .where(EtfDividend.pay_date <= end_date)
                .order_by(EtfDividend.record_date, EtfDividend.pay_date, EtfDividend.ts_code)
            ).all()
        return [
            DividendEvent(
                ts_code=row.ts_code,
                record_date=row.record_date,
                pay_date=row.pay_date,
                div_cash=float(row.div_cash),
            )
            for row in rows
            if row.ts_code and row.record_date and row.pay_date and row.div_cash
        ]

    def load_benchmark(self, ts_code: str, start_date: date, end_date: date) -> pd.DataFrame | None:
        return _load_etf_benchmark(ts_code, start_date, end_date)


def _build_benchmark_df(rows: list, previous_rows: list, value_fn) -> pd.DataFrame | None:
    """Assemble a benchmark DataFrame from query rows using ``value_fn(row) -> float``."""
    if not rows:
        return None
    benchmark_df = pd.DataFrame(
        [{"date": row.trade_date, "value": value_fn(row), "daily_value": value_fn(row)} for row in rows]
    )
    if previous_rows:
        base_value = value_fn(previous_rows[0])
        benchmark_df.attrs["base_value"] = base_value
        benchmark_df.attrs["daily_base_value"] = base_value
    return benchmark_df


def _load_etf_benchmark(ts_code: str, start_date: date, end_date: date) -> pd.DataFrame | None:
    def _query(before_start: bool) -> list:
        stmt = (
            select(EtfDaily.trade_date, EtfDaily.close, EtfAdjFactor.adj_factor)
            .outerjoin(
                EtfAdjFactor,
                (EtfAdjFactor.ts_code == EtfDaily.ts_code) & (EtfAdjFactor.trade_date == EtfDaily.trade_date),
            )
            .where(EtfDaily.ts_code == ts_code)
        )
        if before_start:
            stmt = stmt.where(EtfDaily.trade_date < start_date).order_by(EtfDaily.trade_date.desc()).limit(1)
        else:
            stmt = (
                stmt.where(EtfDaily.trade_date >= start_date)
                .where(EtfDaily.trade_date <= end_date)
                .order_by(EtfDaily.trade_date)
            )
        with session_scope() as sess:
            return list(sess.execute(stmt).all())

    def _value(row) -> float:
        close = float(row.close)
        factor = float(row.adj_factor) if row.adj_factor is not None else 1.0
        return close * factor

    return _build_benchmark_df(_query(False), _query(True), _value)


class StockDataSource(MarketDataSource):
    """A-share stock data from ``daily`` / ``adj_factor`` / ``dividend``.

    Unlike ETFs there is no NAV. Share-count changes on the ex-dividend day come
    from the explicit ``stk_div`` (每股送转) field rather than an ``adj_factor`` jump,
    because the adj_factor bundles in the cash dividend and would over-state the
    pure share multiplier (e.g. 1.2169 vs the true 1.2 for a 10送2). Cash dividends
    are handled separately via :class:`DividendEvent` using the after-tax ``cash_div``.
    """

    def load_ohlcv(self, ts_codes: list[str], start_date: date, end_date: date) -> pd.DataFrame:
        if not ts_codes:
            return _empty_ohlcv()
        start_str = start_date.strftime("%Y%m%d")
        end_str = end_date.strftime("%Y%m%d")
        with session_scope() as sess:
            rows = sess.execute(
                select(
                    StockDaily.ts_code,
                    StockDaily.trade_date,
                    StockDaily.open,
                    StockDaily.high,
                    StockDaily.low,
                    StockDaily.close,
                    StockDaily.vol,
                    StockDaily.amount,
                    StockDaily.pre_close,
                    StockDaily.pct_chg,
                    AdjFactor.adj_factor,
                )
                .outerjoin(
                    AdjFactor,
                    (AdjFactor.ts_code == StockDaily.ts_code)
                    & (AdjFactor.trade_date == StockDaily.trade_date),
                )
                .where(StockDaily.ts_code.in_(ts_codes))
                .where(StockDaily.trade_date >= start_str)
                .where(StockDaily.trade_date <= end_str)
                .order_by(StockDaily.ts_code, StockDaily.trade_date)
            ).all()
        if not rows:
            return _empty_ohlcv()
        df = pd.DataFrame(
            rows,
            columns=[
                "ts_code",
                "date",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "amount",
                "pre_close",
                "pct_chg",
                "adj_factor",
            ],
        )
        df["date"] = pd.to_datetime(df["date"])
        df["adj_factor"] = pd.to_numeric(df["adj_factor"], errors="coerce").fillna(1.0)
        df = df.sort_values(["ts_code", "date"]).reset_index(drop=True)
        df["split_ratio"] = self._split_ratios(df, start_date, end_date)
        return _finalize_ohlcv(df)

    def _bonus_share_map(
        self, ts_codes: list[str], start_date: date, end_date: date
    ) -> dict[tuple[str, date], float]:
        """{(ts_code, ex_date): stk_div} for executed 送转 events in the window."""
        with session_scope() as sess:
            rows = sess.execute(
                select(StockDividend.ts_code, StockDividend.ex_date, StockDividend.stk_div)
                .where(StockDividend.ts_code.in_(ts_codes))
                .where(StockDividend.div_proc.like("实施%"))
                .where(StockDividend.ex_date >= start_date)
                .where(StockDividend.ex_date <= end_date)
                .where(StockDividend.stk_div > 0)
            ).all()
        out: dict[tuple[str, date], float] = {}
        for row in rows:
            if row.ex_date is None or row.stk_div is None:
                continue
            # Same (code, ex_date) should be unique among 实施 rows; sum defensively.
            out[(row.ts_code, row.ex_date)] = out.get((row.ts_code, row.ex_date), 0.0) + float(row.stk_div)
        return out

    def _split_ratios(self, df: pd.DataFrame, start_date: date, end_date: date) -> list[float]:
        codes = df["ts_code"].unique().tolist()
        bonus = self._bonus_share_map(codes, start_date, end_date)
        if not bonus:
            return [1.0] * len(df)
        ratios: list[float] = []
        for row in df.itertuples(index=False):
            d = row.date.date() if hasattr(row.date, "date") else row.date
            stk_div = bonus.get((row.ts_code, d))
            ratios.append(1.0 + stk_div if stk_div else 1.0)
        return ratios

    def load_dividends(self, ts_codes: list[str], start_date: date, end_date: date) -> list[DividendEvent]:
        if not ts_codes:
            return []
        with session_scope() as sess:
            rows = sess.execute(
                select(
                    StockDividend.ts_code,
                    StockDividend.record_date,
                    StockDividend.pay_date,
                    StockDividend.cash_div,
                )
                .where(StockDividend.ts_code.in_(ts_codes))
                .where(StockDividend.div_proc.like("实施%"))
                .where(StockDividend.record_date >= start_date)
                .where(StockDividend.record_date <= end_date)
                .where(StockDividend.pay_date >= start_date)
                .where(StockDividend.pay_date <= end_date)
                .where(StockDividend.cash_div > 0)
                .order_by(StockDividend.record_date, StockDividend.pay_date, StockDividend.ts_code)
            ).all()
        return [
            DividendEvent(
                ts_code=row.ts_code,
                record_date=row.record_date,
                pay_date=row.pay_date,
                div_cash=float(row.cash_div),
            )
            for row in rows
            if row.ts_code and row.record_date and row.pay_date and row.cash_div
        ]

    def load_benchmark(self, ts_code: str, start_date: date, end_date: date) -> pd.DataFrame | None:
        def _query(before_start: bool) -> list:
            stmt = (
                select(StockDaily.trade_date, StockDaily.close, AdjFactor.adj_factor)
                .outerjoin(
                    AdjFactor,
                    (AdjFactor.ts_code == StockDaily.ts_code)
                    & (AdjFactor.trade_date == StockDaily.trade_date),
                )
                .where(StockDaily.ts_code == ts_code)
            )
            if before_start:
                stmt = (
                    stmt.where(StockDaily.trade_date < start_date)
                    .order_by(StockDaily.trade_date.desc())
                    .limit(1)
                )
            else:
                stmt = (
                    stmt.where(StockDaily.trade_date >= start_date)
                    .where(StockDaily.trade_date <= end_date)
                    .order_by(StockDaily.trade_date)
                )
            with session_scope() as sess:
                return list(sess.execute(stmt).all())

        def _value(row) -> float:
            close = float(row.close)
            factor = float(row.adj_factor) if row.adj_factor is not None else 1.0
            return close * factor

        return _build_benchmark_df(_query(False), _query(True), _value)


class IndexDataSource(MarketDataSource):
    """Index data from ``index_daily``. No adjustment, splits or dividends.

    Indices are typically only used as a benchmark, but ``load_ohlcv`` is provided
    for completeness so an index could in principle be charted alongside holdings.
    """

    def load_ohlcv(self, ts_codes: list[str], start_date: date, end_date: date) -> pd.DataFrame:
        if not ts_codes:
            return _empty_ohlcv()
        with session_scope() as sess:
            rows = sess.execute(
                select(
                    IndexDaily.ts_code,
                    IndexDaily.trade_date,
                    IndexDaily.open,
                    IndexDaily.high,
                    IndexDaily.low,
                    IndexDaily.close,
                    IndexDaily.vol,
                    IndexDaily.amount,
                    IndexDaily.pre_close,
                    IndexDaily.pct_chg,
                )
                .where(IndexDaily.ts_code.in_(ts_codes))
                .where(IndexDaily.trade_date >= start_date)
                .where(IndexDaily.trade_date <= end_date)
                .order_by(IndexDaily.ts_code, IndexDaily.trade_date)
            ).all()
        if not rows:
            return _empty_ohlcv()
        df = pd.DataFrame(
            rows,
            columns=[
                "ts_code",
                "date",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "amount",
                "pre_close",
                "pct_chg",
            ],
        )
        df["adj_factor"] = 1.0
        df["split_ratio"] = 1.0
        return _finalize_ohlcv(df)

    def load_dividends(self, ts_codes: list[str], start_date: date, end_date: date) -> list[DividendEvent]:
        return []

    def load_benchmark(self, ts_code: str, start_date: date, end_date: date) -> pd.DataFrame | None:
        def _query(before_start: bool) -> list:
            stmt = select(IndexDaily.trade_date, IndexDaily.close).where(IndexDaily.ts_code == ts_code)
            if before_start:
                stmt = (
                    stmt.where(IndexDaily.trade_date < start_date)
                    .order_by(IndexDaily.trade_date.desc())
                    .limit(1)
                )
            else:
                stmt = (
                    stmt.where(IndexDaily.trade_date >= start_date)
                    .where(IndexDaily.trade_date <= end_date)
                    .order_by(IndexDaily.trade_date)
                )
            with session_scope() as sess:
                return list(sess.execute(stmt).all())

        return _build_benchmark_df(_query(False), _query(True), lambda row: float(row.close))


class CompositeDataSource(MarketDataSource):
    """Routes each code to the concrete source matching its asset class.

    A single backtest may mix ETFs, stocks and (for benchmarks) indices; this
    splits the requested universe by :func:`classify_asset`, delegates to the
    right source, and concatenates the canonical frames back together.
    """

    def __init__(self) -> None:
        self._sources: dict[str, MarketDataSource] = {
            "etf": EtfDataSource(),
            "stock": StockDataSource(),
            "index": IndexDataSource(),
        }

    def _bucket(self, ts_codes: list[str]) -> dict[str, list[str]]:
        buckets: dict[str, list[str]] = {}
        for code in ts_codes:
            buckets.setdefault(classify_asset(code), []).append(code)
        return buckets

    def load_ohlcv(self, ts_codes: list[str], start_date: date, end_date: date) -> pd.DataFrame:
        frames = []
        for asset, codes in self._bucket(ts_codes).items():
            part = self._sources[asset].load_ohlcv(codes, start_date, end_date)
            if not part.empty:
                frames.append(part)
        if not frames:
            return _empty_ohlcv()
        df = pd.concat(frames, ignore_index=True)
        return df.sort_values(["ts_code", "date"]).reset_index(drop=True)

    def load_dividends(self, ts_codes: list[str], start_date: date, end_date: date) -> list[DividendEvent]:
        events: list[DividendEvent] = []
        for asset, codes in self._bucket(ts_codes).items():
            events.extend(self._sources[asset].load_dividends(codes, start_date, end_date))
        return events

    def load_benchmark(self, ts_code: str, start_date: date, end_date: date) -> pd.DataFrame | None:
        return self._sources[classify_asset(ts_code)].load_benchmark(ts_code, start_date, end_date)
