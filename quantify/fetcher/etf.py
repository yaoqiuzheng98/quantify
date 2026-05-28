"""ETF data fetcher: pulls all ETF-related Tushare datasets into MySQL.

Tushare endpoints used (market = 'E' for exchange-traded funds):
    fund_basic, fund_daily, fund_nav, fund_adj,
    fund_div, fund_share, fund_portfolio, fund_manager

Usage
-----
>>> from quantify.fetcher.etf import EtfFetcher
>>> EtfFetcher().fetch_all()
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Iterable

import pandas as pd
from sqlalchemy import func, select

from quantify.database.engine import session_scope
from quantify.database.models import (
    EtfAdjFactor,
    EtfBasic,
    EtfDaily,
    EtfDividend,
    EtfManager,
    EtfNav,
    EtfPortfolio,
    EtfShare,
)
from quantify.database.upsert import upsert_dataframe
from quantify.tushare_client.client import TushareClient, get_client
from quantify.utils.logger import log


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
DATE_COLUMNS = {
    "trade_date",
    "end_date",
    "nav_date",
    "ann_date",
    "found_date",
    "due_date",
    "list_date",
    "issue_date",
    "delist_date",
    "purc_startdate",
    "redm_startdate",
    "ex_date",
    "pay_date",
    "record_date",
    "base_date",
    "begin_date",
}


def _normalize_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Convert Tushare 'YYYYMMDD' string date columns into python dates."""
    if df is None or df.empty:
        return df
    for col in df.columns:
        if col in DATE_COLUMNS:
            df[col] = pd.to_datetime(df[col], format="%Y%m%d", errors="coerce").dt.date
    return df


def _today_str() -> str:
    return datetime.now().strftime("%Y%m%d")


# ---------------------------------------------------------------------------
# Fetcher
# ---------------------------------------------------------------------------
@dataclass
class FetchSummary:
    name: str
    rows: int


class EtfFetcher:
    """Pull every ETF-related dataset from Tushare into MySQL."""

    DEFAULT_START_DATE = "20000101"

    def __init__(self, client: TushareClient | None = None) -> None:
        self.client = client or get_client()

    # ------------------------------------------------------------------
    # Top-level orchestration
    # ------------------------------------------------------------------
    def fetch_all(
        self,
        *,
        incremental: bool = True,
        ts_codes: Iterable[str] | None = None,
        skip: Iterable[str] | None = None,
    ) -> list[FetchSummary]:
        """Run all sub-fetchers in order.

        Parameters
        ----------
        incremental:
            If True, time-series fetchers query the max date already stored
            and only request data afterwards.
        ts_codes:
            If provided, restrict the universe to these codes (otherwise read
            all from ``etf_basic``).
        skip:
            Names of stages to skip, e.g. ``{"portfolio", "manager"}``.
        """
        skip = set(skip or [])
        results: list[FetchSummary] = []

        # 1. Basic info first - everything else depends on its ts_codes.
        if "basic" not in skip:
            n = self.fetch_basic()
            results.append(FetchSummary("basic", n))

        codes = list(ts_codes) if ts_codes else self._load_universe()
        log.info(f"ETF universe size: {len(codes)}")

        stage_methods = [
            ("daily", self.fetch_daily),
            ("nav", self.fetch_nav),
            ("adj", self.fetch_adj),
            ("dividend", self.fetch_dividend),
            ("share", self.fetch_share),
            ("portfolio", self.fetch_portfolio),
            ("manager", self.fetch_manager),
        ]
        for name, method in stage_methods:
            if name in skip:
                log.info(f"[skip] {name}")
                continue
            n = method(ts_codes=codes, incremental=incremental)
            results.append(FetchSummary(name, n))

        log.info("=== ETF fetch summary ===")
        for r in results:
            log.info(f"  {r.name:<10s}: {r.rows} rows")
        return results

    # ------------------------------------------------------------------
    # Universe
    # ------------------------------------------------------------------
    def _load_universe(self) -> list[str]:
        with session_scope() as sess:
            rows = (
                sess.execute(
                    select(EtfBasic.ts_code).where(EtfBasic.status != "D")
                )
                .scalars()
                .all()
            )
        return list(rows)

    # ------------------------------------------------------------------
    # 1) fund_basic
    # ------------------------------------------------------------------
    def fetch_basic(self) -> int:
        """Pull listed + delisted + issuing ETFs."""
        log.info("Fetching fund_basic (market=E) ...")
        frames = []
        for status in ("L", "D", "I"):
            df = self.client.call("fund_basic", market="E", status=status)
            log.debug(f"fund_basic status={status}: {len(df)} rows")
            if df is not None and not df.empty:
                frames.append(df)
        if not frames:
            log.warning("fund_basic returned no rows")
            return 0
        df = pd.concat(frames, ignore_index=True)
        df = df.drop_duplicates(subset=["ts_code"], keep="last")
        df = _normalize_dates(df)
        return upsert_dataframe(EtfBasic, df)

    # ------------------------------------------------------------------
    # Generic per-code time-series helper
    # ------------------------------------------------------------------
    def _fetch_per_code_timeseries(
        self,
        *,
        api: str,
        model,
        date_field_in_db: str,
        ts_codes: Iterable[str],
        incremental: bool,
        start_param: str = "start_date",
        end_param: str = "end_date",
        api_extra: dict | None = None,
    ) -> int:
        """Loop over ts_codes, call ``api`` for each and upsert."""
        api_extra = api_extra or {}
        end_str = _today_str()

        # Pre-compute per-ts_code start_date for incremental mode.
        starts: dict[str, str] = {}
        if incremental:
            db_col = getattr(model, date_field_in_db)
            with session_scope() as sess:
                rows = sess.execute(
                    select(model.ts_code, func.max(db_col)).group_by(model.ts_code)
                ).all()
            for code, mx in rows:
                if mx is None:
                    continue
                next_day = mx + timedelta(days=1)
                starts[code] = next_day.strftime("%Y%m%d")

        total = 0
        codes = list(ts_codes)
        n = len(codes)
        log.info(f"Fetching {api} for {n} ETFs (incremental={incremental}) ...")
        for i, code in enumerate(codes, start=1):
            start_str = starts.get(code, self.DEFAULT_START_DATE)
            if start_str >= end_str:
                continue  # already up-to-date
            try:
                df = self.client.call(
                    api,
                    ts_code=code,
                    **{start_param: start_str, end_param: end_str},
                    **api_extra,
                )
            except Exception as e:  # noqa: BLE001
                log.error(f"{api} failed for {code}: {e}")
                continue
            if df is None or df.empty:
                log.info(f"  {api} [{i}/{n}] {code} empty")
                continue
            df = _normalize_dates(df)
            written = upsert_dataframe(model, df)
            total += written
            log.info(f"  {api} [{i}/{n}] {code} +{written} rows (total={total})")
        log.info(f"{api} done. total rows={total}")
        return total

    # ------------------------------------------------------------------
    # 2) fund_daily
    # ------------------------------------------------------------------
    def fetch_daily(self, *, ts_codes: Iterable[str], incremental: bool = True) -> int:
        return self._fetch_per_code_timeseries(
            api="fund_daily",
            model=EtfDaily,
            date_field_in_db="trade_date",
            ts_codes=ts_codes,
            incremental=incremental,
        )

    # ------------------------------------------------------------------
    # 3) fund_nav
    # ------------------------------------------------------------------
    def fetch_nav(self, *, ts_codes: Iterable[str], incremental: bool = True) -> int:
        return self._fetch_per_code_timeseries(
            api="fund_nav",
            model=EtfNav,
            date_field_in_db="nav_date",
            ts_codes=ts_codes,
            incremental=incremental,
        )

    # ------------------------------------------------------------------
    # 4) fund_adj
    # ------------------------------------------------------------------
    def fetch_adj(self, *, ts_codes: Iterable[str], incremental: bool = True) -> int:
        return self._fetch_per_code_timeseries(
            api="fund_adj",
            model=EtfAdjFactor,
            date_field_in_db="trade_date",
            ts_codes=ts_codes,
            incremental=incremental,
        )

    # ------------------------------------------------------------------
    # 5) fund_div
    # ------------------------------------------------------------------
    def fetch_dividend(self, *, ts_codes: Iterable[str], incremental: bool = True) -> int:
        # fund_div doesn't strictly support start/end_date in older versions; the
        # generic helper still works because Tushare ignores unknown args.
        # Dividend records are sparse; full pull is cheap, so we always full-pull.
        del incremental
        codes = list(ts_codes)
        n = len(codes)
        total = 0
        log.info(f"Fetching fund_div for {n} ETFs ...")
        for i, code in enumerate(codes, start=1):
            try:
                df = self.client.call("fund_div", ts_code=code)
            except Exception as e:  # noqa: BLE001
                log.error(f"fund_div failed for {code}: {e}")
                continue
            if df is None or df.empty:
                log.info(f"  fund_div [{i}/{n}] {code} empty")
                continue
            # ts_code might be missing in response - inject it.
            if "ts_code" not in df.columns:
                df["ts_code"] = code
            df = _normalize_dates(df)
            # PK requires non-null ex_date / base_date - drop offending rows.
            df = df.dropna(subset=["ex_date", "base_date"], how="any")
            if df.empty:
                log.info(f"  fund_div [{i}/{n}] {code} empty (after pk filter)")
                continue
            written = upsert_dataframe(EtfDividend, df)
            total += written
            log.info(f"  fund_div [{i}/{n}] {code} +{written} rows (total={total})")
        log.info(f"fund_div done. total rows={total}")
        return total

    # ------------------------------------------------------------------
    # 6) fund_share
    # ------------------------------------------------------------------
    def fetch_share(self, *, ts_codes: Iterable[str], incremental: bool = True) -> int:
        return self._fetch_per_code_timeseries(
            api="fund_share",
            model=EtfShare,
            date_field_in_db="trade_date",
            ts_codes=ts_codes,
            incremental=incremental,
        )

    # ------------------------------------------------------------------
    # 7) fund_portfolio
    # ------------------------------------------------------------------
    def fetch_portfolio(self, *, ts_codes: Iterable[str], incremental: bool = True) -> int:
        del incremental  # portfolio is reported quarterly; full pull each time.
        codes = list(ts_codes)
        total = 0
        log.info(f"Fetching fund_portfolio for {len(codes)} ETFs ...")
        for i, code in enumerate(codes, start=1):
            try:
                df = self.client.call("fund_portfolio", ts_code=code)
            except Exception as e:  # noqa: BLE001
                log.error(f"fund_portfolio failed for {code}: {e}")
                continue
            if df is None or df.empty:
                continue
            if "ts_code" not in df.columns:
                df["ts_code"] = code
            df = _normalize_dates(df)
            df = df.dropna(subset=["end_date", "symbol"], how="any")
            if df.empty:
                continue
            total += upsert_dataframe(EtfPortfolio, df)
            if i % 100 == 0 or i == len(codes):
                log.info(f"  fund_portfolio progress {i}/{len(codes)}, rows={total}")
        log.info(f"fund_portfolio done. total rows={total}")
        return total

    # ------------------------------------------------------------------
    # 8) fund_manager
    # ------------------------------------------------------------------
    def fetch_manager(self, *, ts_codes: Iterable[str], incremental: bool = True) -> int:
        del incremental
        codes = list(ts_codes)
        total = 0
        log.info(f"Fetching fund_manager for {len(codes)} ETFs ...")
        for i, code in enumerate(codes, start=1):
            try:
                df = self.client.call("fund_manager", ts_code=code)
            except Exception as e:  # noqa: BLE001
                log.error(f"fund_manager failed for {code}: {e}")
                continue
            if df is None or df.empty:
                continue
            if "ts_code" not in df.columns:
                df["ts_code"] = code
            df = _normalize_dates(df)
            total += upsert_dataframe(EtfManager, df)
            if i % 100 == 0 or i == len(codes):
                log.info(f"  fund_manager progress {i}/{len(codes)}, rows={total}")
        log.info(f"fund_manager done. total rows={total}")
        return total
