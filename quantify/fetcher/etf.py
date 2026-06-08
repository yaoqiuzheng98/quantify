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

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Callable, Iterable

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
    "account_date",
    "earpay_date",
    "imp_anndate",
    "net_ex_date",
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


def _date_from_str(value: str) -> date:
    return datetime.strptime(value, "%Y%m%d").date()


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
    # Tushare 实测并发上限为 2，超过会触发"并发请求过多"错误并可能返回空。
    MAX_WORKERS = 2
    # fund_daily/fund_nav/fund_adj 单次行数上限（Tushare 通常单次 ≤8000 行）。
    # 达到该阈值视为可能被接口截断，需要缩小日期窗口重拉。
    TIMESERIES_ROW_CAP = 7800

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
            rows = sess.execute(select(EtfBasic.ts_code).where(EtfBasic.status != "D")).scalars().all()
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
    # Shared concurrent fetch helper
    # ------------------------------------------------------------------
    def _fetch_concurrent(
        self,
        *,
        api: str,
        model,
        ts_codes: Iterable[str],
        fetch_one: Callable[[int, str], pd.DataFrame | None],
        log_extra: str = "",
    ) -> int:
        codes = list(ts_codes)
        n = len(codes)
        log.info(f"Fetching {api} for {n} ETFs{log_extra} ...")

        counter_lock = threading.Lock()
        total = 0

        def _run_one(idx_code: tuple[int, str]) -> int:
            nonlocal total
            i, code = idx_code
            while True:
                try:
                    df = fetch_one(i, code)
                    break
                except Exception as e:  # noqa: BLE001
                    log.error(f"{api} failed for {code}: {e}, retrying in 5s ...")
                    time.sleep(5)
            if df is None or df.empty:
                log.info(f"  {api} [{i}/{n}] {code} empty")
                return 0
            df = _normalize_dates(df)
            written = upsert_dataframe(model, df)
            with counter_lock:
                total += written
                log.info(f"  {api} [{i}/{n}] {code} +{written} rows (total={total})")
            return written

        with ThreadPoolExecutor(max_workers=self.MAX_WORKERS) as executor:
            list(executor.map(_run_one, enumerate(codes, start=1)))

        log.info(f"{api} done. total rows={total}")
        return total

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
        api_extra = api_extra or {}
        end_str = _today_str()

        starts: dict[str, str] = {}
        if incremental:
            db_col = getattr(model, date_field_in_db)
            with session_scope() as sess:
                rows = sess.execute(select(model.ts_code, func.max(db_col)).group_by(model.ts_code)).all()
            for code, mx in rows:
                if mx is None:
                    continue
                next_day = mx + timedelta(days=1)
                starts[code] = next_day.strftime("%Y%m%d")

        def fetch_one(i: int, code: str) -> pd.DataFrame | None:
            del i
            start_str = starts.get(code, self.DEFAULT_START_DATE)
            if start_str >= end_str:
                return None
            return self._fetch_timeseries_range(
                api,
                code,
                start_str,
                end_str,
                start_param=start_param,
                end_param=end_param,
                api_extra=api_extra,
            )

        return self._fetch_concurrent(
            api=api,
            model=model,
            ts_codes=ts_codes,
            fetch_one=fetch_one,
            log_extra=f" (incremental={incremental})",
        )

    def _fetch_timeseries_range(
        self,
        api: str,
        code: str,
        start: str,
        end: str,
        *,
        start_param: str = "start_date",
        end_param: str = "end_date",
        api_extra: dict | None = None,
        depth: int = 0,
    ) -> pd.DataFrame | None:
        """Fetch one time-series date range, splitting if the row cap is hit.

        Tushare silently caps a single response (commonly ~8000 rows). When a
        response reaches ``TIMESERIES_ROW_CAP`` we recursively split the date
        range in half so no rows are silently dropped on long histories.
        """
        api_extra = api_extra or {}
        df = self.client.call(
            api,
            ts_code=code,
            **{start_param: start, end_param: end},
            **api_extra,
        )
        if df is None or df.empty:
            return df
        if len(df) < self.TIMESERIES_ROW_CAP:
            return df

        start_d = _date_from_str(start)
        end_d = _date_from_str(end)
        if start_d >= end_d or depth >= 12:
            log.warning(f"{api} {code} {start}..{end} hit row cap and cannot split (rows={len(df)})")
            return df

        mid = start_d + (end_d - start_d) // 2
        log.info(f"{api} {code} {start}..{end} hit row cap (rows={len(df)}); splitting at {mid}")
        left = self._fetch_timeseries_range(
            api,
            code,
            start,
            mid.strftime("%Y%m%d"),
            start_param=start_param,
            end_param=end_param,
            api_extra=api_extra,
            depth=depth + 1,
        )
        right = self._fetch_timeseries_range(
            api,
            code,
            (mid + timedelta(days=1)).strftime("%Y%m%d"),
            end,
            start_param=start_param,
            end_param=end_param,
            api_extra=api_extra,
            depth=depth + 1,
        )
        frames = [f for f in (left, right) if f is not None and not f.empty]
        if not frames:
            return df
        return pd.concat(frames, ignore_index=True)

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

    def _fetch_per_code_full(
        self,
        *,
        api: str,
        model,
        ts_codes: Iterable[str],
        pk_dropna: list[str] | None = None,
    ) -> int:
        def fetch_one(i: int, code: str) -> pd.DataFrame | None:
            df = self.client.call(api, ts_code=code)
            if df is None:
                return None
            if "ts_code" not in df.columns:
                df["ts_code"] = code
            if pk_dropna:
                missing = [c for c in pk_dropna if c not in df.columns]
                if missing:
                    log.warning(f"  {api} [{i}/?] {code} missing pk column(s): {missing}")
                    return None
                df = df.dropna(subset=pk_dropna, how="any")
                if df.empty:
                    return None
            return df

        return self._fetch_concurrent(
            api=api,
            model=model,
            ts_codes=ts_codes,
            fetch_one=fetch_one,
        )

    # ------------------------------------------------------------------
    # 5) fund_div
    # ------------------------------------------------------------------
    def fetch_dividend(self, *, ts_codes: Iterable[str], incremental: bool = True) -> int:
        del incremental
        return self._fetch_per_code_full(
            api="fund_div",
            model=EtfDividend,
            ts_codes=ts_codes,
            pk_dropna=["base_date"],
        )

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
        return self._fetch_per_code_full(
            api="fund_portfolio",
            model=EtfPortfolio,
            ts_codes=ts_codes,
            pk_dropna=["end_date", "symbol"],
        )

    # ------------------------------------------------------------------
    # 8) fund_manager
    # ------------------------------------------------------------------
    def fetch_manager(self, *, ts_codes: Iterable[str], incremental: bool = True) -> int:
        del incremental
        return self._fetch_per_code_full(
            api="fund_manager",
            model=EtfManager,
            ts_codes=ts_codes,
        )
