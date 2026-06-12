"""Futures data fetcher for Tushare futures-theme datasets.

Covers: fut_basic, fut_daily, fut_holding, fut_wsr, fut_settle.
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
from quantify.database.models import FutBasic, FutDaily, FutHolding, FutSettle, FutWsr
from quantify.database.upsert import upsert_dataframe
from quantify.tushare_client.client import TushareClient, get_client
from quantify.utils.logger import log


DATE_COLUMNS = {
    "trade_date",
    "list_date",
    "delist_date",
    "last_ddate",
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


@dataclass
class FetchSummary:
    name: str
    rows: int


class FuturesFetcher:
    """Pull Tushare futures-theme datasets into MySQL."""

    DEFAULT_START_DATE = "20000101"
    MAX_WORKERS = 2
    DAILY_ROW_CAP = 7800
    DEFAULT_SKIP_STAGES = frozenset({"fut_holding", "fut_wsr", "fut_settle"})

    def __init__(self, client: TushareClient | None = None) -> None:
        self.client = client or get_client()

    # ------------------------------------------------------------------
    # Top-level orchestration
    # ------------------------------------------------------------------
    def fetch_all(
        self,
        *,
        incremental: bool = True,
        skip: Iterable[str] | None = None,
    ) -> list[FetchSummary]:
        """Run all futures sub-fetchers in order."""
        skip = set(skip or []) | self.DEFAULT_SKIP_STAGES
        results: list[FetchSummary] = []

        if "fut_basic" not in skip:
            n = self.fetch_basic()
            results.append(FetchSummary("fut_basic", n))

        symbols = self._load_symbols()
        log.info(f"Futures symbols: {len(symbols)}")

        if "fut_daily" not in skip:
            n = self.fetch_daily(symbols=symbols, incremental=incremental)
            results.append(FetchSummary("fut_daily", n))

        if "fut_holding" not in skip:
            n = self.fetch_holding(symbols=symbols, incremental=incremental)
            results.append(FetchSummary("fut_holding", n))

        if "fut_wsr" not in skip:
            n = self.fetch_wsr(symbols=symbols, incremental=incremental)
            results.append(FetchSummary("fut_wsr", n))

        if "fut_settle" not in skip:
            n = self.fetch_settle(symbols=symbols, incremental=incremental)
            results.append(FetchSummary("fut_settle", n))

        log.info("=== Futures fetch summary ===")
        for r in results:
            log.info(f"  {r.name:<18s}: {r.rows} rows")
        return results

    # ------------------------------------------------------------------
    # Universe
    # ------------------------------------------------------------------
    def _load_symbols(self) -> list[str]:
        """Load distinct product symbols from fut_basic."""
        with session_scope() as sess:
            rows = sess.execute(select(FutBasic.fut_code).distinct()).scalars().all()
        return sorted(c for c in rows if c)

    # ------------------------------------------------------------------
    # 1) fut_basic (期货合约列表)
    # ------------------------------------------------------------------
    def fetch_basic(self) -> int:
        """Pull all futures contracts across exchanges."""
        log.info("Fetching fut_basic ...")
        frames = []
        for exchange in ("DCE", "CZCE", "SHFE", "CFFEX", "INE", "GFEX"):
            df = self.client.call("fut_basic", exchange=exchange, fut_type="1")
            if df is not None and not df.empty:
                frames.append(df)
            time.sleep(0.2)
            # Also fetch main/continuous contracts (fut_type=2)
            df2 = self.client.call("fut_basic", exchange=exchange, fut_type="2")
            if df2 is not None and not df2.empty:
                frames.append(df2)
        if not frames:
            log.warning("fut_basic returned no rows")
            return 0
        df = pd.concat(frames, ignore_index=True)
        df = df.drop_duplicates(subset=["ts_code"], keep="last")
        df = _normalize_dates(df)
        return upsert_dataframe(FutBasic, df)

    # ------------------------------------------------------------------
    # 2) fut_daily (期货日线)
    # ------------------------------------------------------------------
    def fetch_daily(self, *, symbols: Iterable[str], incremental: bool = True) -> int:
        codes = list(symbols)
        end_str = _today_str()
        starts = self._incremental_starts(FutDaily, "trade_date") if incremental else {}

        def fetch_one(i: int, code: str) -> pd.DataFrame | None:
            del i
            start_str = starts.get(code, self.DEFAULT_START_DATE)
            if start_str >= end_str:
                return None
            return self._fetch_range("fut_daily", code, start_str, end_str, row_cap=self.DAILY_ROW_CAP)

        return self._fetch_concurrent(
            api="fut_daily",
            model=FutDaily,
            codes=codes,
            fetch_one=fetch_one,
            log_extra=f" (incremental={incremental})",
        )

    # ------------------------------------------------------------------
    # 3) fut_holding (每日成交持仓排名)
    # ------------------------------------------------------------------
    def fetch_holding(self, *, symbols: Iterable[str], incremental: bool = True) -> int:
        codes = list(symbols)
        end_str = _today_str()
        starts = (
            self._incremental_starts(FutHolding, "trade_date", code_field="symbol") if incremental else {}
        )

        def fetch_one(i: int, code: str) -> pd.DataFrame | None:
            del i
            start_str = starts.get(code, self.DEFAULT_START_DATE)
            if start_str >= end_str:
                return None
            return self._fetch_range(
                "fut_holding",
                code,
                start_str,
                end_str,
                code_param="symbol",
                row_cap=self.DAILY_ROW_CAP,
            )

        return self._fetch_concurrent(
            api="fut_holding",
            model=FutHolding,
            codes=codes,
            fetch_one=fetch_one,
            log_extra=f" (incremental={incremental})",
        )

    # ------------------------------------------------------------------
    # 4) fut_wsr (仓单日报)
    # ------------------------------------------------------------------
    def fetch_wsr(self, *, symbols: Iterable[str], incremental: bool = True) -> int:
        codes = list(symbols)
        end_str = _today_str()
        starts = self._incremental_starts(FutWsr, "trade_date", code_field="symbol") if incremental else {}

        def fetch_one(i: int, code: str) -> pd.DataFrame | None:
            del i
            start_str = starts.get(code, self.DEFAULT_START_DATE)
            if start_str >= end_str:
                return None
            return self._fetch_range(
                "fut_wsr",
                code,
                start_str,
                end_str,
                code_param="symbol",
                row_cap=self.DAILY_ROW_CAP,
            )

        return self._fetch_concurrent(
            api="fut_wsr",
            model=FutWsr,
            codes=codes,
            fetch_one=fetch_one,
            log_extra=f" (incremental={incremental})",
        )

    # ------------------------------------------------------------------
    # 5) fut_settle (结算参数)
    # ------------------------------------------------------------------
    def fetch_settle(self, *, symbols: Iterable[str], incremental: bool = True) -> int:
        codes = list(symbols)
        end_str = _today_str()
        starts = self._incremental_starts(FutSettle, "trade_date") if incremental else {}

        def fetch_one(i: int, code: str) -> pd.DataFrame | None:
            del i
            start_str = starts.get(code, self.DEFAULT_START_DATE)
            if start_str >= end_str:
                return None
            return self._fetch_range("fut_settle", code, start_str, end_str, row_cap=self.DAILY_ROW_CAP)

        return self._fetch_concurrent(
            api="fut_settle",
            model=FutSettle,
            codes=codes,
            fetch_one=fetch_one,
            log_extra=f" (incremental={incremental})",
        )

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------
    def _fetch_concurrent(
        self,
        *,
        api: str,
        model,
        codes: Iterable[str],
        fetch_one: Callable[[int, str], pd.DataFrame | None],
        log_extra: str = "",
    ) -> int:
        code_list = list(codes)
        n = len(code_list)
        log.info(f"Fetching {api} for {n} symbols{log_extra} ...")
        total = 0
        lock = threading.Lock()
        EMPTY_RETRIES = 3

        def run_one(idx_code: tuple[int, str]) -> int:
            nonlocal total
            i, code = idx_code
            df = None
            attempt = 0
            while True:
                try:
                    df = fetch_one(i, code)
                except Exception as e:  # noqa: BLE001
                    log.error(f"{api} failed for {code}: {e}, retrying in 5s ...")
                    time.sleep(5)
                    continue
                # None = deliberate skip; non-empty = success. Only an empty
                # DataFrame (possible transient HTTP error) is retried.
                if df is None or not df.empty:
                    break
                attempt += 1
                if attempt > EMPTY_RETRIES:
                    break
                time.sleep(2)
            if df is None or df.empty:
                log.info(f"  {api} [{i}/{n}] {code} empty")
                return 0
            df = _normalize_dates(df)
            written = upsert_dataframe(model, df)
            with lock:
                total += written
                log.info(f"  {api} [{i}/{n}] {code} +{written} rows (total={total})")
            return written

        with ThreadPoolExecutor(max_workers=self.MAX_WORKERS) as executor:
            list(executor.map(run_one, enumerate(code_list, start=1)))

        log.info(f"{api} done. total rows={total}")
        return total

    def _fetch_range(
        self,
        api: str,
        code: str,
        start: str,
        end: str,
        *,
        code_param: str = "ts_code",
        row_cap: int,
        depth: int = 0,
    ) -> pd.DataFrame | None:
        """Fetch one date range, splitting if the row cap is hit."""
        df = self.client.call(api, **{code_param: code}, start_date=start, end_date=end)
        if df is None or df.empty:
            return df
        if len(df) < row_cap:
            return df
        start_d = _date_from_str(start)
        end_d = _date_from_str(end)
        if start_d >= end_d or depth >= 12:
            log.warning(f"{api} {code} {start}..{end} hit row cap and cannot split (rows={len(df)})")
            return df
        mid = start_d + (end_d - start_d) // 2
        log.info(f"{api} {code} {start}..{end} hit row cap (rows={len(df)}); splitting at {mid}")
        left = self._fetch_range(
            api,
            code,
            start,
            mid.strftime("%Y%m%d"),
            code_param=code_param,
            row_cap=row_cap,
            depth=depth + 1,
        )
        right = self._fetch_range(
            api,
            code,
            (mid + timedelta(days=1)).strftime("%Y%m%d"),
            end,
            code_param=code_param,
            row_cap=row_cap,
            depth=depth + 1,
        )
        frames = [f for f in (left, right) if f is not None and not f.empty]
        if not frames:
            return df
        return pd.concat(frames, ignore_index=True)

    def _incremental_starts(
        self,
        model,
        date_field: str,
        code_field: str = "ts_code",
    ) -> dict[str, str]:
        """Build {code: 'YYYYMMDD'} map of next-day-after-last-stored."""
        starts: dict[str, str] = {}
        db_col = getattr(model, date_field)
        db_code = getattr(model, code_field)
        with session_scope() as sess:
            rows = sess.execute(select(db_code, func.max(db_col)).group_by(db_code)).all()
        for code, mx in rows:
            if mx is None:
                continue
            next_day = mx + timedelta(days=1)
            starts[code] = max(next_day.strftime("%Y%m%d"), self.DEFAULT_START_DATE)
        return starts
