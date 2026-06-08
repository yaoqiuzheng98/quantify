"""Index data fetcher for Tushare index-theme datasets.

Covers: index_basic, index_daily, index_dailybasic, index_weight,
moneyflow_ind_dc. Table names mirror their Tushare endpoint names.
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
    IndexBasic,
    IndexDaily,
    IndexDailyBasic,
    IndexWeight,
    MoneyflowIndDc,
)
from quantify.database.upsert import upsert_dataframe
from quantify.tushare_client.client import TushareClient, get_client
from quantify.utils.logger import log


DATE_COLUMNS = {"trade_date", "base_date", "list_date", "exp_date"}


def _normalize_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Convert Tushare 'YYYYMMDD' string date columns into python dates."""
    if df is None or df.empty:
        return df
    for column in df.columns:
        if column in DATE_COLUMNS:
            df[column] = pd.to_datetime(df[column], format="%Y%m%d", errors="coerce").dt.date
    return df


def _today_str() -> str:
    return datetime.now().strftime("%Y%m%d")


def _date_from_str(value: str) -> date:
    return datetime.strptime(value, "%Y%m%d").date()


def _date_chunks(start_date: str, end_date: str, *, max_days: int) -> Iterable[tuple[str, str]]:
    start = _date_from_str(start_date)
    end = _date_from_str(end_date)
    while start <= end:
        chunk_end = min(start + timedelta(days=max_days - 1), end)
        yield start.strftime("%Y%m%d"), chunk_end.strftime("%Y%m%d")
        start = chunk_end + timedelta(days=1)


@dataclass
class FetchSummary:
    name: str
    rows: int


class IndexFetcher:
    """Pull Tushare index-theme datasets into MySQL."""

    DEFAULT_START_DATE = "20000101"
    MAX_WORKERS = 2
    # index_daily 单次上限 8000 行；窗口按交易日折算需远低于该值。
    MAX_DAILY_RANGE_DAYS = 1000
    DAILY_ROW_CAP = 7800
    # index_dailybasic 单次上限 3000 行。
    BASIC_RANGE_DAYS = 800
    BASIC_ROW_CAP = 2800
    # 默认拉取每日指标的宽基指数(接口仅支持这几个)。
    DAILYBASIC_CODES = (
        "000001.SH",
        "000300.SH",
        "000905.SH",
        "399001.SZ",
        "399005.SZ",
        "399006.SZ",
        "399016.SZ",
        "000016.SH",
    )

    def __init__(self, client: TushareClient | None = None) -> None:
        self.client = client or get_client()

    # ------------------------------------------------------------------
    # index_basic
    # ------------------------------------------------------------------
    def fetch_index_basic(self, *, markets: Iterable[str] | None = None) -> int:
        """Pull index basic info across markets."""
        market_list = list(markets) if markets else ["SSE", "SZSE", "CSI", "SW", "MSCI", "CICC", "OTH"]
        frames = []
        for market in market_list:
            log.info(f"Fetching index_basic (market={market}) ...")
            df = self.client.call("index_basic", market=market)
            if df is not None and not df.empty:
                frames.append(df)
        if not frames:
            log.warning("index_basic returned no rows")
            return 0
        df = pd.concat(frames, ignore_index=True)
        df = df.drop_duplicates(subset=["ts_code"], keep="last")
        df = _normalize_dates(df)
        return upsert_dataframe(IndexBasic, df)

    # ------------------------------------------------------------------
    # shared concurrent + range-splitting helpers
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
        log.info(f"Fetching {api} for {len(code_list)} codes{log_extra} ...")
        total = 0
        lock = threading.Lock()

        def run_one(position: int, code: str) -> int:
            while True:
                try:
                    df = fetch_one(position, code)
                    break
                except Exception as exc:  # noqa: BLE001
                    log.warning(f"  {api} [{position}] {code} failed: {exc}; retrying in 3s")
                    time.sleep(3)
            if df is None or df.empty:
                return 0
            df = _normalize_dates(df)
            written = upsert_dataframe(model, df)
            with lock:
                nonlocal total
                total += written
                log.info(f"  {api} [{position}/{len(code_list)}] {code} +{written} rows (total={total})")
            return written

        with ThreadPoolExecutor(max_workers=self.MAX_WORKERS) as executor:
            futures = [executor.submit(run_one, i + 1, c) for i, c in enumerate(code_list)]
            for future in futures:
                future.result()
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
        """Fetch one date range, splitting in half if the row cap is hit."""
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

    # ------------------------------------------------------------------
    # universe helpers
    # ------------------------------------------------------------------
    def _load_index_codes(self, *, markets: Iterable[str] | None = None) -> list[str]:
        """Load index ts_codes from the index_basic table."""
        with session_scope() as session:
            stmt = select(IndexBasic.ts_code)
            if markets:
                stmt = stmt.where(IndexBasic.market.in_(list(markets)))
            rows = session.execute(stmt).scalars().all()
        return list(rows)

    def _incremental_starts(self, model, default_start: str) -> dict[str, str]:
        starts: dict[str, str] = {}
        with session_scope() as session:
            rows = session.execute(
                select(model.ts_code, func.max(model.trade_date)).group_by(model.ts_code)
            ).all()
        for code, max_date in rows:
            if max_date is None:
                continue
            next_day = max_date + timedelta(days=1)
            starts[code] = max(next_day.strftime("%Y%m%d"), default_start)
        return starts

    # ------------------------------------------------------------------
    # index_daily
    # ------------------------------------------------------------------
    def fetch_index_daily(
        self,
        *,
        ts_codes: Iterable[str] | None = None,
        markets: Iterable[str] | None = None,
        incremental: bool = True,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> int:
        codes = list(ts_codes) if ts_codes else self._load_index_codes(markets=markets)
        if not codes:
            log.warning("index_daily: no index codes - run index_basic first")
            return 0
        end_str = end_date or _today_str()
        default_start = start_date or self.DEFAULT_START_DATE
        starts = self._incremental_starts(IndexDaily, default_start) if incremental else {}

        def fetch_one(position: int, code: str) -> pd.DataFrame | None:
            del position
            code_start = starts.get(code, default_start)
            if code_start > end_str:
                return None
            frames = []
            for chunk_start, chunk_end in _date_chunks(
                code_start, end_str, max_days=self.MAX_DAILY_RANGE_DAYS
            ):
                df = self._fetch_range(
                    "index_daily", code, chunk_start, chunk_end, row_cap=self.DAILY_ROW_CAP
                )
                if df is not None and not df.empty:
                    frames.append(df)
            return pd.concat(frames, ignore_index=True) if frames else None

        return self._fetch_concurrent(
            api="index_daily",
            model=IndexDaily,
            codes=codes,
            fetch_one=fetch_one,
            log_extra=f" (incremental={incremental})",
        )

    # ------------------------------------------------------------------
    # index_dailybasic (broad indices only)
    # ------------------------------------------------------------------
    def fetch_index_dailybasic(
        self,
        *,
        ts_codes: Iterable[str] | None = None,
        incremental: bool = True,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> int:
        codes = list(ts_codes) if ts_codes else list(self.DAILYBASIC_CODES)
        end_str = end_date or _today_str()
        default_start = start_date or self.DEFAULT_START_DATE
        starts = self._incremental_starts(IndexDailyBasic, default_start) if incremental else {}

        def fetch_one(position: int, code: str) -> pd.DataFrame | None:
            del position
            code_start = starts.get(code, default_start)
            if code_start > end_str:
                return None
            frames = []
            for chunk_start, chunk_end in _date_chunks(code_start, end_str, max_days=self.BASIC_RANGE_DAYS):
                df = self._fetch_range(
                    "index_dailybasic", code, chunk_start, chunk_end, row_cap=self.BASIC_ROW_CAP
                )
                if df is not None and not df.empty:
                    frames.append(df)
            return pd.concat(frames, ignore_index=True) if frames else None

        return self._fetch_concurrent(
            api="index_dailybasic",
            model=IndexDailyBasic,
            codes=codes,
            fetch_one=fetch_one,
            log_extra=f" (incremental={incremental})",
        )

    # ------------------------------------------------------------------
    # index_weight (monthly constituents & weights)
    # ------------------------------------------------------------------
    def fetch_index_weight(
        self,
        *,
        index_codes: Iterable[str] | None = None,
        markets: Iterable[str] | None = None,
        incremental: bool = True,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> int:
        codes = list(index_codes) if index_codes else self._load_index_codes(markets=markets)
        if not codes:
            log.warning("index_weight: no index codes - run index_basic first")
            return 0
        end_str = end_date or _today_str()
        default_start = start_date or self.DEFAULT_START_DATE

        starts: dict[str, str] = {}
        if incremental:
            with session_scope() as session:
                rows = session.execute(
                    select(IndexWeight.index_code, func.max(IndexWeight.trade_date)).group_by(
                        IndexWeight.index_code
                    )
                ).all()
            for code, max_date in rows:
                if max_date is None:
                    continue
                next_day = max_date + timedelta(days=1)
                starts[code] = max(next_day.strftime("%Y%m%d"), default_start)

        def fetch_one(position: int, code: str) -> pd.DataFrame | None:
            del position
            code_start = starts.get(code, default_start)
            if code_start > end_str:
                return None
            frames = []
            for chunk_start, chunk_end in _date_chunks(
                code_start, end_str, max_days=self.MAX_DAILY_RANGE_DAYS
            ):
                df = self._fetch_range(
                    "index_weight",
                    code,
                    chunk_start,
                    chunk_end,
                    code_param="index_code",
                    row_cap=self.DAILY_ROW_CAP,
                )
                if df is not None and not df.empty:
                    frames.append(df)
            return pd.concat(frames, ignore_index=True) if frames else None

        return self._fetch_concurrent(
            api="index_weight",
            model=IndexWeight,
            codes=codes,
            fetch_one=fetch_one,
            log_extra=f" (incremental={incremental})",
        )

    # ------------------------------------------------------------------
    # moneyflow_ind_dc (per trade date)
    # ------------------------------------------------------------------
    def fetch_moneyflow_ind_dc(
        self,
        *,
        trade_dates: Iterable[str] | None = None,
        incremental: bool = True,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> int:
        dates = list(trade_dates) if trade_dates else self._open_dates(start_date, end_date, incremental)
        if not dates:
            log.info("moneyflow_ind_dc: no trade dates to fetch")
            return 0

        def fetch_one(position: int, trade_date: str) -> pd.DataFrame | None:
            del position
            return self.client.call("moneyflow_ind_dc", trade_date=trade_date)

        return self._fetch_concurrent(
            api="moneyflow_ind_dc",
            model=MoneyflowIndDc,
            codes=dates,
            fetch_one=fetch_one,
            log_extra=f" (incremental={incremental})",
        )

    def _open_dates(self, start_date: str | None, end_date: str | None, incremental: bool) -> list[str]:
        """Open trade dates from trade_cal (SSE), optionally after last stored."""
        from quantify.database.models import TradeCalendar

        end_str = end_date or _today_str()
        default_start = start_date or self.DEFAULT_START_DATE
        start_str = default_start
        if incremental:
            with session_scope() as session:
                max_date = session.execute(select(func.max(MoneyflowIndDc.trade_date))).scalar()
            if max_date is not None:
                start_str = max((max_date + timedelta(days=1)).strftime("%Y%m%d"), default_start)
        if start_str > end_str:
            return []
        with session_scope() as session:
            rows = (
                session.execute(
                    select(TradeCalendar.cal_date)
                    .where(TradeCalendar.exchange == "SSE")
                    .where(TradeCalendar.is_open == 1)
                    .where(TradeCalendar.cal_date >= _date_from_str(start_str))
                    .where(TradeCalendar.cal_date <= _date_from_str(end_str))
                    .order_by(TradeCalendar.cal_date)
                )
                .scalars()
                .all()
            )
        return [d.strftime("%Y%m%d") for d in rows]

    # ------------------------------------------------------------------
    # orchestration
    # ------------------------------------------------------------------
    def fetch_all(
        self,
        *,
        incremental: bool = True,
        start_date: str | None = None,
        end_date: str | None = None,
        skip: Iterable[str] | None = None,
    ) -> list[FetchSummary]:
        skip_set = {s.strip().lower() for s in (skip or [])}
        results: list[FetchSummary] = []

        if "index_basic" not in skip_set:
            results.append(FetchSummary("index_basic", self.fetch_index_basic()))
        if "index_daily" not in skip_set:
            results.append(
                FetchSummary(
                    "index_daily",
                    self.fetch_index_daily(incremental=incremental, start_date=start_date, end_date=end_date),
                )
            )
        if "index_dailybasic" not in skip_set:
            results.append(
                FetchSummary(
                    "index_dailybasic",
                    self.fetch_index_dailybasic(
                        incremental=incremental, start_date=start_date, end_date=end_date
                    ),
                )
            )
        if "index_weight" not in skip_set:
            results.append(
                FetchSummary(
                    "index_weight",
                    self.fetch_index_weight(
                        incremental=incremental, start_date=start_date, end_date=end_date
                    ),
                )
            )
        if "moneyflow_ind_dc" not in skip_set:
            results.append(
                FetchSummary(
                    "moneyflow_ind_dc",
                    self.fetch_moneyflow_ind_dc(
                        incremental=incremental, start_date=start_date, end_date=end_date
                    ),
                )
            )

        log.info("=== Index fetch summary ===")
        for result in results:
            log.info(f"  {result.name:<18s}: {result.rows} rows")
        return results
