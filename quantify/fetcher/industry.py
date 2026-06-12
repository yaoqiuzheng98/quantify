"""Industry data fetcher for Tushare SW/CITIC classification datasets."""

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
    CiticIndustryDaily,
    CiticIndustryMember,
    SwIndustryClassify,
    SwIndustryDaily,
    SwIndustryMember,
    TradeCalendar,
)
from quantify.database.upsert import upsert_dataframe
from quantify.tushare_client.client import TushareClient, get_client
from quantify.utils.logger import log


DATE_COLUMNS = {"trade_date", "in_date", "out_date", "cal_date", "pretrade_date"}


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


class IndustryFetcher:
    """Pull SW/CITIC industry classification, members and daily quotes."""

    DEFAULT_START_DATE = "20000101"
    # Tushare 实测并发上限为 2，超过会触发"并发请求过多"错误并可能返回空。
    MAX_WORKERS = 2
    # ci_daily/sw_daily 单次最多约 4000 行；窗口按交易日折算需远低于该值。
    # ~800 自然日 ≈ 550 个交易日，单段稳定低于上限。
    MAX_DAILY_RANGE_DAYS = 800
    # 单段返回行数达到该阈值视为可能被接口截断，需要缩小窗口重拉。
    DAILY_ROW_CAP = 3800

    def __init__(self, client: TushareClient | None = None) -> None:
        self.client = client or get_client()

    def fetch_trade_cal(
        self,
        *,
        exchange: str = "SSE",
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> int:
        """Pull exchange trade calendar (trade_cal) into MySQL.

        The calendar is the authoritative basis for data-completeness checks:
        for any open day a daily series should have a row (unless the index
        itself was suspended/de-listed on that day).
        """
        start = start_date or self.DEFAULT_START_DATE
        end = end_date or _today_str()
        log.info(f"Fetching trade_cal (exchange={exchange}) {start}..{end} ...")
        df = self.client.call("trade_cal", exchange=exchange, start_date=start, end_date=end)
        if df is None or df.empty:
            log.warning(f"trade_cal returned no rows for {exchange}")
            return 0
        df = _normalize_dates(df)
        return upsert_dataframe(TradeCalendar, df)

    def fetch_all(
        self,
        *,
        provider: str = "sw",
        incremental: bool = True,
        start_date: str | None = None,
        end_date: str | None = None,
        sw_src: str = "SW2021",
    ) -> list[FetchSummary]:
        """Run industry fetchers for one provider or both providers."""
        provider = provider.lower()
        if provider not in {"sw", "ci", "all"}:
            raise ValueError("provider must be one of: sw, ci, all")

        results: list[FetchSummary] = []
        if provider in {"sw", "all"}:
            results.append(FetchSummary("sw_classify", self.fetch_sw_classify(src=sw_src)))
            results.append(FetchSummary("sw_member", self.fetch_sw_member(src=sw_src)))
            results.append(
                FetchSummary(
                    "sw_daily",
                    self.fetch_sw_daily(
                        src=sw_src,
                        incremental=incremental,
                        start_date=start_date,
                        end_date=end_date,
                    ),
                )
            )

        if provider in {"ci", "all"}:
            results.append(FetchSummary("ci_member", self.fetch_ci_member()))
            results.append(
                FetchSummary(
                    "ci_daily",
                    self.fetch_ci_daily(
                        incremental=incremental,
                        start_date=start_date,
                        end_date=end_date,
                    ),
                )
            )

        log.info("=== Industry fetch summary ===")
        for result in results:
            log.info(f"  {result.name:<14s}: {result.rows} rows")
        return results

    def fetch_sw_classify(self, *, src: str = "SW2021") -> int:
        """Pull SW industry classification metadata."""
        log.info(f"Fetching index_classify (src={src}) ...")
        df = self.client.call("index_classify", src=src)
        if df is None or df.empty:
            log.warning(f"index_classify src={src} returned no rows")
            return 0
        return upsert_dataframe(SwIndustryClassify, df)

    def fetch_sw_member(self, *, src: str = "SW2021", latest: bool = True) -> int:
        """Pull SW industry stock members by L3 industry code."""
        codes = self._load_sw_codes(src=src, level="L3", published_only=False)
        if not codes:
            self.fetch_sw_classify(src=src)
            codes = self._load_sw_codes(src=src, level="L3", published_only=False)
        if not codes:
            log.warning("SW industry classification is empty; cannot fetch members")
            return 0

        api_kwargs = {"is_new": "Y"} if latest else {}

        def fetch_one(position: int, code: str) -> pd.DataFrame | None:
            del position
            return self.client.call("index_member_all", l3_code=code, **api_kwargs)

        return self._fetch_concurrent(
            api="index_member_all",
            model=SwIndustryMember,
            codes=codes,
            fetch_one=fetch_one,
            pk_dropna=["ts_code", "l3_code", "in_date"],
        )

    def fetch_sw_daily(
        self,
        *,
        index_codes: Iterable[str] | None = None,
        src: str = "SW2021",
        incremental: bool = True,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> int:
        """Pull SW industry daily quotes.

        The traversal universe is the union of published classification codes and
        the codes actually returned by ``sw_daily`` (e.g. 申万50/中小/A指 等风格规模
        指数并不在 ``index_classify`` 分类表里，但 ``sw_daily`` 会返回行情)。
        """
        if index_codes:
            codes = list(index_codes)
        else:
            classify_codes = self._load_sw_codes(src=src, published_only=True)
            if not classify_codes:
                self.fetch_sw_classify(src=src)
                classify_codes = self._load_sw_codes(src=src, published_only=True)
            codes = sorted(set(classify_codes) | set(self._discover_sw_daily_codes()))
        if not codes:
            log.warning("SW industry classification is empty; cannot fetch daily quotes")
            return 0
        return self._fetch_index_daily(
            api="sw_daily",
            model=SwIndustryDaily,
            codes=codes,
            incremental=incremental,
            start_date=start_date,
            end_date=end_date,
        )

    def fetch_ci_member(self, *, latest: bool = True) -> int:
        """Pull CITIC industry stock members by discovered L1 industry codes."""
        api_kwargs = {"is_new": "Y"} if latest else {}
        seed = self.client.call("ci_index_member", **api_kwargs)
        if seed is None or seed.empty:
            log.warning("ci_index_member returned no rows")
            return 0

        l1_codes = sorted(str(code) for code in seed["l1_code"].dropna().unique())
        if not l1_codes:
            return 0

        def fetch_one(position: int, code: str) -> pd.DataFrame | None:
            del position
            return self.client.call("ci_index_member", l1_code=code, **api_kwargs)

        return self._fetch_concurrent(
            api="ci_index_member",
            model=CiticIndustryMember,
            codes=l1_codes,
            fetch_one=fetch_one,
            pk_dropna=["ts_code", "l3_code", "in_date"],
        )

    def fetch_ci_daily(
        self,
        *,
        index_codes: Iterable[str] | None = None,
        incremental: bool = True,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> int:
        """Pull CITIC industry daily quotes."""
        codes = list(index_codes) if index_codes else self._load_citic_codes()
        if not codes:
            self.fetch_ci_member()
            codes = self._load_citic_codes()
        if not codes:
            log.warning("CITIC industry member table is empty; cannot infer daily index codes")
            return 0
        return self._fetch_index_daily(
            api="ci_daily",
            model=CiticIndustryDaily,
            codes=codes,
            incremental=incremental,
            start_date=start_date,
            end_date=end_date,
        )

    def _discover_sw_daily_codes(self) -> list[str]:
        """Discover the full sw_daily index universe from a recent trading day.

        ``sw_daily`` covers style/scale indices (申万50/中小/A指 等) that are not
        present in ``index_classify``; querying a recent trade day returns the
        full set so the daily fetch does not silently miss them.
        """
        end_str = _today_str()
        start_str = (datetime.now() - timedelta(days=20)).strftime("%Y%m%d")
        for chunk_start, chunk_end in reversed(list(_date_chunks(start_str, end_str, max_days=1))):
            del chunk_end
            try:
                df = self.client.call("sw_daily", trade_date=chunk_start)
            except Exception as exc:  # noqa: BLE001
                log.warning(f"sw_daily probe {chunk_start} failed: {exc}")
                continue
            if df is not None and not df.empty and "ts_code" in df.columns:
                codes = sorted(str(code) for code in df["ts_code"].dropna().unique())
                log.info(f"sw_daily universe discovered on {chunk_start}: {len(codes)} indices")
                return codes
        log.warning("Could not discover sw_daily universe from recent trading days")
        return []

    def _load_sw_codes(
        self,
        *,
        src: str,
        level: str | None = None,
        published_only: bool = False,
    ) -> list[str]:
        conditions = [SwIndustryClassify.src == src]
        if level:
            conditions.append(SwIndustryClassify.level == level)
        if published_only:
            conditions.append(SwIndustryClassify.is_pub == "1")
        with session_scope() as session:
            rows = session.execute(
                select(SwIndustryClassify.index_code)
                .where(*conditions)
                .order_by(SwIndustryClassify.index_code)
            ).scalars()
            return list(rows)

    def _load_citic_codes(self) -> list[str]:
        with session_scope() as session:
            rows = session.execute(
                select(
                    CiticIndustryMember.l1_code,
                    CiticIndustryMember.l2_code,
                    CiticIndustryMember.l3_code,
                ).where(CiticIndustryMember.is_new == "Y")
            ).all()
        codes = {code for row in rows for code in row if code}
        return sorted(codes)

    def _fetch_daily_range(
        self,
        api: str,
        code: str,
        start: str,
        end: str,
        *,
        depth: int = 0,
    ) -> pd.DataFrame | None:
        """Fetch one daily date range, splitting if the row cap is hit.

        ``ci_daily``/``sw_daily`` silently cap a single response at ~4000 rows.
        When a response reaches ``DAILY_ROW_CAP`` we recursively split the date
        range in half so no rows are silently dropped.
        """
        df = self.client.call(api, ts_code=code, start_date=start, end_date=end)
        if df is None or df.empty:
            return df

        if len(df) < self.DAILY_ROW_CAP:
            return df

        start_d = _date_from_str(start)
        end_d = _date_from_str(end)
        if start_d >= end_d or depth >= 12:
            # Cannot split further; return what we have.
            log.warning(f"{api} {code} {start}..{end} hit row cap and cannot split (rows={len(df)})")
            return df

        mid = start_d + (end_d - start_d) // 2
        log.info(f"{api} {code} {start}..{end} hit row cap (rows={len(df)}); splitting at {mid}")
        left = self._fetch_daily_range(api, code, start, mid.strftime("%Y%m%d"), depth=depth + 1)
        right = self._fetch_daily_range(
            api, code, (mid + timedelta(days=1)).strftime("%Y%m%d"), end, depth=depth + 1
        )
        frames = [f for f in (left, right) if f is not None and not f.empty]
        if not frames:
            return df
        return pd.concat(frames, ignore_index=True)

    def _fetch_index_daily(
        self,
        *,
        api: str,
        model,
        codes: Iterable[str],
        incremental: bool,
        start_date: str | None,
        end_date: str | None,
    ) -> int:
        code_list = list(codes)
        end_str = end_date or _today_str()
        default_start = start_date or self.DEFAULT_START_DATE
        starts: dict[str, str] = {}

        if incremental:
            with session_scope() as session:
                rows = session.execute(
                    select(model.ts_code, func.max(model.trade_date)).group_by(model.ts_code)
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
                code_start,
                end_str,
                max_days=self.MAX_DAILY_RANGE_DAYS,
            ):
                df = self._fetch_daily_range(api, code, chunk_start, chunk_end)
                if df is not None and not df.empty:
                    frames.append(df)
            if not frames:
                return None
            return pd.concat(frames, ignore_index=True)

        return self._fetch_concurrent(
            api=api,
            model=model,
            codes=code_list,
            fetch_one=fetch_one,
            pk_dropna=["ts_code", "trade_date"],
            log_extra=f" (incremental={incremental})",
        )

    def _fetch_concurrent(
        self,
        *,
        api: str,
        model,
        codes: Iterable[str],
        fetch_one: Callable[[int, str], pd.DataFrame | None],
        pk_dropna: list[str] | None = None,
        log_extra: str = "",
    ) -> int:
        code_list = list(codes)
        total_codes = len(code_list)
        log.info(f"Fetching {api} for {total_codes} codes{log_extra} ...")

        counter_lock = threading.Lock()
        total = 0
        EMPTY_RETRIES = 3

        def run_one(index_code: tuple[int, str]) -> int:
            nonlocal total
            position, code = index_code
            df = None
            attempt = 0
            while True:
                try:
                    df = fetch_one(position, code)
                except Exception as exc:  # noqa: BLE001
                    log.error(f"{api} failed for {code}: {exc}, retrying in 5s ...")
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
                log.info(f"  {api} [{position}/{total_codes}] {code} empty")
                return 0

            if pk_dropna:
                missing_columns = [column for column in pk_dropna if column not in df.columns]
                if missing_columns:
                    log.warning(
                        f"  {api} [{position}/{total_codes}] {code} missing columns: {missing_columns}"
                    )
                    return 0
                df = df.dropna(subset=pk_dropna, how="any")
                if df.empty:
                    log.info(f"  {api} [{position}/{total_codes}] {code} empty after pk filter")
                    return 0

            df = _normalize_dates(df)
            written = upsert_dataframe(model, df)
            with counter_lock:
                total += written
                log.info(f"  {api} [{position}/{total_codes}] {code} +{written} rows (total={total})")
            return written

        with ThreadPoolExecutor(max_workers=self.MAX_WORKERS) as executor:
            list(executor.map(run_one, enumerate(code_list, start=1)))

        log.info(f"{api} done. total rows={total}")
        return total
