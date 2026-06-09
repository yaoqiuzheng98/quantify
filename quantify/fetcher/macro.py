"""Macro / cross-asset fetchers for All-Weather style strategies.

Datasets (接口名 == 表名):
- ``yc_cb``       中债国债收益率曲线 (即期/到期，多期限)
- ``index_global`` 国际主要指数日线
- ``us_tycr``     美国国债名义收益率曲线利率 (日频)
- ``us_trycr``    美国国债实际收益率曲线利率 (日频)
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Iterable

import pandas as pd
from sqlalchemy import func, select

from quantify.database.engine import session_scope
from quantify.database.models import IndexGlobal, UsTrycr, UsTycr, YcCb
from quantify.database.upsert import upsert_dataframe
from quantify.tushare_client.client import TushareClient, get_client
from quantify.utils.logger import log


DATE_COLUMNS = {"trade_date", "date"}


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


class MacroFetcher:
    """Pull macro / cross-asset datasets from Tushare into MySQL."""

    DEFAULT_START_DATE = "20000101"
    # 镜像站并发硬上限为 2。这些接口标的少，串行即可，无需线程池。
    # yc_cb 单次上限 2000 行；一天约 1010 行(两种 curve_type)，故每窗 1 天。
    YC_CB_RANGE_DAYS = 1
    YC_CB_CODE = "1001.CB"
    # index_global 单次上限 4000 行；按指数+日期窗，每窗 ~10 年(约 2500 交易日)。
    GLOBAL_RANGE_DAYS = 3600
    GLOBAL_ROW_CAP = 3900
    # us_tycr/us_trycr 单次上限 2000 行(日频)；每窗 ~6 年。
    US_RANGE_DAYS = 2000
    US_ROW_CAP = 1900
    # 国际指数代码(文档列出的全部 22 个)。
    GLOBAL_CODES = (
        "XIN9",
        "HSI",
        "HKTECH",
        "HKAH",
        "DJI",
        "SPX",
        "IXIC",
        "FTSE",
        "FCHI",
        "GDAXI",
        "N225",
        "KS11",
        "AS51",
        "SENSEX",
        "IBOVESPA",
        "RTS",
        "TWII",
        "CKLSE",
        "SPTSX",
        "CSX5P",
        "RUT",
    )

    def __init__(self, client: TushareClient | None = None) -> None:
        self.client = client or get_client()

    # ------------------------------------------------------------------
    # generic range fetch with row-cap bisection
    # ------------------------------------------------------------------
    def _fetch_range(
        self,
        api: str,
        start: str,
        end: str,
        *,
        row_cap: int,
        api_extra: dict | None = None,
        depth: int = 0,
    ) -> pd.DataFrame | None:
        """Fetch [start, end] for a date-keyed endpoint; bisect if row-capped."""
        kwargs = {"start_date": start, "end_date": end}
        if api_extra:
            kwargs.update(api_extra)
        df = self.client.call(api, **kwargs)
        if df is None or df.empty:
            return None
        # 触顶疑似截断：二分日期窗口重拉。
        if len(df) >= row_cap and depth < 12 and start < end:
            mid = _date_from_str(start) + (_date_from_str(end) - _date_from_str(start)) // 2
            left = self._fetch_range(
                api, start, mid.strftime("%Y%m%d"), row_cap=row_cap, api_extra=api_extra, depth=depth + 1
            )
            right = self._fetch_range(
                api,
                (mid + timedelta(days=1)).strftime("%Y%m%d"),
                end,
                row_cap=row_cap,
                api_extra=api_extra,
                depth=depth + 1,
            )
            frames = [f for f in (left, right) if f is not None and not f.empty]
            if not frames:
                return df
            return pd.concat(frames, ignore_index=True)
        return df

    def _single_table_start(self, model, date_col, default_start: str) -> str:
        """Incremental start = max(date) + 1 day, else default."""
        with session_scope() as session:
            mx = session.execute(select(func.max(date_col))).scalar()
        if mx is None:
            return default_start
        return max((mx + timedelta(days=1)).strftime("%Y%m%d"), default_start)

    # ------------------------------------------------------------------
    # yc_cb (中债国债收益率曲线)
    # ------------------------------------------------------------------
    def fetch_yc_cb(
        self,
        *,
        incremental: bool = True,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> int:
        end_str = end_date or _today_str()
        default_start = start_date or self.DEFAULT_START_DATE
        if incremental:
            default_start = self._single_table_start(YcCb, YcCb.trade_date, default_start)
        if default_start > end_str:
            log.info("yc_cb: up to date")
            return 0
        log.info(f"Fetching yc_cb ({default_start}..{end_str}) ...")
        total = 0
        # 逐窗(默认 1 天)拉取两种曲线类型。
        for chunk_start, chunk_end in _date_chunks(default_start, end_str, max_days=self.YC_CB_RANGE_DAYS):
            frames = []
            for ctype in ("0", "1"):
                df = self.client.call(
                    "yc_cb",
                    ts_code=self.YC_CB_CODE,
                    curve_type=ctype,
                    start_date=chunk_start,
                    end_date=chunk_end,
                )
                if df is not None and not df.empty:
                    frames.append(df)
            if not frames:
                continue
            df = pd.concat(frames, ignore_index=True)
            df = _normalize_dates(df)
            total += upsert_dataframe(YcCb, df)
        log.info(f"yc_cb done. rows={total}")
        return total

    # ------------------------------------------------------------------
    # index_global (国际主要指数)
    # ------------------------------------------------------------------
    def fetch_index_global(
        self,
        *,
        ts_codes: Iterable[str] | None = None,
        incremental: bool = True,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> int:
        codes = list(ts_codes) if ts_codes else list(self.GLOBAL_CODES)
        end_str = end_date or _today_str()
        default_start = start_date or self.DEFAULT_START_DATE

        starts: dict[str, str] = {}
        if incremental:
            with session_scope() as session:
                rows = session.execute(
                    select(IndexGlobal.ts_code, func.max(IndexGlobal.trade_date)).group_by(
                        IndexGlobal.ts_code
                    )
                ).all()
            for code, mx in rows:
                if mx is not None:
                    starts[code] = max((mx + timedelta(days=1)).strftime("%Y%m%d"), default_start)

        log.info(f"Fetching index_global for {len(codes)} codes (incremental={incremental}) ...")
        total = 0
        for i, code in enumerate(codes, 1):
            code_start = starts.get(code, default_start)
            if code_start > end_str:
                continue
            frames = []
            for chunk_start, chunk_end in _date_chunks(code_start, end_str, max_days=self.GLOBAL_RANGE_DAYS):
                df = self._fetch_range(
                    "index_global",
                    chunk_start,
                    chunk_end,
                    row_cap=self.GLOBAL_ROW_CAP,
                    api_extra={"ts_code": code},
                )
                if df is not None and not df.empty:
                    frames.append(df)
            if not frames:
                continue
            df = pd.concat(frames, ignore_index=True)
            df = _normalize_dates(df)
            n = upsert_dataframe(IndexGlobal, df)
            total += n
            log.info(f"  index_global [{i}/{len(codes)}] {code} +{n} rows (total={total})")
        log.info(f"index_global done. total rows={total}")
        return total

    # ------------------------------------------------------------------
    # us_tycr / us_trycr (美债名义/实际收益率)
    # ------------------------------------------------------------------
    def _fetch_us_curve(
        self,
        api: str,
        model,
        *,
        incremental: bool,
        start_date: str | None,
        end_date: str | None,
    ) -> int:
        end_str = end_date or _today_str()
        default_start = start_date or self.DEFAULT_START_DATE
        if incremental:
            default_start = self._single_table_start(model, model.trade_date, default_start)
        if default_start > end_str:
            log.info(f"{api}: up to date")
            return 0
        log.info(f"Fetching {api} ({default_start}..{end_str}) ...")
        total = 0
        for chunk_start, chunk_end in _date_chunks(default_start, end_str, max_days=self.US_RANGE_DAYS):
            df = self._fetch_range(api, chunk_start, chunk_end, row_cap=self.US_ROW_CAP)
            if df is None or df.empty:
                continue
            df = _normalize_dates(df)
            total += upsert_dataframe(model, df)
        log.info(f"{api} done. rows={total}")
        return total

    def fetch_us_tycr(
        self, *, incremental: bool = True, start_date: str | None = None, end_date: str | None = None
    ) -> int:
        return self._fetch_us_curve(
            "us_tycr", UsTycr, incremental=incremental, start_date=start_date, end_date=end_date
        )

    def fetch_us_trycr(
        self, *, incremental: bool = True, start_date: str | None = None, end_date: str | None = None
    ) -> int:
        return self._fetch_us_curve(
            "us_trycr", UsTrycr, incremental=incremental, start_date=start_date, end_date=end_date
        )

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

        stages = [
            ("yc_cb", self.fetch_yc_cb),
            ("index_global", self.fetch_index_global),
            ("us_tycr", self.fetch_us_tycr),
            ("us_trycr", self.fetch_us_trycr),
        ]
        for name, method in stages:
            if name in skip_set:
                log.info(f"[skip] {name}")
                continue
            n = method(incremental=incremental, start_date=start_date, end_date=end_date)
            results.append(FetchSummary(name, n))

        log.info("=== macro fetch summary ===")
        for r in results:
            log.info(f"  {r.name}: {r.rows}")
        return results
