"""Stock data fetcher for Tushare A-share datasets.

Covers: stock_basic, daily, adj_factor, daily_basic, weekly, monthly,
suspend_d, namechange, income, balancesheet, cashflow, fina_indicator,
forecast, express, dividend, moneyflow_hsgt, margin, margin_detail,
stk_factor, broker_recommend.
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
    AdjFactor,
    BalanceSheet,
    BrokerRecommend,
    CashFlow,
    DailyBasic,
    Express,
    FinaIndicator,
    Forecast,
    Income,
    Margin,
    MarginDetail,
    MoneyflowHsgt,
    NameChange,
    StkFactor,
    StockBasic,
    StockDaily,
    StockDividend,
    StockMonthly,
    StockWeekly,
    SuspendD,
    TradeCalendar,
)
from quantify.database.upsert import upsert_dataframe
from quantify.tushare_client.client import TushareClient, get_client
from quantify.utils.logger import log


DATE_COLUMNS = {
    "trade_date",
    "end_date",
    "ann_date",
    "f_ann_date",
    "list_date",
    "delist_date",
    "start_date",
    "ex_date",
    "record_date",
    "first_ann_date",
    "pay_date",
    "div_listdate",
    "imp_ann_date",
    "base_date",
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


class StockFetcher:
    """Pull A-share stock datasets from Tushare into MySQL."""

    DEFAULT_START_DATE = "19900101"
    MAX_WORKERS = 2
    # daily 单次上限 8000 行
    DAILY_ROW_CAP = 7800
    # daily_basic 单次上限 6000 行
    BASIC_ROW_CAP = 5800
    # 财务接口(income/balancesheet/cashflow/fina_indicator)单次上限 4000 行
    FIN_ROW_CAP = 3800
    # fetch_all 时默认跳过的阶段
    DEFAULT_SKIP_STAGES = frozenset(
        {"margin_detail", "stk_factor", "stk_factor_pro", "broker_recommend", "weekly", "monthly"}
    )

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
        """Run all stock sub-fetchers in order."""
        skip = set(skip or []) | self.DEFAULT_SKIP_STAGES
        results: list[FetchSummary] = []

        # 1. Basic info first
        if "basic" not in skip:
            n = self.fetch_basic()
            results.append(FetchSummary("basic", n))

        codes = list(ts_codes) if ts_codes else self._load_universe()
        log.info(f"Stock universe size: {len(codes)}")

        if not codes:
            log.warning("Stock universe is empty — run `quantify fetch stock basic` first.")
            return results

        # 2. Time-series stages (per code, incremental supported)
        ts_stages = [
            ("daily", self.fetch_daily),
            ("adj_factor", self.fetch_adj_factor),
        ]
        for name, method in ts_stages:
            if name in skip:
                log.info(f"[skip] {name}")
                continue
            n = method(ts_codes=codes, incremental=incremental)
            results.append(FetchSummary(name, n))

        # 3. Optional time-series stages
        opt_ts_stages = [
            ("weekly", self.fetch_weekly),
            ("monthly", self.fetch_monthly),
            ("daily_basic", self.fetch_daily_basic),
        ]
        for name, method in opt_ts_stages:
            if name in skip:
                log.info(f"[skip] {name}")
                continue
            n = method(ts_codes=codes, incremental=incremental)
            results.append(FetchSummary(name, n))

        # 4. Full-pull stages (no incremental concept)
        full_stages = [
            ("suspend", self.fetch_suspend),
            ("namechange", self.fetch_namechange),
            ("stk_factor", self.fetch_stk_factor),
            ("broker_recommend", self.fetch_broker_recommend),
        ]
        for name, method in full_stages:
            if name in skip:
                log.info(f"[skip] {name}")
                continue
            n = method(ts_codes=codes)
            results.append(FetchSummary(name, n))

        # 5. Financial statement stages (per code, full pull)
        fin_stages = [
            ("income", self.fetch_income),
            ("balancesheet", self.fetch_balancesheet),
            ("cashflow", self.fetch_cashflow),
            ("fina_indicator", self.fetch_fina_indicator),
            ("forecast", self.fetch_forecast),
            ("express", self.fetch_express),
            ("dividend", self.fetch_dividend),
        ]
        for name, method in fin_stages:
            if name in skip:
                log.info(f"[skip] {name}")
                continue
            n = method(ts_codes=codes)
            results.append(FetchSummary(name, n))

        # 6. Date-keyed market-wide stages
        mkt_stages = [
            ("moneyflow_hsgt", self.fetch_moneyflow_hsgt),
            ("margin", self.fetch_margin),
            ("margin_detail", self.fetch_margin_detail),
        ]
        for name, method in mkt_stages:
            if name in skip:
                log.info(f"[skip] {name}")
                continue
            n = method(incremental=incremental)
            results.append(FetchSummary(name, n))

        log.info("=== Stock fetch summary ===")
        for r in results:
            log.info(f"  {r.name:<20s}: {r.rows} rows")
        return results

    # ------------------------------------------------------------------
    # Universe
    # ------------------------------------------------------------------
    def _load_universe(self) -> list[str]:
        """Load listed stock codes from stock_basic."""
        with session_scope() as sess:
            rows = (
                sess.execute(select(StockBasic.ts_code).where(StockBasic.list_status == "L")).scalars().all()
            )
        return list(rows)

    def _load_all_codes(self) -> list[str]:
        """Load all stock codes (listed + delisted) from stock_basic."""
        with session_scope() as sess:
            rows = sess.execute(select(StockBasic.ts_code)).scalars().all()
        return list(rows)

    # ------------------------------------------------------------------
    # Shared helpers
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
        log.info(f"Fetching {api} for {n} stocks{log_extra} ...")
        total = 0
        lock = threading.Lock()

        EMPTY_RETRIES = 3

        def _run_one(idx_code: tuple[int, str]) -> int:
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
                # None = deliberate skip (already up-to-date); non-empty = success.
                # Neither is retried. Only an empty DataFrame (possible transient HTTP
                # error that returns nothing without raising) is retried a few times.
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
            list(executor.map(_run_one, enumerate(codes, start=1)))

        log.info(f"{api} done. total rows={total}")
        return total

    def _fetch_timeseries_range(
        self,
        api: str,
        code: str,
        start: str,
        end: str,
        *,
        api_extra: dict | None = None,
        row_cap: int = 7800,
        depth: int = 0,
    ) -> pd.DataFrame | None:
        """Fetch one time-series date range, splitting if the row cap is hit."""
        api_extra = api_extra or {}
        df = self.client.call(api, ts_code=code, start_date=start, end_date=end, **api_extra)
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
        left = self._fetch_timeseries_range(
            api, code, start, mid.strftime("%Y%m%d"), api_extra=api_extra, row_cap=row_cap, depth=depth + 1
        )
        right = self._fetch_timeseries_range(
            api,
            code,
            (mid + timedelta(days=1)).strftime("%Y%m%d"),
            end,
            api_extra=api_extra,
            row_cap=row_cap,
            depth=depth + 1,
        )
        frames = [f for f in (left, right) if f is not None and not f.empty]
        if not frames:
            return df
        return pd.concat(frames, ignore_index=True)

    def _fetch_per_code_full(
        self,
        *,
        api: str,
        model,
        ts_codes: Iterable[str],
        pk_dropna: list[str] | None = None,
        api_extra: dict | None = None,
    ) -> int:
        api_extra = api_extra or {}

        def fetch_one(i: int, code: str) -> pd.DataFrame | None:
            df = self.client.call(api, ts_code=code, **api_extra)
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
    # 1) stock_basic
    # ------------------------------------------------------------------
    def fetch_basic(self) -> int:
        """Pull listed + delisted + suspended A-share stocks."""
        log.info("Fetching stock_basic ...")
        frames = []
        for status in ("L", "D", "P"):
            for exchange in ("SSE", "SZSE", "BSE"):
                df = self.client.call("stock_basic", exchange=exchange, list_status=status)
                if df is not None and not df.empty:
                    frames.append(df)
        if not frames:
            log.warning("stock_basic returned no rows")
            return 0
        df = pd.concat(frames, ignore_index=True)
        df = df.drop_duplicates(subset=["ts_code"], keep="last")
        df = _normalize_dates(df)
        return upsert_dataframe(StockBasic, df)

    # ------------------------------------------------------------------
    # 2) daily (日线行情)
    # ------------------------------------------------------------------
    def fetch_daily(self, *, ts_codes: Iterable[str], incremental: bool = True) -> int:
        end_str = _today_str()
        starts = self._incremental_starts(StockDaily, "trade_date") if incremental else {}

        def fetch_one(i: int, code: str) -> pd.DataFrame | None:
            del i
            start_str = starts.get(code, self.DEFAULT_START_DATE)
            if start_str >= end_str:
                return None
            return self._fetch_timeseries_range("daily", code, start_str, end_str, row_cap=self.DAILY_ROW_CAP)

        return self._fetch_concurrent(
            api="daily",
            model=StockDaily,
            ts_codes=ts_codes,
            fetch_one=fetch_one,
            log_extra=f" (incremental={incremental})",
        )

    # ------------------------------------------------------------------
    # 3) adj_factor (复权因子)
    # ------------------------------------------------------------------
    def fetch_adj_factor(self, *, ts_codes: Iterable[str], incremental: bool = True) -> int:
        end_str = _today_str()
        starts = self._incremental_starts(AdjFactor, "trade_date") if incremental else {}

        def fetch_one(i: int, code: str) -> pd.DataFrame | None:
            del i
            start_str = starts.get(code, self.DEFAULT_START_DATE)
            if start_str >= end_str:
                return None
            return self._fetch_timeseries_range(
                "adj_factor", code, start_str, end_str, row_cap=self.DAILY_ROW_CAP
            )

        return self._fetch_concurrent(
            api="adj_factor",
            model=AdjFactor,
            ts_codes=ts_codes,
            fetch_one=fetch_one,
            log_extra=f" (incremental={incremental})",
        )

    # ------------------------------------------------------------------
    # 4) daily_basic (每日指标: PE/PB/换手率/市值等)
    # ------------------------------------------------------------------
    def fetch_daily_basic(self, *, ts_codes: Iterable[str], incremental: bool = True) -> int:
        end_str = _today_str()
        starts = self._incremental_starts(DailyBasic, "trade_date") if incremental else {}

        def fetch_one(i: int, code: str) -> pd.DataFrame | None:
            del i
            start_str = starts.get(code, self.DEFAULT_START_DATE)
            if start_str >= end_str:
                return None
            return self._fetch_timeseries_range(
                "daily_basic", code, start_str, end_str, row_cap=self.BASIC_ROW_CAP
            )

        return self._fetch_concurrent(
            api="daily_basic",
            model=DailyBasic,
            ts_codes=ts_codes,
            fetch_one=fetch_one,
            log_extra=f" (incremental={incremental})",
        )

    # ------------------------------------------------------------------
    # 5/6) weekly / monthly
    # ------------------------------------------------------------------
    def fetch_weekly(self, *, ts_codes: Iterable[str], incremental: bool = True) -> int:
        end_str = _today_str()
        starts = self._incremental_starts(StockWeekly, "trade_date") if incremental else {}

        def fetch_one(i: int, code: str) -> pd.DataFrame | None:
            del i
            start_str = starts.get(code, self.DEFAULT_START_DATE)
            if start_str >= end_str:
                return None
            return self._fetch_timeseries_range(
                "weekly", code, start_str, end_str, row_cap=self.DAILY_ROW_CAP
            )

        return self._fetch_concurrent(
            api="weekly",
            model=StockWeekly,
            ts_codes=ts_codes,
            fetch_one=fetch_one,
            log_extra=f" (incremental={incremental})",
        )

    def fetch_monthly(self, *, ts_codes: Iterable[str], incremental: bool = True) -> int:
        end_str = _today_str()
        starts = self._incremental_starts(StockMonthly, "trade_date") if incremental else {}

        def fetch_one(i: int, code: str) -> pd.DataFrame | None:
            del i
            start_str = starts.get(code, self.DEFAULT_START_DATE)
            if start_str >= end_str:
                return None
            return self._fetch_timeseries_range(
                "monthly", code, start_str, end_str, row_cap=self.DAILY_ROW_CAP
            )

        return self._fetch_concurrent(
            api="monthly",
            model=StockMonthly,
            ts_codes=ts_codes,
            fetch_one=fetch_one,
            log_extra=f" (incremental={incremental})",
        )

    # ------------------------------------------------------------------
    # 7) suspend_d (停复牌)
    # ------------------------------------------------------------------
    def fetch_suspend(self, *, ts_codes: Iterable[str]) -> int:
        return self._fetch_per_code_full(api="suspend_d", model=SuspendD, ts_codes=ts_codes)

    # ------------------------------------------------------------------
    # 8) namechange (曾用名)
    # ------------------------------------------------------------------
    def fetch_namechange(self, *, ts_codes: Iterable[str]) -> int:
        return self._fetch_per_code_full(api="namechange", model=NameChange, ts_codes=ts_codes)

    # ------------------------------------------------------------------
    # 9) income (利润表)
    # ------------------------------------------------------------------
    def fetch_income(self, *, ts_codes: Iterable[str]) -> int:
        return self._fetch_per_code_full(
            api="income",
            model=Income,
            ts_codes=ts_codes,
            pk_dropna=["end_date", "report_type"],
        )

    # ------------------------------------------------------------------
    # 10) balancesheet (资产负债表)
    # ------------------------------------------------------------------
    def fetch_balancesheet(self, *, ts_codes: Iterable[str]) -> int:
        return self._fetch_per_code_full(
            api="balancesheet",
            model=BalanceSheet,
            ts_codes=ts_codes,
            pk_dropna=["end_date", "report_type"],
        )

    # ------------------------------------------------------------------
    # 11) cashflow (现金流量表)
    # ------------------------------------------------------------------
    def fetch_cashflow(self, *, ts_codes: Iterable[str]) -> int:
        return self._fetch_per_code_full(
            api="cashflow",
            model=CashFlow,
            ts_codes=ts_codes,
            pk_dropna=["end_date", "report_type"],
        )

    # ------------------------------------------------------------------
    # 12) fina_indicator (财务指标)
    # ------------------------------------------------------------------
    def fetch_fina_indicator(self, *, ts_codes: Iterable[str]) -> int:
        return self._fetch_per_code_full(
            api="fina_indicator",
            model=FinaIndicator,
            ts_codes=ts_codes,
            pk_dropna=["end_date"],
        )

    # ------------------------------------------------------------------
    # 13) forecast (业绩预告)
    # ------------------------------------------------------------------
    def fetch_forecast(self, *, ts_codes: Iterable[str]) -> int:
        return self._fetch_per_code_full(
            api="forecast",
            model=Forecast,
            ts_codes=ts_codes,
            pk_dropna=["end_date", "ann_date"],
        )

    # ------------------------------------------------------------------
    # 14) express (业绩快报)
    # ------------------------------------------------------------------
    def fetch_express(self, *, ts_codes: Iterable[str]) -> int:
        return self._fetch_per_code_full(
            api="express",
            model=Express,
            ts_codes=ts_codes,
            pk_dropna=["end_date"],
        )

    # ------------------------------------------------------------------
    # 15) dividend — stock dividend/split (分红送股)
    # ------------------------------------------------------------------
    def fetch_dividend(self, *, ts_codes: Iterable[str]) -> int:
        return self._fetch_per_code_full(
            api="dividend",
            model=StockDividend,
            ts_codes=ts_codes,
            pk_dropna=["end_date", "div_proc"],
        )

    # ------------------------------------------------------------------
    # 16) moneyflow_hsgt (沪深港通资金流向)
    # ------------------------------------------------------------------
    def fetch_moneyflow_hsgt(self, *, incremental: bool = True) -> int:
        end_str = _today_str()
        start_str = self.DEFAULT_START_DATE
        if incremental:
            start_str = self._single_table_max(
                MoneyflowHsgt, MoneyflowHsgt.trade_date, self.DEFAULT_START_DATE
            )
        if start_str > end_str:
            log.info("moneyflow_hsgt: up to date")
            return 0

        log.info(f"Fetching moneyflow_hsgt ({start_str}..{end_str}) ...")
        df = self.client.call("moneyflow_hsgt", start_date=start_str, end_date=end_str)
        if df is None or df.empty:
            log.info("moneyflow_hsgt: no new data")
            return 0
        df = _normalize_dates(df)
        n = upsert_dataframe(MoneyflowHsgt, df)
        log.info(f"moneyflow_hsgt done. rows={n}")
        return n

    # ------------------------------------------------------------------
    # 17) margin (融资融券交易汇总)
    # ------------------------------------------------------------------
    def fetch_margin(self, *, incremental: bool = True) -> int:
        end_str = _today_str()
        start_str = self.DEFAULT_START_DATE
        if incremental:
            start_str = self._single_table_max(Margin, Margin.trade_date, self.DEFAULT_START_DATE)
        if start_str > end_str:
            log.info("margin: up to date")
            return 0

        log.info(f"Fetching margin ({start_str}..{end_str}) ...")
        total = 0
        for exchange_id in ("SSE", "SZSE"):
            df = self.client.call("margin", exchange_id=exchange_id, start_date=start_str, end_date=end_str)
            if df is None or df.empty:
                log.info(f"  margin {exchange_id}: empty")
                continue
            df = _normalize_dates(df)
            n = upsert_dataframe(Margin, df)
            total += n
            log.info(f"  margin {exchange_id} +{n} rows (total={total})")
        log.info(f"margin done. rows={total}")
        return total

    # ------------------------------------------------------------------
    # 18) margin_detail (融资融券交易明细) — per trade date
    # ------------------------------------------------------------------
    def fetch_margin_detail(self, *, incremental: bool = True) -> int:
        dates = self._open_trade_dates(
            MarginDetail,
            MarginDetail.trade_date,
            self.DEFAULT_START_DATE,
            incremental,
        )
        if not dates:
            log.info("margin_detail: no trade dates to fetch")
            return 0

        def fetch_one(i: int, trade_date_str: str) -> pd.DataFrame | None:
            del i
            return self.client.call("margin_detail", trade_date=trade_date_str)

        return self._fetch_concurrent(
            api="margin_detail",
            model=MarginDetail,
            ts_codes=dates,
            fetch_one=fetch_one,
            log_extra=f" (incremental={incremental})",
        )

    # ------------------------------------------------------------------
    # 19) stk_factor (每日技术/估值因子)
    # ------------------------------------------------------------------
    def fetch_stk_factor(self, *, ts_codes: Iterable[str]) -> int:
        end_str = _today_str()
        starts = self._incremental_starts(StkFactor, "trade_date")

        def fetch_one(i: int, code: str) -> pd.DataFrame | None:
            del i
            start_str = starts.get(code, self.DEFAULT_START_DATE)
            if start_str >= end_str:
                return None
            return self._fetch_timeseries_range(
                "stk_factor", code, start_str, end_str, row_cap=self.DAILY_ROW_CAP
            )

        return self._fetch_concurrent(
            api="stk_factor",
            model=StkFactor,
            ts_codes=ts_codes,
            fetch_one=fetch_one,
        )

    # ------------------------------------------------------------------
    # 20) broker_recommend (券商金股)
    # ------------------------------------------------------------------
    def fetch_broker_recommend(self, *, ts_codes: Iterable[str]) -> int:
        """Pull broker monthly gold-stock picks for the current month."""
        del ts_codes
        month = datetime.now().strftime("%Y%m")
        log.info(f"Fetching broker_recommend (month={month}) ...")
        df = self.client.call("broker_recommend", month=month)
        if df is None or df.empty:
            log.info("broker_recommend: empty")
            return 0
        df = _normalize_dates(df)
        n = upsert_dataframe(BrokerRecommend, df)
        log.info(f"broker_recommend done. rows={n}")
        return n

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _incremental_starts(self, model, date_field: str) -> dict[str, str]:
        """Build {ts_code: 'YYYYMMDD'} map of next-day-after-last-stored."""
        starts: dict[str, str] = {}
        db_col = getattr(model, date_field)
        with session_scope() as sess:
            rows = sess.execute(select(model.ts_code, func.max(db_col)).group_by(model.ts_code)).all()
        for code, mx in rows:
            if mx is None:
                continue
            next_day = mx + timedelta(days=1)
            starts[code] = max(next_day.strftime("%Y%m%d"), self.DEFAULT_START_DATE)
        return starts

    def _single_table_max(self, model, date_col, default_start: str) -> str:
        """Incremental start = max(date) + 1 day, else default."""
        with session_scope() as sess:
            mx = sess.execute(select(func.max(date_col))).scalar()
        if mx is None:
            return default_start
        return max((mx + timedelta(days=1)).strftime("%Y%m%d"), default_start)

    def _open_trade_dates(
        self,
        model,
        date_col,
        default_start: str,
        incremental: bool,
    ) -> list[str]:
        """Get open SSE trade dates from the last stored date forward."""
        end_str = _today_str()
        start_str = default_start
        if incremental:
            start_str = self._single_table_max(model, date_col, default_start)
        if start_str > end_str:
            return []
        with session_scope() as sess:
            rows = (
                sess.execute(
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
