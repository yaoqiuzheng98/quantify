"""Core backtest event loop — matches strategies against historical data, bar by bar."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Callable

import pandas as pd
from sqlalchemy import select

from quantify.database.engine import session_scope
from quantify.database.models import EtfAdjFactor, EtfDaily, EtfDividend
from quantify.utils.logger import log

from .broker import Broker, make_commission, make_slippage, zero_slippage
from .codes import normalize_codes, to_tushare_code
from .context import Bar, Context, DataProxy, Portfolio
from .joinquant import JoinQuantCompat, make_jqdata_module
from .metrics import BacktestMetrics, compute_metrics
from .reporting import build_report_payload


def _load_data(
    ts_codes: list[str],
    start_date: date,
    end_date: date,
) -> pd.DataFrame:
    """Load OHLCV data from the database for the given codes and date range.

    Returns a DataFrame with columns: ts_code, date, open, high, low, close,
    volume, amount, pre_close, pct_chg.  Data is sorted by (ts_code, date).
    """
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
            )
            .outerjoin(
                EtfAdjFactor,
                (EtfAdjFactor.ts_code == EtfDaily.ts_code)
                & (EtfAdjFactor.trade_date == EtfDaily.trade_date),
            )
            .where(EtfDaily.ts_code.in_(ts_codes))
            .where(EtfDaily.trade_date >= start_str)
            .where(EtfDaily.trade_date <= end_str)
            .order_by(EtfDaily.ts_code, EtfDaily.trade_date)
        ).all()

    if not rows:
        return pd.DataFrame(
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
            ]
        )

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
    # Missing adjustment factors default to 1.0 (no adjustment).
    df["adj_factor"] = pd.to_numeric(df["adj_factor"], errors="coerce").fillna(1.0)
    df = df.sort_values(["ts_code", "date"]).reset_index(drop=True)
    return df


@dataclass(frozen=True)
class DividendEvent:
    ts_code: str
    record_date: date
    pay_date: date
    div_cash: float


@dataclass(frozen=True)
class DividendPayment:
    ts_code: str
    pay_date: date
    amount: int
    div_cash: float
    cash: float


def _load_dividends(ts_codes: list[str], start_date: date, end_date: date) -> list[DividendEvent]:
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


def _load_benchmark_data(ts_code: str, start_date: date, end_date: date) -> pd.DataFrame | None:
    def _query_rows(before_start: bool) -> list:
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

    previous_rows = _query_rows(before_start=True)
    rows = _query_rows(before_start=False)
    if not rows:
        return None

    def _benchmark_value(row) -> float:
        close = float(row.close)
        factor = float(row.adj_factor) if row.adj_factor is not None else 1.0
        return close * factor

    benchmark_df = pd.DataFrame(
        [
            {
                "date": row.trade_date,
                "value": _benchmark_value(row),
                "daily_value": _benchmark_value(row),
            }
            for row in rows
        ]
    )
    if previous_rows:
        base_value = _benchmark_value(previous_rows[0])
        benchmark_df.attrs["base_value"] = base_value
        benchmark_df.attrs["daily_base_value"] = base_value
    return benchmark_df


def _group_to_bars(df: pd.DataFrame) -> dict[str, list[Bar]]:
    """Convert a DataFrame of OHLCV rows into per-code Bar lists."""
    bars: dict[str, list[Bar]] = {}
    for ts_code, group in df.groupby("ts_code"):
        bars[ts_code] = [
            Bar(
                ts_code=ts_code,
                date=row.date.date() if hasattr(row.date, "date") else row.date,
                open=float(row.open),
                high=float(row.high),
                low=float(row.low),
                close=float(row.close),
                volume=float(row.volume),
                amount=float(row.amount),
                pre_close=float(row.pre_close),
                pct_chg=float(row.pct_chg),
                adj_factor=float(row.adj_factor),
            )
            for row in group.itertuples(index=False)
        ]
    return bars


def _portfolio_value_at_close(
    portfolio: Portfolio,
    all_bars: dict[str, list[Bar]],
    next_indices: dict[str, int],
) -> float:
    value = portfolio.cash
    for code, position in portfolio.positions.items():
        bars = all_bars.get(code, [])
        idx = next_indices.get(code, -1)
        price = position.current_price
        if 0 <= idx < len(bars):
            price = bars[idx].close
        value += position.amount * price
    return value


@dataclass
class StrategyRuntime:
    initialize: Callable
    handle_data: Callable | None
    compat: JoinQuantCompat

    def handle_functions(self) -> list[Callable]:
        if self.compat.daily_functions:
            return list(self.compat.daily_functions)
        return [self.handle_data] if self.handle_data is not None else []


def _load_strategy(source: str) -> StrategyRuntime:
    """Parse a strategy source string and extract initialize & handle_data.

    Parameters
    ----------
    source:
        Python source code containing ``initialize(context)`` and
        ``handle_data(context)`` function definitions.  May also contain
        imports, helper functions, and global variables.

    Returns the initialize function, optional handle_data function, and the
    JoinQuant compatibility layer bound to this strategy namespace.
    """
    compat = JoinQuantCompat()
    ns: dict = compat.namespace()
    jqdata_module = make_jqdata_module(compat)
    previous_jqdata = sys.modules.get("jqdata")
    sys.modules["jqdata"] = jqdata_module
    try:
        exec(source, ns)
    finally:
        if previous_jqdata is None:
            sys.modules.pop("jqdata", None)
        else:
            sys.modules["jqdata"] = previous_jqdata

    init_fn = ns.get("initialize")
    handle_fn = ns.get("handle_data")

    if init_fn is None:
        raise ValueError("Strategy source must define initialize(context)")

    return StrategyRuntime(initialize=init_fn, handle_data=handle_fn, compat=compat)


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class BacktestEngine:
    """Run a strategy against historical data and produce metrics + daily series.

    Typical usage::

        engine = BacktestEngine(
            strategy_source=open("my_strategy.py").read(),
            ts_codes=["510300.SH"],
            start_date="2020-01-01",
            end_date="2024-12-31",
            initial_cash=100000,
            commission_rate=0.0001,   # 0.01% per trade
            commission_min=0,          # no minimum
        )
        result = engine.run()
        print(result.metrics.to_llm_prompt())
    """

    def __init__(
        self,
        strategy_source: str,
        ts_codes: list[str],
        start_date: str,
        end_date: str,
        initial_cash: float = 100000.0,
        benchmark_code: str | None = None,
        commission_rate: float = 0.0005,
        commission_min: float = 0.5,
        slippage_rate: float = 0.0,
        override_strategy_costs: bool = False,
        history_lookback_days: int = 365,
    ) -> None:
        self.strategy_source = strategy_source
        self.ts_codes = normalize_codes(ts_codes)
        self.start_date = _parse_date(start_date)
        self.end_date = _parse_date(end_date)
        self.initial_cash = initial_cash
        self.benchmark_code = to_tushare_code(benchmark_code) if benchmark_code else None
        self.commission_fn = make_commission(rate=commission_rate, minimum=commission_min)
        self.slippage_fn = make_slippage(rate=slippage_rate) if slippage_rate > 0 else zero_slippage
        self.override_strategy_costs = override_strategy_costs
        self.history_lookback_days = history_lookback_days

    # ------------------------------------------------------------------
    def run(self) -> BacktestResult:
        """Execute the backtest and return a complete result bundle."""
        log.info(
            f"Backtest: {len(self.ts_codes)} codes, "
            f"{self.start_date} → {self.end_date}, "
            f"cash={self.initial_cash:,.0f}"
        )

        # 1. Load strategy
        runtime = _load_strategy(self.strategy_source)
        log.info("Strategy loaded successfully")

        # 2. Load market data
        data_start_date = self.start_date - timedelta(days=self.history_lookback_days)
        raw_df = _load_data(self.ts_codes, data_start_date, self.end_date)
        if raw_df.empty:
            log.warning("No market data found for the given date range")
            return _empty_result(self.initial_cash)

        log.info(f"Loaded {len(raw_df)} OHLCV rows")

        # 3. Convert to Bar structures
        all_bars = _group_to_bars(raw_df)
        log.info(f"  {len(all_bars)} codes with bar data")

        dividend_events = _load_dividends(self.ts_codes, self.start_date, self.end_date)
        dividends_by_record: dict[date, list[DividendEvent]] = {}
        dividends_by_pay: dict[date, list[DividendEvent]] = {}
        for event in dividend_events:
            dividends_by_record.setdefault(event.record_date, []).append(event)
            dividends_by_pay.setdefault(event.pay_date, []).append(event)

        # 4. Build unified date index (sorted union of all bar dates)
        unified_dates = sorted(
            {
                bar.date
                for bars in all_bars.values()
                for bar in bars
                if self.start_date <= bar.date <= self.end_date
            }
        )

        # 5. Set up backtest context
        data_proxy = DataProxy()
        for code, bars in all_bars.items():
            data_proxy._load(code, bars)

        portfolio = Portfolio(initial_cash=self.initial_cash, cash=self.initial_cash)
        broker = Broker(
            data=data_proxy,
            commission_fn=self.commission_fn,
            slippage_fn=self.slippage_fn,
        )
        context = Context(
            portfolio=portfolio,
            broker=broker,
            data=data_proxy,
            start_date=self.start_date,
            end_date=self.end_date,
        )
        context.set_benchmark(self.benchmark_code or self.ts_codes[0])

        # Override submit_order reference so context.order* methods work
        context._broker = broker  # noqa: SLF001

        # 6. Run initialize
        runtime.compat.bind(context)
        runtime.initialize(context)
        if self.benchmark_code:
            context.set_benchmark(self.benchmark_code)
        if self.override_strategy_costs:
            broker.set_commission_fn(self.commission_fn)
            broker.set_slippage_fn(self.slippage_fn)

        handle_functions = runtime.handle_functions()
        if not handle_functions:
            raise ValueError(
                "Strategy must define handle_data(context) or register run_daily(..., time='open')"
            )
        benchmark_df = (
            _load_benchmark_data(context.benchmark_code, self.start_date, self.end_date)
            if context.benchmark_code
            else None
        )

        # 7. Main event loop — bar by bar
        equity_records: list[dict] = []
        next_indices = {code: 0 for code in all_bars}
        dividend_entitlements: dict[DividendEvent, int] = {}
        dividend_payments: list[DividendPayment] = []

        log.info(f"Running {len(unified_dates)} trading days ...")

        for bar_date in unified_dates:
            # Expose current date JoinQuant-style as ``context.current_dt`` so
            # strategies written for JoinQuant run unmodified on this engine.
            context.current_dt = datetime(bar_date.year, bar_date.month, bar_date.day)

            # Advance data proxy to this date
            for code, bars in all_bars.items():
                idx = next_indices[code]
                while idx < len(bars) and bars[idx].date < bar_date:
                    idx += 1
                next_indices[code] = idx
                if idx < len(bars) and bars[idx].date == bar_date:
                    data_proxy._current_idx[code] = idx  # noqa: SLF001
                else:
                    data_proxy._current_idx[code] = -1  # noqa: SLF001

            # Mark current holdings at today's open before sizing orders.
            for code, pos in list(portfolio.positions.items()):
                bar = data_proxy.current(code)
                if isinstance(bar, Bar):
                    pos.current_price = bar.open

            for event in dividends_by_pay.get(bar_date, []):
                entitled_amount = dividend_entitlements.pop(event, 0)
                if entitled_amount > 0:
                    cash = entitled_amount * event.div_cash
                    portfolio.cash += cash
                    dividend_payments.append(
                        DividendPayment(
                            ts_code=event.ts_code,
                            pay_date=event.pay_date,
                            amount=entitled_amount,
                            div_cash=event.div_cash,
                            cash=cash,
                        )
                    )

            # Call user strategy with today's open + completed history only.
            for handle_fn in handle_functions:
                handle_fn(context)

            # Execute orders generated by today's strategy at today's open.
            broker.execute_pending(portfolio)

            # Update position prices after fills.
            for code, pos in list(portfolio.positions.items()):
                bar = data_proxy.current(code)
                if isinstance(bar, Bar):
                    pos.current_price = bar.open

            for event in dividends_by_record.get(bar_date, []):
                position = portfolio.positions.get(event.ts_code)
                if position is not None and position.amount > 0:
                    dividend_entitlements[event] = position.amount

            # Record daily snapshot
            equity_records.append(
                {
                    "date": bar_date,
                    "value": _portfolio_value_at_close(portfolio, all_bars, next_indices),
                }
            )

        cancelled = broker.cancel_pending()
        if cancelled:
            log.info(f"Cancelled {cancelled} unfilled order(s) after final bar")

        # 8. Build results
        equity_df = pd.DataFrame(equity_records)

        metrics = compute_metrics(
            equity_df,
            initial_cash=self.initial_cash,
            commission=portfolio.total_commission,
            slippage=portfolio.total_slippage,
            trade_count=portfolio.trade_count,
        )

        log.info("Backtest complete.")
        log.info(f"\n{metrics.to_llm_prompt()}")

        return BacktestResult(
            metrics=metrics,
            equity_df=equity_df,
            benchmark_df=benchmark_df,
            trades=broker.trades,
            dividends=dividend_payments,
        )


class BacktestResult:
    """Bundles everything produced by a backtest run."""

    def __init__(
        self,
        metrics: BacktestMetrics,
        equity_df: pd.DataFrame,
        benchmark_df: pd.DataFrame | None,
        trades: list,
        dividends: list | None = None,
    ) -> None:
        self.metrics = metrics
        self.equity_df = equity_df
        self.benchmark_df = benchmark_df
        self.trades = trades
        self.dividends = dividends or []

    def to_report_dict(self) -> dict:
        """Return the canonical report payload shared by Web and LLM outputs."""
        return build_report_payload(
            self.equity_df,
            self.benchmark_df,
            self.metrics,
            self.trades,
            self.dividends,
        )

    def to_llm_dict(self) -> dict:
        """Return the same canonical payload consumed by the Web dashboard."""
        return self.to_report_dict()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_date(s: str) -> date:
    for fmt in ("%Y-%m-%d", "%Y%m%d"):
        try:
            return datetime.strptime(s.strip(), fmt).date()
        except ValueError:
            pass
    raise ValueError(f"Cannot parse date: {s}")


def _empty_result(initial_cash: float) -> BacktestResult:
    return BacktestResult(
        metrics=compute_metrics(pd.DataFrame(), initial_cash=initial_cash),
        equity_df=pd.DataFrame(),
        benchmark_df=None,
        trades=[],
        dividends=[],
    )
