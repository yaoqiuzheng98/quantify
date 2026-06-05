"""Core backtest event loop — matches strategies against historical data, bar by bar."""

from __future__ import annotations

from datetime import date, datetime
from typing import Callable

import pandas as pd
from sqlalchemy import select

from quantify.database.engine import session_scope
from quantify.database.models import EtfDaily
from quantify.utils.logger import log

from .broker import Broker, make_commission, make_slippage, zero_slippage
from .context import Bar, Context, DataProxy, Portfolio
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
        ],
    )
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["ts_code", "date"]).reset_index(drop=True)
    return df


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
            )
            for row in group.itertuples(index=False)
        ]
    return bars


def _load_strategy(source: str) -> tuple[Callable, Callable]:
    """Parse a strategy source string and extract initialize & handle_data.

    Parameters
    ----------
    source:
        Python source code containing ``initialize(context)`` and
        ``handle_data(context)`` function definitions.  May also contain
        imports, helper functions, and global variables.

    Returns
    -------
    (initialize, handle_data) callables.
    """
    ns: dict = {}
    exec(source, ns)

    init_fn = ns.get("initialize")
    handle_fn = ns.get("handle_data")

    if init_fn is None or handle_fn is None:
        raise ValueError(
            "Strategy source must define both initialize(context) and handle_data(context) functions"
        )

    return init_fn, handle_fn


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
        commission_rate: float = 0.00015,
        commission_min: float = 5.0,
        slippage_rate: float = 0.0,
    ) -> None:
        self.strategy_source = strategy_source
        self.ts_codes = list(ts_codes)
        self.start_date = _parse_date(start_date)
        self.end_date = _parse_date(end_date)
        self.initial_cash = initial_cash
        self.benchmark_code = benchmark_code
        self.commission_fn = make_commission(rate=commission_rate, minimum=commission_min)
        self.slippage_fn = make_slippage(rate=slippage_rate) if slippage_rate > 0 else zero_slippage

    # ------------------------------------------------------------------
    def run(self) -> BacktestResult:
        """Execute the backtest and return a complete result bundle."""
        log.info(
            f"Backtest: {len(self.ts_codes)} codes, "
            f"{self.start_date} → {self.end_date}, "
            f"cash={self.initial_cash:,.0f}"
        )

        # 1. Load strategy
        initialize_fn, handle_data_fn = _load_strategy(self.strategy_source)
        log.info("Strategy loaded successfully")

        # 2. Load market data
        raw_df = _load_data(self.ts_codes, self.start_date, self.end_date)
        if raw_df.empty:
            log.warning("No market data found for the given date range")
            return _empty_result(self.initial_cash)

        log.info(f"Loaded {len(raw_df)} OHLCV rows")

        # 3. Convert to Bar structures
        all_bars = _group_to_bars(raw_df)
        log.info(f"  {len(all_bars)} codes with bar data")

        # 4. Build unified date index (sorted union of all bar dates)
        unified_dates = sorted(set(b.date for bars in all_bars.values() for b in bars))

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
        initialize_fn(context)

        # 7. Main event loop — bar by bar
        equity_records: list[dict] = []
        benchmark_records: list[dict] = []
        next_indices = {code: 0 for code in all_bars}

        log.info(f"Running {len(unified_dates)} trading days ...")

        for bar_date in unified_dates:
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

            # Call user strategy with today's open + completed history only.
            handle_data_fn(context)

            # Execute orders generated by today's strategy at today's open.
            broker.execute_pending(portfolio)

            # Update position prices after fills.
            for code, pos in list(portfolio.positions.items()):
                bar = data_proxy.current(code)
                if isinstance(bar, Bar):
                    pos.current_price = bar.open

            # Record daily snapshot
            equity_records.append(
                {
                    "date": bar_date,
                    "value": portfolio.total_value,
                }
            )

            # Benchmark tracking
            if context.benchmark_code:
                bm_bar = data_proxy.current(context.benchmark_code)
                if isinstance(bm_bar, Bar):
                    benchmark_records.append(
                        {
                            "date": bar_date,
                            "value": bm_bar.open,
                        }
                    )

        cancelled = broker.cancel_pending()
        if cancelled:
            log.info(f"Cancelled {cancelled} unfilled order(s) after final bar")

        # 8. Build results
        equity_df = pd.DataFrame(equity_records)
        bm_df = pd.DataFrame(benchmark_records) if benchmark_records else None

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
            benchmark_df=bm_df,
            trades=broker.trades,
        )


class BacktestResult:
    """Bundles everything produced by a backtest run."""

    def __init__(
        self,
        metrics: BacktestMetrics,
        equity_df: pd.DataFrame,
        benchmark_df: pd.DataFrame | None,
        trades: list,
    ) -> None:
        self.metrics = metrics
        self.equity_df = equity_df
        self.benchmark_df = benchmark_df
        self.trades = trades

    def to_report_dict(self) -> dict:
        """Return the canonical report payload shared by Web and LLM outputs."""
        return build_report_payload(self.equity_df, self.benchmark_df, self.metrics, self.trades)

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
    )
