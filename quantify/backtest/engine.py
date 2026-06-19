"""Core backtest event loop — matches strategies against historical data, bar by bar."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Callable

import pandas as pd

from quantify.utils.logger import log

from .broker import Broker, make_commission, make_slippage, zero_slippage
from .codes import normalize_codes, to_tushare_code
from .context import Bar, Context, DataProxy, Portfolio
from .datasource import CompositeDataSource, DividendEvent, MarketDataSource
from .joinquant import JoinQuantCompat, make_jqdata_module
from .metrics import BacktestMetrics, compute_metrics
from .reporting import build_report_payload


@dataclass(frozen=True)
class DividendPayment:
    ts_code: str
    pay_date: date
    amount: int
    div_cash: float
    cash: float


def _group_to_bars(df: pd.DataFrame) -> dict[str, list[Bar]]:
    """Convert a DataFrame of OHLCV rows into per-code Bar lists.

    The ``split_ratio`` column is precomputed by the data source (NAV-based for
    ETFs, ``stk_div`` for stocks), so this is a pure structural conversion.
    """
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
                split_ratio=float(row.split_ratio),
            )
            for row in group.itertuples(index=False)
        ]
    return bars


def _position_value_at_close(
    portfolio: Portfolio,
    all_bars: dict[str, list[Bar]],
    next_indices: dict[str, int],
    bar_date: date,
) -> float:
    """当日收盘时的持仓市值(不含现金)。"""
    total = 0.0
    for code, position in portfolio.positions.items():
        if position.amount == 0:
            continue
        bars = all_bars.get(code, [])
        idx = next_indices.get(code, -1)
        price = position.current_price
        # Only mark to the bar at ``idx`` when it is *today's* bar. If the code
        # has no data on ``bar_date`` (e.g. suspended/missing day), ``idx`` may
        # point at a *future* bar whose close already reflects a share split not
        # yet applied to the holding — using it would distort the equity curve.
        if 0 <= idx < len(bars) and bars[idx].date == bar_date:
            price = bars[idx].close
        total += position.amount * price
    return total


def _schedule_fire_dates(unified_dates: list, period: str) -> dict[int, set]:
    """按"第 N 个交易日"语义，预计算各 day 偏移对应的触发日期集合。

    period="weekly"  以 ISO 周(年, 周号)分组；period="monthly" 以(年, 月)分组。
    返回 {day_offset: {date, ...}}：day_offset>0 取组内正数第 N 个交易日(1-based)，
    day_offset<0 取倒数第 |N| 个。聚宽 run_weekly(weekday=)/run_monthly(monthday=)
    即此语义(1=该周期首个交易日)。仅为已可能用到的偏移构建集合(惰性按需在引擎查询)。
    """
    groups: dict[tuple, list] = {}
    order: list[tuple] = []
    for d in unified_dates:
        if period == "weekly":
            iso = d.isocalendar()
            key = (iso[0], iso[1])
        else:
            key = (d.year, d.month)
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(d)

    fire: dict[int, set] = {}
    for key in order:
        days = groups[key]
        n = len(days)
        for pos, d in enumerate(days):
            pos_offset = pos + 1  # 正数第 N 个交易日(1-based)
            neg_offset = -(n - pos)  # 倒数第 N 个交易日
            fire.setdefault(pos_offset, set()).add(d)
            fire.setdefault(neg_offset, set()).add(d)
    return fire


@dataclass
class StrategyRuntime:
    initialize: Callable
    handle_data: Callable | None
    compat: JoinQuantCompat

    def handle_functions(self) -> list[Callable]:
        if self.compat.scheduled:
            return [func for func, _freq, _day in self.compat.scheduled]
        return [self.handle_data] if self.handle_data is not None else []

    def scheduled_tasks(self) -> list[tuple[Callable, str, int]]:
        if self.compat.scheduled:
            return list(self.compat.scheduled)
        if self.handle_data is not None:
            return [(self.handle_data, "daily", 0)]
        return []


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
        data_source: MarketDataSource | None = None,
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
        # Routes each code (ETF / stock / index) to the right tables. A custom
        # source can be injected for testing or alternative data backends.
        self.data_source = data_source or CompositeDataSource()

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
        raw_df = self.data_source.load_ohlcv(self.ts_codes, data_start_date, self.end_date)
        if raw_df.empty:
            log.warning("No market data found for the given date range")
            return _empty_result(self.initial_cash)

        log.info(f"Loaded {len(raw_df)} OHLCV rows")

        # 3. Convert to Bar structures
        all_bars = _group_to_bars(raw_df)
        log.info(f"  {len(all_bars)} codes with bar data")

        dividend_events = self.data_source.load_dividends(self.ts_codes, self.start_date, self.end_date)
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

        scheduled_tasks = runtime.scheduled_tasks()
        if not scheduled_tasks:
            raise ValueError(
                "Strategy must define handle_data(context) or register run_daily(..., time='open')"
            )
        # 预计算每个交易日触发哪些 weekly/monthly 任务：按"第 N 个交易日"语义
        # (monthday/weekday 为正=正数第 N 个交易日，为负=倒数第 N 个)。daily 任务每天触发。
        weekly_fire = _schedule_fire_dates(unified_dates, period="weekly")
        monthly_fire = _schedule_fire_dates(unified_dates, period="monthly")
        benchmark_df = (
            self.data_source.load_benchmark(context.benchmark_code, self.start_date, self.end_date)
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

            # T+1 解锁：新交易日开盘前，把上一日买入而锁定的股数全部释放为可卖。
            # 个股买入时 broker 累加 locked_amount，隔夜即可卖出，符合 A 股 T+1。
            # ETF/指数不会被锁定(broker 按资产类型判定)，此处清零对其无副作用。
            for pos in portfolio.positions.values():
                pos.locked_amount = 0

            # 份额折算/送转股处理(对齐聚宽 use_real_price=True 的动态复权账户处理)：
            # 在除权日开盘前，按 split_ratio 调整持仓数量与成本价，使持仓市值在除权
            # 前后保持连续(只反映当日真实涨跌)。除权不涉及现金，与分红现金链路相互
            # 独立，故不会双算。split_ratio 由数据源预计算：ETF 取自 accum_nav/unit_nav
            # (剔除分红污染)，个股取自 dividend.stk_div(纯送转比例)，均比 adj_factor 更准。
            for code, pos in list(portfolio.positions.items()):
                if pos.amount == 0:
                    continue
                idx = next_indices[code]
                bars = all_bars.get(code, [])
                if not (0 <= idx < len(bars) and bars[idx].date == bar_date):
                    continue
                ratio = bars[idx].split_ratio
                if ratio and abs(ratio - 1.0) > 1e-6:
                    pos.amount = int(round(pos.amount * ratio))
                    pos.avg_cost = pos.avg_cost / ratio

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
            # 按注册频率触发：daily 每天；weekly/monthly 仅在当周/当月第 N 个交易日。
            for handle_fn, freq, day in scheduled_tasks:
                if freq == "daily":
                    handle_fn(context)
                elif freq == "weekly":
                    if bar_date in weekly_fire.get(day, ()):
                        handle_fn(context)
                elif freq == "monthly":
                    if bar_date in monthly_fire.get(day, ()):
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
            position_value = _position_value_at_close(portfolio, all_bars, next_indices, bar_date)
            equity_records.append(
                {
                    "date": bar_date,
                    "value": portfolio.cash + position_value,
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
            tax=portfolio.total_tax,
            trade_count=portfolio.trade_count,
            trades=broker.trades,
            dividends=dividend_events,
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
