"""Core backtest event loop — matches strategies against historical data, bar by bar."""

from __future__ import annotations

import re
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

# Matches security codes like 000300.XSHG, 600000.SH, 159915.SZ in strategy source
_CODE_RE = re.compile(r"\b(\d{6})\.(XSHG|XSHE|SH|SZ|BJ)\b", re.IGNORECASE)


def _extract_referenced_codes(strategy_source: str) -> list[str]:
    """Extract all security codes referenced in strategy source code.

    This catches codes used in ``attribute_history("000852.XSHG", ...)``,
    ``set_benchmark("000300.XSHG")``, ``get_index_stocks("000300.XSHG")``,
    and any other string literal containing a security code — ensuring the
    engine pre-loads data for all codes the strategy will access at runtime.
    """
    seen: dict[str, None] = {}
    for num, suffix in _CODE_RE.findall(strategy_source):
        seen[f"{num}.{suffix.upper()}"] = None
    return normalize_codes(list(seen))


@dataclass(frozen=True)
class DividendPayment:
    ts_code: str
    pay_date: date
    amount: int
    div_cash: float
    cash: float


@dataclass(frozen=True)
class SplitEvent:
    """A share split/折算 on ``ex_date`` multiplying share count by ``ratio``."""

    ts_code: str
    ex_date: date
    ratio: float


def _load_df_into_proxy(df: pd.DataFrame, data_proxy: DataProxy) -> list[str]:
    """Load an OHLCV DataFrame into DataProxy using fast numpy array conversion.

    Replaces the old ``_group_to_bars`` + ``data_proxy._load`` pair.  Avoids
    creating any ``Bar`` objects — the DataFrame is split by ``ts_code`` and each
    group is handed directly to ``DataProxy._load_df`` which converts columns to
    numpy arrays in one vectorised pass.

    Returns the list of ``ts_code`` values that were loaded (preserving the order
    they first appear in ``df``).
    """
    loaded: list[str] = []
    # Ensure date column is datetime so _load_df can call .dt.date
    if not pd.api.types.is_datetime64_any_dtype(df["date"]):
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])
    for ts_code, group in df.groupby("ts_code", sort=False):
        data_proxy._load_df(str(ts_code), group.reset_index(drop=True))
        loaded.append(str(ts_code))
    return loaded


def _position_value_at_close(
    portfolio: Portfolio,
    data_proxy: DataProxy,
    next_indices: dict[str, int],
    bar_date: date,
) -> float:
    """当日收盘时的持仓市值(不含现金)。"""
    total = 0.0
    for code, position in portfolio.positions.items():
        if position.amount == 0:
            continue
        idx = next_indices.get(code, -1)
        price = position.current_price
        # Only mark to the bar at ``idx`` when it is *today's* bar. If the code
        # has no data on ``bar_date`` (e.g. suspended/missing day), ``idx`` may
        # point at a *future* bar whose close already reflects a share split not
        # yet applied to the holding — using it would distort the equity curve.
        if idx >= 0 and data_proxy.get_date(code, idx) == bar_date:
            arrays = data_proxy._arrays.get(code)
            if arrays is not None and idx < len(arrays["close"]):
                price = float(arrays["close"][idx])
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


def _load_strategy(source: str, strategy_log: StrategyLogCollector | None = None) -> StrategyRuntime:
    """Parse a strategy source string and extract initialize & handle_data.

    Parameters
    ----------
    source:
        Python source code containing ``initialize(context)`` and
        ``handle_data(context)`` function definitions.  May also contain
        imports, helper functions, and global variables.
    strategy_log:
        Optional log collector — if provided, replaces the ``log`` object in
        the strategy namespace so ``log.info()`` calls are captured for the
        Web UI.

    Returns the initialize function, optional handle_data function, and the
    JoinQuant compatibility layer bound to this strategy namespace.
    """
    compat = JoinQuantCompat()
    ns: dict = compat.namespace()
    if strategy_log is not None:
        ns["log"] = strategy_log
    jqdata_module = make_jqdata_module(compat)
    # ``from jqdata import *`` would overwrite ns["log"] with the module's log,
    # so remove ``log`` from jqdata's exports to preserve the strategy_log collector.
    if strategy_log is not None and "log" in jqdata_module.__dict__:
        del jqdata_module.__dict__["log"]
        if "log" in getattr(jqdata_module, "__all__", []):
            jqdata_module.__all__.remove("log")
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
# Strategy log collector — captures log.info()/log.warning() from strategy code
# ---------------------------------------------------------------------------


class StrategyLogCollector:
    """A drop-in replacement for ``log`` that collects strategy output lines.

    Strategy code calls ``log.info("msg")`` / ``log.warning("msg")`` — this
    collector stores each line as ``"[INFO] 2024-01-05: msg"`` (prefixed with
    the current backtest date if available) so the Web UI can display them
    after the backtest finishes.
    """

    def __init__(self) -> None:
        self._lines: list[str] = []
        self._current_date: str = ""

    def set_date(self, date_str: str) -> None:
        self._current_date = date_str

    def _add(self, level: str, msg: str) -> None:
        prefix = f"[{level}]"
        if self._current_date:
            prefix += f" {self._current_date}"
        self._lines.append(f"{prefix}: {msg}")

    def info(self, msg: str, *args: object) -> None:
        self._add("INFO", str(msg).format(*args) if args else str(msg))

    def warning(self, msg: str, *args: object) -> None:
        self._add("WARN", str(msg).format(*args) if args else str(msg))

    def error(self, msg: str, *args: object) -> None:
        self._add("ERROR", str(msg).format(*args) if args else str(msg))

    def debug(self, msg: str, *args: object) -> None:
        self._add("DEBUG", str(msg).format(*args) if args else str(msg))

    @property
    def lines(self) -> list[str]:
        return self._lines


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

        # Auto-inject codes referenced in strategy source but not in ts_codes.
        # The engine only loads data for ts_codes into the DataProxy; if a
        # strategy calls attribute_history("000852.XSHG", ...) for market
        # timing but 000852.SH is not in ts_codes, the call returns empty
        # arrays and the timing logic silently fails.  Scan the source for all
        # security codes (including the benchmark) and append any missing ones
        # so the engine pre-loads their OHLCV data.
        referenced = set(_extract_referenced_codes(strategy_source))
        if self.benchmark_code:
            referenced.add(self.benchmark_code)
        missing = referenced - set(self.ts_codes)
        if missing:
            self.ts_codes = normalize_codes(self.ts_codes + list(missing))
            log.info(f"Auto-injected {len(missing)} code(s) from strategy source: {sorted(missing)}")

    # ------------------------------------------------------------------
    def run(self) -> BacktestResult:
        """Execute the backtest and return a complete result bundle."""
        log.info(
            f"Backtest: {len(self.ts_codes)} codes, "
            f"{self.start_date} → {self.end_date}, "
            f"cash={self.initial_cash:,.0f}"
        )

        # 1. Load strategy
        strategy_log = StrategyLogCollector()
        runtime = _load_strategy(self.strategy_source, strategy_log=strategy_log)
        log.info("Strategy loaded successfully")

        # 2. Load market data
        data_start_date = self.start_date - timedelta(days=self.history_lookback_days)
        raw_df = self.data_source.load_ohlcv(self.ts_codes, data_start_date, self.end_date)
        if raw_df.empty:
            log.warning("No market data found for the given date range")
            return _empty_result(self.initial_cash)

        log.info(f"Loaded {len(raw_df)} OHLCV rows")

        # 3. Load into DataProxy (fast numpy path — no Bar objects created)
        data_proxy = DataProxy()
        all_codes = _load_df_into_proxy(raw_df, data_proxy)
        log.info(f"  {len(all_codes)} codes with bar data")

        dividend_events = self.data_source.load_dividends(self.ts_codes, self.start_date, self.end_date)
        dividends_by_record: dict[date, list[DividendEvent]] = {}
        dividends_by_pay: dict[date, list[DividendEvent]] = {}
        for event in dividend_events:
            dividends_by_record.setdefault(event.record_date, []).append(event)
            dividends_by_pay.setdefault(event.pay_date, []).append(event)

        # Extract share-split events from DataProxy arrays (split_ratio != 1.0).
        split_events: list[SplitEvent] = []
        for code in all_codes:
            sr_arr = data_proxy._split_ratios.get(code)
            dates = data_proxy._dates.get(code, [])
            if sr_arr is not None:
                for i, ratio in enumerate(sr_arr):
                    if abs(float(ratio) - 1.0) > 1e-6:
                        split_events.append(SplitEvent(ts_code=code, ex_date=dates[i], ratio=float(ratio)))

        # 4. Build unified date index from DataProxy date lists
        unified_dates = sorted(
            {
                d
                for code in all_codes
                for d in data_proxy._dates.get(code, [])
                if self.start_date <= d <= self.end_date
            }
        )

        # 5. Set up backtest context (DataProxy already populated above)

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
        next_indices = {code: 0 for code in all_codes}
        dividend_entitlements: dict[DividendEvent, int] = {}
        dividend_payments: list[DividendPayment] = []

        log.info(f"Running {len(unified_dates)} trading days ...")

        for bar_date in unified_dates:
            # Expose current date JoinQuant-style as ``context.current_dt`` so
            # strategies written for JoinQuant run unmodified on this engine.
            context.current_dt = datetime(bar_date.year, bar_date.month, bar_date.day)
            strategy_log.set_date(bar_date.isoformat())

            # Advance data proxy to this date (pointer-chase through sorted date list)
            for code in all_codes:
                dates = data_proxy._dates.get(code)  # noqa: SLF001
                if not dates:
                    data_proxy._current_idx[code] = -1  # noqa: SLF001
                    continue
                idx = next_indices[code]
                n = len(dates)
                while idx < n and dates[idx] < bar_date:
                    idx += 1
                next_indices[code] = idx
                data_proxy._current_idx[code] = idx if (idx < n and dates[idx] == bar_date) else -1  # noqa: SLF001

            # T+1 解锁：新交易日开盘前，把上一日买入而锁定的股数全部释放为可卖。
            # 个股买入时 broker 累加 locked_amount，隔夜即可卖出，符合 A 股 T+1。
            # ETF/指数不会被锁定(broker 按资产类型判定)，此处清零对其无副作用。
            for pos in portfolio.positions.values():
                pos.locked_amount = 0

            # 融资融券每日计息：interest += cash_liability × 日利率
            # 聚宽在每日结算时自动计息，日利率 = 年化/360
            if portfolio.margin_enabled and portfolio.cash_liability > 0:
                daily_rate = portfolio.margin_interest_rate / 360.0
                portfolio.interest += portfolio.cash_liability * daily_rate

            # 份额折算/送转股处理(对齐聚宽 use_real_price=True 的动态复权账户处理)：
            # 在除权日开盘前，按 split_ratio 调整持仓数量与成本价，使持仓市值在除权
            # 前后保持连续(只反映当日真实涨跌)。除权不涉及现金，与分红现金链路相互
            # 独立，故不会双算。split_ratio 由数据源预计算：ETF 取自 accum_nav/unit_nav
            # (剔除分红污染)，个股取自 dividend.stk_div(纯送转比例)，均比 adj_factor 更准。
            for code, pos in list(portfolio.positions.items()):
                if pos.amount == 0:
                    continue
                idx = next_indices.get(code, -1)
                if idx < 0 or data_proxy.get_date(code, idx) != bar_date:
                    continue
                ratio = data_proxy.get_split_ratio(code, idx)
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
            # 融资账户：equity = net_value = 总资产 - 总负债（对齐聚宽 net_value 口径）
            # 普通账户：equity = cash + position_value（= total_value，无负债）
            position_value = _position_value_at_close(portfolio, data_proxy, next_indices, bar_date)
            total_value = portfolio.cash + position_value
            if portfolio.margin_enabled and portfolio.total_liability > 0:
                equity_value = total_value - portfolio.total_liability
            else:
                equity_value = total_value
            equity_records.append(
                {
                    "date": bar_date,
                    "value": equity_value,
                }
            )

            # 融资融券维持担保比例检查：低于阈值时警告（不强制平仓，仅记录）
            if portfolio.margin_enabled and portfolio.total_liability > 0:
                mmr = portfolio.maintenance_margin_rate
                if mmr < portfolio.maintenance_margin_limit:
                    log.warning(
                        f"{bar_date} 维持担保比例 {mmr:.2f} 低于阈值 "
                        f"{portfolio.maintenance_margin_limit:.2f}，"
                        f"总资产={portfolio.total_value:,.0f} 总负债={portfolio.total_liability:,.0f}"
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
            splits=split_events,
        )

        log.info("Backtest complete.")
        log.info(f"\n{metrics.to_llm_prompt()}")

        return BacktestResult(
            metrics=metrics,
            equity_df=equity_df,
            benchmark_df=benchmark_df,
            trades=broker.trades,
            dividends=dividend_payments,
            splits=split_events,
            strategy_logs=strategy_log.lines,
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
        splits: list | None = None,
        strategy_logs: list[str] | None = None,
    ) -> None:
        self.metrics = metrics
        self.equity_df = equity_df
        self.benchmark_df = benchmark_df
        self.trades = trades
        self.dividends = dividends or []
        self.splits = splits or []
        self.strategy_logs = strategy_logs or []

    def to_report_dict(self) -> dict:
        """Return the canonical report payload shared by Web and LLM outputs."""
        return build_report_payload(
            self.equity_df,
            self.benchmark_df,
            self.metrics,
            self.trades,
            self.dividends,
            self.splits,
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
        splits=[],
    )
