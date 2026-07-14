"""Strategy execution context and portfolio tracking, modelled on JoinQuant API."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any

import numpy as np
import pandas as pd

from .codes import to_tushare_code

# Fields that are price-based and require adj_factor adjustment
_PRICE_FIELDS = frozenset({"open", "high", "low", "close", "pre_close", "vwap"})
# All OHLCV fields stored as numpy arrays
_ARRAY_FIELDS = ("open", "high", "low", "close", "pre_close", "volume", "amount", "pct_chg", "turnover_rate", "vwap")

# ---------------------------------------------------------------------------
# Portfolio
# ---------------------------------------------------------------------------


@dataclass
class Position:
    """Snapshot of a single asset holding."""

    ts_code: str
    amount: int = 0
    avg_cost: float = 0.0
    current_price: float = 0.0
    # 当日买入、受 T+1 限制当日不可卖出的股数。每个交易日开盘前由引擎清零
    # (隔夜后解锁)，股票买入成交时累加。ETF/指数不设限，恒为 0。
    locked_amount: int = 0

    @property
    def market_value(self) -> float:
        return self.amount * self.current_price

    @property
    def value(self) -> float:
        """JoinQuant-compatible alias of ``market_value`` (持仓市值)."""
        return self.market_value

    @property
    def total_amount(self) -> int:
        return self.amount

    @property
    def closeable_amount(self) -> int:
        """可平仓(可卖出)数量 = 总持仓 - 当日锁定 (聚宽同名字段语义)。"""
        return max(0, self.amount - self.locked_amount)

    @property
    def pnl(self) -> float:
        return self.amount * (self.current_price - self.avg_cost)

    @property
    def pnl_pct(self) -> float:
        if self.avg_cost == 0:
            return 0.0
        return (self.current_price / self.avg_cost - 1) * 100

    @property
    def price(self) -> float:
        """JoinQuant-compatible alias of ``current_price`` (最新行情价格)."""
        return self.current_price


class PositionBook(dict[str, Position]):
    """Position mapping that accepts both JoinQuant and Tushare codes."""

    def __contains__(self, key: object) -> bool:
        if isinstance(key, str):
            key = to_tushare_code(key)
        return super().__contains__(key)

    def __getitem__(self, key: str) -> Position:
        return super().__getitem__(to_tushare_code(key))

    def __setitem__(self, key: str, value: Position) -> None:
        super().__setitem__(to_tushare_code(key), value)

    def get(self, key: str, default: Position | None = None) -> Position | None:  # type: ignore[override]
        return super().get(to_tushare_code(key), default)


@dataclass
class Portfolio:
    """Tracks cash, positions, and total value throughout the backtest.

    When *margin_enabled* is True (JoinQuant ``stock_margin`` account), buying
    power exceeds cash — the broker can borrow on the investor's behalf, with
    the borrowed amount tracked in *cash_liability* and accrued interest in
    *interest*.  This mirrors the JoinQuant margin account semantics so that
    strategies using ``set_subportfolios([SubPortfolioConfig(..., type='stock_margin')])``
    run identically locally and in the cloud.
    """

    initial_cash: float
    cash: float
    positions: PositionBook = field(default_factory=PositionBook)
    total_commission: float = 0.0
    total_slippage: float = 0.0
    total_tax: float = 0.0
    trade_count: int = 0

    # --- margin (融资融券) fields ---
    margin_enabled: bool = False
    cash_liability: float = 0.0  # 融资负债（向券商借入的资金本金）
    interest: float = 0.0  # 累计未还利息
    # 融资年化利率（聚宽默认 ~8.6%，日利率 = annual/360）
    margin_interest_rate: float = 0.086
    # 维持担保比例下限，低于此值触发强制平仓
    maintenance_margin_limit: float = 1.30

    @property
    def total_value(self) -> float:
        """总资产 = 现金 + 证券市值（含融资买入的证券）。"""
        pos_val = sum(p.market_value for p in self.positions.values())
        return self.cash + pos_val

    @property
    def total_liability(self) -> float:
        """总负债 = 融资负债 + 融券负债 + 利息（对齐聚宽）。"""
        return self.cash_liability + self.interest

    @property
    def net_value(self) -> float:
        """净资产 = 总资产 - 总负债。"""
        return self.total_value - self.total_liability

    @property
    def available_cash(self) -> float:
        """可用资金 = 现金（聚宽口径：已扣除融资负债的部分）。"""
        return max(self.cash, 0.0)

    @property
    def maintenance_margin_rate(self) -> float:
        """维持担保比例 = 总资产 / 总负债。"""
        if self.total_liability <= 0:
            return float("inf")
        return self.total_value / self.total_liability

    @property
    def starting_cash(self) -> float:
        """聚宽兼容：初始资金。"""
        return self.initial_cash

    @property
    def total_pnl(self) -> float:
        return self.net_value - self.initial_cash

    @property
    def total_return_pct(self) -> float:
        if self.initial_cash == 0:
            return 0.0
        return (self.net_value / self.initial_cash - 1) * 100

    def get_position(self, ts_code: str) -> Position:
        ts_code = to_tushare_code(ts_code)
        if ts_code not in self.positions:
            self.positions[ts_code] = Position(ts_code=ts_code)
        return self.positions[ts_code]


# ---------------------------------------------------------------------------
# Order
# ---------------------------------------------------------------------------

ORDER_STATUS_NEW = "new"
ORDER_STATUS_FILLED = "filled"
ORDER_STATUS_REJECTED = "rejected"


@dataclass
class Order:
    """A single order record."""

    ts_code: str
    amount: int
    order_price: float | None
    created_date: date
    status: str = ORDER_STATUS_NEW
    filled_date: date | None = None
    filled_price: float | None = None
    filled_amount: int = 0
    commission: float = 0.0
    slippage: float = 0.0
    tax: float = 0.0


# ---------------------------------------------------------------------------
# Execution Context
# ---------------------------------------------------------------------------


@dataclass
class Bar:
    """A single bar of market data for one asset."""

    ts_code: str
    date: date
    open: float
    high: float
    low: float
    close: float
    volume: float
    amount: float
    pre_close: float
    pct_chg: float
    adj_factor: float = 1.0
    # 当日发生 ETF 份额折算时的折算比例(份额×ratio、价格÷ratio)；
    # 非折算日为 1.0。由 _group_to_bars 预计算(见引擎)。
    split_ratio: float = 1.0

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)


class DataProxy:
    """Access current and historical market data within a strategy.

    Internally stores data as numpy arrays for O(1) slicing in ``history()``,
    which is the hot path when strategies call ``attribute_history`` on many codes.
    The old ``list[Bar]`` storage is gone; Bar objects are reconstructed on demand
    only for ``current()`` (called only on held positions, not on all codes).
    """

    def __init__(self) -> None:
        # Per-code numpy arrays: ts_code -> {field: np.ndarray}
        self._arrays: dict[str, dict[str, np.ndarray]] = {}
        # adj_factor array (separate because it's used in every price history call)
        self._adj: dict[str, np.ndarray] = {}
        # Date list for each code (used by engine for date-pointer advance)
        self._dates: dict[str, list[date]] = {}
        # split_ratio array (used by engine for corporate-action adjustments)
        self._split_ratios: dict[str, np.ndarray] = {}
        # ts_code of the Bar object (needed to reconstruct Bar in current())
        self._current_idx: dict[str, int] = {}

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _load(self, ts_code: str, bars: list[Bar]) -> None:
        """Load from a ``list[Bar]`` — kept for backward compatibility."""
        ts_code = to_tushare_code(ts_code)
        if not bars:
            return
        n = len(bars)
        arrays: dict[str, np.ndarray] = {}
        for f in _ARRAY_FIELDS:
            arrays[f] = np.fromiter((getattr(b, f, 0.0) for b in bars), dtype=np.float64, count=n)
        self._arrays[ts_code] = arrays
        self._adj[ts_code] = np.fromiter((b.adj_factor for b in bars), dtype=np.float64, count=n)
        self._dates[ts_code] = [b.date for b in bars]
        self._split_ratios[ts_code] = np.fromiter((b.split_ratio for b in bars), dtype=np.float64, count=n)
        self._current_idx[ts_code] = -1

    def _load_df(self, ts_code: str, group: pd.DataFrame) -> None:
        """Fast load path: build numpy arrays directly from a DataFrame group.

        Skips ``Bar`` object creation entirely — avoids iterating over rows in
        Python and constructing per-row dataclass instances.  Equivalent to
        ``_load`` but ~10× faster for large groups.
        """
        ts_code = to_tushare_code(ts_code)
        if group.empty:
            return
        n = len(group)
        arrays: dict[str, np.ndarray] = {}
        for f in _ARRAY_FIELDS:
            col = group[f] if f in group.columns else pd.Series(np.zeros(n))
            arrays[f] = col.to_numpy(dtype=np.float64, na_value=0.0)
        self._arrays[ts_code] = arrays
        adj_col = group["adj_factor"] if "adj_factor" in group.columns else pd.Series(np.ones(n))
        self._adj[ts_code] = adj_col.to_numpy(dtype=np.float64, na_value=1.0)
        # dates: convert once to Python date objects (engine uses them for pointer advance)
        self._dates[ts_code] = group["date"].dt.date.tolist()
        split_col = group["split_ratio"] if "split_ratio" in group.columns else pd.Series(np.ones(n))
        self._split_ratios[ts_code] = split_col.to_numpy(dtype=np.float64, na_value=1.0)
        self._current_idx[ts_code] = -1

    # ------------------------------------------------------------------
    # Engine interface
    # ------------------------------------------------------------------

    def _advance(self, ts_code: str) -> None:
        ts_code = to_tushare_code(ts_code)
        if ts_code in self._current_idx:
            self._current_idx[ts_code] += 1

    def get_date(self, ts_code: str, idx: int) -> date | None:
        """Return the date at position ``idx`` for ``ts_code``."""
        dates = self._dates.get(ts_code)
        if dates is None or idx < 0 or idx >= len(dates):
            return None
        return dates[idx]

    def get_split_ratio(self, ts_code: str, idx: int) -> float:
        """Return the split_ratio at position ``idx`` for ``ts_code``."""
        sr = self._split_ratios.get(ts_code)
        if sr is None or idx < 0 or idx >= len(sr):
            return 1.0
        return float(sr[idx])

    def code_count(self, ts_code: str) -> int:
        """Total number of bars loaded for ``ts_code``."""
        dates = self._dates.get(ts_code)
        return len(dates) if dates else 0

    # ------------------------------------------------------------------
    # Strategy interface
    # ------------------------------------------------------------------

    def current(self, ts_code: str, field: str | None = None) -> float | Bar | None:
        ts_code = to_tushare_code(ts_code)
        idx = self._current_idx.get(ts_code, -1)
        arrays = self._arrays.get(ts_code)
        if idx < 0 or arrays is None:
            return None
        n = len(next(iter(arrays.values())))
        if idx >= n:
            return None
        # Strategy code runs at the open: only the open price is known for the current bar.
        open_price = float(arrays["open"][idx])
        pre_close = float(arrays["pre_close"][idx])
        bar_date = self._dates[ts_code][idx] if self._dates.get(ts_code) else None
        visible_bar = Bar(
            ts_code=ts_code,
            date=bar_date,
            open=open_price,
            high=open_price,
            low=open_price,
            close=open_price,
            volume=0.0,
            amount=0.0,
            pre_close=pre_close,
            pct_chg=0.0,
        )
        if field is None:
            return visible_bar
        return getattr(visible_bar, field, None)

    def history(self, ts_code: str, count: int, field: str = "close") -> np.ndarray:
        """Return up to ``count`` historical bars of ``field``, front-adjusted.

        Returns a numpy array (compatible with ``pd.DataFrame`` and standard
        pandas/numpy operations).  The hot path uses O(1) array slicing instead
        of the old O(count) Python loop with ``getattr``.
        """
        ts_code = to_tushare_code(ts_code)
        idx = self._current_idx.get(ts_code, -1)
        arrays = self._arrays.get(ts_code)
        if idx < 0 or arrays is None:
            return np.array([], dtype=np.float64)
        end = idx - 1  # history is up to yesterday (strategy runs at open)
        if end < 0:
            return np.array([], dtype=np.float64)
        start = max(0, end - count + 1)

        arr = arrays.get(field)
        if arr is None:
            return np.array([], dtype=np.float64)

        slice_ = arr[start : end + 1]

        # Price fields are front-adjusted (前复权) to the current bar's basis so
        # that dividend/ex-rights gaps don't create spurious returns — matching
        # JoinQuant's ``attribute_history`` under ``use_real_price=True``.
        if field in _PRICE_FIELDS:
            adj = self._adj.get(ts_code)
            if adj is not None and idx < len(adj):
                base_factor = adj[idx]
                if base_factor <= 0.0:
                    base_factor = 1.0
                slice_ = slice_ * adj[start : end + 1] / base_factor

        return slice_

    @property
    def today(self) -> date | None:
        for code, idx in self._current_idx.items():
            dates = self._dates.get(code)
            if dates and 0 <= idx < len(dates):
                return dates[idx]
        return None


class Context:
    """Strategy execution context — the ``context`` object passed to strategies."""

    def __init__(
        self,
        portfolio: Portfolio,
        broker,
        data: DataProxy,
        start_date: date,
        end_date: date,
    ) -> None:
        self.portfolio = portfolio
        self._broker = broker
        self.data = data
        self.start_date = start_date
        self.end_date = end_date
        self.benchmark_code: str | None = None
        self._user_attrs: dict[str, Any] = {}

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        try:
            return self._user_attrs[name]
        except KeyError:
            raise AttributeError(name) from None

    def __setattr__(self, name: str, value: Any) -> None:
        if name in {
            "portfolio",
            "data",
            "start_date",
            "end_date",
            "benchmark_code",
            "_broker",
            "_user_attrs",
        }:
            super().__setattr__(name, value)
        else:
            self._user_attrs[name] = value

    def __delattr__(self, name: str) -> None:
        if name in self._user_attrs:
            del self._user_attrs[name]
        else:
            super().__delattr__(name)

    # -- strategy API helpers -------------------------------------------------

    def set_benchmark(self, ts_code: str) -> None:
        self.benchmark_code = to_tushare_code(ts_code)

    def order_value(self, ts_code: str, value: float) -> Order | None:
        ts_code = to_tushare_code(ts_code)
        price = self._broker.current_price(ts_code)
        if price is None or price <= 0:
            return None
        amount = int(value / price)
        return self.order(ts_code, amount)

    def order_target_value(self, ts_code: str, target_value: float) -> Order | None:
        """对齐聚宽口径:先算 diff=(target_value-current_value)/price,再向零取整到 lot。

        聚宽 order_target_value 的计算方式是:
        1. delta = int((target_value - current_value) / price)  # int() 向零截断
        2. delta = int(delta / lot_size) * lot_size  # 向零取整到 lot

        **特例**: ``target_value == 0``（清仓）时聚宽不做 lot 取整，直接卖出全部持仓
        （含送转产生的零股），避免残留零股持仓。参见聚宽文档"要卖出全部股票时,
        可以使用 order_target_value(security, 0), 不需要考虑零股问题"。

        不能用"先算 target_shares 再减持仓"——对卖出时 floor() 向负无穷截断会多卖 100 股
        (如 delta=-1990, floor→-2000, 而 int()→-1900)。
        """
        ts_code = to_tushare_code(ts_code)
        price = self._broker.current_price(ts_code)
        if price is None or price <= 0:
            return None
        pos = self.portfolio.get_position(ts_code)
        lot_size = self._broker._lot_size  # noqa: SLF001
        current_value = pos.amount * price
        if target_value == 0 and pos.amount > 0:
            # 清仓特例:卖出全部持仓(含零股),不取整到 lot。
            return self.order(ts_code, -pos.amount)
        delta = int((target_value - current_value) / price)
        delta = int(delta / lot_size) * lot_size
        if delta == 0:
            return None
        return self.order(ts_code, delta)

    def order_target_percent(self, ts_code: str, pct: float) -> Order | None:
        ts_code = to_tushare_code(ts_code)
        target_value = self.portfolio.total_value * pct
        return self.order_target_value(ts_code, target_value)

    def order(self, ts_code: str, amount: int) -> Order | None:
        ts_code = to_tushare_code(ts_code)
        return self._broker.submit_order(ts_code, amount)
