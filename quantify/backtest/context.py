"""Strategy execution context and portfolio tracking, modelled on JoinQuant API."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any

from .codes import to_tushare_code

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
    """Tracks cash, positions, and total value throughout the backtest."""

    initial_cash: float
    cash: float
    positions: PositionBook = field(default_factory=PositionBook)
    total_commission: float = 0.0
    total_slippage: float = 0.0
    total_tax: float = 0.0
    trade_count: int = 0

    @property
    def total_value(self) -> float:
        pos_val = sum(p.market_value for p in self.positions.values())
        return self.cash + pos_val

    @property
    def total_pnl(self) -> float:
        return self.total_value - self.initial_cash

    @property
    def total_return_pct(self) -> float:
        if self.initial_cash == 0:
            return 0.0
        return (self.total_value / self.initial_cash - 1) * 100

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
    """Access current and historical market data within a strategy."""

    def __init__(self) -> None:
        self._bars: dict[str, list[Bar]] = {}
        self._current_idx: dict[str, int] = {}

    def _load(self, ts_code: str, bars: list[Bar]) -> None:
        ts_code = to_tushare_code(ts_code)
        self._bars[ts_code] = bars
        self._current_idx[ts_code] = -1

    def _advance(self, ts_code: str) -> None:
        ts_code = to_tushare_code(ts_code)
        if ts_code in self._current_idx:
            self._current_idx[ts_code] += 1

    def current(self, ts_code: str, field: str | None = None) -> float | Bar | None:
        ts_code = to_tushare_code(ts_code)
        bars = self._bars.get(ts_code, [])
        idx = self._current_idx.get(ts_code, -1)
        if idx < 0 or idx >= len(bars):
            return None
        bar = bars[idx]
        # Strategy code runs at the open: only the open price is known for the current bar.
        visible_bar = Bar(
            ts_code=bar.ts_code,
            date=bar.date,
            open=bar.open,
            high=bar.open,
            low=bar.open,
            close=bar.open,
            volume=0.0,
            amount=0.0,
            pre_close=bar.pre_close,
            pct_chg=0.0,
        )
        if field is None:
            return visible_bar
        return getattr(visible_bar, field, None)

    def history(self, ts_code: str, count: int, field: str = "close") -> list[float]:
        ts_code = to_tushare_code(ts_code)
        bars = self._bars.get(ts_code, [])
        idx = self._current_idx.get(ts_code, -1)
        if idx < 0 or idx >= len(bars):
            return []
        end = idx - 1
        if end < 0:
            return []
        start = max(0, end - count + 1)

        # Price fields are front-adjusted (前复权) to the current bar's basis so
        # that dividend/ex-rights gaps don't create spurious returns — matching
        # JoinQuant's ``attribute_history`` under ``use_real_price=True``.
        # Non-price fields (volume/amount/pct_chg) are returned as-is.
        price_fields = {"open", "high", "low", "close", "pre_close"}
        if field not in price_fields:
            return [getattr(bars[i], field, 0.0) for i in range(start, end + 1)]

        base_factor = getattr(bars[idx], "adj_factor", 1.0) or 1.0
        out: list[float] = []
        for i in range(start, end + 1):
            raw = getattr(bars[i], field, 0.0)
            factor = getattr(bars[i], "adj_factor", 1.0) or 1.0
            out.append(raw * factor / base_factor)
        return out

    @property
    def today(self) -> date | None:
        for code, bars in self._bars.items():
            idx = self._current_idx.get(code, -1)
            if 0 <= idx < len(bars):
                return bars[idx].date
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
