"""Strategy execution context and portfolio tracking, modelled on JoinQuant API."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any

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

    @property
    def market_value(self) -> float:
        return self.amount * self.current_price

    @property
    def pnl(self) -> float:
        return self.amount * (self.current_price - self.avg_cost)

    @property
    def pnl_pct(self) -> float:
        if self.avg_cost == 0:
            return 0.0
        return (self.current_price / self.avg_cost - 1) * 100


@dataclass
class Portfolio:
    """Tracks cash, positions, and total value throughout the backtest."""

    initial_cash: float
    cash: float
    positions: dict[str, Position] = field(default_factory=dict)
    total_commission: float = 0.0
    total_slippage: float = 0.0
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

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)


class DataProxy:
    """Access current and historical market data within a strategy."""

    def __init__(self) -> None:
        self._bars: dict[str, list[Bar]] = {}
        self._current_idx: dict[str, int] = {}

    def _load(self, ts_code: str, bars: list[Bar]) -> None:
        self._bars[ts_code] = bars
        self._current_idx[ts_code] = -1

    def _advance(self, ts_code: str) -> None:
        if ts_code in self._current_idx:
            self._current_idx[ts_code] += 1

    def current(self, ts_code: str, field: str | None = None) -> float | Bar | None:
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
        bars = self._bars.get(ts_code, [])
        idx = self._current_idx.get(ts_code, -1)
        if idx < 0 or idx >= len(bars):
            return []
        end = idx - 1
        if end < 0:
            return []
        start = max(0, end - count + 1)
        return [getattr(bars[i], field, 0.0) for i in range(start, end + 1)]

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
        self.benchmark_code = ts_code

    def order_value(self, ts_code: str, value: float) -> Order | None:
        price = self._broker.current_price(ts_code)
        if price is None or price <= 0:
            return None
        amount = int(value / price)
        return self.order(ts_code, amount)

    def order_target_value(self, ts_code: str, target_value: float) -> Order | None:
        current_value = self.portfolio.get_position(ts_code).market_value
        diff = target_value - current_value
        if abs(diff) < 1:
            return None
        return self.order_value(ts_code, diff)

    def order_target_percent(self, ts_code: str, pct: float) -> Order | None:
        target_value = self.portfolio.total_value * pct
        return self.order_target_value(ts_code, target_value)

    def order(self, ts_code: str, amount: int) -> Order | None:
        return self._broker.submit_order(ts_code, amount)
