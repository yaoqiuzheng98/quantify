"""Order execution, commission, and slippage modelling."""

from __future__ import annotations

from datetime import date

from quantify.utils.logger import log

from .codes import classify_asset, to_tushare_code
from .context import Bar, DataProxy, Order, ORDER_STATUS_FILLED, ORDER_STATUS_REJECTED


# A-share stamp duty (印花税): single-sided, charged on the SELL leg only.
# Cut from 0.1% to 0.05% on 2023-08-28; we use the current 0.05%.
STOCK_STAMP_DUTY_RATE = 0.0005
# A-share daily price-limit band: ±10% for the main boards (vs prev close).
# STAR/ChiNext are ±20% and BSE ±30%, but 10% is a safe conservative default
# that blocks the egregious limit-board fills追涨打板 strategies would otherwise get.
STOCK_PRICE_LIMIT_PCT = 0.10
PRICE_LIMIT_EPS = 1e-3


# ---------------------------------------------------------------------------
# Commission & slippage models
# ---------------------------------------------------------------------------


def make_commission(rate: float = 0.0005, minimum: float = 0.5):
    """Build a commission function from a rate and minimum fee.

    The returned callable has signature ``(trade_value: float) -> float``
    and computes ``max(rate * trade_value, minimum)``.

    Set ``rate=0, minimum=0`` for zero-commission backtests.
    """

    def _fn(trade_value: float) -> float:
        fee = abs(trade_value) * rate
        return max(fee, minimum)

    return _fn


default_etf_commission = make_commission(rate=0.0005, minimum=0.5)
"""Default ETF commission: 0.05 % per trade, min 0.5 CNY."""


def make_stamp_duty(rate: float = STOCK_STAMP_DUTY_RATE):
    """Build an A-share stamp-duty function (charged on sells only).

    Returns ``(trade_value) -> tax`` computing ``abs(trade_value) * rate``.
    ETFs/funds are exempt, so the broker only calls this for stock codes.
    """

    def _fn(trade_value: float) -> float:
        return abs(trade_value) * rate

    return _fn


def zero_slippage(_price: float, _amount: int) -> float:
    return _price


def make_slippage(rate: float = 0.0):
    """Build a JoinQuant-style price-related slippage function.

    ``PriceRelatedSlippage(0.002)`` moves the executable price by half the
    spread: buy at ``price * 1.001`` and sell at ``price * 0.999``.
    """

    def _fn(price: float, amount: int) -> float:
        if amount > 0:
            return price * (1 + rate / 2)
        if amount < 0:
            return price * (1 - rate / 2)
        return price

    return _fn


# ---------------------------------------------------------------------------
# Broker
# ---------------------------------------------------------------------------


class Broker:
    """Manages order submission, execution, and P&L tracking."""

    def __init__(
        self,
        data: DataProxy,
        commission_fn=default_etf_commission,
        slippage_fn=zero_slippage,
        lot_size: int = 100,
        price_tick: float = 0.001,
        stamp_duty_rate: float = STOCK_STAMP_DUTY_RATE,
        enforce_t_plus_1: bool = True,
        enforce_price_limit: bool = True,
    ) -> None:
        self._data = data
        self._commission_fn = commission_fn
        self._slippage_fn = slippage_fn
        self._lot_size = lot_size
        self._price_tick = price_tick
        # A-share-only frictions. These are applied solely to codes that
        # ``classify_asset`` deems a stock, so ETF/index backtests are unaffected.
        self._stamp_duty_fn = make_stamp_duty(stamp_duty_rate)
        self._enforce_t_plus_1 = enforce_t_plus_1
        self._enforce_price_limit = enforce_price_limit
        self._pending_orders: list[Order] = []
        self._trades: list[Order] = []

    @staticmethod
    def _is_stock(ts_code: str) -> bool:
        return classify_asset(ts_code) == "stock"

    def _sell_tax(self, ts_code: str, trade_value: float) -> float:
        """Stamp duty for a sell. Zero for non-stock instruments (ETF/index)."""
        if not self._is_stock(ts_code):
            return 0.0
        return float(self._stamp_duty_fn(trade_value))

    @property
    def trades(self) -> list[Order]:
        return list(self._trades)

    def set_commission_fn(self, commission_fn) -> None:
        self._commission_fn = commission_fn

    def set_slippage_fn(self, slippage_fn) -> None:
        self._slippage_fn = slippage_fn

    def set_stamp_duty_rate(self, rate: float) -> None:
        """Override the sell-side stamp-duty rate (stocks only)."""
        self._stamp_duty_fn = make_stamp_duty(rate)

    def current_price(self, ts_code: str) -> float | None:
        ts_code = to_tushare_code(ts_code)
        bar = self._data.current(ts_code)
        if not isinstance(bar, Bar):
            return None
        return bar.open

    def submit_order(self, ts_code: str, amount: int) -> Order | None:
        ts_code = to_tushare_code(ts_code)
        amount = self._round_to_lot(amount)
        if amount == 0:
            return None
        price = self.current_price(ts_code)
        if price is None:
            log.warning(f"submit_order({ts_code}, {amount}): no price data")
            return None

        order = Order(
            ts_code=ts_code,
            amount=amount,
            order_price=price,
            created_date=self._data.today or date.today(),
        )
        self._pending_orders.append(order)
        return order

    def _round_to_lot(self, amount: int) -> int:
        if self._lot_size <= 1:
            return amount
        sign = 1 if amount > 0 else -1
        lots = abs(amount) // self._lot_size
        return sign * lots * self._lot_size

    def _round_price(self, price: float) -> float:
        if self._price_tick <= 0:
            return price
        return round(round(price / self._price_tick) * self._price_tick, 10)

    def _execution_price(self, price: float, amount: int) -> float:
        return self._round_price(float(self._slippage_fn(price, amount)))

    def _price_limit_blocks(self, order: Order, bar: Bar) -> bool:
        """True if a stock order can't fill at ``bar.open`` due to 涨跌停.

        Buys are blocked when the open is pinned at/above the upper limit
        (涨停, no sellers); sells are blocked at/below the lower limit (跌停, no
        buyers). Limits are derived from ``pre_close`` with a small tolerance.
        Only applies to stocks and only when ``pre_close`` is usable.
        """
        if not self._enforce_price_limit or not self._is_stock(order.ts_code):
            return False
        pre_close = getattr(bar, "pre_close", 0.0)
        if not pre_close or pre_close <= 0:
            return False
        band = pre_close * STOCK_PRICE_LIMIT_PCT
        upper = pre_close + band
        lower = pre_close - band
        if order.amount > 0 and bar.open >= upper - PRICE_LIMIT_EPS:
            log.debug(f"{order.ts_code} limit-up at open {bar.open}, buy rejected")
            return True
        if order.amount < 0 and bar.open <= lower + PRICE_LIMIT_EPS:
            log.debug(f"{order.ts_code} limit-down at open {bar.open}, sell rejected")
            return True
        return False

    def execute_pending(self, portfolio) -> None:
        """Execute all pending orders at the current bar's open price."""
        remaining_orders: list[Order] = []
        for order in self._pending_orders:
            bar = self._data.current(order.ts_code)
            if not isinstance(bar, Bar):
                remaining_orders.append(order)
                continue
            if self._price_limit_blocks(order, bar):
                order.status = ORDER_STATUS_REJECTED
                continue
            base_price = bar.open
            exec_price = self._execution_price(base_price, order.amount)
            if self._apply_fill(order, portfolio, base_price, exec_price):
                order.status = ORDER_STATUS_FILLED
                order.filled_date = bar.date
                order.filled_price = exec_price
                self._trades.append(order)
            else:
                order.status = ORDER_STATUS_REJECTED

        self._pending_orders = remaining_orders

    def cancel_pending(self) -> int:
        """Cancel orders that cannot be executed because no next bar exists."""
        count = len(self._pending_orders)
        for order in self._pending_orders:
            order.status = ORDER_STATUS_REJECTED
        self._pending_orders.clear()
        return count

    def _buy_total_cost(self, price: float, amount: int) -> tuple[float, float, float, float]:
        exec_price = self._execution_price(price, amount)
        trade_value = amount * exec_price
        commission = float(self._commission_fn(trade_value))
        slippage = abs((exec_price - price) * amount)
        return trade_value + commission, commission, slippage, exec_price

    def _max_affordable_buy_amount(self, requested: int, cash: float, price: float) -> int:
        if price <= 0:
            return 0

        low = 0
        high = min(requested, int(cash / price))
        while low < high:
            mid = (low + high + 1) // 2
            total_cost, _, _, _ = self._buy_total_cost(price, mid)
            if total_cost <= cash:
                low = mid
            else:
                high = mid - 1
        return self._round_to_lot(low)

    def _apply_fill(self, order: Order, portfolio, base_price: float, price: float) -> bool:
        pos = portfolio.get_position(order.ts_code)
        if order.amount > 0:  # buy
            fill_amount = self._max_affordable_buy_amount(order.amount, portfolio.cash, base_price)
            if fill_amount <= 0:
                log.debug(f"Insufficient cash for {order.ts_code}: have {portfolio.cash:.2f}")
                return False
            if fill_amount != order.amount:
                log.debug(f"Partial fill {order.ts_code}: {order.amount} -> {fill_amount}")

            total_cost, commission, slippage, exec_price = self._buy_total_cost(base_price, fill_amount)

            old_val = pos.amount * pos.avg_cost
            pos.avg_cost = (
                (old_val + fill_amount * exec_price) / (pos.amount + fill_amount)
                if (pos.amount + fill_amount) > 0
                else 0
            )
            pos.amount += fill_amount
            # T+1: shares bought today cannot be sold until the next session.
            # Locked only for stocks; the engine clears this each new day.
            if self._enforce_t_plus_1 and self._is_stock(order.ts_code):
                pos.locked_amount += fill_amount
            portfolio.cash -= total_cost
            portfolio.total_commission += commission
            portfolio.total_slippage += slippage
            portfolio.trade_count += 1
            order.amount = fill_amount
            order.filled_amount = fill_amount
            order.commission = commission
            order.slippage = slippage
            return True
        else:  # sell
            # T+1: only the closeable (unlocked) portion may be sold for stocks.
            sellable = pos.amount
            if self._enforce_t_plus_1 and self._is_stock(order.ts_code):
                sellable = pos.closeable_amount
            sell_amount = min(sellable, -order.amount)
            if sell_amount <= 0:
                return False
            filled_amount = -sell_amount
            trade_value = sell_amount * price
            commission = float(self._commission_fn(trade_value))
            tax = self._sell_tax(order.ts_code, trade_value)
            slippage = abs((price - base_price) * sell_amount)
            order.amount = -sell_amount
            order.filled_amount = filled_amount
            order.commission = commission
            order.slippage = slippage
            order.tax = tax
            revenue = trade_value - commission - tax
            pos.amount -= sell_amount
            if pos.amount == 0:
                pos.avg_cost = 0.0
            portfolio.cash += revenue
            portfolio.total_commission += commission
            portfolio.total_slippage += slippage
            portfolio.total_tax += tax
            portfolio.trade_count += 1
            return True
