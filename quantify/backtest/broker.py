"""Order execution, commission, and slippage modelling."""

from __future__ import annotations

from datetime import date

from quantify.utils.logger import log

from .context import Bar, DataProxy, Order, ORDER_STATUS_FILLED, ORDER_STATUS_REJECTED


# ---------------------------------------------------------------------------
# Commission & slippage models
# ---------------------------------------------------------------------------


def make_commission(rate: float = 0.00015, minimum: float = 5.0):
    """Build a commission function from a rate and minimum fee.

    The returned callable has signature ``(trade_value: float) -> float``
    and computes ``max(rate * trade_value, minimum)``.

    Set ``rate=0, minimum=0`` for zero-commission backtests.
    """

    def _fn(trade_value: float) -> float:
        fee = abs(trade_value) * rate
        return max(fee, minimum)

    return _fn


default_etf_commission = make_commission(rate=0.00015, minimum=5.0)
"""Default ETF commission: 0.015 % per trade, min 5 CNY."""


def zero_slippage(_price: float, _amount: int) -> float:
    return 0.0


def make_slippage(rate: float = 0.0):
    """Build a slippage function that applies a proportional spread.

    ``cost = abs(price * amount) * rate``
    """

    def _fn(price: float, amount: int) -> float:
        return abs(price * amount) * rate

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
    ) -> None:
        self._data = data
        self._commission_fn = commission_fn
        self._slippage_fn = slippage_fn
        self._pending_orders: list[Order] = []
        self._trades: list[Order] = []

    @property
    def trades(self) -> list[Order]:
        return list(self._trades)

    def current_price(self, ts_code: str) -> float | None:
        bar = self._data.current(ts_code)
        if not isinstance(bar, Bar):
            return None
        return bar.close

    def submit_order(self, ts_code: str, amount: int) -> Order | None:
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

    def execute_pending(self, portfolio) -> None:
        """Execute all pending orders at next bar's open price.

        For simplicity, executes at the *current* bar close for buy orders
        (liquidity assumption). This can be tuned later.
        """
        remaining_orders: list[Order] = []
        for order in self._pending_orders:
            bar = self._data.current(order.ts_code)
            if not isinstance(bar, Bar):
                remaining_orders.append(order)
                continue
            exec_price = bar.close
            if self._apply_fill(order, portfolio, exec_price):
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

    def _buy_total_cost(self, price: float, amount: int) -> tuple[float, float, float]:
        trade_value = amount * price
        commission = float(self._commission_fn(trade_value))
        slippage = abs(float(self._slippage_fn(price, amount)))
        return trade_value + commission + slippage, commission, slippage

    def _max_affordable_buy_amount(self, requested: int, cash: float, price: float) -> int:
        if price <= 0:
            return 0

        low = 0
        high = min(requested, int(cash / price))
        while low < high:
            mid = (low + high + 1) // 2
            total_cost, _, _ = self._buy_total_cost(price, mid)
            if total_cost <= cash:
                low = mid
            else:
                high = mid - 1
        return low

    def _apply_fill(self, order: Order, portfolio, price: float) -> bool:
        pos = portfolio.get_position(order.ts_code)
        if order.amount > 0:  # buy
            fill_amount = self._max_affordable_buy_amount(order.amount, portfolio.cash, price)
            if fill_amount <= 0:
                log.debug(f"Insufficient cash for {order.ts_code}: have {portfolio.cash:.2f}")
                return False
            if fill_amount != order.amount:
                log.debug(f"Partial fill {order.ts_code}: {order.amount} -> {fill_amount}")

            total_cost, commission, slippage = self._buy_total_cost(price, fill_amount)

            old_val = pos.amount * pos.avg_cost
            pos.avg_cost = (
                (old_val + fill_amount * price) / (pos.amount + fill_amount)
                if (pos.amount + fill_amount) > 0
                else 0
            )
            pos.amount += fill_amount
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
            sell_amount = min(pos.amount, -order.amount)
            if sell_amount <= 0:
                return False
            filled_amount = -sell_amount
            trade_value = sell_amount * price
            commission = float(self._commission_fn(trade_value))
            slippage = abs(float(self._slippage_fn(price, filled_amount)))
            order.amount = -sell_amount
            order.filled_amount = filled_amount
            order.commission = commission
            order.slippage = slippage
            revenue = trade_value - commission - slippage
            pos.amount -= sell_amount
            if pos.amount == 0:
                pos.avg_cost = 0.0
            portfolio.cash += revenue
            portfolio.total_commission += commission
            portfolio.total_slippage += slippage
            portfolio.trade_count += 1
            return True
