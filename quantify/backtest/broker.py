"""Order execution, commission, and slippage modelling."""

from __future__ import annotations

from datetime import date

from quantify.utils.logger import log

from .context import DataProxy, Order, ORDER_STATUS_FILLED


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
        if bar is None:
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
        for order in self._pending_orders:
            order.status = ORDER_STATUS_FILLED
            order.filled_date = self._data.today
            bar = self._data.current(order.ts_code)
            if bar is None:
                order.status = "rejected"
                continue
            exec_price = bar.close
            slippage_cost = self._slippage_fn(exec_price, order.amount)
            order.filled_price = exec_price
            order.filled_amount = order.amount
            order.slippage = slippage_cost

            trade_value = abs(order.amount) * exec_price
            commission = self._commission_fn(trade_value)
            order.commission = commission

            self._apply_fill(order, portfolio, exec_price, commission)
            self._trades.append(order)

        self._pending_orders.clear()

    @staticmethod
    def _apply_fill(order: Order, portfolio, price: float, commission: float) -> None:
        pos = portfolio.get_position(order.ts_code)
        if order.amount > 0:  # buy
            total_cost = order.amount * price + commission
            if portfolio.cash < total_cost:
                affordable = int((portfolio.cash - commission) / price)
                if affordable <= 0:
                    log.debug(
                        f"Insufficient cash for {order.ts_code}: need {total_cost:.2f}, have {portfolio.cash:.2f}"
                    )
                    return
                log.debug(f"Partial fill {order.ts_code}: {order.amount} -> {affordable}")
                order.amount = affordable
                order.filled_amount = affordable
                total_cost = order.amount * price + commission

            old_val = pos.amount * pos.avg_cost
            pos.avg_cost = (
                (old_val + order.amount * price) / (pos.amount + order.amount)
                if (pos.amount + order.amount) > 0
                else 0
            )
            pos.amount += order.amount
            portfolio.cash -= total_cost
            portfolio.total_commission += commission
            portfolio.total_slippage += order.slippage
            portfolio.trade_count += 1
        else:  # sell
            sell_amount = min(pos.amount, -order.amount)
            if sell_amount <= 0:
                return
            order.amount = -sell_amount
            order.filled_amount = -sell_amount
            revenue = sell_amount * price - commission
            pos.amount -= sell_amount
            if pos.amount == 0:
                pos.avg_cost = 0.0
            portfolio.cash += revenue
            portfolio.total_commission += commission
            portfolio.total_slippage += order.slippage
            portfolio.trade_count += 1
