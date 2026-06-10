"""Small JoinQuant compatibility layer for strategy execution."""

from __future__ import annotations

from dataclasses import dataclass
from types import ModuleType
from typing import Any, Callable

import pandas as pd

from .broker import make_commission, make_slippage
from .codes import to_tushare_code


@dataclass
class OrderCost:
    open_tax: float = 0.0
    close_tax: float = 0.0
    open_commission: float = 0.0
    close_commission: float = 0.0
    close_today_commission: float = 0.0
    min_commission: float = 0.0


@dataclass
class PriceRelatedSlippage:
    rate: float = 0.0


class JoinQuantCompat:
    """Expose common JoinQuant globals against the local Context object."""

    def __init__(self) -> None:
        self.context: Any | None = None
        # 已注册的调度任务，每项为 (func, freq, day)：
        #   freq="daily"   day 忽略，每个交易日触发
        #   freq="weekly"  day=weekday，第 N 个交易日(每周)，负数表示倒数
        #   freq="monthly" day=monthday，第 N 个交易日(每月)，负数表示倒数
        self.scheduled: list[tuple[Callable, str, int]] = []
        self.options: dict[str, Any] = {}

    @property
    def daily_functions(self) -> list[Callable]:
        """Backward-compatible view of registered daily tasks."""
        return [func for func, freq, _day in self.scheduled if freq == "daily"]

    def bind(self, context: Any) -> None:
        self.context = context

    def _require_context(self) -> Any:
        if self.context is None:
            raise RuntimeError("JoinQuant API called before context is bound")
        return self.context

    def set_option(self, key: str, value: Any) -> None:
        self.options[key] = value

    def set_benchmark(self, security: str) -> None:
        self._require_context().set_benchmark(security)

    def run_daily(self, func: Callable, time: str = "open", **_kwargs: Any) -> None:
        if time not in ("open", "every_bar"):
            raise NotImplementedError("Local engine currently supports run_daily(..., time='open') only")
        self.scheduled.append((func, "daily", 0))

    def run_weekly(self, func: Callable, weekday: int = 1, time: str = "open", **_kwargs: Any) -> None:
        if time not in ("open", "every_bar"):
            raise NotImplementedError("Local engine currently supports time='open' only")
        self.scheduled.append((func, "weekly", weekday))

    def run_monthly(self, func: Callable, monthday: int = 1, time: str = "open", **_kwargs: Any) -> None:
        if time not in ("open", "every_bar"):
            raise NotImplementedError("Local engine currently supports time='open' only")
        self.scheduled.append((func, "monthly", monthday))

    def attribute_history(
        self,
        security: str,
        count: int,
        unit: str = "1d",
        fields: str | list[str] | tuple[str, ...] = "close",
        skip_paused: bool = True,
        df: bool = True,
        **_kwargs: Any,
    ) -> pd.DataFrame | dict[str, list[float]]:
        del skip_paused
        if unit != "1d":
            raise NotImplementedError("Local engine currently supports 1d history only")

        context = self._require_context()
        field_names = [fields] if isinstance(fields, str) else list(fields)
        data = {
            field: context.data.history(to_tushare_code(security), count=count, field=field)
            for field in field_names
        }
        return pd.DataFrame(data) if df else data

    def order(self, security: str, amount: int):
        return self._require_context().order(security, amount)

    def order_value(self, security: str, value: float):
        return self._require_context().order_value(security, value)

    def order_target_value(self, security: str, value: float):
        return self._require_context().order_target_value(security, value)

    def order_target_percent(self, security: str, percent: float):
        return self._require_context().order_target_percent(security, percent)

    def set_order_cost(self, order_cost: OrderCost, type: str = "stock") -> None:  # noqa: A002
        del type
        rate = max(order_cost.open_commission, order_cost.close_commission)
        self._require_context()._broker.set_commission_fn(  # noqa: SLF001
            make_commission(rate=rate, minimum=order_cost.min_commission)
        )

    def set_slippage(self, slippage: PriceRelatedSlippage) -> None:
        self._require_context()._broker.set_slippage_fn(make_slippage(slippage.rate))  # noqa: SLF001

    def namespace(self) -> dict[str, Any]:
        return {
            "OrderCost": OrderCost,
            "PriceRelatedSlippage": PriceRelatedSlippage,
            "set_option": self.set_option,
            "set_benchmark": self.set_benchmark,
            "set_order_cost": self.set_order_cost,
            "set_slippage": self.set_slippage,
            "run_daily": self.run_daily,
            "run_weekly": self.run_weekly,
            "run_monthly": self.run_monthly,
            "attribute_history": self.attribute_history,
            "order": self.order,
            "order_value": self.order_value,
            "order_target_value": self.order_target_value,
            "order_target_percent": self.order_target_percent,
        }


def make_jqdata_module(compat: JoinQuantCompat) -> ModuleType:
    module = ModuleType("jqdata")
    namespace = compat.namespace()
    module.__dict__.update(namespace)
    module.__all__ = list(namespace.keys())
    return module
