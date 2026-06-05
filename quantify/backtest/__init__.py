"""Backtest engine — strategy-driven simulation against historical data.

API modelled after JoinQuant for familiarity:
    - ``initialize(context)``  — set up strategy parameters
    - ``handle_data(context)`` — called on every bar, place orders here
"""

from __future__ import annotations

from .broker import make_commission, make_slippage
from .engine import BacktestEngine, BacktestResult
from .examples import DEFAULT_STRATEGY_SOURCE

__all__ = ["BacktestEngine", "BacktestResult", "DEFAULT_STRATEGY_SOURCE", "make_commission", "make_slippage"]
