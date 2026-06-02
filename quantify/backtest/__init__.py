"""Backtest engine — strategy-driven simulation against historical data.

API modelled after JoinQuant for familiarity:
    - ``initialize(context)``  — set up strategy parameters
    - ``handle_data(context)`` — called on every bar, place orders here
"""

from __future__ import annotations

from .broker import make_commission, make_slippage
from .engine import BacktestEngine, BacktestResult

__all__ = ["BacktestEngine", "BacktestResult", "make_commission", "make_slippage"]
