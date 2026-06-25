"""Runtime support for ML-generated strategies.

Provides functions that ML strategies call at runtime inside the BacktestEngine
to compute factor values and model predictions.  The strategy source code
imports from this module::

    from quantify.ml.strategy_runtime import RuntimeContext

This module handles:
1. Loading the saved ML model from disk
2. Computing Qlib factor expressions from ``attribute_history()`` data
3. Running model predictions and selecting top-N stocks
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from quantify.utils.logger import log

from .qlib_eval import parse_expression

# ---------------------------------------------------------------------------
# Model storage
# ---------------------------------------------------------------------------

_MODELS_DIR = Path(__file__).resolve().parent.parent.parent / "models"


def save_model(model, factor_exprs: list[str], config: dict, name: str) -> Path:
    """Save a trained ML model + metadata to disk.

    Parameters
    ----------
    model : object
        Trained model (must be picklable, e.g. XGBoost / sklearn).
    factor_exprs : list[str]
        Qlib factor expressions used as features (order matters).
    config : dict
        Strategy configuration (top_n, rebalance_days, universe, etc.).
    name : str
        Base name for the model file (e.g. ``"ml_xgboost_000300"``).

    Returns
    -------
    Path
        Path to the saved ``.pkl`` file.
    """
    _MODELS_DIR.mkdir(parents=True, exist_ok=True)
    path = _MODELS_DIR / f"{name}.pkl"
    payload = {
        "model": model,
        "factor_exprs": factor_exprs,
        "config": config,
    }
    with open(path, "wb") as f:
        pickle.dump(payload, f)
    log.info(f"模型已保存: {path}")
    return path


def load_model(name: str) -> dict:
    """Load a saved ML model + metadata from disk.

    Returns dict with keys: model, factor_exprs, config.
    """
    path = _MODELS_DIR / f"{name}.pkl"
    if not path.exists():
        raise FileNotFoundError(f"模型文件不存在: {path}")
    with open(path, "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------------------
# Runtime factor computation
# ---------------------------------------------------------------------------

# Fields available from attribute_history (engine OHLCV)
_OHLCV_FIELDS = {"open", "high", "low", "close", "volume", "amount", "pre_close", "pct_chg", "adj_factor"}

# Fields that need get_fundamentals
_FUNDAMENTAL_FIELDS = {"pe", "pb", "ps", "turn", "total_mv", "circ_mv"}

# vwap is computed from amount/volume
_COMPUTED_FIELDS = {"vwap"}


def _extract_fields(exprs: list[str]) -> set[str]:
    """Extract all field names (without $) referenced in expressions."""
    import re

    fields = set()
    for expr in exprs:
        for m in re.findall(r"\$([a-zA-Z_]+)", expr):
            fields.add(m)
    return fields


def _get_ohlcv_data(attribute_history, code: str, count: int, fields: set[str]) -> dict[str, np.ndarray]:
    """Get OHLCV data from engine's attribute_history."""
    ohlcv_needed = fields & _OHLCV_FIELDS
    if not ohlcv_needed:
        return {}

    df = attribute_history(code, count, "1d", list(ohlcv_needed))
    data = {}
    for col in df.columns:
        data[col] = df[col].to_numpy(dtype=float)
    return data


def _compute_vwap(amount: np.ndarray, volume: np.ndarray) -> np.ndarray:
    """vwap = amount * 10 / volume (matching Qlib dump-data convention)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        vwap = (amount * 10.0) / np.where(volume != 0, volume, np.nan)
    return np.nan_to_num(vwap, nan=0.0)


def _get_fundamentals_batch(
    get_fundamentals_fn, query_fn, valuation_obj, stocks: list[str], date_str: str
) -> dict[str, dict]:
    """Get fundamental data (pe/pb/ps/turn/total_mv/circ_mv) for a batch of stocks.

    Returns {stock_code: {field: value}}.
    """
    field_map = {
        "pe": "pe_ratio",
        "pb": "pb_ratio",
        "ps": "ps_ratio",
        "turn": "turnover_ratio",
        "total_mv": "market_cap",
        "circ_mv": "circulating_market_cap",
    }

    try:
        q = query_fn(valuation_obj.code).filter(valuation_obj.code.in_(stocks))
        df = get_fundamentals_fn(q, date=date_str)
    except Exception:
        return {}

    result = {}
    if df is None or df.empty:
        return {}

    for _, row in df.iterrows():
        code = row.get("code")
        if not code:
            continue
        result[code] = {}
        for qlib_field, jq_field in field_map.items():
            val = row.get(jq_field)
            if val is not None and not pd.isna(val):
                result[code][qlib_field] = float(val)
    return result


class RuntimeContext:
    """Runtime context for ML strategy execution.

    Encapsulates model loading, factor computation, and prediction logic.
    Created in strategy's ``initialize()`` and used in ``rebalance()``.
    """

    def __init__(self, model_name: str, universe_code: str, top_n: int = 20, rebalance_days: int = 5):
        """Load model and pre-parse factor expressions.

        Parameters
        ----------
        model_name : str
            Name of the saved model (without .pkl extension).
        universe_code : str
            JoinQuant-format index code for get_index_stocks (e.g. "000300.XSHG").
        top_n : int
            Number of stocks to select.
        rebalance_days : int
            Rebalance frequency in trading days.
        """
        payload = load_model(model_name)
        self.model = payload["model"]
        self.factor_exprs: list[str] = payload["factor_exprs"]
        self.config: dict = payload["config"]
        self.universe_code = universe_code
        self.top_n = top_n
        self.rebalance_days = rebalance_days
        self.day_count = 0

        # Pre-parse expressions for speed
        self._parsed = [parse_expression(e) for e in self.factor_exprs]

        # Determine which fields are needed
        self._needed_fields = _extract_fields(self.factor_exprs)
        self._ohlcv_needed = self._needed_fields & _OHLCV_FIELDS
        self._fundamental_needed = self._needed_fields & _FUNDAMENTAL_FIELDS
        self._needs_vwap = "vwap" in self._needed_fields

        # Max rolling window in expressions (determines history count)
        self._max_window = self._estimate_max_window()

        log.info(
            f"RuntimeContext loaded: model={model_name}, "
            f"factors={len(self.factor_exprs)}, fields={self._needed_fields}, "
            f"max_window={self._max_window}"
        )

    def _estimate_max_window(self) -> int:
        """Estimate the maximum rolling window needed across all expressions."""
        import re

        max_w = 20
        for expr in self.factor_exprs:
            # Find numbers that are likely window sizes (after commas in rolling funcs)
            for m in re.findall(
                r"(?:Mean|Std|Sum|Min|Max|Skew|Kurt|Var|Median|Rank|Corr|Cov|EMA|WMA|Slope|Resi|Delta|Ref|Quantile|IdxMax)\([^)]*,\s*(\d+)\s*\)",
                expr,
            ):
                w = int(m)
                if w > max_w:
                    max_w = w
        return max_w + 10  # safety margin

    def compute_scores(
        self,
        stocks: list[str],
        attribute_history_fn,
        get_fundamentals_fn=None,
        query_fn=None,
        valuation_obj=None,
        current_date=None,
    ) -> dict[str, float]:
        """Compute ML model scores for a list of stocks.

        Parameters
        ----------
        stocks : list[str]
            Stock codes in JoinQuant format (e.g. "000001.XSHE").
        attribute_history_fn : callable
            Engine's ``attribute_history(code, count, "1d", fields)``.
        get_fundamentals_fn, query_fn, valuation_obj : callable
            For fundamental data (optional, only if factors use pe/pb/turn etc.).
        current_date : str, optional
            Date string for get_fundamentals (e.g. "2024-01-05").

        Returns
        -------
        dict[str, float]
            Maps stock code → model score. Higher = better.
        """
        count = self._max_window
        ohlcv_fields = list(self._ohlcv_needed)
        if self._needs_vwap:
            # Need amount and volume to compute vwap
            if "amount" not in ohlcv_fields:
                ohlcv_fields.append("amount")
            if "volume" not in ohlcv_fields:
                ohlcv_fields.append("volume")

        # Batch fetch fundamentals if needed
        fund_data = {}
        if self._fundamental_needed and get_fundamentals_fn and query_fn and valuation_obj:
            date_str = current_date or ""
            fund_data = _get_fundamentals_batch(
                get_fundamentals_fn, query_fn, valuation_obj, stocks, date_str
            )

        from .qlib_eval import evaluate

        features_list = []
        valid_stocks = []

        for code in stocks:
            data = {}

            # OHLCV data
            if ohlcv_fields:
                try:
                    df = attribute_history_fn(code, count, "1d", ohlcv_fields)
                    for col in df.columns:
                        data[col] = df[col].to_numpy(dtype=float)
                except Exception:
                    continue

            # Compute vwap if needed
            if self._needs_vwap and "amount" in data and "volume" in data:
                data["vwap"] = _compute_vwap(data["amount"], data["volume"])

            # Fundamental data (single value, broadcast to array length)
            if self._fundamental_needed:
                stock_fund = fund_data.get(code, {})
                ref_len = len(next(iter(data.values()))) if data else 1
                for fld in self._fundamental_needed:
                    val = stock_fund.get(fld, 0.0)
                    data[fld] = np.full(ref_len, val, dtype=float)

            # Check we have all needed fields
            if not all(f in data for f in self._needed_fields):
                continue

            # Compute each factor
            factor_values = []
            for ast in self._parsed:
                try:
                    result = evaluate(ast, data)
                    val = float(result[-1]) if len(result) > 0 else float("nan")
                    if not np.isfinite(val):
                        val = 0.0
                    factor_values.append(val)
                except Exception:
                    factor_values.append(0.0)

            features_list.append(factor_values)
            valid_stocks.append(code)

        if not valid_stocks:
            return {}

        # Predict
        X = np.array(features_list, dtype=float)
        # Fill NaN with 0
        X = np.nan_to_num(X, nan=0.0)

        try:
            predictions = self.model.predict(X)
        except Exception as e:
            log.warning(f"模型预测失败: {e}")
            return {}

        return {code: float(pred) for code, pred in zip(valid_stocks, predictions, strict=False)}

    def select_top_stocks(self, scores: dict[str, float]) -> dict[str, float]:
        """Select top-N stocks by score, equal weight.

        Returns {stock_code: weight}.
        """
        if not scores:
            return {}

        sorted_stocks = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        top = sorted_stocks[: self.top_n]

        if not top:
            return {}

        weight = 1.0 / len(top)
        return {code: weight for code, _ in top}
