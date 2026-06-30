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

import json
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

    For XGBoost/LightGBM models, saves the model in native JSON format (fast
    loading, ~1000x faster than pickle on WSL).  Metadata (factor expressions,
    config) is saved as a separate JSON file.

    Parameters
    ----------
    model : object
        Trained model (XGBoost / LightGBM / sklearn).
    factor_exprs : list[str]
        Qlib factor expressions used as features (order matters).
    config : dict
        Strategy configuration (top_n, rebalance_days, universe, etc.).
    name : str
        Base name for the model file (e.g. ``"ml_xgboost_000300"``).

    Returns
    -------
    Path
        Path to the saved model file.
    """
    _MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Check if model supports native save_model (XGBoost, LightGBM)
    model_type = type(model).__module__.split(".")[0]
    if hasattr(model, "save_model") and model_type in ("xgboost", "lightgbm"):
        # Save model in native format
        model_path = _MODELS_DIR / f"{name}.json"
        model.save_model(model_path)
    else:
        # Fallback: pickle for sklearn models
        model_path = _MODELS_DIR / f"{name}.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(model, f)

    # Save metadata as JSON (always)
    meta_path = _MODELS_DIR / f"{name}.meta.json"
    meta = {
        "factor_exprs": factor_exprs,
        "config": config,
        "model_type": model_type,
        "model_file": model_path.name,
        "best_iteration": getattr(model, "best_iteration", None),
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    log.info(f"模型已保存: {model_path}")
    return model_path


def load_model(name: str) -> dict:
    """Load a saved ML model + metadata from disk.

    Returns dict with keys: model, factor_exprs, config.
    """
    meta_path = _MODELS_DIR / f"{name}.meta.json"
    if meta_path.exists():
        # New format: metadata JSON + native model file
        with open(meta_path) as f:
            meta = json.load(f)
        model_file = meta["model_file"]
        model_path = _MODELS_DIR / model_file
        if not model_path.exists():
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
        model_type = meta.get("model_type", "")
        if model_type == "xgboost":
            import xgboost as xgb

            model = xgb.XGBRegressor()
            model.load_model(model_path)
        elif model_type == "lightgbm":
            import lightgbm as lgb

            model = lgb.LGBMRegressor()
            model.load_model(str(model_path))
        else:
            with open(model_path, "rb") as f:
                model = pickle.load(f)
        # Restore best_iteration (lost during native save_model/load_model)
        best_iter = meta.get("best_iteration")
        if best_iter is not None and hasattr(model, "best_iteration"):
            model.best_iteration = best_iter
        return {
            "model": model,
            "factor_exprs": meta["factor_exprs"],
            "config": meta["config"],
        }

    # Legacy format: single .pkl file with everything
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

        # Cache: bars converted to numpy arrays, keyed by ts_code
        # {ts_code: {field: np.ndarray}}, built on first rebalance
        self._bars_cache: dict[str, dict[str, np.ndarray]] = {}
        self._adj_cache: dict[str, np.ndarray] = {}  # {ts_code: adj_factor array}

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
        context=None,
    ) -> dict[str, float]:
        """Compute ML model scores for a list of stocks.

        When ``context`` is provided, directly accesses ``context.data._bars``
        for batch data retrieval (10x faster than calling attribute_history
        per stock). Falls back to attribute_history_fn otherwise.
        """
        if context is not None:
            return self._compute_scores_from_bars(
                stocks, context, get_fundamentals_fn, query_fn, valuation_obj, current_date
            )
        return self._compute_scores_slow(
            stocks, attribute_history_fn, get_fundamentals_fn, query_fn, valuation_obj, current_date
        )

    def _ensure_bars_cache(self, ts_codes: list[str], bars_map: dict) -> None:
        """Convert Bar objects to numpy arrays for cached stocks. Called once per stock."""
        all_fields = self._ohlcv_needed | ({"amount", "volume"} if self._needs_vwap else set())
        for ts_code in ts_codes:
            if ts_code in self._bars_cache:
                continue
            bars = bars_map.get(ts_code)
            if bars is None or len(bars) == 0:
                continue
            cache: dict[str, np.ndarray] = {}
            for fld in all_fields:
                cache[fld] = np.array([getattr(b, fld, 0.0) for b in bars], dtype=float)
            self._adj_cache[ts_code] = np.array(
                [getattr(b, "adj_factor", 1.0) or 1.0 for b in bars], dtype=float
            )
            self._bars_cache[ts_code] = cache

    def _compute_scores_from_bars(
        self,
        stocks: list[str],
        context,
        get_fundamentals_fn=None,
        query_fn=None,
        valuation_obj=None,
        current_date=None,
    ) -> dict[str, float]:
        """Fast path: vectorized across all stocks in one pass.

        Builds 2D arrays (time × stocks) and evaluates each factor once
        on the full array, instead of looping per stock.
        """
        from quantify.backtest.codes import to_tushare_code

        from .qlib_eval_2d import evaluate_2d

        count = self._max_window
        data_proxy = context.data
        bars_map = data_proxy._bars  # noqa: SLF001
        idx_map = data_proxy._current_idx  # noqa: SLF001

        price_fields = {"open", "high", "low", "close", "pre_close"}
        ohlcv_needed = self._ohlcv_needed
        if self._needs_vwap:
            ohlcv_needed = ohlcv_needed | {"amount", "volume"}

        # Batch fetch fundamentals if needed
        fund_data = {}
        if self._fundamental_needed and get_fundamentals_fn and query_fn and valuation_obj:
            date_str = current_date or ""
            fund_data = _get_fundamentals_batch(
                get_fundamentals_fn, query_fn, valuation_obj, stocks, date_str
            )

        # Collect valid stocks and their bar windows
        valid = []  # (jq_code, ts_code, idx, end)
        ts_codes_to_cache = []
        for jq_code in stocks:
            ts_code = to_tushare_code(jq_code)
            bars = bars_map.get(ts_code)
            idx = idx_map.get(ts_code, -1)
            if bars is None or idx < 0 or idx >= len(bars):
                continue
            end = idx - 1
            if end < 0:
                continue
            valid.append((jq_code, ts_code, idx, end))
            ts_codes_to_cache.append(ts_code)

        if not valid:
            return {}

        # Build numpy cache from Bar objects (only first time per stock)
        self._ensure_bars_cache(ts_codes_to_cache, bars_map)

        n_stocks = len(valid)
        # Determine window length (right-aligned, all same length)
        max_wlen = min(count, max(end - max(0, end - count + 1) + 1 for _, _, _, end in valid))

        # Build 2D arrays from cached numpy arrays (fast slicing, no getattr)
        data_2d: dict[str, np.ndarray] = {}
        for fld in ohlcv_needed:
            arr = np.zeros((max_wlen, n_stocks), dtype=float)
            for col, (_, ts_code, idx, end) in enumerate(valid):
                if ts_code not in self._bars_cache:
                    continue
                cached = self._bars_cache[ts_code]
                start = max(0, end - count + 1)
                wlen = end - start + 1
                offset = max_wlen - wlen
                if fld in price_fields:
                    adj = self._adj_cache[ts_code]
                    base = adj[idx]
                    raw = cached[fld][start : end + 1]
                    arr[offset : offset + wlen, col] = raw * adj[start : end + 1] / base
                else:
                    arr[offset : offset + wlen, col] = cached[fld][start : end + 1]
            data_2d[fld] = arr

        # Compute vwap
        if self._needs_vwap and "amount" in data_2d and "volume" in data_2d:
            amt, vol = data_2d["amount"], data_2d["volume"]
            with np.errstate(divide="ignore", invalid="ignore"):
                data_2d["vwap"] = np.nan_to_num((amt * 10.0) / np.where(vol != 0, vol, np.nan), nan=0.0)

        # Fundamental data: broadcast per-stock value across all time rows
        if self._fundamental_needed:
            for fld in self._fundamental_needed:
                arr = np.zeros((max_wlen, n_stocks), dtype=float)
                for col, (jq_code, _, _, _) in enumerate(valid):
                    arr[:, col] = fund_data.get(jq_code, {}).get(fld, 0.0)
                data_2d[fld] = arr

        # Check all needed fields present
        if not all(f in data_2d for f in self._needed_fields):
            return self._compute_scores_slow_stocks(valid, fund_data, price_fields, ohlcv_needed, count)

        # Evaluate all factors on 2D arrays — last row = today's values for all stocks
        features = np.zeros((n_stocks, len(self._parsed)), dtype=float)
        for f_idx, ast in enumerate(self._parsed):
            try:
                result = evaluate_2d(ast, data_2d)
                last_row = result[-1] if result.ndim == 2 else result
                features[:, f_idx] = np.nan_to_num(last_row, nan=0.0)
            except Exception:
                features[:, f_idx] = 0.0

        # Predict
        X = np.nan_to_num(features, nan=0.0)
        try:
            predictions = self.model.predict(X)
        except Exception as e:
            log.warning(f"模型预测失败: {e}")
            return {}

        scores = {code: float(pred) for code, pred in zip([v[0] for v in valid], predictions, strict=False)}
        return self._normalize_scores(scores)

    def _compute_scores_slow_stocks(
        self, valid, fund_data, price_fields, ohlcv_needed, count
    ) -> dict[str, float]:
        """Fallback: per-stock evaluation when 2D path fails."""

        from .qlib_eval import evaluate

        features_list = []
        valid_stocks = []
        for jq_code, ts_code, idx, end in valid:
            bars = self._bars_cache.get(ts_code)
            if bars is None:
                continue
            start = max(0, end - count + 1)
            adj = self._adj_cache[ts_code]
            base = adj[idx]
            data = {}
            for fld in ohlcv_needed:
                raw = bars[fld][start : end + 1]
                if fld in price_fields:
                    data[fld] = raw * adj[start : end + 1] / base
                else:
                    data[fld] = raw.copy()
            if self._needs_vwap and "amount" in data and "volume" in data:
                data["vwap"] = _compute_vwap(data["amount"], data["volume"])
            if self._fundamental_needed:
                stock_fund = fund_data.get(jq_code, {})
                ref_len = len(next(iter(data.values()))) if data else 1
                for fld in self._fundamental_needed:
                    data[fld] = np.full(ref_len, stock_fund.get(fld, 0.0), dtype=float)
            factor_values = []
            for ast in self._parsed:
                try:
                    result = evaluate(ast, data)
                    val = float(result[-1]) if len(result) > 0 else float("nan")
                    factor_values.append(val if np.isfinite(val) else 0.0)
                except Exception:
                    factor_values.append(0.0)
            features_list.append(factor_values)
            valid_stocks.append(jq_code)
        if not valid_stocks:
            return {}
        X = np.nan_to_num(np.array(features_list, dtype=float), nan=0.0)
        try:
            predictions = self.model.predict(X)
        except Exception as e:
            log.warning(f"模型预测失败: {e}")
            return {}
        scores = {code: float(pred) for code, pred in zip(valid_stocks, predictions, strict=False)}
        return self._normalize_scores(scores)

    def _compute_scores_slow(
        self,
        stocks: list[str],
        attribute_history_fn,
        get_fundamentals_fn=None,
        query_fn=None,
        valuation_obj=None,
        current_date=None,
    ) -> dict[str, float]:
        """Slow path: use attribute_history per stock (fallback)."""
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

        scores = {code: float(pred) for code, pred in zip(valid_stocks, predictions, strict=False)}
        return self._normalize_scores(scores)

    def _normalize_scores(self, scores: dict[str, float]) -> dict[str, float]:
        """Apply cross-sectional z-score normalization to align with training."""
        if not scores:
            return scores
        vals = np.array(list(scores.values()), dtype=float)
        mu = vals.mean()
        sigma = vals.std()
        if sigma == 0:
            sigma = 1.0
        normalized = (vals - mu) / sigma
        return dict(zip(scores.keys(), normalized, strict=False))

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
