"""Phase 1: ML factor synthesis.

Uses existing factor library as features, trains ML models (XGBoost /
LightGBM / sklearn) to predict cross-sectional forward returns, and outputs
daily stock scores for vectorized backtesting.

Usage::

    from quantify.ml.factor_synthesis import MLSynthesizer

    synth = MLSynthesizer(
        universe="000300.SH",
        forward_period=5,
        model_type="xgboost",
    )
    result = synth.run()
    print(result.summary())
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from quantify.database.factor_store import list_factors
from quantify.utils.logger import log

from .backtest import VectorBacktestResult, compute_ic, vectorized_backtest
from .data import FactorDataset, build_dataset, load_forward_returns


@dataclass
class MLSynthConfig:
    """Configuration for ML factor synthesis."""

    universe: str | list[str] | None = None
    start_date: str | None = None
    end_date: str | None = None
    forward_period: int = 5
    test_ratio: float = 0.3
    top_n: int = 20
    rebalance_days: int = 5
    weight_method: str = "equal"  # "equal" or "score"
    model_type: str = "xgboost"  # xgboost / lightgbm / ridge / lasso / rf / gbdt
    # Model hyperparameters (optional overrides)
    model_params: dict = field(default_factory=dict)
    # Factor selection
    min_icir: float = 0.3  # only use factors with |ICIR| >= this
    max_factors: int = 20  # cap number of features


@dataclass
class MLSynthResult:
    """Result of ML factor synthesis."""

    model: object
    dataset: FactorDataset
    train_scores: pd.DataFrame  # (date × asset) predicted scores on train set
    test_scores: pd.DataFrame  # (date × asset) predicted scores on test set
    train_ic: dict
    test_ic: dict
    train_backtest: VectorBacktestResult
    test_backtest: VectorBacktestResult
    feature_importance: dict[str, float]
    config: MLSynthConfig

    def summary(self) -> str:
        lines = [
            "=== ML Factor Synthesis Results ===",
            f"Model: {self.config.model_type}",
            f"Features: {len(self.dataset.feature_names)}",
            f"Train: {self.dataset.n_train} rows, {len(self.dataset.dates_train)} days",
            f"Test:  {self.dataset.n_test} rows, {len(self.dataset.dates_test)} days",
            "",
            "--- IC Metrics ---",
            f"Train IC={self.train_ic.get('ic_mean', 0):.4f} IR={self.train_ic.get('icir', 0):.4f}",
            f"Test  IC={self.test_ic.get('ic_mean', 0):.4f} IR={self.test_ic.get('icir', 0):.4f}",
            "",
            "--- Train Backtest ---",
            f"Total return: {self.train_backtest.total_return:.2f}%",
            f"Sharpe: {self.train_backtest.sharpe:.2f}",
            f"Max DD: {self.train_backtest.max_drawdown:.2f}%",
            "",
            "--- Test Backtest ---",
            f"Total return: {self.test_backtest.total_return:.2f}%",
            f"Sharpe: {self.test_backtest.sharpe:.2f}",
            f"Max DD: {self.test_backtest.max_drawdown:.2f}%",
            "",
            "--- Top Feature Importance ---",
        ]
        sorted_imp = sorted(self.feature_importance.items(), key=lambda x: x[1], reverse=True)
        for name, imp in sorted_imp[:10]:
            lines.append(f"  {imp:.4f}  {name[:80]}")
        return "\n".join(lines)


class MLSynthesizer:
    """ML-based factor synthesis from existing factor library."""

    def __init__(self, config: MLSynthConfig | None = None) -> None:
        self.config = config or MLSynthConfig()
        self._extra_expressions: list[str] = []  # GP-discovered factors injected from CLI

    def _select_factors(self) -> list[str]:
        """Select factors from the library based on |ICIR| threshold + correlation filter."""
        factors = list_factors()
        # Only use single factors (not composed)
        single = [f for f in factors if (f.factor_type or "single") == "single"]
        if not single:
            raise RuntimeError("因子库中没有单因子，请先运行 `quantify factor mine`")

        # Filter by |ICIR| and sort
        qualified = [f for f in single if f.icir is not None and abs(f.icir) >= self.config.min_icir]
        if not qualified:
            log.warning(f"无因子满足 |ICIR|>={self.config.min_icir}，使用全部单因子")
            qualified = single

        qualified.sort(key=lambda f: abs(f.icir or 0), reverse=True)
        selected = qualified[: self.config.max_factors]
        expressions = [f.expression for f in selected]
        log.info(f"从因子库选取 {len(expressions)} 个因子 (from {len(single)} single factors)")
        return expressions

    @staticmethod
    def _filter_correlated(expressions: list[str], panels: dict, threshold: float = 0.85) -> list[str]:
        """Remove highly correlated factors (keep the one that appears first)."""
        if len(expressions) <= 1:
            return expressions
        # Stack all factor panels into a single DataFrame
        all_data = pd.DataFrame()
        for expr in expressions:
            panel = panels[expr]
            # Flatten to 1D (date, stock) → single column
            stacked = panel.stack()
            all_data[expr] = stacked

        corr_matrix = all_data.corr().abs()
        # Select factors with correlation < threshold
        selected = []
        dropped = set()
        for expr in expressions:
            if expr in dropped:
                continue
            selected.append(expr)
            # Drop factors highly correlated with this one
            for other in expressions:
                if other == expr or other in dropped or other in selected:
                    continue
                if corr_matrix.loc[expr, other] > threshold:
                    dropped.add(other)

        if len(dropped) > 0:
            log.info(f"相关性过滤: 移除 {len(dropped)} 个高相关因子 (threshold={threshold})")
        return selected

    def _build_model(self) -> object:
        """Build the ML model based on config.model_type."""
        model_type = self.config.model_type
        params = self.config.model_params

        if model_type == "xgboost":
            import xgboost as xgb

            return xgb.XGBRegressor(
                n_estimators=params.get("n_estimators", 200),
                max_depth=params.get("max_depth", 4),
                learning_rate=params.get("learning_rate", 0.05),
                subsample=params.get("subsample", 0.8),
                colsample_bytree=params.get("colsample_bytree", 0.8),
                reg_alpha=params.get("reg_alpha", 0.1),
                reg_lambda=params.get("reg_lambda", 1.0),
                min_child_weight=params.get("min_child_weight", 10),
                random_state=42,
                n_jobs=-1,
                early_stopping_rounds=20,
            )

        if model_type == "lightgbm":
            import lightgbm as lgb

            return lgb.LGBMRegressor(
                n_estimators=params.get("n_estimators", 200),
                max_depth=params.get("max_depth", 4),
                learning_rate=params.get("learning_rate", 0.05),
                subsample=params.get("subsample", 0.8),
                colsample_bytree=params.get("colsample_bytree", 0.8),
                reg_alpha=params.get("reg_alpha", 0.1),
                reg_lambda=params.get("reg_lambda", 1.0),
                min_child_weight=params.get("min_child_weight", 10),
                random_state=42,
                n_jobs=-1,
                verbose=-1,
            )

        if model_type == "ridge":
            from sklearn.linear_model import Ridge

            return Ridge(alpha=params.get("alpha", 1.0))

        if model_type == "lasso":
            from sklearn.linear_model import Lasso

            return Lasso(alpha=params.get("alpha", 0.001), max_iter=10000)

        if model_type == "rf":
            from sklearn.ensemble import RandomForestRegressor

            return RandomForestRegressor(
                n_estimators=params.get("n_estimators", 200),
                max_depth=params.get("max_depth", 6),
                random_state=42,
                n_jobs=-1,
            )

        if model_type == "gbdt":
            from sklearn.ensemble import GradientBoostingRegressor

            return GradientBoostingRegressor(
                n_estimators=params.get("n_estimators", 200),
                max_depth=params.get("max_depth", 4),
                learning_rate=params.get("learning_rate", 0.05),
                random_state=42,
            )

        raise ValueError(f"Unknown model_type: {model_type}")

    def _get_feature_importance(self, model: object) -> dict[str, float]:
        """Extract feature importance from the trained model."""
        feature_names = self._dataset.feature_names
        if hasattr(model, "feature_importances_"):
            imp = model.feature_importances_
            return dict(zip(feature_names, imp.tolist(), strict=False))
        if hasattr(model, "coef_"):
            coef = np.abs(model.coef_)
            return dict(zip(feature_names, coef.tolist(), strict=False))
        return {}

    def run(self) -> MLSynthResult:
        """Run the full ML synthesis pipeline.

        Steps:
        1. Select factors from library
        2. Build train/test dataset
        3. Train ML model
        4. Predict scores on train and test
        5. Compute IC metrics
        6. Run vectorized backtest
        """
        from quantify.factor.evaluator import evaluation_window_default

        cfg = self.config
        # Fill in default dates if not provided
        if not cfg.start_date or not cfg.end_date:
            ds, de = evaluation_window_default()
            cfg.start_date = cfg.start_date or ds
            cfg.end_date = cfg.end_date or de

        # 1. Select factors
        expressions = self._select_factors()
        # Add GP-discovered expressions (injected from CLI pipeline)
        if self._extra_expressions:
            # Avoid duplicates
            existing = set(expressions)
            for expr in self._extra_expressions:
                if expr not in existing:
                    expressions.append(expr)
                    existing.add(expr)
            log.info(f"加入 {len(self._extra_expressions)} 个 GP 因子，总因子数={len(expressions)}")

        # 2. Build dataset
        dataset = build_dataset(
            expressions,
            universe=cfg.universe,
            start_date=cfg.start_date,
            end_date=cfg.end_date,
            forward_period=cfg.forward_period,
            test_ratio=cfg.test_ratio,
        )
        self._dataset = dataset

        # Apply correlation filter on loaded panels
        # (panels are already loaded in build_dataset, but we need to reload for filtering)
        # Skip if only 1 factor
        if len(expressions) > 1:
            from quantify.ml.data import load_factor_panels

            panels = load_factor_panels(expressions, cfg.universe, cfg.start_date, cfg.end_date)
            expressions = self._filter_correlated(expressions, panels)
            # Rebuild dataset with filtered factors
            dataset = build_dataset(
                expressions,
                universe=cfg.universe,
                start_date=cfg.start_date,
                end_date=cfg.end_date,
                forward_period=cfg.forward_period,
                test_ratio=cfg.test_ratio,
            )
            self._dataset = dataset

        if dataset.n_train < 100 or dataset.n_test < 50:
            raise RuntimeError(
                f"数据量不足: train={dataset.n_train}, test={dataset.n_test}. "
                "请扩大回测区间或减少 test_ratio."
            )

        # 3. Train model
        log.info(f"训练 {cfg.model_type} 模型...")
        model = self._build_model()

        # Handle NaN in features: fill with cross-sectional median
        X_train = dataset.X_train.fillna(dataset.X_train.median()).fillna(0)
        X_test = dataset.X_test.fillna(dataset.X_train.median()).fillna(0)
        # Drop any remaining NaN targets (shouldn't happen after _stack filtering)
        y_train = dataset.y_train
        valid_y = y_train.notna()
        if not valid_y.all():
            log.warning(f"丢弃 {(~valid_y).sum()} 行 NaN 目标值")
            X_train = X_train[valid_y]
            y_train = y_train[valid_y]

        # Train with early stopping for tree models
        # Carve a validation set from the END of training data (chronological)
        fit_kwargs = {}
        if cfg.model_type in ("xgboost", "lightgbm"):
            # Use last 15% of training data as validation (chronological split)
            val_split = int(len(X_train) * 0.85)
            X_tr, X_val = X_train.iloc[:val_split], X_train.iloc[val_split:]
            y_tr, y_val = y_train.iloc[:val_split], y_train.iloc[val_split:]
            fit_kwargs["eval_set"] = [(X_val, y_val.fillna(0))]
            if cfg.model_type == "lightgbm":
                import lightgbm as lgb

                fit_kwargs["callbacks"] = [
                    lgb.early_stopping(20),
                    lgb.log_evaluation(0),
                ]
            log.info(f"Early stopping: train={len(X_tr)}, val={len(X_val)}")
        else:
            X_tr, y_tr = X_train, y_train

        model.fit(X_tr, y_tr, **fit_kwargs)
        if hasattr(model, "best_iteration") and model.best_iteration is not None:
            log.info(f"模型训练完成: best_iteration={model.best_iteration}")
        else:
            log.info("模型训练完成")

        # 4. Predict
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)

        # 5. Rebuild (date × asset) score panels with cross-sectional z-score normalization
        train_scores = self._rebuild_panel(train_pred, dataset, is_train=True)
        test_scores = self._rebuild_panel(test_pred, dataset, is_train=False)
        # Cross-sectional z-score per date (normalizes score distribution across dates)
        train_scores = train_scores.sub(train_scores.mean(axis=1), axis=0).div(
            train_scores.std(axis=1).replace(0, 1.0), axis=0
        )
        test_scores = test_scores.sub(test_scores.mean(axis=1), axis=0).div(
            test_scores.std(axis=1).replace(0, 1.0), axis=0
        )

        # 6. Load forward returns and close prices for IC + backtest
        fwd_returns = load_forward_returns(
            universe=cfg.universe,
            start_date=cfg.start_date,
            end_date=cfg.end_date,
            period=cfg.forward_period,
        )

        # Load close prices for backtest
        from qlib.data import D

        from quantify.factor.evaluator import _resolve_universe
        from quantify.factor.qlib_data import init_qlib, qlib_to_ts_code

        init_qlib()
        instruments = _resolve_universe(cfg.universe, cfg.start_date or "", cfg.end_date or "")
        raw = D.features(instruments, ["$close"], start_time=cfg.start_date, end_time=cfg.end_date)
        close_panel = raw["$close"].unstack(level=0)
        close_panel.index = close_panel.index.strftime("%Y-%m-%d")
        close_panel.columns = [qlib_to_ts_code(c) for c in close_panel.columns]

        # 7. Compute IC
        train_ic = compute_ic(train_scores, fwd_returns)
        test_ic = compute_ic(test_scores, fwd_returns)
        log.info(f"Train IC={train_ic['ic_mean']:.4f} IR={train_ic['icir']:.4f}")
        log.info(f"Test  IC={test_ic['ic_mean']:.4f} IR={test_ic['icir']:.4f}")

        # 8. Vectorized backtest
        train_bt = vectorized_backtest(
            train_scores,
            close_panel,
            top_n=cfg.top_n,
            rebalance_days=cfg.rebalance_days,
            weight_method=cfg.weight_method,
        )
        test_bt = vectorized_backtest(
            test_scores,
            close_panel,
            top_n=cfg.top_n,
            rebalance_days=cfg.rebalance_days,
            weight_method=cfg.weight_method,
        )
        log.info(f"Train: return={train_bt.total_return:.2f}% sharpe={train_bt.sharpe:.2f}")
        log.info(f"Test:  return={test_bt.total_return:.2f}% sharpe={test_bt.sharpe:.2f}")

        # 9. Feature importance
        importance = self._get_feature_importance(model)

        result = MLSynthResult(
            model=model,
            dataset=dataset,
            train_scores=train_scores,
            test_scores=test_scores,
            train_ic=train_ic,
            test_ic=test_ic,
            train_backtest=train_bt,
            test_backtest=test_bt,
            feature_importance=importance,
            config=cfg,
        )

        # 10. Save model + generate reusable strategy
        self._save_model_and_strategy(model, expressions, cfg, result)

        return result

    def _save_model_and_strategy(
        self,
        model: object,
        expressions: list[str],
        cfg: MLSynthConfig,
        result: MLSynthResult,
    ) -> None:
        """Save the trained model to disk and generate a reusable strategy source.

        The strategy computes factor values at runtime using attribute_history()
        and feeds them to the saved model for predictions — no hardcoded holdings.
        """

        from quantify.factor.llm import _to_jq_code

        from .strategy_runtime import save_model

        # Save model
        universe_str = (cfg.universe or "all").replace(".", "_").replace("/", "_")
        model_name = f"ml_{cfg.model_type}_{universe_str}"
        config_dict = {
            "model_type": cfg.model_type,
            "universe": cfg.universe,
            "top_n": cfg.top_n,
            "rebalance_days": cfg.rebalance_days,
            "forward_period": cfg.forward_period,
            "test_ic": result.test_ic,
            "train_ic": result.train_ic,
        }
        save_model(model, expressions, config_dict, model_name)

        # Generate strategy source
        jq_universe = _to_jq_code(cfg.universe) if cfg.universe and cfg.universe != "all" else "000300.XSHG"
        self._saved_model_name = model_name
        self._strategy_source = _generate_ml_strategy_source(
            model_name=model_name,
            jq_universe=jq_universe,
            top_n=cfg.top_n,
            rebalance_days=cfg.rebalance_days,
            factor_exprs=expressions,
        )

    @property
    def strategy_source(self) -> str:
        """Generated reusable strategy source code (available after run())."""
        return getattr(self, "_strategy_source", "")

    @property
    def saved_model_name(self) -> str:
        """Name of the saved model file (available after run())."""
        return getattr(self, "_saved_model_name", "")

    def _rebuild_panel(
        self,
        predictions: np.ndarray,
        dataset: FactorDataset,
        is_train: bool,
    ) -> pd.DataFrame:
        """Rebuild (date × asset) panel from flat predictions.

        Uses the MultiIndex preserved in the dataset (date, asset) to place
        predictions back into a 2D panel.
        """
        X = dataset.X_train if is_train else dataset.X_test
        dates = dataset.dates_train if is_train else dataset.dates_test
        assets = dataset.assets or list(X.index.get_level_values("asset").unique())

        # Create a Series with the same MultiIndex, then unstack
        pred_series = pd.Series(predictions, index=X.index, name="score")
        panel = pred_series.unstack(level="asset")

        # Reindex to cover all dates and assets (fill missing with NaN)
        panel = panel.reindex(index=dates, columns=assets)
        return panel


# ---------------------------------------------------------------------------
# Strategy source generation
# ---------------------------------------------------------------------------


def _generate_ml_strategy_source(
    model_name: str,
    jq_universe: str,
    top_n: int,
    rebalance_days: int,
    factor_exprs: list[str],
) -> str:
    """Generate a reusable JoinQuant-format strategy source code.

    The strategy:
    1. Loads the saved ML model at initialize()
    2. Each rebalance day, gets the stock universe via get_index_stocks()
    3. For each stock, computes factor values from attribute_history() / get_fundamentals()
    4. Feeds factor values to the model → predictions
    5. Selects top-N stocks by prediction score, equal-weight rebalance

    No hardcoded holdings — works on any date range.
    """
    # Check if any factors use fundamental fields
    import re

    all_fields = set()
    for expr in factor_exprs:
        all_fields.update(re.findall(r"\$([a-zA-Z_]+)", expr))
    fundamental_fields = all_fields & {"pe", "pb", "ps", "turn", "total_mv", "circ_mv"}

    # Build factor expressions list for embedding
    exprs_json = json.dumps(factor_exprs)

    has_fundamentals = bool(fundamental_fields)

    source = f'''from jqdata import *
import builtins
sum = builtins.sum
max = builtins.max
min = builtins.min
abs = builtins.abs
round = builtins.round
import json
import numpy as np

from quantify.ml.strategy_runtime import RuntimeContext

_MODEL_NAME = "{model_name}"
_UNIVERSE = "{jq_universe}"
_TOP_N = {top_n}
_REBALANCE_DAYS = {rebalance_days}
_FACTOR_EXPRS = json.loads('{exprs_json}')

_rt = None

def initialize(context):
    set_option("use_real_price", True)
    set_option("avoid_future_data", True)
    set_benchmark("{jq_universe}")
    set_order_cost(OrderCost(
        open_tax=0, close_tax=0,
        open_commission=0.0005, close_commission=0.0005,
        min_commission=0.5,
    ), type="stock")
    set_slippage(PriceRelatedSlippage(0.002))

    global _rt
    _rt = RuntimeContext(
        model_name=_MODEL_NAME,
        universe_code=_UNIVERSE,
        top_n=_TOP_N,
        rebalance_days=_REBALANCE_DAYS,
    )
    context.day_count = 0
    run_daily(rebalance, time="open")


def rebalance(context):
    context.day_count += 1
    if context.day_count % _REBALANCE_DAYS != 0:
        return

    dt_str = str(context.current_dt.date())

    # Get stock universe
    stocks = get_index_stocks(_UNIVERSE)
    if not stocks:
        log.warning(f"{{dt_str}} 股票池为空")
        return

    log.info(f"{{dt_str}} 调仓: 股票池={{len(stocks)}} 只")

    # Compute ML scores (pass context for fast batch data access)
    scores = _rt.compute_scores(
        stocks=stocks,
        attribute_history_fn=attribute_history,
        context=context,
'''

    if has_fundamentals:
        source += """        get_fundamentals_fn=get_fundamentals,
        query_fn=query,
        valuation_obj=valuation,
        current_date=dt_str,
"""
    else:
        source += """        current_date=dt_str,
"""

    source += """    )

    if not scores:
        log.warning(f"{dt_str} 无有效评分股票")
        return

    # Select top-N
    target = _rt.select_top_stocks(scores)
    if not target:
        log.warning(f"{dt_str} 选股为空")
        return

    log.info(f"{dt_str} 评分有效={len(scores)} 目标持仓={len(target)}")
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    for code, score in sorted_scores[:5]:
        log.info(f"  {code}: score={score:.4f}")

    # Sell positions not in target
    current_positions = list(context.portfolio.positions.keys())
    for code in current_positions:
        if code not in target:
            order_target_value(code, 0)

    # Buy / adjust target positions
    total_value = context.portfolio.total_value * 0.95
    for code, weight in target.items():
        try:
            order_target_value(code, total_value * weight)
        except Exception as e:
            log.warning(f"下单失败 {code}: {e}")
"""

    return source
