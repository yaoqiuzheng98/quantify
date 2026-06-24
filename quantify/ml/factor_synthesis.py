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

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from quantify.database.factor_store import list_factors
from quantify.utils.logger import log

from .backtest import VectorBacktestResult, compute_ic, vectorized_backtest
from .data import FactorDataset, build_dataset, load_factor_panels, load_forward_returns


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
    min_icir: float = 0.0  # only use factors with |ICIR| >= this
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

    def _select_factors(self) -> list[str]:
        """Select factors from the library based on |ICIR| threshold."""
        factors = list_factors()
        # Only use single factors (not composed)
        single = [f for f in factors if (f.factor_type or "single") == "single"]
        if not single:
            raise RuntimeError("因子库中没有单因子，请先运行 `quantify factor mine`")

        # Filter by |ICIR| and sort
        qualified = [f for f in single if f.icir is not None and abs(f.icir) >= self.config.min_icir]
        if not qualified:
            # Fall back to all single factors if none pass the threshold
            log.warning(f"无因子满足 |ICIR|>={self.config.min_icir}，使用全部单因子")
            qualified = single

        qualified.sort(key=lambda f: abs(f.icir or 0), reverse=True)
        selected = qualified[: self.config.max_factors]
        expressions = [f.expression for f in selected]
        log.info(f"从因子库选取 {len(expressions)} 个因子 (from {len(single)} single factors)")
        return expressions

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
                random_state=42,
                n_jobs=-1,
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
        y_train = dataset.y_train.fillna(0)

        model.fit(X_train, y_train)
        log.info("模型训练完成")

        # 4. Predict
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)

        # 5. Rebuild (date × asset) score panels
        train_scores = self._rebuild_panel(train_pred, dataset, is_train=True)
        test_scores = self._rebuild_panel(test_pred, dataset, is_train=False)

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

        return MLSynthResult(
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

    def _rebuild_panel(
        self,
        predictions: np.ndarray,
        dataset: FactorDataset,
        is_train: bool,
    ) -> pd.DataFrame:
        """Rebuild (date × asset) panel from flat predictions.

        The dataset was stacked with ignore_index=True, losing the (date, asset)
        mapping. We rebuild by re-loading the factor panels and using their
        structure to place predictions.
        """
        # Re-load the first factor panel to get the (date, asset) structure
        panels = load_factor_panels(
            dataset.feature_names[:1],
            universe=self.config.universe,
            start_date=self.config.start_date,
            end_date=self.config.end_date,
        )
        if not panels:
            raise RuntimeError("无法重建因子面板")

        ref_panel = list(panels.values())[0]
        dates = dataset.dates_train if is_train else dataset.dates_test
        assets = list(ref_panel.columns)

        panel = pd.DataFrame(np.nan, index=ref_panel.index, columns=assets)

        # We need to map predictions back. Since the dataset was built by
        # iterating dates and filtering valid rows, we need to replicate that.
        # Load forward returns to know which (date, asset) pairs were valid.
        fwd = load_forward_returns(
            universe=self.config.universe,
            start_date=self.config.start_date,
            end_date=self.config.end_date,
            period=dataset.forward_period,
        )

        pred_idx = 0
        for dt in dates:
            if dt not in fwd.index:
                continue
            y_dt = fwd.loc[dt, assets]
            valid = y_dt.notna()
            n_valid = int(valid.sum())
            if n_valid == 0:
                continue
            if pred_idx + n_valid > len(predictions):
                break
            panel.loc[dt, valid.index[valid]] = predictions[pred_idx : pred_idx + n_valid]
            pred_idx += n_valid

        return panel.loc[dates]
