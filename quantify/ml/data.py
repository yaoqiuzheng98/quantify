"""Data layer for ML/DL factor mining.

Loads factor panels and forward returns from Qlib, constructs train/test
splits, and provides ready-to-use datasets for ML/DL models.

All functions lazily init Qlib — call from within functions, not at module
level.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from quantify.utils.logger import log


@dataclass
class FactorDataset:
    """A ready-to-train dataset for ML/DL models.

    Attributes
    ----------
    X_train, y_train : pd.DataFrame / pd.Series
        Training features and labels (cross-sectional, stacked by date).
    X_test, y_test : pd.DataFrame / pd.Series
        Test features and labels.
    dates_train, dates_test : list[str]
        Unique dates in train / test splits.
    feature_names : list[str]
        Column names of X (factor expressions).
    forward_period : int
        Forward return horizon in trading days.
    """

    X_train: pd.DataFrame
    y_train: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series
    dates_train: list[str]
    dates_test: list[str]
    feature_names: list[str]
    forward_period: int
    assets: list[str] = None  # all asset codes in the dataset

    @property
    def n_train(self) -> int:
        return len(self.X_train)

    @property
    def n_test(self) -> int:
        return len(self.X_test)

    @property
    def n_features(self) -> int:
        return len(self.feature_names)


def load_factor_panels(
    expressions: list[str],
    universe: str | list[str] | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
) -> dict[str, pd.DataFrame]:
    """Load factor value panels from Qlib for a list of expressions.

    Returns
    -------
    dict[str, pd.DataFrame]
        Maps expression → (date × asset) DataFrame of factor values.
    """
    from qlib.data import D

    from quantify.factor.evaluator import _resolve_universe, evaluation_window_default
    from quantify.factor.qlib_data import init_qlib, qlib_to_ts_code

    init_qlib()

    if not start_date or not end_date:
        ds, de = evaluation_window_default()
        start_date = start_date or ds
        end_date = end_date or de

    instruments = _resolve_universe(universe, start_date, end_date)
    if not instruments:
        raise RuntimeError("股票池为空，请先 dump-data 并确认 universe")

    panels: dict[str, pd.DataFrame] = {}

    # Batch load: try all expressions in one Qlib call for speed
    try:
        raw_all = D.features(instruments, expressions, start_time=start_date, end_time=end_date)
    except Exception as exc:
        log.warning(f"批量因子求值失败: {exc}，回退到逐个加载")
        raw_all = None

    if raw_all is not None and not raw_all.empty:
        for expr in expressions:
            if expr not in raw_all.columns:
                log.warning(f"因子面板为空: {expr}")
                continue
            panel = raw_all[expr].unstack(level=0)
            panel.index = panel.index.strftime("%Y-%m-%d")
            panel.columns = [qlib_to_ts_code(c) for c in panel.columns]
            panels[expr] = panel
            log.info(f"  加载因子面板: {expr}  shape={panel.shape}")
        return panels

    # Fallback: load one by one
    for expr in expressions:
        try:
            raw = D.features(instruments, [expr], start_time=start_date, end_time=end_date)
        except Exception as exc:
            log.warning(f"因子求值失败: {expr} -> {exc}")
            continue
        if raw is None or raw.empty:
            log.warning(f"因子面板为空: {expr}")
            continue
        col = raw.columns[0]
        panel = raw[col].unstack(level=0)
        panel.index = panel.index.strftime("%Y-%m-%d")
        panel.columns = [qlib_to_ts_code(c) for c in panel.columns]
        panels[expr] = panel
        log.info(f"  加载因子面板: {expr}  shape={panel.shape}")

    return panels


def load_forward_returns(
    universe: str | list[str] | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    period: int = 5,
) -> pd.DataFrame:
    """Load forward returns panel from Qlib close prices.

    Returns
    -------
    pd.DataFrame
        (date × asset) DataFrame of forward returns.  ``fwd_ret[t]`` is the
        return from ``t`` to ``t+period``.  Last ``period`` rows are NaN.
    """
    from qlib.data import D

    from quantify.factor.evaluator import _resolve_universe, evaluation_window_default
    from quantify.factor.qlib_data import init_qlib, qlib_to_ts_code

    init_qlib()

    if not start_date or not end_date:
        ds, de = evaluation_window_default()
        start_date = start_date or ds
        end_date = end_date or de

    instruments = _resolve_universe(universe, start_date, end_date)
    if not instruments:
        raise RuntimeError("股票池为空")

    raw = D.features(instruments, ["$close"], start_time=start_date, end_time=end_date)
    if raw is None or raw.empty:
        raise RuntimeError("无法加载收盘价")

    close = raw["$close"].unstack(level=0)  # (date, instrument)
    close.index = close.index.strftime("%Y-%m-%d")
    close.columns = [qlib_to_ts_code(c) for c in close.columns]

    # Forward return: close[t+period] / close[t] - 1
    fwd = close.shift(-period) / close - 1
    return fwd


def build_dataset(
    expressions: list[str],
    universe: str | list[str] | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    forward_period: int = 5,
    test_ratio: float = 0.3,
) -> FactorDataset:
    """Build a train/test dataset from factor expressions.

    Stacks cross-sectional data: each row = (date, stock) with factor values
    as features and forward return as label.

    Parameters
    ----------
    expressions : list[str]
        Qlib factor expressions to use as features.
    universe : str | list[str] | None
        Stock universe (index code or "all").
    start_date, end_date : str | None
        Evaluation window.
    forward_period : int
        Forward return horizon in trading days.
    test_ratio : float
        Fraction of dates reserved for testing (chronological split).
    """
    from quantify.factor.evaluator import evaluation_window_default

    if not start_date or not end_date:
        ds, de = evaluation_window_default()
        start_date = start_date or ds
        end_date = end_date or de

    log.info(
        f"构建 ML 数据集: {len(expressions)} 个因子, universe={universe}, "
        f"period={forward_period}, test_ratio={test_ratio}"
    )

    # Load factor panels
    panels = load_factor_panels(expressions, universe, start_date, end_date)
    if not panels:
        raise RuntimeError("无有效因子面板")

    # Load forward returns
    fwd = load_forward_returns(universe, start_date, end_date, forward_period)

    # Align dates and assets
    common_dates = sorted(set.intersection(*[set(p.index) for p in panels.values()]) & set(fwd.index))
    if not common_dates:
        raise RuntimeError("因子面板与前瞻收益无交集日期")

    common_assets = sorted(set.intersection(*[set(p.columns) for p in panels.values()]) & set(fwd.columns))
    if not common_assets:
        raise RuntimeError("因子面板与前瞻收益无交集股票")

    # Chronological train/test split
    split_idx = int(len(common_dates) * (1 - test_ratio))
    dates_train = common_dates[:split_idx]
    dates_test = common_dates[split_idx:]

    feature_names = list(panels.keys())

    def _stack(dates: list[str]) -> tuple[pd.DataFrame, pd.Series]:
        """Stack cross-sectional data for given dates into (X, y).

        Preserves a (date, asset) MultiIndex so predictions can be reshaped
        back into (date × asset) panels without reloading data.
        """
        X_rows = []
        y_rows = []
        idx = []
        for dt in dates:
            # Forward returns
            y_dt = fwd.loc[dt, common_assets]
            # Factor values
            x_dt = pd.DataFrame({expr: panels[expr].loc[dt, common_assets] for expr in feature_names})
            # Drop rows where y is NaN or any X is NaN
            valid = y_dt.notna() & x_dt.notna().all(axis=1)
            if valid.sum() == 0:
                continue
            X_rows.append(x_dt.loc[valid])
            y_rows.append(y_dt.loc[valid])
            idx.extend([(dt, a) for a in y_dt.index[valid]])

        if not X_rows:
            return pd.DataFrame(columns=feature_names), pd.Series(dtype=float)

        multi_idx = pd.MultiIndex.from_tuples(idx, names=["date", "asset"])
        X = pd.concat(X_rows, ignore_index=True)
        y = pd.concat(y_rows, ignore_index=True)
        X.index = multi_idx
        y.index = multi_idx
        return X, y

    X_train, y_train = _stack(dates_train)
    X_test, y_test = _stack(dates_test)

    log.info(
        f"数据集构建完成: train={len(X_train)} rows, test={len(X_test)} rows, "
        f"features={len(feature_names)}, train_dates={len(dates_train)}, test_dates={len(dates_test)}"
    )

    return FactorDataset(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        dates_train=dates_train,
        dates_test=dates_test,
        feature_names=feature_names,
        forward_period=forward_period,
        assets=common_assets,
    )


def load_raw_ohlcv(
    universe: str | list[str] | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    fields: tuple[str, ...] = ("open", "high", "low", "close", "volume", "amount"),
) -> dict[str, pd.DataFrame]:
    """Load raw OHLCV panels from Qlib for DL models.

    Returns
    -------
    dict[str, pd.DataFrame]
        Maps field name → (date × asset) DataFrame.
    """
    from qlib.data import D

    from quantify.factor.evaluator import _resolve_universe, evaluation_window_default
    from quantify.factor.qlib_data import init_qlib, qlib_to_ts_code

    init_qlib()

    if not start_date or not end_date:
        ds, de = evaluation_window_default()
        start_date = start_date or ds
        end_date = end_date or de

    instruments = _resolve_universe(universe, start_date, end_date)
    if not instruments:
        raise RuntimeError("股票池为空")

    exprs = [f"${f}" for f in fields]
    raw = D.features(instruments, exprs, start_time=start_date, end_time=end_date)
    if raw is None or raw.empty:
        raise RuntimeError("无法加载 OHLCV 数据")

    panels: dict[str, pd.DataFrame] = {}
    for i, field in enumerate(fields):
        panel = raw[exprs[i]].unstack(level=0)
        panel.index = panel.index.strftime("%Y-%m-%d")
        panel.columns = [qlib_to_ts_code(c) for c in panel.columns]
        panels[field] = panel

    return panels
