"""Phase 2: Genetic Programming factor discovery.

Uses gplearn to evolve cross-sectional factor combinations.  GP terminals are
**pre-computed atomic factors** (Qlib expressions like ``Mean($close, 20)``,
``Std($volume, 10)``, ``Corr($close, $turn, 20)``), loaded via Qlib's
``D.features``.  GP only does **cross-sectional arithmetic** (add/sub/mul/div/
abs/neg/sign/log) to combine these atoms into new composite factors.

This is the standard approach in factor mining: rolling/time-series operations
are expensive and need per-stock sequences, so they're pre-computed once.  GP
then searches the combinatorial space cheaply using element-wise operations on
stacked (date, stock) arrays.

Output: Qlib expressions (strings) that can be fed into the existing
``factor_library`` pipeline for evaluation, strategy generation, etc.

Usage::

    from quantify.ml.gp_miner import GPMiner, GPConfig

    miner = GPMiner(GPConfig(universe="000300.SH", population=500, generations=30))
    results = miner.run()
    for expr, ic in results[:5]:
        print(f"IC={ic:.4f}  {expr}")
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from quantify.utils.logger import log

# ---------------------------------------------------------------------------
# Atomic factor definitions — pre-computed by Qlib, used as GP terminals
# ---------------------------------------------------------------------------

# Each entry: (terminal_name, qlib_expression)
# These are the building blocks GP will combine.  All rolling/time-series
# computation is done by Qlib upfront; GP only does cross-sectional arithmetic.
ATOMIC_FACTORS: list[tuple[str, str]] = [
    # ── Returns ──
    ("ret_1d", "Div(Sub($close, Ref($close, 1)), Add(Ref($close, 1), 1e-8))"),
    ("ret_5d", "Div(Sub($close, Ref($close, 5)), Add(Ref($close, 5), 1e-8))"),
    ("ret_10d", "Div(Sub($close, Ref($close, 10)), Add(Ref($close, 10), 1e-8))"),
    ("ret_20d", "Div(Sub($close, Ref($close, 20)), Add(Ref($close, 20), 1e-8))"),
    # ── Volatility ──
    ("vol_5d", "Std(Div(Sub($close, Ref($close, 1)), Add(Ref($close, 1), 1e-8)), 5)"),
    ("vol_10d", "Std(Div(Sub($close, Ref($close, 1)), Add(Ref($close, 1), 1e-8)), 10)"),
    ("vol_20d", "Std(Div(Sub($close, Ref($close, 1)), Add(Ref($close, 1), 1e-8)), 20)"),
    # ── Volume ratios ──
    ("vol_ratio_5", "Div(Sub($volume, Mean($volume, 5)), Add(Std($volume, 5), 1e-8))"),
    ("vol_ratio_10", "Div(Sub($volume, Mean($volume, 10)), Add(Std($volume, 10), 1e-8))"),
    ("vol_ratio_20", "Div(Sub($volume, Mean($volume, 20)), Add(Std($volume, 20), 1e-8))"),
    # ── Price range ──
    ("range_hl", "Div(Sub($high, $low), Add($close, 1e-8))"),
    ("range_co", "Div(Sub($close, $open), Add($open, 1e-8))"),
    ("range_oc", "Div(Sub($open, $close), Add(Ref($close, 1), 1e-8))"),
    # ── Rolling means (normalized) ──
    ("close_ma5_dev", "Div(Sub($close, Mean($close, 5)), Add(Mean($close, 5), 1e-8))"),
    ("close_ma10_dev", "Div(Sub($close, Mean($close, 10)), Add(Mean($close, 10), 1e-8))"),
    ("close_ma20_dev", "Div(Sub($close, Mean($close, 20)), Add(Mean($close, 20), 1e-8))"),
    # ── Rolling std ──
    ("close_std_5", "Std($close, 5)"),
    ("close_std_10", "Std($close, 10)"),
    ("close_std_20", "Std($close, 20)"),
    ("vol_std_5", "Std($volume, 5)"),
    ("vol_std_10", "Std($volume, 10)"),
    ("vol_std_20", "Std($volume, 20)"),
    # ── Skew / Kurt ──
    ("close_skew_20", "Skew($close, 20)"),
    ("vol_skew_20", "Skew($volume, 20)"),
    ("close_kurt_20", "Kurt($close, 20)"),
    ("vol_kurt_20", "Kurt($volume, 20)"),
    # ── VWAP deviation ──
    ("vwap_dev", "Div(Sub($close, $vwap), Add($vwap, 1e-8))"),
    ("vwap_dev_ma5", "Mean(Div(Sub($close, $vwap), Add($vwap, 1e-8)), 5)"),
    # ── Turnover ──
    ("turn_ma5", "Mean($turn, 5)"),
    ("turn_ma10", "Mean($turn, 10)"),
    ("turn_ma20", "Mean($turn, 20)"),
    ("turn_std_5", "Std($turn, 5)"),
    ("turn_std_20", "Std($turn, 20)"),
    # ── Correlations ──
    ("corr_cv_5", "Corr($close, $volume, 5)"),
    ("corr_cv_10", "Corr($close, $volume, 10)"),
    ("corr_cv_20", "Corr($close, $volume, 20)"),
    ("corr_ct_5", "Corr($close, $turn, 5)"),
    ("corr_ct_20", "Corr($close, $turn, 20)"),
    ("corr_hlv_20", "Corr(Sub($high, $low), $volume, 20)"),
    # ── Fundamentals ──
    ("ep", "Div(1, Add($pe, 1e-8))"),
    ("bp", "Div(1, Add($pb, 1e-8))"),
    ("sp", "Div(1, Add($ps, 1e-8))"),
    # ── EMA deviation ──
    ("ema5_dev", "Div(Sub($close, EMA($close, 5)), Add(EMA($close, 5), 1e-8))"),
    ("ema10_dev", "Div(Sub($close, EMA($close, 10)), Add(EMA($close, 10), 1e-8))"),
    ("ema20_dev", "Div(Sub($close, EMA($close, 20)), Add(EMA($close, 20), 1e-8))"),
    # ── Price position ──
    ("pos_20", "Div(Sub($close, Min($low, 20)), Add(Sub(Max($high, 20), Min($low, 20)), 1e-8))"),
    ("pos_40", "Div(Sub($close, Min($low, 40)), Add(Sub(Max($high, 40), Min($low, 40)), 1e-8))"),
    # ── Rank ──
    ("rank_close_20", "Rank($close, 20)"),
    ("rank_vol_20", "Rank($volume, 20)"),
    # ── Amount ──
    ("amt_ma5_dev", "Div(Sub($amount, Mean($amount, 5)), Add(Mean($amount, 5), 1e-8))"),
    ("amt_ma20_dev", "Div(Sub($amount, Mean($amount, 20)), Add(Mean($amount, 20), 1e-8))"),
    # ── Market cap (new data field) ──
    ("log_circ_mv", "Log(Add($circ_mv, 1))"),
    ("log_total_mv", "Log(Add($total_mv, 1))"),
    ("mv_ret_5d", "Div(Sub($circ_mv, Ref($circ_mv, 5)), Add(Ref($circ_mv, 5), 1e-8))"),
    ("mv_ret_20d", "Div(Sub($circ_mv, Ref($circ_mv, 20)), Add(Ref($circ_mv, 20), 1e-8))"),
    ("mv_turn_ratio", "Div($circ_mv, Add(Mul($turn, $close), 1e-8))"),
    # ── Delta (price/volume changes, new operator) ──
    ("delta_close_5", "Delta($close, 5)"),
    ("delta_close_10", "Delta($close, 10)"),
    ("delta_vol_5", "Delta($volume, 5)"),
    ("delta_turn_5", "Delta($turn, 5)"),
    ("delta_amt_5", "Delta($amount, 5)"),
    # ── Resi (residual from MA, new operator) ──
    ("resi_close_10", "Resi($close, 10)"),
    ("resi_close_20", "Resi($close, 20)"),
    ("resi_vol_20", "Resi($volume, 20)"),
    ("resi_turn_20", "Resi($turn, 20)"),
    ("resi_amt_20", "Resi($amount, 20)"),
    # ── WMA deviation (weighted MA, new operator) ──
    ("wma5_dev", "Div(Sub($close, WMA($close, 5)), Add(WMA($close, 5), 1e-8))"),
    ("wma10_dev", "Div(Sub($close, WMA($close, 10)), Add(WMA($close, 10), 1e-8))"),
    ("wma20_dev", "Div(Sub($close, WMA($close, 20)), Add(WMA($close, 20), 1e-8))"),
    # ── Slope (trend strength, new operator) ──
    ("slope_close_20", "Slope($close, 20)"),
    ("slope_vol_20", "Slope($volume, 20)"),
    ("slope_turn_20", "Slope($turn, 20)"),
    # ── Rsquare (trend consistency, new operator) ──
    ("rsq_close_20", "Rsquare($close, 20)"),
    ("rsq_vol_20", "Rsquare($volume, 20)"),
    # ── IdxMax / IdxMin (timing of peaks/troughs, new operator) ──
    ("idxmax_close_20", "IdxMax($close, 20)"),
    ("idxmin_close_20", "IdxMin($close, 20)"),
    ("idxmax_vol_20", "IdxMax($volume, 20)"),
    ("idxmin_vol_20", "IdxMin($volume, 20)"),
    # ── Mad (robust volatility, new operator) ──
    ("mad_close_20", "Mad($close, 20)"),
    ("mad_vol_20", "Mad($volume, 20)"),
    ("mad_turn_20", "Mad($turn, 20)"),
    # ── Med deviation (median, robust to outliers, new operator) ──
    ("med_close_dev_20", "Div(Sub($close, Med($close, 20)), Add(Med($close, 20), 1e-8))"),
    ("med_vol_dev_20", "Div(Sub($volume, Med($volume, 20)), Add(Med($volume, 20), 1e-8))"),
    # ── Sum (cumulative, new operator) ──
    ("sum_turn_5", "Sum($turn, 5)"),
    ("sum_turn_20", "Sum($turn, 20)"),
    ("sum_vol_5", "Sum($volume, 5)"),
    # ── Cov (covariance, new operator) ──
    ("cov_cv_20", "Cov($close, $volume, 20)"),
    ("cov_ct_20", "Cov($close, $turn, 20)"),
]


# ---------------------------------------------------------------------------
# GP helper functions (cross-sectional only — no rolling!)
# ---------------------------------------------------------------------------

_CLIP_RANGE = 1e6


def _safe_div(x, y):
    """Element-wise safe division, clipped to avoid overflow."""
    result = np.divide(x, y, out=np.zeros_like(x, dtype=float), where=np.abs(y) > 1e-10)
    return np.clip(result, -_CLIP_RANGE, _CLIP_RANGE)


def _safe_log(x):
    return np.log(np.abs(x) + 1e-10)


def _safe_mul(x, y):
    return np.clip(np.multiply(x, y), -_CLIP_RANGE, _CLIP_RANGE)


def _safe_sub(x, y):
    return np.clip(np.subtract(x, y), -_CLIP_RANGE, _CLIP_RANGE)


def _safe_add(x, y):
    return np.clip(np.add(x, y), -_CLIP_RANGE, _CLIP_RANGE)


# ---------------------------------------------------------------------------
# GP Config & Result
# ---------------------------------------------------------------------------


@dataclass
class GPConfig:
    """Configuration for GP factor discovery."""

    universe: str | list[str] | None = None
    start_date: str | None = None
    end_date: str | None = None
    forward_period: int = 5
    # GP parameters
    population: int = 500
    generations: int = 30
    tournament_size: int = 20
    p_crossover: float = 0.7
    p_subtree_mutation: float = 0.1
    p_hoist_mutation: float = 0.05
    p_point_mutation: float = 0.1
    max_depth: int = 5
    init_depth: tuple[int, int] = (2, 6)
    # Fitness
    metric: str = "ic"  # "ic" or "rank_ic"
    # Data
    test_ratio: float = 0.3
    val_ratio: float = 0.15  # validation set ratio (from train)
    # How many top expressions to return
    top_k: int = 10
    # Atomic factors to use as terminals (defaults to ATOMIC_FACTORS)
    atomic_factors: list[tuple[str, str]] = field(default_factory=lambda: list(ATOMIC_FACTORS))
    # Random seed
    random_state: int = 42
    # Cross-sectional standardize features (z-score per day)
    standardize: bool = True
    # Multi-period forward returns for fitness (IC averaged across periods)
    forward_periods: tuple[int, ...] = (5,)  # (1, 5, 10) for multi-period


@dataclass
class GPResult:
    """Result of GP factor discovery."""

    expressions: list[str]  # top-k Qlib expressions
    fitness: list[float]  # corresponding IC values
    test_fitness: list[float]  # out-of-sample fitness
    history: object  # gplearn estimator (for debugging)


# ---------------------------------------------------------------------------
# GP Miner
# ---------------------------------------------------------------------------


class GPMiner:
    """Genetic Programming factor discovery using gplearn.

    GP terminals are pre-computed atomic factors (Qlib expressions).  GP
    combines them cross-sectionally using arithmetic operators only.
    """

    def __init__(self, config: GPConfig | None = None) -> None:
        self.config = config or GPConfig()
        # terminal_name → Qlib expression mapping
        self._terminal_exprs: dict[str, str] = {}

    def _build_function_set(self) -> tuple:
        """Build gplearn function set — cross-sectional ops only."""
        from gplearn.functions import make_function

        functions = [
            make_function(function=_safe_add, name="add", arity=2),
            make_function(function=_safe_sub, name="sub", arity=2),
            make_function(function=_safe_mul, name="mul", arity=2),
            make_function(function=_safe_div, name="div", arity=2),
            make_function(function=np.abs, name="abs", arity=1),
            make_function(function=np.negative, name="neg", arity=1),
            make_function(function=np.sign, name="sign", arity=1),
            make_function(function=_safe_log, name="log", arity=1),
        ]
        return tuple(functions)

    def _load_industry_map(self, ts_codes: list[str]) -> dict[str, str]:
        """Load SW L1 industry mapping {ts_code: industry_name}."""
        from sqlalchemy import text as sa_text

        from quantify.database.engine import session_scope

        if not ts_codes:
            return {}
        codes_str = "','".join(ts_codes)
        # Get ALL historical industry memberships (not just current).
        # Using is_new='Y' AND out_date IS NULL introduces look-ahead bias
        # because it only returns stocks currently in the index.
        # Instead, get all members and let the neutralization use whatever
        # industry each stock belongs to. For stocks that changed industry,
        # this uses the latest assignment which is a minor approximation.
        query = sa_text(
            f"""
            SELECT ts_code, l1_name
            FROM index_member_all
            WHERE ts_code IN ('{codes_str}')
              AND is_new = 'Y'
            """
        )
        mapping: dict[str, str] = {}
        with session_scope() as sess:
            rows = sess.execute(query).fetchall()
        # Deduplicate: if a stock appears multiple times (changed industry),
        # keep the last occurrence (most recent assignment)
        for ts_code, l1_name in rows:
            mapping[ts_code] = l1_name
        return mapping

    @staticmethod
    def _industry_neutralize(panel: pd.DataFrame, industry_map: dict[str, str]) -> pd.DataFrame:
        """Subtract daily industry mean from each stock's factor value.

        panel: (date × stock) DataFrame
        industry_map: {ts_code: industry_name}
        Returns neutralized (date × stock) DataFrame.
        """
        # Vectorized: for each date, group by industry and subtract mean
        # stack to long format, groupby, unstack back
        long = panel.stack()
        long.name = "value"
        long = long.reset_index()
        long.columns = ["date", "asset", "value"]
        long["industry"] = long["asset"].map(industry_map)
        # Industry mean per date
        ind_mean = long.groupby(["date", "industry"])["value"].transform("mean")
        long["neutral"] = long["value"] - ind_mean
        result = long.pivot(index="date", columns="asset", values="neutral")
        # Restore original column order and fill NaN
        result = result.reindex(columns=panel.columns, index=panel.index)
        return result.fillna(0.0)

    def _load_training_data(
        self,
    ) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        """Load pre-computed atomic factor panels and forward returns.

        Returns stacked (date, stock) datasets: each row = one observation,
        columns = atomic factor values.  All rolling computation is done by
        Qlib upfront.
        """
        from qlib.data import D

        from quantify.factor.evaluator import _resolve_universe, evaluation_window_default
        from quantify.factor.qlib_data import init_qlib, qlib_to_ts_code

        from .data import load_forward_returns

        init_qlib()

        cfg = self.config
        if not cfg.start_date or not cfg.end_date:
            ds, de = evaluation_window_default()
            cfg.start_date = cfg.start_date or ds
            cfg.end_date = cfg.end_date or de

        instruments = _resolve_universe(cfg.universe, cfg.start_date, cfg.end_date)
        if not instruments:
            raise RuntimeError("股票池为空")

        # Build terminal name → Qlib expression mapping
        atomics = cfg.atomic_factors
        self._terminal_exprs = {name: expr for name, expr in atomics}
        qlib_exprs = [expr for _, expr in atomics]
        terminal_names = [name for name, _ in atomics]

        log.info(f"GP 加载 {len(atomics)} 个原子因子 (Qlib 预计算)")

        # Load all atomic factors in one Qlib call
        raw = D.features(instruments, qlib_exprs, start_time=cfg.start_date, end_time=cfg.end_date)
        if raw is None or raw.empty:
            raise RuntimeError("无法加载原子因子数据")

        # Load forward returns
        fwd = load_forward_returns(cfg.universe, cfg.start_date, cfg.end_date, cfg.forward_period)

        # Build panels: (date × stock) for each atomic factor
        common_dates = sorted(set(raw.index.get_level_values(1).strftime("%Y-%m-%d")) & set(fwd.index))
        common_assets = sorted(
            set(qlib_to_ts_code(c) for c in raw.index.get_level_values(0).unique()) & set(fwd.columns)
        )

        panels = {}
        for i, (name, expr) in enumerate(atomics):
            panel = raw[qlib_exprs[i]].unstack(level=0)
            panel.index = panel.index.strftime("%Y-%m-%d")
            panel.columns = [qlib_to_ts_code(c) for c in panel.columns]
            panels[name] = panel

        # ── Expand terminals with cross-sectional rank (CSRank) versions ──
        # For each atomic factor, add a csrank_* terminal = daily cross-sectional
        # percentile rank (0..1). This gives GP access to rank-based combinations
        # which are more robust to outliers than raw values.
        csrank_names = []
        for name, _ in atomics:
            cs_name = f"csrank_{name}"
            panels[cs_name] = panels[name].rank(axis=1, pct=True)
            csrank_names.append(cs_name)
            self._terminal_exprs[cs_name] = f"CSRank({self._terminal_exprs[name]})"

        # ── Expand terminals with industry-neutralized versions ──
        # For each atomic factor, subtract the daily industry mean (SW L1),
        # giving a neutral factor that removes sector bias.
        industry_names = []
        industry_map = self._load_industry_map(common_assets)
        if industry_map:
            log.info(f"GP 行业中性化: {len(set(industry_map.values()))} 个申万一级行业")
            for name, _ in atomics:
                neu_name = f"neu_{name}"
                panels[neu_name] = self._industry_neutralize(panels[name], industry_map)
                industry_names.append(neu_name)
                self._terminal_exprs[neu_name] = f"Neu({self._terminal_exprs[name]})"
        else:
            log.warning("无法加载行业映射，跳过行业中性化因子")

        # Update terminal list to include csrank_* and neu_* terminals
        all_terminal_names = terminal_names + csrank_names + industry_names
        log.info(
            f"GP 终端: {len(atomics)} 原子 + {len(csrank_names)} CSRank + {len(industry_names)} 行业中性 = {len(all_terminal_names)}"
        )

        # Chronological split: train → val → test
        n_dates = len(common_dates)
        n_test = int(n_dates * cfg.test_ratio)
        n_val = int(n_dates * cfg.val_ratio)
        n_train = n_dates - n_test - n_val
        dates_train = common_dates[:n_train]
        dates_val = common_dates[n_train : n_train + n_val]
        dates_test = common_dates[n_train + n_val :]

        def _stack(dates):
            """Stack cross-sectional data for given dates.

            Returns X with (date, asset) MultiIndex so we can group by date
            for daily IC computation.
            """
            X_rows = []
            y_rows = []
            idx = []
            for dt in dates:
                y_dt = fwd.loc[dt, common_assets]
                x_dt = pd.DataFrame(
                    {name: panels[name].loc[dt, common_assets] for name in all_terminal_names}
                )
                valid = y_dt.notna() & x_dt.notna().all(axis=1)
                if valid.sum() == 0:
                    continue
                X_rows.append(x_dt.loc[valid])
                y_rows.append(y_dt.loc[valid])
                idx.extend([(dt, a) for a in y_dt.index[valid]])
            if not X_rows:
                return (
                    pd.DataFrame(columns=all_terminal_names),
                    pd.Series(dtype=float),
                    pd.Index([]),
                )
            X = pd.concat(X_rows, ignore_index=True)
            y = pd.concat(y_rows, ignore_index=True)
            # Clip extreme values to prevent overflow in GP arithmetic
            X = X.clip(-_CLIP_RANGE, _CLIP_RANGE).fillna(0.0)
            y = y.fillna(0.0)
            dates_idx = pd.Index([d for d, _ in idx])
            return X, y, dates_idx

        X_train, y_train, dates_train_idx = _stack(dates_train)
        X_val, y_val, dates_val_idx = _stack(dates_val)
        X_test, y_test, dates_test_idx = _stack(dates_test)

        # Cross-sectional standardization (z-score per day)
        if cfg.standardize:

            def _standardize(X, dates_idx):
                if X.empty:
                    return X
                X_std = X.copy()
                for dt in dates_idx.unique():
                    mask = dates_idx == dt
                    row = X_std[mask]
                    mu = row.mean()
                    sigma = row.std().replace(0, 1.0)
                    X_std[mask] = (row - mu) / sigma
                return X_std.fillna(0.0)

            X_train = _standardize(X_train, dates_train_idx)
            X_val = _standardize(X_val, dates_val_idx)
            X_test = _standardize(X_test, dates_test_idx)

        log.info(
            f"GP 数据: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}, "
            f"terminals={len(all_terminal_names)}"
        )
        # Store date indices for daily IC computation
        self._dates_train_idx = dates_train_idx
        self._dates_val_idx = dates_val_idx
        self._dates_test_idx = dates_test_idx
        return X_train, y_train, X_val, y_val, X_test, y_test

    @staticmethod
    def _daily_ic(y_pred: np.ndarray, y: np.ndarray, dates_idx: pd.Index) -> float:
        """Compute mean daily Spearman rank IC (cross-sectional IC averaged over dates).

        This is the correct IC metric for factor evaluation: each day's
        cross-sectional rank correlation between predicted scores and actual
        returns, averaged across all dates.
        """
        from scipy.stats import spearmanr

        if len(y_pred) != len(dates_idx):
            return 0.0
        ics = []
        for dt in dates_idx.unique():
            mask = dates_idx == dt
            yp = y_pred[mask]
            ya = y[mask]
            valid = np.isfinite(yp) & np.isfinite(ya)
            if valid.sum() < 10:
                continue
            if np.std(yp[valid]) < 1e-12 or np.std(ya[valid]) < 1e-12:
                continue
            corr, _ = spearmanr(yp[valid], ya[valid])
            if np.isfinite(corr):
                ics.append(float(corr))
        return float(np.mean(ics)) if ics else 0.0

    def run(self) -> GPResult:
        """Run GP evolution and return top-k expressions.

        Returns
        -------
        GPResult
            Top-k expressions with train/val/test fitness.
        """
        from gplearn.fitness import make_fitness
        from gplearn.genetic import SymbolicRegressor

        cfg = self.config
        log.info(
            f"GP 因子发现: population={cfg.population}, generations={cfg.generations}, "
            f"atoms={len(cfg.atomic_factors)}"
        )

        # Load data (now includes validation set)
        X_train, y_train, X_val, y_val, X_test, y_test = self._load_training_data()
        if len(X_train) < 500:
            raise RuntimeError(f"训练数据太少: {len(X_train)} rows")

        # Build function set (cross-sectional only)
        function_set = self._build_function_set()

        # Fitness: mean daily Spearman rank IC (correct cross-sectional IC)
        dates_train_idx = self._dates_train_idx

        def _ic_metric(y, y_pred, w):
            return self._daily_ic(y_pred, y, dates_train_idx)

        ic_fitness = make_fitness(function=_ic_metric, greater_is_better=True, wrap=False)

        est = SymbolicRegressor(
            population_size=cfg.population,
            generations=cfg.generations,
            tournament_size=cfg.tournament_size,
            function_set=function_set,
            metric=ic_fitness,
            p_crossover=cfg.p_crossover,
            p_subtree_mutation=cfg.p_subtree_mutation,
            p_hoist_mutation=cfg.p_hoist_mutation,
            p_point_mutation=cfg.p_point_mutation,
            init_depth=cfg.init_depth,
            const_range=None,  # no constants — pure factor combinations
            parsimony_coefficient=0.001,  # light complexity penalty
            max_samples=1.0,
            n_jobs=-1,
            verbose=1,
            random_state=cfg.random_state,
            stopping_criteria=0.03,  # stop if daily IC > 0.03
        )

        log.info("开始 GP 进化...")
        est.fit(X_train, y_train)
        log.info("GP 进化完成")

        # Get all programs from final population
        programs = est._programs[-1]
        programs = [p for p in programs if p is not None]

        # Evaluate ALL programs on train IC first (fast, no val/test needed)
        # Then take top 100 by re-computed train IC for val/test evaluation
        X_train_np = X_train.to_numpy()
        X_val_np = X_val.to_numpy() if len(X_val) > 0 else None
        X_test_np = X_test.to_numpy()

        # Phase 1: compute train IC for all programs (vectorized execute is fast)
        train_ic_list = []
        for prog in programs:
            train_ic_list.append(
                self._daily_ic(prog.execute(X_train_np), y_train.to_numpy(), dates_train_idx)
            )

        # Sort by re-computed train IC (not raw_fitness_, which may differ)
        prog_ic = list(zip(programs, train_ic_list))
        prog_ic.sort(key=lambda x: x[1], reverse=True)

        # Phase 2: evaluate top 100 by train IC on val/test
        evaluated = []
        seen_exprs = set()
        for prog, _ in prog_ic[:100]:
            expr = self._program_to_qlib(prog)
            # Deduplicate by expression string
            if expr in seen_exprs:
                continue
            seen_exprs.add(expr)

            train_ic = self._daily_ic(prog.execute(X_train_np), y_train.to_numpy(), dates_train_idx)
            val_ic = (
                self._daily_ic(prog.execute(X_val_np), y_val.to_numpy(), self._dates_val_idx)
                if X_val_np is not None
                else 0.0
            )
            test_ic = self._daily_ic(prog.execute(X_test_np), y_test.to_numpy(), self._dates_test_idx)
            evaluated.append((prog, expr, train_ic, val_ic, test_ic))

        # Sort by validation IC (primary), then train IC (tiebreaker)
        evaluated.sort(key=lambda x: (x[3], x[2]), reverse=True)

        top = evaluated[: cfg.top_k]

        expressions = []
        train_fitness = []
        test_fitness = []
        for _, expr, tr_ic, val_ic, te_ic in top:
            expressions.append(expr)
            train_fitness.append(tr_ic)
            test_fitness.append(te_ic)

        for i, (expr, tr, te) in enumerate(zip(expressions, train_fitness, test_fitness, strict=False)):
            val_ic = top[i][3] if i < len(top) else 0.0
            log.info(f"  GP #{i + 1}: train_IC={tr:.4f} val_IC={val_ic:.4f} test_IC={te:.4f}  {expr[:80]}")

        return GPResult(
            expressions=expressions,
            fitness=train_fitness,
            test_fitness=test_fitness,
            history=est,
        )

    def _program_to_qlib(self, program) -> str:
        """Convert a gplearn program tree to a Qlib expression string.

        Terminal X0, X1, ... map to atomic factor expressions (not raw $field).
        Function names map to Qlib operators.
        """
        # gplearn function name → Qlib operator
        name_map = {
            "add": "Add",
            "sub": "Sub",
            "mul": "Mul",
            "div": "Div",
            "abs": "Abs",
            "neg": "0 - ",  # Qlib has no unary minus
            "sign": "Sign",
            "log": "Log",
        }

        # Terminal index → Qlib expression (includes atomic, csrank, neu terminals)
        # self._terminal_exprs is built in _load_training_data with all terminal names
        # gplearn assigns X0, X1, ... in order of the DataFrame columns
        if self._terminal_exprs:
            # Map by order: atomics first, then csrank_*, then neu_*
            atomics = self.config.atomic_factors
            all_names = (
                [name for name, _ in atomics]
                + [f"csrank_{name}" for name, _ in atomics]
                + [f"neu_{name}" for name, _ in atomics if f"neu_{name}" in self._terminal_exprs]
            )
            terminal_map = {f"X{i}": self._terminal_exprs[name] for i, name in enumerate(all_names)}
        else:
            terminal_map = {f"X{i}": expr for i, (_, expr) in enumerate(atomics)}

        raw_str = str(program)

        def convert(s: str) -> str:
            s = s.strip()
            # Terminal?
            if s in terminal_map:
                return f"({terminal_map[s]})"
            # Number?
            try:
                float(s)
                return s
            except ValueError:
                pass
            # Function call?
            paren_idx = s.find("(")
            if paren_idx == -1:
                return s

            func_name = s[:paren_idx].strip()
            args_str = s[paren_idx + 1 : s.rfind(")")].strip()
            args = _split_args(args_str)
            converted_args = [convert(a) for a in args]

            qlib_name = name_map.get(func_name, func_name.capitalize())

            # neg → "0 - X"
            if func_name == "neg":
                return f"0 - {converted_args[0]}"

            return f"{qlib_name}({', '.join(converted_args)})"

        return convert(raw_str)


def _split_args(s: str) -> list[str]:
    """Split function arguments by top-level commas (not inside nested parens)."""
    args = []
    depth = 0
    current = []
    for ch in s:
        if ch == "(":
            depth += 1
            current.append(ch)
        elif ch == ")":
            depth -= 1
            current.append(ch)
        elif ch == "," and depth == 0:
            args.append("".join(current).strip())
            current = []
        else:
            current.append(ch)
    if current:
        args.append("".join(current).strip())
    return args
