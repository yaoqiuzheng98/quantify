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
    init_depth: tuple[int, int] = (2, 4)
    # Fitness
    metric: str = "ic"  # "ic" or "rank_ic"
    # Data
    test_ratio: float = 0.3
    # How many top expressions to return
    top_k: int = 10
    # Atomic factors to use as terminals (defaults to ATOMIC_FACTORS)
    atomic_factors: list[tuple[str, str]] = field(default_factory=lambda: list(ATOMIC_FACTORS))
    # Random seed
    random_state: int = 42


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

    def _load_training_data(self) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
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

        # Chronological split
        split_idx = int(len(common_dates) * (1 - cfg.test_ratio))
        dates_train = common_dates[:split_idx]
        dates_test = common_dates[split_idx:]

        def _stack(dates):
            X_rows = []
            y_rows = []
            for dt in dates:
                y_dt = fwd.loc[dt, common_assets]
                x_dt = pd.DataFrame({name: panels[name].loc[dt, common_assets] for name in terminal_names})
                valid = y_dt.notna() & x_dt.notna().all(axis=1)
                if valid.sum() == 0:
                    continue
                X_rows.append(x_dt.loc[valid])
                y_rows.append(y_dt.loc[valid])
            if not X_rows:
                return pd.DataFrame(columns=terminal_names), pd.Series(dtype=float)
            X = pd.concat(X_rows, ignore_index=True)
            y = pd.concat(y_rows, ignore_index=True)
            # Clip extreme values to prevent overflow in GP arithmetic
            X = X.clip(-_CLIP_RANGE, _CLIP_RANGE).fillna(0.0)
            y = y.fillna(0.0)
            return X, y

        X_train, y_train = _stack(dates_train)
        X_test, y_test = _stack(dates_test)

        log.info(f"GP 数据: train={len(X_train)}, test={len(X_test)}, atoms={len(terminal_names)}")
        return X_train, y_train, X_test, y_test

    def run(self) -> GPResult:
        """Run GP evolution and return top-k expressions.

        Returns
        -------
        GPResult
            Top-k expressions with train/test fitness.
        """
        from gplearn.fitness import make_fitness
        from gplearn.genetic import SymbolicRegressor

        cfg = self.config
        log.info(
            f"GP 因子发现: population={cfg.population}, generations={cfg.generations}, "
            f"atoms={len(cfg.atomic_factors)}"
        )

        # Load data
        X_train, y_train, X_test, y_test = self._load_training_data()
        if len(X_train) < 500:
            raise RuntimeError(f"训练数据太少: {len(X_train)} rows")

        # Build function set (cross-sectional only)
        function_set = self._build_function_set()

        # Fitness: Spearman rank IC
        def _ic_metric(y, y_pred, w):
            from scipy.stats import spearmanr

            mask = np.isfinite(y_pred) & np.isfinite(y)
            if mask.sum() < 10:
                return 0.0
            yp = y_pred[mask]
            if np.std(yp) < 1e-12 or np.std(y[mask]) < 1e-12:
                return 0.0
            corr, _ = spearmanr(yp, y[mask])
            return float(corr) if np.isfinite(corr) else 0.0

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
            parsimony_coefficient=0.01,  # penalize complex trees
            max_samples=1.0,
            n_jobs=-1,
            verbose=1,
            random_state=cfg.random_state,
            stopping_criteria=0.05,  # stop if IC > 0.05
        )

        log.info("开始 GP 进化...")
        est.fit(X_train, y_train)
        log.info("GP 进化完成")

        # Get top-k programs by fitness from final population
        programs = est._programs[-1]
        programs = [p for p in programs if p is not None]
        programs.sort(key=lambda p: p.raw_fitness_, reverse=True)

        top_programs = programs[: cfg.top_k]

        expressions = []
        train_fitness = []
        test_fitness = []

        for prog in top_programs:
            expr = self._program_to_qlib(prog)
            expressions.append(expr)
            train_fitness.append(prog.raw_fitness_)
            test_pred = prog.execute(X_test.to_numpy())
            test_ic = self._compute_ic(test_pred, y_test.to_numpy())
            test_fitness.append(test_ic)

        for i, (expr, tr, te) in enumerate(zip(expressions, train_fitness, test_fitness, strict=False)):
            log.info(f"  GP #{i + 1}: train_IC={tr:.4f} test_IC={te:.4f}  {expr[:80]}")

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

        # Terminal index → atomic factor Qlib expression
        atomics = self.config.atomic_factors
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

    def _compute_ic(self, pred: np.ndarray, actual: np.ndarray) -> float:
        """Compute Spearman rank IC between predictions and actual returns."""
        from scipy.stats import spearmanr

        mask = np.isfinite(pred) & np.isfinite(actual)
        if mask.sum() < 10:
            return 0.0
        p, a = pred[mask], actual[mask]
        if np.std(p) < 1e-12 or np.std(a) < 1e-12:
            return 0.0
        corr, _ = spearmanr(p, a)
        return float(corr) if np.isfinite(corr) else 0.0


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
