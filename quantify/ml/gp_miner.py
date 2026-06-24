"""Phase 2: Genetic Programming factor discovery.

Uses gplearn to evolve Qlib factor expressions.  GP searches the expression
space by mutating and recombining syntax trees, with fitness = cross-sectional
IC against forward returns.

Output: Qlib expressions (strings) that can be fed into the existing
``factor_library`` pipeline for evaluation, strategy generation, etc.

Usage::

    from quantify.ml.gp_miner import GPMiner, GPConfig

    miner = GPMiner(GPConfig(universe="000300.SH", population=500, generations=50))
    results = miner.run()
    for expr, ic in results[:5]:
        print(f"IC={ic:.4f}  {expr}")
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from quantify.utils.logger import log

# ---------------------------------------------------------------------------
# Numpy helper functions for GP
# ---------------------------------------------------------------------------


def _safe_div(x, y):
    result = np.divide(x, y, out=np.zeros_like(x, dtype=float), where=np.abs(y) > 1e-10)
    return result + 1e-10  # avoid all-zero output for gplearn closure check


def _safe_log(x):
    return np.log(np.abs(x) + 1e-10)


def _rolling(x, window, func):
    """Apply a rolling function to a 1-D array."""
    s = pd.Series(x)
    return s.rolling(window, min_periods=1).apply(func, raw=True).to_numpy() + 1e-10


def _rolling_delta(x, n):
    """x[t] - x[t-n]."""
    s = pd.Series(x)
    return (s - s.shift(n)).to_numpy() + 1e-10


def _shift(x, n):
    """x[t-n]."""
    return pd.Series(x).shift(n).to_numpy() + 1e-10


def _rolling_rank(x, window):
    """Rolling percentile rank (0-1)."""
    s = pd.Series(x)
    return s.rolling(window, min_periods=1).rank(pct=True).to_numpy() + 1e-10


def _rolling_corr(x, y, window):
    """Rolling correlation of two series."""
    sx = pd.Series(x)
    sy = pd.Series(y)
    return sx.rolling(window, min_periods=5).corr(sy).to_numpy() + 1e-10


# ---------------------------------------------------------------------------
# Qlib field/operator definitions for GP
# ---------------------------------------------------------------------------

# Fields available as GP terminals (must match QLIB_FIELDS)
GP_FIELDS: tuple[str, ...] = (
    "open",
    "high",
    "low",
    "close",
    "volume",
    "amount",
    "vwap",
    "turn",
    "pe",
    "pb",
    "ps",
    "total_mv",
    "circ_mv",
)

# Rolling window sizes to try
GP_WINDOWS: tuple[int, ...] = (5, 10, 20, 40, 60)

# Operators that gplearn will use as function set.
# Each maps to a numpy function operating on pandas Series.
# We define them to work on 1-D arrays (single stock time series).
GP_FUNCTION_MAP: dict[str, tuple] = {
    # (arity, function)
    "add": (2, np.add),
    "sub": (2, np.subtract),
    "mul": (2, np.multiply),
    "div": (2, _safe_div),
    "mean_5": (1, lambda x: _rolling(x, 5, np.nanmean)),
    "mean_10": (1, lambda x: _rolling(x, 10, np.nanmean)),
    "mean_20": (1, lambda x: _rolling(x, 20, np.nanmean)),
    "std_5": (1, lambda x: _rolling(x, 5, np.nanstd)),
    "std_10": (1, lambda x: _rolling(x, 10, np.nanstd)),
    "std_20": (1, lambda x: _rolling(x, 20, np.nanstd)),
    "max_5": (1, lambda x: _rolling(x, 5, np.nanmax)),
    "max_10": (1, lambda x: _rolling(x, 10, np.nanmax)),
    "max_20": (1, lambda x: _rolling(x, 20, np.nanmax)),
    "min_5": (1, lambda x: _rolling(x, 5, np.nanmin)),
    "min_10": (1, lambda x: _rolling(x, 10, np.nanmin)),
    "min_20": (1, lambda x: _rolling(x, 20, np.nanmin)),
    "delta_5": (1, lambda x: _rolling_delta(x, 5)),
    "delta_10": (1, lambda x: _rolling_delta(x, 10)),
    "delta_20": (1, lambda x: _rolling_delta(x, 20)),
    "ref_5": (1, lambda x: _shift(x, 5)),
    "ref_10": (1, lambda x: _shift(x, 10)),
    "ref_20": (1, lambda x: _shift(x, 20)),
    "rank_20": (1, lambda x: _rolling_rank(x, 20)),
    "rank_60": (1, lambda x: _rolling_rank(x, 60)),
    "abs": (1, np.abs),
    "neg": (1, np.negative),
    "sign": (1, np.sign),
    "log": (1, _safe_log),
    "corr_10": (2, lambda x, y: _rolling_corr(x, y, 10)),
    "corr_20": (2, lambda x, y: _rolling_corr(x, y, 20)),
}


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
    population: int = 1000
    generations: int = 50
    tournament_size: int = 20
    p_crossover: float = 0.7
    p_subtree_mutation: float = 0.1
    p_hoist_mutation: float = 0.05
    p_point_mutation: float = 0.1
    max_depth: int = 8
    init_depth: tuple[int, int] = (2, 6)
    # Fitness
    metric: str = "ic"  # "ic" or "rank_ic"
    # Data
    test_ratio: float = 0.3
    # How many top expressions to return
    top_k: int = 10
    # Fields to use as terminals
    fields: tuple[str, ...] = GP_FIELDS
    # Random seed
    random_state: int = 42


@dataclass
class GPResult:
    """Result of GP factor discovery."""

    expressions: list[str]  # top-k Qlib expressions
    fitness: list[float]  # corresponding IC/ICIR values
    test_fitness: list[float]  # out-of-sample fitness
    history: object  # gplearn _program history (for debugging)


# ---------------------------------------------------------------------------
# GP Miner
# ---------------------------------------------------------------------------


class GPMiner:
    """Genetic Programming factor discovery using gplearn."""

    def __init__(self, config: GPConfig | None = None) -> None:
        self.config = config or GPConfig()

    def _build_function_set(self) -> tuple:
        """Build gplearn function set from GP_FUNCTION_MAP."""
        from gplearn.functions import make_function

        functions = []
        for name, (arity, func) in GP_FUNCTION_MAP.items():
            wrapped = make_function(function=func, name=name, arity=arity)
            functions.append(wrapped)
        return tuple(functions)

    def _load_training_data(self) -> tuple[pd.DataFrame, pd.Series]:
        """Load stacked (date, stock) factor features and forward returns.

        Instead of using raw OHLCV (which requires per-stock time series),
        we use pre-computed Qlib fields as GP terminals.  Each "feature" is
        a raw field value ($close, $volume, etc.) for a given (date, stock).

        GP then evolves combinations of these fields using rolling operators.
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

        # Load raw fields as features
        field_exprs = [f"${f}" for f in cfg.fields]
        log.info(f"GP 加载 {len(field_exprs)} 个字段: {cfg.fields}")
        raw = D.features(instruments, field_exprs, start_time=cfg.start_date, end_time=cfg.end_date)
        if raw is None or raw.empty:
            raise RuntimeError("无法加载数据")

        # Load forward returns
        fwd = load_forward_returns(cfg.universe, cfg.start_date, cfg.end_date, cfg.forward_period)

        # Build stacked dataset: each row = (date, stock), columns = field values
        # Qlib returns (instrument, datetime) MultiIndex
        common_dates = sorted(set(raw.index.get_level_values(1).strftime("%Y-%m-%d")) & set(fwd.index))
        common_assets = sorted(
            set(qlib_to_ts_code(c) for c in raw.index.get_level_values(0).unique()) & set(fwd.columns)
        )

        # Build field panels and stack
        field_panels = {}
        for i, fld in enumerate(cfg.fields):
            panel = raw[field_exprs[i]].unstack(level=0)
            panel.index = panel.index.strftime("%Y-%m-%d")
            panel.columns = [qlib_to_ts_code(c) for c in panel.columns]
            field_panels[fld] = panel

        # Chronological split
        split_idx = int(len(common_dates) * (1 - cfg.test_ratio))
        dates_train = common_dates[:split_idx]
        dates_test = common_dates[split_idx:]

        def _stack(dates):
            X_rows = []
            y_rows = []
            for dt in dates:
                y_dt = fwd.loc[dt, common_assets]
                x_dt = pd.DataFrame({f: field_panels[f].loc[dt, common_assets] for f in cfg.fields})
                valid = y_dt.notna() & x_dt.notna().any(axis=1)
                if valid.sum() == 0:
                    continue
                X_rows.append(x_dt.loc[valid])
                y_rows.append(y_dt.loc[valid])
            if not X_rows:
                return pd.DataFrame(columns=cfg.fields), pd.Series(dtype=float)
            return pd.concat(X_rows, ignore_index=True), pd.concat(y_rows, ignore_index=True)

        X_train, y_train = _stack(dates_train)
        X_test, y_test = _stack(dates_test)

        log.info(f"GP 数据: train={len(X_train)}, test={len(X_test)}, fields={len(cfg.fields)}")
        return X_train, y_train, X_test, y_test

    def run(self) -> GPResult:
        """Run GP evolution and return top-k expressions.

        Returns
        -------
        GPResult
            Top-k expressions with train/test fitness.
        """
        from gplearn.genetic import SymbolicRegressor

        cfg = self.config
        log.info(
            f"GP 因子发现: population={cfg.population}, generations={cfg.generations}, "
            f"fields={len(cfg.fields)}"
        )

        # Load data
        X_train, y_train, X_test, y_test = self._load_training_data()
        if len(X_train) < 500:
            raise RuntimeError(f"训练数据太少: {len(X_train)} rows")

        # Build function set
        function_set = self._build_function_set()

        # Fitness: cross-sectional IC (Spearman rank correlation)
        # gplearn maximizes fitness for 'ic' (we define a custom metric)
        # gplearn's built-in metrics are MSE-based (minimized). We want to
        # maximize IC, so we use negative MSE as a proxy and also define
        # a custom IC metric.
        from gplearn.fitness import make_fitness

        def _ic_metric(y, y_pred, w):
            """Custom fitness: Spearman rank IC (higher = better)."""
            from scipy.stats import spearmanr

            mask = np.isfinite(y_pred) & np.isfinite(y)
            if mask.sum() < 10:
                return 0.0
            corr, _ = spearmanr(y_pred[mask], y[mask])
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
            const_range=None,  # no constants — pure field combinations
            parsimony_coefficient=0.001,  # penalize overly complex trees
            max_samples=1.0,
            n_jobs=-1,
            verbose=1,
            random_state=cfg.random_state,
            stopping_criteria=0.1,  # stop if IC > 0.1
        )

        log.info("开始 GP 进化...")
        est.fit(X_train, y_train)
        log.info("GP 进化完成")

        # Get top-k programs by fitness
        # gplearn stores the best program in est._program, but we want top-k
        # from the final population
        programs = est._programs[-1]  # final generation programs
        programs = [p for p in programs if p is not None]
        programs.sort(key=lambda p: p.raw_fitness_, reverse=True)

        top_programs = programs[: cfg.top_k]

        expressions = []
        train_fitness = []
        test_fitness = []

        for prog in top_programs:
            # Convert gplearn program tree to Qlib expression
            expr = self._program_to_qlib(prog)
            expressions.append(expr)
            train_fitness.append(prog.raw_fitness_)
            # Evaluate on test set
            test_pred = prog.predict(X_test)
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

        Maps gplearn function names back to Qlib operator names.
        """
        # gplearn program has an `__str__` method that produces a Lisp-like
        # expression: e.g. "sub(mean_5(close), std_20(volume))"
        # We need to convert this to Qlib syntax: "Sub(Mean($close,5), Std($volume,20))"

        # Build a mapping from GP function names to Qlib operators
        name_map = {
            "add": "Add",
            "sub": "Sub",
            "mul": "Mul",
            "div": "Div",
            "mean_5": "Mean",
            "mean_10": "Mean",
            "mean_20": "Mean",
            "std_5": "Std",
            "std_10": "Std",
            "std_20": "Std",
            "max_5": "Max",
            "max_10": "Max",
            "max_20": "Max",
            "min_5": "Min",
            "min_10": "Min",
            "min_20": "Min",
            "delta_5": "Delta",
            "delta_10": "Delta",
            "delta_20": "Delta",
            "ref_5": "Ref",
            "ref_10": "Ref",
            "ref_20": "Ref",
            "rank_20": "Rank",
            "rank_60": "Rank",
            "abs": "Abs",
            "neg": "0 - ",  # Qlib doesn't support unary minus, use 0 - X
            "sign": "Sign",
            "log": "Log",
            "corr_10": "Corr",
            "corr_20": "Corr",
        }

        # Window size mapping
        window_map = {
            "mean_5": 5,
            "mean_10": 10,
            "mean_20": 20,
            "std_5": 5,
            "std_10": 10,
            "std_20": 20,
            "max_5": 5,
            "max_10": 10,
            "max_20": 20,
            "min_5": 5,
            "min_10": 10,
            "min_20": 20,
            "delta_5": 5,
            "delta_10": 10,
            "delta_20": 20,
            "ref_5": 5,
            "ref_10": 10,
            "ref_20": 20,
            "rank_20": 20,
            "rank_60": 60,
            "corr_10": 10,
            "corr_20": 20,
        }

        # Parse the gplearn string representation
        # gplearn uses a simplified Lisp syntax: func(arg1, arg2, ...)
        raw_str = str(program)

        # Recursively convert
        def convert(s: str) -> str:
            s = s.strip()
            # Check if it's a terminal (field name)
            if s in self.config.fields:
                return f"${s}"
            # Check if it's a number
            try:
                float(s)
                return s
            except ValueError:
                pass
            # Parse function call: func(args)
            # Find the outermost function name
            paren_idx = s.find("(")
            if paren_idx == -1:
                return s  # bare token

            func_name = s[:paren_idx].strip()
            args_str = s[paren_idx + 1 : s.rfind(")")].strip()

            # Split args by top-level commas
            args = _split_args(args_str)
            converted_args = [convert(a) for a in args]

            # Map to Qlib
            qlib_name = name_map.get(func_name, func_name.capitalize())

            # Handle neg specially
            if func_name == "neg":
                return f"0 - {converted_args[0]}"

            # Add window parameter for rolling functions
            if func_name in window_map:
                window = window_map[func_name]
                if func_name.startswith("corr_"):
                    # Corr takes two series + window
                    return f"{qlib_name}({converted_args[0]}, {converted_args[1]}, {window})"
                return f"{qlib_name}({converted_args[0]}, {window})"

            return f"{qlib_name}({', '.join(converted_args)})"

        return convert(raw_str)

    def _compute_ic(self, pred: np.ndarray, actual: np.ndarray) -> float:
        """Compute Spearman rank IC between predictions and actual returns."""
        from scipy.stats import spearmanr

        mask = np.isfinite(pred) & np.isfinite(actual)
        if mask.sum() < 10:
            return 0.0
        corr, _ = spearmanr(pred[mask], actual[mask])
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
