"""Closed-loop LLM factor-mining pipeline (two-phase).

Phase 1 — single-factor mining:
    1. ask the LLM for N candidate factors (seeded with the existing library);
    2. statically validate + de-duplicate them;
    3. evaluate survivors with Qlib + Alphalens;
    4. persist **all** evaluated factors into ``factor_library`` (no quality
       gate — ``status`` distinguishes passed/evaluated);
    5. for each factor: LLM generates a strategy → BacktestEngine runs it →
       strategy saved to ``strategy`` table → ``factor_library.strategy_id``
       linked back.

Phase 2 — composite-factor mining:
    1. LLM inspects the single-factor library (+ backtest feedback) and decides
       which factors to combine and how (equal/ic/icir weighting);
    2. compute the composite score panel → evaluate IC/IR via Alphalens;
    3. persist as ``factor_library`` row (``factor_type=composed``,
       ``parent_factor_ids`` listing the source factors);
    4. LLM generates a strategy for the composite → backtest → persist → link.
    Repeat M times, each time feeding the previous result back to the LLM.

All errors are raised — no silent swallowing. If any step fails (LLM, evaluate,
backtest, compose), the exception propagates to the caller.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field

from quantify.database.factor_store import (
    FactorRecord,
    existing_expressions,
    metrics_to_json,
    save_factor,
)
from quantify.factor.evaluator import (
    FactorEvaluation,
    QualityThresholds,
    evaluate_expression,
    evaluation_window_default,
)
from quantify.factor.llm import FactorCandidate, LLMClient
from quantify.factor.validator import validate_expression
from quantify.utils.logger import log


@dataclass
class MiningConfig:
    # Phase 1: single-factor mining — how many factors to generate
    n_factors: int = 15
    # Phase 2: composite-factor mining — how many composite factors to build
    n_compose: int = 2
    # Common
    universe: str | list[str] | None = None
    start_date: str | None = None
    end_date: str | None = None
    periods: tuple[int, ...] = (1, 5, 10)
    quantiles: int = 5
    primary_period: int | None = 5
    thresholds: QualityThresholds = field(default_factory=QualityThresholds)
    extra_instruction: str | None = None
    # Strategy backtest
    backtest_top_n: int = 20
    backtest_rebalance_days: int = 5
    backtest_initial_cash: float = 1_000_000
    backtest_max_retries: int = 3


@dataclass
class MiningResult:
    saved: list[FactorRecord] = field(default_factory=list)
    evaluations: list[FactorEvaluation] = field(default_factory=list)
    composed: list[FactorRecord] = field(default_factory=list)

    @property
    def n_passed(self) -> int:
        return len(self.saved)

    @property
    def n_evaluated(self) -> int:
        return len(self.evaluations)


def _slugify(name: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_]+", "_", name).strip("_").lower()
    return slug or "factor"


def _unique_name(candidate: FactorCandidate) -> str:
    digest = hashlib.sha1(candidate.expression.encode("utf-8")).hexdigest()[:6]  # noqa: S324
    return f"{_slugify(candidate.name)}_{digest}"


def _normalize_expr(expression: str) -> str:
    return re.sub(r"\s+", "", expression)


def _to_record(
    candidate: FactorCandidate,
    evaluation: FactorEvaluation,
    config: MiningConfig,
) -> FactorRecord:
    universe = config.universe if isinstance(config.universe, str) else "custom"
    return FactorRecord(
        name=_unique_name(candidate),
        expression=evaluation.expression,
        hypothesis=candidate.hypothesis or None,
        category=candidate.category or None,
        universe=universe or "all",
        start_date=_as_date(config.start_date),
        end_date=_as_date(config.end_date),
        periods=",".join(str(p) for p in config.periods),
        ic_mean=evaluation.ic_mean,
        ic_std=evaluation.ic_std,
        icir=evaluation.icir,
        ic_tstat=evaluation.ic_tstat,
        rank_ic_mean=evaluation.rank_ic_mean,
        rank_icir=evaluation.rank_icir,
        quantile_spread=evaluation.quantile_spread,
        turnover=evaluation.turnover,
        coverage=evaluation.coverage,
        status="passed" if evaluation.passed else "evaluated",
        factor_type="single",
        metrics_json=metrics_to_json(evaluation.to_dict()),
    )


def _as_date(value: str | None):
    if not value:
        return None
    import pandas as pd

    return pd.Timestamp(value).date()


def _backtest_factor(
    factor: FactorRecord,
    config: MiningConfig,
    feedback: str | None = None,
) -> str:
    """Generate strategy, backtest, persist, link. Returns feedback text.

    Raises on any failure — errors propagate to the caller.
    """
    from quantify.factor.strategy_gen import generate_and_backtest_strategy

    result = generate_and_backtest_strategy(
        factor,
        universe=config.universe if isinstance(config.universe, str) else "all",
        start_date=config.start_date or "",
        end_date=config.end_date or "",
        top_n=config.backtest_top_n,
        rebalance_days=config.backtest_rebalance_days,
        initial_cash=config.backtest_initial_cash,
        feedback=feedback,
        max_retries=config.backtest_max_retries,
    )
    m = result.metrics
    return (
        f"策略#{result.strategy_id}: 总收益={m.get('total_return_pct', 0):.2f}%, "
        f"年化={m.get('annual_return_pct', 0):.2f}%, "
        f"夏普={m.get('sharpe_ratio', 0):.2f}, "
        f"回撤={m.get('max_drawdown_pct', 0):.2f}%"
    )


def mine_factors(config: MiningConfig | None = None) -> MiningResult:
    """Run the two-phase closed-loop factor-mining pipeline.

    All errors are raised. If any step fails (LLM generation, factor evaluation,
    strategy backtest, composition), the exception propagates immediately.
    """
    config = config or MiningConfig()
    if not config.start_date or not config.end_date:
        default_start, default_end = evaluation_window_default()
        config.start_date = config.start_date or default_start
        config.end_date = config.end_date or default_end

    result = MiningResult()
    llm = LLMClient()

    seen: set[str] = {_normalize_expr(e) for e in existing_expressions()}

    # ===================================================================
    # Phase 1: Single-factor mining + strategy backtest
    # ===================================================================
    log.info(f"=== 单因子挖掘：生成 {config.n_factors} 个候选 ===")
    candidates = llm.generate_factors(
        config.n_factors,
        existing=sorted(seen)[:40],
        feedback=None,
        extra_instruction=config.extra_instruction,
    )

    for candidate in candidates:
        norm = _normalize_expr(candidate.expression)
        if norm in seen:
            log.info(f"  跳过重复因子: {candidate.expression}")
            continue
        seen.add(norm)

        syntax = validate_expression(candidate.expression)
        if not syntax.ok:
            log.info(f"  语法不通过: {candidate.expression} -> {syntax.error}")
            continue

        log.info(f"  评估: {candidate.expression}")
        evaluation = evaluate_expression(
            candidate.expression,
            universe=config.universe,
            start_date=config.start_date,
            end_date=config.end_date,
            periods=config.periods,
            quantiles=config.quantiles,
            primary_period=config.primary_period,
            thresholds=config.thresholds,
        )
        result.evaluations.append(evaluation)

        # 无门槛入库
        record = _to_record(candidate, evaluation, config)
        saved = save_factor(record)
        result.saved.append(saved)
        if evaluation.passed:
            log.info(f"  ✓ 入库(passed) {saved.name}: IC={evaluation.ic_mean:.4f} IR={evaluation.icir:.4f}")
        else:
            log.info(
                f"  ✓ 入库(evaluated) {saved.name}: IC={evaluation.ic_mean:.4f} IR={evaluation.icir:.4f}  ({evaluation.reason})"
            )

        # Strategy backtest — raises on failure
        bt_feedback = _backtest_factor(saved, config)
        log.info(f"  {bt_feedback}")

    log.info(f"单因子挖掘完成：评估 {result.n_evaluated} 个，入库 {result.n_passed} 个。")

    # ===================================================================
    # Phase 2: Composite-factor mining + strategy backtest
    # ===================================================================
    if config.n_compose > 0 and result.saved:
        from quantify.factor.compose import compose_factors_llm

        compose_feedback: str | None = None
        for comp_idx in range(1, config.n_compose + 1):
            log.info(f"=== 合成因子挖掘 第 {comp_idx}/{config.n_compose} 个 ===")
            plan, composite_panel, eval_metrics = compose_factors_llm(
                universe=config.universe,
                start_date=config.start_date,
                end_date=config.end_date,
                feedback=compose_feedback,
                extra_instruction=config.extra_instruction,
            )

            # Save composite factor to library
            parent_ids_str = ",".join(str(fid) for fid in plan.factor_ids)
            comp_name = (
                f"composed_{_slugify(plan.name)}_{hashlib.sha1(parent_ids_str.encode()).hexdigest()[:6]}"
            )
            comp_record = FactorRecord(
                name=comp_name,
                expression=f"COMPOSED({parent_ids_str}, {plan.weight_method})",
                hypothesis=plan.hypothesis,
                category="composed",
                universe=config.universe if isinstance(config.universe, str) else "custom",
                start_date=_as_date(config.start_date),
                end_date=_as_date(config.end_date),
                periods=",".join(str(p) for p in config.periods),
                ic_mean=eval_metrics.get("ic_mean"),
                ic_std=eval_metrics.get("ic_std"),
                icir=eval_metrics.get("icir"),
                rank_ic_mean=eval_metrics.get("rank_ic_mean"),
                rank_icir=eval_metrics.get("rank_icir"),
                coverage=None,
                status="composed",
                factor_type="composed",
                parent_factor_ids=parent_ids_str,
                metrics_json=metrics_to_json(eval_metrics),
            )
            saved_comp = save_factor(comp_record)
            result.composed.append(saved_comp)
            log.info(
                f"  ✓ 合成因子入库 {saved_comp.name}: IC={eval_metrics.get('ic_mean', 'NA')} "
                f"IR={eval_metrics.get('icir', 'NA')}  父因子={parent_ids_str}"
            )

            # Strategy backtest for composite factor — raises on failure
            bt_feedback = _backtest_factor(saved_comp, config)
            ic = eval_metrics.get("ic_mean", "NA")
            ir = eval_metrics.get("icir", "NA")
            compose_feedback = (
                f"合成因子: {plan.name} (权重={plan.weight_method}, 因子={plan.factor_ids})\n"
                f"评估: IC={ic}, IR={ir}\n{bt_feedback}"
            )

    log.info(
        f"挖掘完成：单因子 {result.n_evaluated} 个(入库 {result.n_passed})，"
        f"合成因子 {len(result.composed)} 个。"
    )
    return result
