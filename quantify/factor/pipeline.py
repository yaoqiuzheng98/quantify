"""Closed-loop LLM factor-mining pipeline (two-phase).

Phase 1 — single-factor mining (N rounds):
    1. ask the LLM for candidate factors (seeded with the existing library and
       the previous round's evaluation feedback);
    2. statically validate + de-duplicate them;
    3. evaluate survivors with Qlib + Alphalens;
    4. persist **all** evaluated factors into ``factor_library`` (no quality
       gate — ``status`` distinguishes passed/evaluated);
    5. for each factor: LLM generates a strategy → BacktestEngine runs it →
       strategy saved to ``strategy`` table → ``factor_library.strategy_id``
       linked back;
    6. summarize all results as feedback for the next round.

Phase 2 — composite-factor mining (M rounds):
    1. LLM inspects the single-factor library (+ backtest feedback) and decides
       which factors to combine and how (equal/ic/icir weighting);
    2. compute the composite score panel → evaluate IC/IR via Alphalens;
    3. persist as ``factor_library`` row (``factor_type=composed``,
       ``parent_factor_ids`` listing the source factors);
    4. LLM generates a strategy for the composite → backtest → persist → link.
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
    # Phase 1: single-factor mining
    single_rounds: int = 3
    per_round: int = 5
    # Phase 2: composite-factor mining
    compose_rounds: int = 2
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
    # Legacy compat: if single_rounds=0 and rounds>0, use old-style rounds
    rounds: int = 0


@dataclass
class MiningResult:
    saved: list[FactorRecord] = field(default_factory=list)
    evaluations: list[FactorEvaluation] = field(default_factory=list)
    rounds_run: int = 0
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
    iteration: int,
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
        iteration=iteration,
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
) -> str | None:
    """Generate strategy, backtest, persist, link. Returns feedback text."""
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
    )
    if result.error:
        log.warning(f"  策略回测失败: {result.error}")
        return f"策略回测失败: {result.error}"
    if result.metrics:
        m = result.metrics
        return (
            f"策略#{result.strategy_id}: 总收益={m.get('total_return_pct', 0):.2f}%, "
            f"年化={m.get('annual_return_pct', 0):.2f}%, "
            f"夏普={m.get('sharpe_ratio', 0):.2f}, "
            f"回撤={m.get('max_drawdown_pct', 0):.2f}%"
        )
    return None


def mine_factors(config: MiningConfig | None = None) -> MiningResult:
    """Run the two-phase closed-loop factor-mining pipeline."""
    config = config or MiningConfig()
    if not config.start_date or not config.end_date:
        default_start, default_end = evaluation_window_default()
        config.start_date = config.start_date or default_start
        config.end_date = config.end_date or default_end

    # Legacy compat: if rounds>0 and single_rounds==0, treat rounds as single_rounds
    n_single = config.single_rounds if config.single_rounds > 0 else config.rounds

    result = MiningResult()
    llm = LLMClient()

    seen: set[str] = {_normalize_expr(e) for e in existing_expressions()}
    feedback: str | None = None

    # ===================================================================
    # Phase 1: Single-factor mining + strategy backtest
    # ===================================================================
    for round_idx in range(1, n_single + 1):
        log.info(f"=== 单因子挖掘 第 {round_idx}/{n_single} 轮 ===")
        try:
            candidates = llm.generate_factors(
                config.per_round,
                existing=sorted(seen)[:40],
                feedback=feedback,
                extra_instruction=config.extra_instruction,
            )
        except Exception as exc:  # noqa: BLE001
            log.error(f"LLM 生成失败，提前结束: {exc}")
            break

        round_feedback: list[str] = []
        for candidate in candidates:
            norm = _normalize_expr(candidate.expression)
            if norm in seen:
                log.info(f"  跳过重复因子: {candidate.expression}")
                continue
            seen.add(norm)

            syntax = validate_expression(candidate.expression)
            if not syntax.ok:
                round_feedback.append(f"表达式: {candidate.expression}\n结果: 语法错误（{syntax.error}）")
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
            round_feedback.append(evaluation.to_feedback_text())

            # 无门槛入库
            record = _to_record(candidate, evaluation, config, round_idx)
            saved = save_factor(record)
            result.saved.append(saved)
            if evaluation.passed:
                log.info(
                    f"  ✓ 入库(passed) {saved.name}: IC={evaluation.ic_mean:.4f} IR={evaluation.icir:.4f}"
                )
            else:
                log.info(
                    f"  ✓ 入库(evaluated) {saved.name}: IC={evaluation.ic_mean:.4f} IR={evaluation.icir:.4f}  ({evaluation.reason})"
                )

            # Strategy backtest
            bt_feedback = _backtest_factor(saved, config)
            if bt_feedback:
                round_feedback.append(f"因子: {candidate.expression}\n{bt_feedback}")

        result.rounds_run = round_idx
        feedback = "\n\n".join(round_feedback) if round_feedback else "本轮无有效候选，请尝试全新方向。"

    log.info(f"单因子挖掘完成：评估 {result.n_evaluated} 个，入库 {result.n_passed} 个。")

    # ===================================================================
    # Phase 2: Composite-factor mining + strategy backtest
    # ===================================================================
    if config.compose_rounds > 0 and result.saved:
        from quantify.factor.compose import compose_factors_llm

        compose_feedback: str | None = None
        for comp_round in range(1, config.compose_rounds + 1):
            log.info(f"=== 合成因子挖掘 第 {comp_round}/{config.compose_rounds} 轮 ===")
            try:
                plan, composite_panel, eval_metrics = compose_factors_llm(
                    universe=config.universe,
                    start_date=config.start_date,
                    end_date=config.end_date,
                    feedback=compose_feedback,
                    extra_instruction=config.extra_instruction,
                )
            except Exception as exc:  # noqa: BLE001
                log.error(f"合成因子生成失败: {exc}")
                break

            if plan is None or eval_metrics is None:
                log.warning("  合成因子未产出有效结果，跳过。")
                compose_feedback = "合成未成功，请尝试不同的因子组合。"
                continue

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
                iteration=comp_round,
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

            # Strategy backtest for composite factor
            bt_feedback = _backtest_factor(saved_comp, config)
            ic = eval_metrics.get("ic_mean", "NA")
            ir = eval_metrics.get("icir", "NA")
            compose_feedback = (
                f"合成因子: {plan.name} (权重={plan.weight_method}, 因子={plan.factor_ids})\n"
                f"评估: IC={ic}, IR={ir}\n"
            )
            if bt_feedback:
                compose_feedback += bt_feedback

    log.info(
        f"挖掘完成：单因子 {result.n_evaluated} 个(入库 {result.n_passed})，"
        f"合成因子 {len(result.composed)} 个。"
    )
    return result
