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
    # Phase 1: single-factor mining — iterative rounds
    rounds: int = 3  # 迭代轮数（每轮生成 n_factors 个，根据 IC 反馈改进）
    n_factors: int = 5  # 每轮生成的候选因子数
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


def _build_round_feedback(
    candidates: list[tuple[FactorCandidate, FactorEvaluation]],
) -> str:
    """Build feedback text from one round's evaluation results for the LLM.

    Tells the LLM which directions worked (high |IC|) and which didn't,
    so it can improve in the next round.
    """
    if not candidates:
        return ""

    lines = ["上一轮因子评估结果（按 |IC| 降序）："]
    sorted_results = sorted(
        candidates,
        key=lambda x: abs(x[1].ic_mean) if x[1].ic_mean is not None else 0,
        reverse=True,
    )

    effective = []
    ineffective = []

    for candidate, evaluation in sorted_results:
        ic = evaluation.ic_mean or 0
        ir = evaluation.icir or 0
        passed = evaluation.passed
        status = "✓有效" if passed else "✗无效"
        line = (
            f"  {status} {candidate.expression}\n"
            f"    IC={ic:.4f}, IC_IR={ir:.4f}, "
            f"Rank_IC={evaluation.rank_ic_mean or 0:.4f}, "
            f"分层差={evaluation.quantile_spread or 0:.4f}, "
            f"换手={evaluation.turnover or 0:.4f}\n"
            f"    逻辑: {candidate.hypothesis}"
        )
        lines.append(line)
        if passed:
            effective.append(candidate.expression)
        else:
            ineffective.append(candidate.expression)

    lines.append("")
    lines.append("## 下一轮改进建议")
    if effective:
        lines.append(
            f"有效方向（|IC|>{0.02}或|ICIR|>{0.3}），请在此基础上深入探索：\n"
            f"- 尝试不同窗口长度（更短/更长）\n"
            f"- 用 Rank/标准化变换增强截面区分度\n"
            f"- 与其他字段组合（如价格×成交量、估值×动量）\n"
            f"- 有效因子: {'; '.join(effective[:3])}"
        )
    if ineffective:
        lines.append(
            f"无效方向（|IC|极低），请放弃这些方向，尝试新维度：\n"
            f"- 无效因子: {'; '.join(ineffective[:3])}\n"
            f"- 建议探索：资金流、波动率结构、跨期限结构、非线性变换、条件因子等"
        )
    if not effective:
        lines.append(
            "⚠️ 上一轮没有有效因子。请尝试更复杂的表达式：\n"
            "- 多算子嵌套（如 Rank(Corr(Mean($close,5), Mean($volume,5), 10))）\n"
            "- 条件因子（如 If($volume > Mean($volume,20), $close/Ref($close,5)-1, 0)）\n"
            "- 跨字段比值（如 $close/Mean($close,20) × $turn/Mean($turn,20)）\n"
            "- 非线性变换（如 Power, Sign, Abs 组合）"
        )

    return "\n".join(lines)


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
    # Phase 1: Single-factor mining with iterative rounds
    # Each round: LLM generates → evaluate → feedback IC results → next round
    # ===================================================================
    round_feedback: str | None = None

    for round_idx in range(1, config.rounds + 1):
        log.info(f"=== 单因子挖掘 第 {round_idx}/{config.rounds} 轮：生成 {config.n_factors} 个候选 ===")
        candidates = llm.generate_factors(
            config.n_factors,
            existing=sorted(seen)[:40],
            feedback=round_feedback,
            extra_instruction=config.extra_instruction,
        )

        round_results: list[tuple[FactorCandidate, FactorEvaluation]] = []

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
            round_results.append((candidate, evaluation))

            # 无门槛入库
            record = _to_record(candidate, evaluation, config)
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

            # Strategy backtest — raises on failure
            bt_feedback = _backtest_factor(saved, config)
            log.info(f"  {bt_feedback}")

        # Build feedback for next round
        if round_idx < config.rounds:
            round_feedback = _build_round_feedback(round_results)
            n_passed_round = sum(1 for _, ev in round_results if ev.passed)
            log.info(
                f"--- 第 {round_idx} 轮结束：评估 {len(round_results)} 个，"
                f"有效 {n_passed_round} 个，反馈给 LLM 进行第 {round_idx + 1} 轮 ---"
            )

    log.info(f"单因子挖掘完成：{config.rounds} 轮，评估 {result.n_evaluated} 个，入库 {result.n_passed} 个。")

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
        f"挖掘完成：单因子 {config.rounds} 轮共 {result.n_evaluated} 个(入库 {result.n_passed})，"
        f"合成因子 {len(result.composed)} 个。"
    )
    return result
