"""Closed-loop LLM factor-mining pipeline.

Each round:
    1. ask the LLM for candidate factors (seeded with the existing library and
       the previous round's evaluation feedback);
    2. statically validate + de-duplicate them;
    3. evaluate survivors with Qlib + Alphalens;
    4. persist the ones that clear the quality gates into ``factor_library``;
    5. summarize all results (pass & fail) as feedback for the next round.
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
    rounds: int = 3
    per_round: int = 5
    universe: str | list[str] | None = None
    start_date: str | None = None
    end_date: str | None = None
    periods: tuple[int, ...] = (1, 5, 10)
    quantiles: int = 5
    primary_period: int | None = 5
    thresholds: QualityThresholds = field(default_factory=QualityThresholds)
    extra_instruction: str | None = None


@dataclass
class MiningResult:
    saved: list[FactorRecord] = field(default_factory=list)
    evaluations: list[FactorEvaluation] = field(default_factory=list)
    rounds_run: int = 0

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
        status="passed",
        iteration=iteration,
        metrics_json=metrics_to_json(evaluation.to_dict()),
    )


def _as_date(value: str | None):
    if not value:
        return None
    import pandas as pd

    return pd.Timestamp(value).date()


def mine_factors(config: MiningConfig | None = None) -> MiningResult:
    """Run the closed-loop factor-mining pipeline and persist passing factors."""
    config = config or MiningConfig()
    if not config.start_date or not config.end_date:
        default_start, default_end = evaluation_window_default()
        config.start_date = config.start_date or default_start
        config.end_date = config.end_date or default_end

    llm = LLMClient()
    result = MiningResult()

    seen: set[str] = {_normalize_expr(e) for e in existing_expressions()}
    feedback: str | None = None

    for round_idx in range(1, config.rounds + 1):
        log.info(f"=== 因子挖掘第 {round_idx}/{config.rounds} 轮 ===")
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

            if evaluation.passed:
                record = _to_record(candidate, evaluation, config, round_idx)
                saved = save_factor(record)
                result.saved.append(saved)
                log.info(f"  ✓ 入库 {saved.name}: IC={evaluation.ic_mean:.4f} IR={evaluation.icir:.4f}")
            else:
                log.info(f"  ✗ 未通过: {evaluation.reason}")

        result.rounds_run = round_idx
        feedback = "\n\n".join(round_feedback) if round_feedback else "本轮无有效候选，请尝试全新方向。"

    log.info(f"挖掘完成：评估 {result.n_evaluated} 个，入库 {result.n_passed} 个。")
    return result
