"""LLM-driven factor mining subpackage.

Pipeline overview
-----------------
1. ``qlib_data``  — dump MySQL stock data into Qlib ``.bin`` format and init Qlib.
2. ``validator``  — syntax-check Qlib factor expressions before evaluation.
3. ``evaluator``  — compute factor values via Qlib ``D.features`` and grade them
   with statistical-quality gates + Alphalens IC / quantile back-testing.
4. ``llm``        — DeepSeek (OpenAI-compatible) client + prompt construction.
5. ``pipeline``   — closed loop: generate -> validate -> filter -> evaluate ->
   feed results back to the LLM, persisting factors that pass to the DB.

Heavy third-party deps (qlib, alphalens, openai) are imported lazily inside the
modules that need them, so importing this package never forces them to load.
Install them with ``pip install -e ".[mining]"``.
"""

from __future__ import annotations

__all__ = [
    "FactorCandidate",
    "FactorEvaluation",
    "QualityThresholds",
]


def __getattr__(name: str):  # pragma: no cover - thin lazy re-export
    if name in {"FactorEvaluation", "QualityThresholds"}:
        from quantify.factor.evaluator import FactorEvaluation, QualityThresholds

        return {"FactorEvaluation": FactorEvaluation, "QualityThresholds": QualityThresholds}[name]
    if name == "FactorCandidate":
        from quantify.factor.llm import FactorCandidate

        return FactorCandidate
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
