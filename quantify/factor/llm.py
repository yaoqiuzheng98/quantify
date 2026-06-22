"""DeepSeek (OpenAI-compatible) client and factor-mining prompts.

The LLM is asked to act as a quant researcher: propose Qlib factor expressions
either by *combining* known alpha building blocks or by stating a *hypothesis*
grounded in the available data fields, then return them as strict JSON so the
pipeline can parse, validate and evaluate them.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from tenacity import retry, stop_after_attempt, wait_exponential

from quantify.config import get_settings
from quantify.factor.qlib_data import QLIB_FIELDS
from quantify.factor.validator import QLIB_OPERATORS
from quantify.utils.logger import log


@dataclass
class FactorCandidate:
    name: str
    expression: str
    hypothesis: str = ""
    category: str = ""


_SYSTEM_PROMPT = """你是一名顶尖的量化研究员，精通 Qlib 因子表达式与 A 股市场微观结构。
你的任务是挖掘**有效的选股因子**：因子在横截面上对未来收益有预测能力（IC、IC_IR 显著）。

# Qlib 表达式语法
- 数据字段以 `$` 前缀引用，可用字段（仅限这些）:
{fields}
  其中价格/vwap 已前复权；$factor 为复权比例；$turn 换手率(%)；$pe/$pb/$ps 估值；$total_mv/$circ_mv 市值(万元)。
- 可用算子（仅限这些，区分大小写）:
{operators}
- 算子用法示例:
  - `Ref($close, 5)` 5 日前收盘；`Mean($close, 20)` 20 日均值；`Std($close, 20)` 20 日波动
  - `Corr($close, $volume, 10)` 量价 10 日相关；`Delta($close, 5)` 5 日差分
  - `Rank($close, 20)` 时序滚动百分位；`($close-Mean($close,20))/Std($close,20)` 标准分
  - 反转因子示例: `-1 * (($close - Ref($close, 20)) / Ref($close, 20))`
  - 量价背离示例: `Corr(Rank($close, 5), Rank($volume, 5), 10)`
- ⚠️ 语法限制：**不支持一元负号 `-X`**（如 `-Delta(...)`、`-$amount` 会报 "bad operand type for unary -"）。需要取反/负向时一律写成 `-1 * X` 或 `0 - X`。
- 算子签名（**参数个数必须严格匹配**，否则报错 "takes N positional arguments"）:
  - 一元 `Op($x)`: Abs, Log, Sign, Not
  - 逐元素二元 `Op($x, $y)`: Add, Sub, Mul, Div, Power, Greater, Less, Gt, Ge, Lt, Le, Eq, Ne, And, Or
  - 单序列+窗口 `Op($x, N)`: Mean, Sum, Std, Var, Max, Min, Med, Mad, Skew, Kurt, Delta, Ref, EMA, WMA, Slope, Rsquare, Resi, Rank, Count, IdxMax, IdxMin —— 注意 **WMA / Slope / Rsquare / Resi 都是单序列**，不接第二个序列、也不接权重（如残差 `Resi($x, N)`、加权均线 `WMA($x, N)`）
  - 双序列+窗口 `Op($x, $y, N)`: **仅 Corr、Cov** 两个
  - 三参数: `If($cond, $a, $b)`、`Quantile($x, N, qscore)`

# 因子设计要求
1. 截面有效：用相对/标准化形式（比值、差分、滚动 zscore、rank），避免量纲依赖的裸价格。
2. **只输出单标的时序表达式**：上面所有算子都在“单只股票的时间序列”上计算（如 Rank/Std 都是滚动窗口口径），**没有任何截面算子**（不存在 CSRank/CSZScore 之类，写了会报错）。截面标准化与横截面 IC 由下游评估器按日自动处理，且对每日单调/仿射变换不变，因此无需、也不要在表达式里做截面归一。
3. 避免未来函数：只用历史数据（Ref/Mean/Std 等滚动算子天然满足）。
4. 多样性：动量、反转、波动率、量价、流动性、估值、资金流等不同维度都可探索。
5. 不要重复给定的「已有因子」。每个因子配一句简短的经济学/行为金融逻辑（hypothesis）。

# 输出格式（严格 JSON，不要任何额外文字、不要 markdown 代码块）
{{"factors": [
  {{"name": "简短英文名", "expression": "Qlib表达式", "hypothesis": "一句话逻辑", "category": "momentum|reversal|volatility|volume_price|liquidity|value|other"}}
]}}
"""


def _system_prompt() -> str:
    fields = "\n".join(f"  - ${f}" for f in QLIB_FIELDS)
    operators = ", ".join(sorted(QLIB_OPERATORS))
    return _SYSTEM_PROMPT.format(fields=fields, operators=operators)


def _user_prompt(
    n: int,
    existing: list[str],
    feedback: str | None,
    extra_instruction: str | None,
) -> str:
    parts = [f"请提出 {n} 个新的候选因子。"]
    if existing:
        shown = existing[:40]
        parts.append("已有因子库（请勿重复，可在其基础上改进/组合）:\n" + "\n".join(f"- {e}" for e in shown))
    if feedback:
        parts.append("上一轮评估反馈（请据此优化：保留并改进表现好的方向，放弃无效方向）:\n" + feedback)
    if extra_instruction:
        parts.append("额外要求:\n" + extra_instruction)
    parts.append("只输出 JSON。")
    return "\n\n".join(parts)


class LLMClient:
    """Thin wrapper over the OpenAI SDK pointed at DeepSeek by default."""

    def __init__(self) -> None:
        cfg = get_settings().llm
        if not cfg.api_key:
            raise RuntimeError("LLM_API_KEY 为空，请在 .env 中配置（DeepSeek 控制台获取）。")
        from openai import OpenAI

        self._cfg = cfg
        self._client = OpenAI(api_key=cfg.api_key, base_url=cfg.base_url, timeout=cfg.timeout)

    @retry(reraise=True, stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=20))
    def _chat(self, messages: list[dict[str, str]]) -> tuple[str, str | None]:
        resp = self._client.chat.completions.create(
            model=self._cfg.model,
            messages=messages,
            temperature=self._cfg.temperature,
            max_tokens=self._cfg.max_tokens,
            response_format={"type": "json_object"},
        )
        choice = resp.choices[0]
        return (choice.message.content or ""), getattr(choice, "finish_reason", None)

    def generate_factors(
        self,
        n: int = 5,
        *,
        existing: list[str] | None = None,
        feedback: str | None = None,
        extra_instruction: str | None = None,
    ) -> list[FactorCandidate]:
        messages = [
            {"role": "system", "content": _system_prompt()},
            {"role": "user", "content": _user_prompt(n, existing or [], feedback, extra_instruction)},
        ]
        content, finish_reason = self._chat(messages)
        candidates = parse_factor_response(content)
        if candidates:
            log.info(f"LLM proposed {len(candidates)} candidate factors")
        else:
            # Surface *why* a round produced nothing instead of failing silently:
            # empty content / truncation (finish_reason="length") / unexpected JSON shape.
            preview = " ".join(content.split())[:500] or "<空回复>"
            log.warning(
                f"LLM 未产出可用候选（finish_reason={finish_reason}, content_len={len(content)}）。"
                f"原始返回片段: {preview}"
            )
            if finish_reason == "length":
                log.warning("回复因 max_tokens 截断（finish_reason=length），建议调大 LLM_MAX_TOKENS。")
        return candidates


def _extract_factor_items(payload: object) -> list:
    """Dig the list of factor dicts out of the assorted JSON shapes LLMs emit.

    Handles ``[...]``, ``{"factors": [...]}`` and common aliases, plus one level
    of nesting like ``{"data": {"factors": [...]}}``; finally falls back to the
    first value that is itself a list of dict-like entries. This keeps a round
    from silently yielding 0 candidates just because the model nested its output.
    """
    if isinstance(payload, list):
        return payload
    if not isinstance(payload, dict):
        return []
    for key in ("factors", "data", "results", "result", "candidates", "items"):
        if key in payload:
            nested = _extract_factor_items(payload[key])
            if nested:
                return nested
    for value in payload.values():
        if isinstance(value, list) and any(isinstance(v, dict) for v in value):
            return value
        if isinstance(value, dict):
            nested = _extract_factor_items(value)
            if nested:
                return nested
    return []


def parse_factor_response(content: str) -> list[FactorCandidate]:
    """Parse the LLM JSON reply into candidates, tolerating minor formatting noise."""
    text = content.strip()
    if not text:
        return []
    # strip ```json ... ``` fences if the model added them despite instructions
    fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, flags=re.DOTALL)
    if fence:
        text = fence.group(1).strip()

    payload: object
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            log.warning("无法从 LLM 回复中解析 JSON")
            return []
        try:
            payload = json.loads(match.group(0))
        except json.JSONDecodeError:
            log.warning("LLM 回复 JSON 解析失败")
            return []

    items = _extract_factor_items(payload)

    candidates: list[FactorCandidate] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        expression = str(item.get("expression", "")).strip()
        if not expression:
            continue
        candidates.append(
            FactorCandidate(
                name=str(item.get("name", "")).strip() or "factor",
                expression=expression,
                hypothesis=str(item.get("hypothesis", "")).strip(),
                category=str(item.get("category", "")).strip(),
            )
        )
    return candidates
