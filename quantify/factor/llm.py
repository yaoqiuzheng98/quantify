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

# 因子设计要求
1. 截面有效：用相对/标准化形式（比值、差分、滚动 zscore、rank），避免量纲依赖的裸价格。
2. 避免未来函数：只用历史数据（Ref/Mean/Std 等滚动算子天然满足）。
3. 多样性：动量、反转、波动率、量价、流动性、估值、资金流等不同维度都可探索。
4. 不要重复给定的「已有因子」。每个因子配一句简短的经济学/行为金融逻辑（hypothesis）。

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
    def _chat(self, messages: list[dict[str, str]]) -> str:
        resp = self._client.chat.completions.create(
            model=self._cfg.model,
            messages=messages,
            temperature=self._cfg.temperature,
            max_tokens=self._cfg.max_tokens,
            response_format={"type": "json_object"},
        )
        return resp.choices[0].message.content or ""

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
        content = self._chat(messages)
        candidates = parse_factor_response(content)
        log.info(f"LLM proposed {len(candidates)} candidate factors")
        return candidates


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

    if isinstance(payload, dict):
        items = payload.get("factors") or payload.get("data") or []
    elif isinstance(payload, list):
        items = payload
    else:
        items = []

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
