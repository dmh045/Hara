from __future__ import annotations

import math
from typing import Any


CONFIDENCE_PROMPT_CONTRACT = (
    "confidence必须为0.0到1.0范围内的JSON number，例如0.85；"
    "禁止使用high、medium、low等定性字符串。"
)

REPORT_LANGUAGE_PROMPT = (
    "\n报告语言：无论输入原文使用何种语言，抽取后的系统说明、边界、运行模式、ODD条件、功能名称、输出说明、失效、场景、"
    "危险事件、因果解释、假设、依据、安全状态和未解决原因均使用简明的简体中文。"
    "原文引用和source_excerpt必须保持原语言原文；契约要求原样复制的身份、字段名、"
    "枚举、方法条目、来源定位及数值单位保持不变。语义字段不属于逐字引用，应忠实译成中文；"
    "翻译不得增加能力、条件、工程批准或改变含义。\n"
)


def parse_confidence(
    value: Any,
    *,
    field_name: str = "confidence",
    default: float | None = 0.0,
) -> float:
    """Parse the shared Semantic confidence contract without qualitative guessing."""
    if value is None:
        if default is None:
            raise ValueError(f"{field_name}不能为空；expected=number[0.0,1.0]")
        value = default
    if isinstance(value, bool):
        raise ValueError(
            f"{field_name}必须为0.0~1.0数值；actual_type=bool；"
            "expected=number[0.0,1.0]"
        )
    if isinstance(value, (int, float)):
        confidence = float(value)
    elif isinstance(value, str):
        try:
            confidence = float(value.strip())
        except ValueError as exc:
            raise ValueError(
                f"{field_name}必须为0.0~1.0数值；actual={value!r}；"
                "expected=number[0.0,1.0]"
            ) from exc
    else:
        raise ValueError(
            f"{field_name}必须为0.0~1.0数值；actual_type={type(value).__name__}；"
            "expected=number[0.0,1.0]"
        )
    if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
        raise ValueError(
            f"{field_name}必须位于0.0~1.0；actual={confidence!r}；"
            "expected=number[0.0,1.0]"
        )
    return confidence
