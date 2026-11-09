from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from hara_agent.models import FunctionDefinition


@dataclass(frozen=True)
class FunctionValidationFinding:
    function_id: str
    code: str
    message: str


class FunctionValidator:
    """Validate textual role and traceability without domain-specific matching."""

    SECTION_LABELS = (
        "潜在后果", "已识别的潜在后果", "质量要求", "性能要求", "可用性要求",
        "potential consequence", "quality requirement", "performance requirement",
    )
    CONDITION_PREFIXES = ("当", "如果", "若", "在以下条件", "when ", "if ", "under ")

    def validate(self, functions: list[FunctionDefinition]) -> list[FunctionValidationFinding]:
        findings = []
        seen_names = set()
        for item in functions:
            name = item.name.strip()
            lowered = name.lower()
            # Bound a label by display width: Latin/German labels need more
            # characters than CJK labels for the same concise function name.
            name_width = sum(2 if unicodedata.east_asian_width(char) in {"W", "F"}
                             else 1 for char in name)
            if name_width > 80:
                findings.append(self._finding(item, "function_name_too_long", "功能名疑似完整需求段落"))
            if name.startswith(("，", "。", "；", ",", ";")):
                findings.append(self._finding(item, "function_fragment", "功能名疑似段落残片"))
            if re.match(r"^\s*\d+[、.．)）-]", name):
                findings.append(self._finding(item, "numbered_item", "编号条目不得直接作为功能"))
            if lowered.startswith(tuple(value.lower() for value in self.CONDITION_PREFIXES)):
                findings.append(self._finding(item, "condition_as_function", "条件或步骤不得作为功能"))
            if any(label.lower() in lowered for label in self.SECTION_LABELS):
                findings.append(self._finding(item, "section_as_function", "章节标题或非功能要求不得作为功能"))
            if name in seen_names:
                findings.append(self._finding(item, "duplicate_function", "功能名称重复"))
            seen_names.add(name)
            if not item.sources or not any(source.location and source.excerpt for source in item.sources):
                findings.append(self._finding(item, "missing_source", "功能缺少可定位的来源和原文摘录"))
        return findings

    def ensure_valid(self, functions: list[FunctionDefinition]):
        findings = self.validate(functions)
        if findings:
            details = "; ".join(
                f"{finding.function_id}:{finding.code}" for finding in findings[:10]
            )
            raise ValueError(f"Function Checker未通过: {details}")

    @staticmethod
    def _finding(item: FunctionDefinition, code: str, message: str) -> FunctionValidationFinding:
        return FunctionValidationFinding(item.function_id, code, message)

