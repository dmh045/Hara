from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class RunConfig:
    item_path: Path
    template_path: Optional[Path]
    output_path: Path
    run_dir: Path
    method_baseline_path: Path = Path("method_assets/fusa_baseline_v1/manifest.yaml")
    report_template_path: Path = Path("references/HARA_Template_AI_20260327.xlsx")
    allow_draft: bool = False
    resume: bool = False
    run_id: str = "hara-run"
    ego_speed_kph: Optional[float] = None
    ego_speed_source: str = ""
    operating_mode: Optional[str] = None
    allow_aggregate_speed_fallback: bool = False
    max_workers: int = 4
    sample_function_limit: int | None = None
    sample_malfunction_limit: int | None = None
    sample_parent_scenario_limit: int | None = None
    sample_function_ids: tuple[str, ...] = ()
    sample_malfunction_ids: tuple[str, ...] = ()
    sample_parent_scenario_ids: tuple[str, ...] = ()
    sample_scenario_pair_limit: int = 32
    provider_attempt_limit: int | None = None

    @property
    def bounded_sample(self) -> bool:
        return self.provider_attempt_limit is not None

    def sample_scope(self) -> dict[str, object]:
        if not self.bounded_sample:
            return {}
        return {
            "version": "bounded-production-sample-v1",
            "function_limit": int(self.sample_function_limit or 0),
            "malfunction_limit": int(self.sample_malfunction_limit or 0),
            "parent_scenario_limit": int(self.sample_parent_scenario_limit or 0),
            "scenario_pair_limit": self.sample_scenario_pair_limit,
            "provider_attempt_limit": int(self.provider_attempt_limit or 0),
            "function_ids": list(self.sample_function_ids),
            "malfunction_ids": list(self.sample_malfunction_ids),
            "parent_scenario_ids": list(self.sample_parent_scenario_ids),
        }

    def validate(self) -> None:
        if not self.item_path.is_file():
            raise FileNotFoundError(f"Item Definition不存在: {self.item_path}")
        if self.template_path is not None:
            if not self.template_path.is_file():
                raise FileNotFoundError(f"HARA模板不存在: {self.template_path}")
            if self.template_path.suffix.lower() != ".xlsx":
                raise ValueError("HARA模板必须为.xlsx文件")
        else:
            if not self.method_baseline_path.is_file():
                raise FileNotFoundError(
                    f"HARA YAML baseline manifest不存在: {self.method_baseline_path}"
                )
            if not self.report_template_path.is_file():
                raise FileNotFoundError(
                    f"HARA报告模板不存在: {self.report_template_path}"
                )
            if self.report_template_path.suffix.lower() != ".xlsx":
                raise ValueError("HARA报告模板必须为.xlsx文件")
        if self.output_path.suffix.lower() != ".xlsx":
            raise ValueError("输出报告必须为.xlsx文件")
        if not self.run_id or any(not (char.isalnum() or char in "-_") for char in self.run_id):
            raise ValueError("run_id只能包含字母、数字、连字符和下划线")
        if self.ego_speed_kph is not None and self.ego_speed_kph < 0:
            raise ValueError("ego_speed_kph不得为负数")
        if self.operating_mode is not None and not self.operating_mode.strip():
            raise ValueError("operating_mode不得为空白字符串")
        if not 1 <= self.max_workers <= 32:
            raise ValueError("max_workers必须在1到32之间")
        sample_limits = (
            self.sample_function_limit,
            self.sample_malfunction_limit,
            self.sample_parent_scenario_limit,
        )
        selectors = (
            self.sample_function_ids,
            self.sample_malfunction_ids,
            self.sample_parent_scenario_ids,
        )
        if self.bounded_sample or any(value is not None for value in sample_limits) or any(selectors):
            if self.provider_attempt_limit is None or self.provider_attempt_limit < 1:
                raise ValueError("Bounded sample requires a positive Provider attempt limit")
            if any(value is None or value < 1 for value in sample_limits):
                raise ValueError("Bounded sample requires positive Function, Malfunction, and parent Scenario limits")
            if self.sample_scenario_pair_limit < 1:
                raise ValueError("Bounded sample Scenario pair limit must be positive")
            if self.max_workers != 1:
                raise ValueError("Bounded sample requires --max-workers 1 for resumable execution")
            for ids, limit in zip(selectors, sample_limits):
                if len(ids) != len(set(ids)) or any(not item.strip() for item in ids):
                    raise ValueError("Bounded sample IDs must be unique and nonblank")
                if ids and len(ids) > int(limit or 0):
                    raise ValueError("Bounded sample IDs exceed their selection limit")
