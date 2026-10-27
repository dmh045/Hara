from __future__ import annotations

import math
import re
from copy import deepcopy
from dataclasses import replace
from typing import Any

from hara_agent.models import (
    FactProvenance,
    ItemDefinitionFacts,
    ReviewStatus,
    SourceRef,
    SpeedEnvelope,
)

from .parsing import parse_confidence


class ItemDefinitionNormalizer:
    """Normalize the canonical item-artifact payload into typed project facts."""

    @staticmethod
    def normalize_source_scope(facts: ItemDefinitionFacts, blocks: list, source_id: str) -> ItemDefinitionFacts:
        """Recover explicit structured environmental rows, without inventing facts.

        Only complete literal table cells under an environmental/ODD heading
        qualify. Unrecognized/qualified prose stays with semantic extraction.
        These limits describe allowed ODD, never the actual grade or heading.
        """
        records = deepcopy(facts.odd_scope_evidence)
        locations = list(facts.odd_locations)
        road_types = list(facts.odd_road_types)
        for block in blocks:
            value = block if isinstance(block, dict) else vars(block)
            if any(any(word in str(part).casefold() for word in ("示例", "参考", "example", "reference"))
                   for part in value.get("section_path", [])):
                continue
            if value.get("kind") != "table_row" or not any(
                    str(part).strip().casefold() in {"环境条件", "运行设计域", "运行范围", "运行条件",
                        "odd", "operational design domain", "operating conditions", "operating domain"}
                    for part in value.get("section_path", [])):
                continue
            cells = list(dict.fromkeys(cell.strip() for cell in value.get("text", "").split("|") if cell.strip()))
            if len(cells) != 2:
                continue
            label, text = cells
            source = dict(source_type="item_definition", source_id=source_id,
                          location=value.get("location", ""), excerpt=value["text"])
            if label in {"停车场", "场所", "运行场所"}:
                parts = [part.strip() for part in re.split(r"[，,、；;]", text)]
                if not parts or not all(re.fullmatch(r"(?:室内|室外|地下|地上)停车场", p) for p in parts):
                    continue
                # The table explicitly declares these as alternatives to the
                # generic heading. Preserve that hierarchy in the evidence.
                locations = list(dict.fromkeys([*locations, *parts]))
                road_types = [p for p in road_types if p not in parts]
                record = dict(field="odd_locations", parent_label=label, values=parts, sources=[source])
            elif label in {"停车位类型", "停车位样式", "标线车位类型", "空间车位类型"}:
                parts = [part.strip() for part in re.split(r"[，,、；;]", text)]
                if not parts or not all(re.fullmatch(r"[\u4e00-\u9fffA-Za-z]{1,16}", p)
                        and not any(word in p for word in ("禁止", "不支持", "不允许", "示例", "例如", "未知", "待定")) for p in parts):
                    continue
                road_types = list(dict.fromkeys([*road_types, *parts]))
                record = dict(field="odd_road_types", parent_label=label, values=parts, sources=[source])
            elif label in {"上下坡", "坡度", "纵向坡度"}:
                parts = [part.strip() for part in re.split(r"[，,、；;]", text)]
                matches = [re.fullmatch(r"(上坡|下坡)\s*(\d+(?:\.\d+)?)\s*%", p) for p in parts]
                if not matches or not all(matches) or not all(math.isfinite(float(m[2])) for m in matches):
                    continue
                record = dict(field="odd_grade_limits", quantity_semantic="ALLOWED_ODD_RANGE",
                    limits=[dict(direction=m[1], upper_pct=float(m[2]), unit="percent",
                                 reference_direction="UNSPECIFIED") for m in matches], sources=[source])
            else:
                continue
            if record not in records:
                records.append(record)
        return replace(facts, odd_locations=locations, odd_road_types=road_types, odd_scope_evidence=records)

    @staticmethod
    def _parse_with_warnings(
        raw: dict[str, Any], source_id: str,
    ) -> tuple[ItemDefinitionFacts, list[str]]:
        odd = raw.get("odd", {})
        if not isinstance(odd, dict):
            raise ValueError("item_definition.odd必须为object")
        speed_min, speed_max, speed_warning = ItemDefinitionNormalizer._speed_range(
            odd.get("speed_range_kph")
        )
        normalization_warnings = [speed_warning] if speed_warning else []
        status = (
            ReviewStatus.FINALIZED
            if str(raw.get("status", "")).upper() == "FINALIZED" and not normalization_warnings
            else ReviewStatus.PENDING
        )
        speed_envelopes, envelope_warnings = ItemDefinitionNormalizer._speed_envelopes(
            raw.get("speed_envelopes"), source_id
        )
        normalization_warnings.extend(envelope_warnings)
        if envelope_warnings:
            status = ReviewStatus.PENDING
        facts = ItemDefinitionFacts(
            system_description=str(raw.get("system_description", "")).strip(),
            item_boundary=str(raw.get("item_boundary", "")).strip(),
            operating_modes=[str(value) for value in ItemDefinitionNormalizer._list(raw.get("operating_modes"), "operating_modes")],
            odd_locations=[str(value) for value in ItemDefinitionNormalizer._list(odd.get("locations"), "odd.locations")],
            odd_road_types=[str(value) for value in ItemDefinitionNormalizer._list(odd.get("road_types"), "odd.road_types")],
            odd_weather_conditions=[str(value) for value in ItemDefinitionNormalizer._list(odd.get("weather_conditions"), "odd.weather_conditions")],
            odd_road_surfaces=[str(value) for value in ItemDefinitionNormalizer._list(odd.get("road_surfaces"), "odd.road_surfaces")],
            speed_min_kph=speed_min,
            speed_max_kph=speed_max,
            speed_envelopes=speed_envelopes,
             sources=[SourceRef(
                 "item_definition", source_id,
                 str(raw.get("source_location", "")),
                 str(raw.get("source_excerpt", "")),
             )],
             status=status,
             confidence=parse_confidence(
                 raw.get("confidence"), field_name="Item Definition confidence",
             ),
        )
        return facts, normalization_warnings

    @staticmethod
    def _speed_envelopes(
        value: Any, source_id: str,
    ) -> tuple[list[SpeedEnvelope], list[str]]:
        """Normalize optional typed envelopes without changing the LLM prompt contract."""
        items = ItemDefinitionNormalizer._list(value, "speed_envelopes")
        envelopes: list[SpeedEnvelope] = []
        warnings: list[str] = []
        for index, item in enumerate(items, start=1):
            if not isinstance(item, dict):
                warnings.append(f"speed_envelopes[{index}] is not an object and was ignored")
                continue
            try:
                provenance = FactProvenance(
                    item.get("provenance", FactProvenance.PROJECT_INPUT.value)
                )
                envelope_status = ReviewStatus(
                    item.get("status", ReviewStatus.PENDING.value)
                )
                sources = [SourceRef(**source) for source in item.get("sources", [])]
                if not sources and (item.get("source_location") or item.get("source_excerpt")):
                    sources = [SourceRef(
                        "item_definition",
                        source_id,
                        str(item.get("source_location", "")),
                        str(item.get("source_excerpt", "")),
                    )]
                envelopes.append(SpeedEnvelope(
                    operating_mode=str(item.get("operating_mode", item.get("mode", ""))),
                    speed_min_kph=(
                        None if item.get("speed_min_kph", item.get("min_kph")) is None
                        else float(item.get("speed_min_kph", item.get("min_kph")))
                    ),
                    speed_max_kph=(
                        None if item.get("speed_max_kph", item.get("max_kph")) is None
                        else float(item.get("speed_max_kph", item.get("max_kph")))
                    ),
                    condition=str(item.get("condition", "")),
                    unit=str(item.get("unit", "km/h")),
                    sources=sources,
                    provenance=provenance,
                    status=envelope_status,
                ))
            except (TypeError, ValueError) as error:
                warnings.append(f"speed_envelopes[{index}] invalid: {error}")
        return envelopes, warnings

    @staticmethod
    def _speed_range(value: Any) -> tuple[float | None, float | None, str]:
        if value is None or value == [None, None]:
            return None, None, ""
        candidate = value
        if isinstance(value, dict):
            candidate = [value.get("min"), value.get("max")]
        elif isinstance(value, str):
            match = re.search(
                r"^\s*[\[(]?\s*(-?\d+(?:\.\d+)?)\s*"
                r"(?:,|，|~|～|至|-)\s*(-?\d+(?:\.\d+)?)",
                value,
            )
            candidate = [match.group(1), match.group(2)] if match else None
        if not isinstance(candidate, list) or len(candidate) != 2:
            return None, None, "speed_range_kph格式无效，已置空并要求评审"
        if candidate[0] is None or candidate[1] is None:
            return None, None, "speed_range_kph不完整，已置空并要求评审"
        try:
            first, second = float(candidate[0]), float(candidate[1])
        except (TypeError, ValueError):
            return None, None, "speed_range_kph不是数值，已置空并要求评审"
        if not math.isfinite(first) or not math.isfinite(second) or first < 0 or second < 0:
            return None, None, "speed_range_kph包含负值或非有限数，已置空并要求评审"
        if first > second:
            return second, first, "speed_range_kph顺序颠倒，已按[min,max]规范化并要求评审"
        return first, second, ""

    @staticmethod
    def _list(value: Any, field: str) -> list[Any]:
        if value is None:
            return []
        if not isinstance(value, list):
            raise ValueError(f"{field}必须为array或null")
        return value
