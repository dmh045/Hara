from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .document_reader import DocumentBlock


SOURCE_SELECTION_SCHEMA_VERSION = "docx-ordered-section-speed-selection-v1"
_SPEED_UNIT = re.compile(r"(?:km\s*/\s*h|kph|kmh)\b", re.IGNORECASE)


@dataclass(frozen=True)
class ExcludedSourceBlock:
    block_id: str
    location: str
    section_path: tuple[str, ...]
    reason: str


@dataclass(frozen=True)
class SourceSelection:
    selected_blocks: tuple[DocumentBlock, ...]
    excluded_blocks: tuple[ExcludedSourceBlock, ...]
    policy_id: str
    policy_version: str
    project_policy_id: str
    project_policy_version: str
    fingerprint: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SOURCE_SELECTION_SCHEMA_VERSION,
            "policy_id": self.policy_id,
            "policy_version": self.policy_version,
            "project_policy_id": self.project_policy_id,
            "project_policy_version": self.project_policy_version,
            "fingerprint": self.fingerprint,
            "selected_block_ids": [block.block_id for block in self.selected_blocks],
            "excluded_blocks": [
                {
                    "block_id": block.block_id,
                    "location": block.location,
                    "section_path": list(block.section_path),
                    "reason": block.reason,
                }
                for block in self.excluded_blocks
            ],
        }


def _section_names(value: Any) -> frozenset[str]:
    if value is None:
        return frozenset()
    if not isinstance(value, (list, tuple)) or any(
        not isinstance(name, str) or not name.strip() for name in value
    ):
        raise ValueError("input selection section names must be nonempty strings")
    return frozenset(" ".join(name.split()).casefold() for name in value)


def select_source_blocks(
    blocks: Sequence[DocumentBlock],
    project_analysis_policy: Mapping[str, Any] | None = None,
) -> SourceSelection:
    """Select scoring evidence by governed section role, retaining raw locators.

    The source document is never edited. Only speed-bearing blocks under the
    configured performance or actuator-capability sections leave the extraction
    view; other environmental and functional content remains available.
    """

    project_policy = dict(project_analysis_policy or {})
    input_policy = project_policy.get("input_selection", {})
    if not isinstance(input_policy, Mapping):
        raise ValueError("project input_selection policy must be an object")
    excluded_sections = _section_names(
        input_policy.get("excluded_sections_for_scoring")
    )
    capability_sections = _section_names(
        input_policy.get("non_operational_speed_sections")
    )
    selected: list[DocumentBlock] = []
    excluded: list[ExcludedSourceBlock] = []
    fingerprint_blocks = []
    for block in blocks:
        names = {" ".join(name.split()).casefold() for name in block.section_path}
        reason = ""
        if re.search(r"\d", block.text) and _SPEED_UNIT.search(block.text):
            if names & excluded_sections:
                reason = "PERFORMANCE_SPEED_EXPECTATION"
            elif names & capability_sections:
                reason = "ACTUATOR_SPEED_CAPABILITY"
        if reason:
            excluded.append(ExcludedSourceBlock(
                block.block_id, block.location, block.section_path, reason,
            ))
        else:
            selected.append(block)
        fingerprint_blocks.append({
            "block_id": block.block_id,
            "kind": block.kind,
            "location": block.location,
            "text": block.text,
            "section_path": block.section_path,
            "selection_reason": reason,
        })
    material = {
        "schema_version": SOURCE_SELECTION_SCHEMA_VERSION,
        "project_policy_id": str(project_policy.get("policy_id", "")),
        "project_policy_version": str(project_policy.get("version", "")),
        "input_selection": dict(input_policy),
        "blocks": fingerprint_blocks,
    }
    fingerprint = hashlib.sha256(json.dumps(
        material, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")).hexdigest()
    return SourceSelection(
        tuple(selected), tuple(excluded),
        str(input_policy.get("policy_id", "")),
        str(input_policy.get("version", "")),
        str(project_policy.get("policy_id", "")),
        str(project_policy.get("version", "")),
        fingerprint,
    )
