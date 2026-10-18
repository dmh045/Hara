from __future__ import annotations

import hashlib
from typing import Any

from hara_agent.contracts import MethodContract


class MethodSafetyGoalService:
    """Create deterministic, review-gated SG/Safe-State proposals."""

    ASIL_RANK = {"QM": 0, "A": 1, "B": 2, "C": 3, "D": 4}

    def __init__(self, method: MethodContract):
        self.method = method
        self.catalog: dict[str, dict[str, Any]] = {}

    @property
    def is_approved(self) -> bool:
        return not (
            self.method.safety_goal_method.semantic_derivation_required
            or self.method.safe_state_method.semantic_derivation_required
        )

    @staticmethod
    def _intent_id(function_name: str, guideword: str, malfunction: str) -> str:
        material = "\n".join(
            item.strip().casefold() for item in (function_name, guideword, malfunction)
        )
        digest = hashlib.sha256(material.encode("utf-8")).hexdigest()[:10].upper()
        return f"SG-METHOD-{digest}"

    def register_intent(
        self,
        *,
        function_name: str,
        malfunction: str,
        guideword: str,
        scenario_id: str,
        scenario_description: str,
        hazard_event: str,
        asil: str,
    ) -> dict[str, str]:
        sg_id = self._intent_id(function_name, guideword, malfunction)
        original_safety_goal = (
            f"避免发生因{guideword}{function_name}而导致"
            f"{scenario_description}发生{hazard_event}。"
        )
        # Scope/parameters belong to associations, not the shared intent sentence.
        safety_goal = "防止" + malfunction.strip().rstrip("。；") + "。"
        safe_state = "待确定"
        entry = self.catalog.setdefault(sg_id, {
            "sg_id": sg_id,
            "safety_goal": safety_goal,
            "original_safety_goal": original_safety_goal,
            "display_basis": "existing malfunction intent; complete member scope in associations",
            "safety_state": safe_state,
            "max_asil": asil,
            "functions": [],
            "associations": [],
            "method_contract_hash": self.method.metadata["template_hash"],
            "safety_goal_method_source": self.method.safety_goal_method.derivation_pattern,
            "safe_state_method_source": self.method.safe_state_method.derivation_pattern,
            "derivation_status": (
                "FINALIZED" if self.is_approved else "NEEDS_REVIEW"
            ),
        })
        if function_name not in entry["functions"]:
            entry["functions"].append(function_name)
        if self.ASIL_RANK.get(asil, -1) > self.ASIL_RANK.get(entry["max_asil"], -1):
            entry["max_asil"] = asil
        association = {
            "function": function_name,
            "malfunction": malfunction,
            "guideword": guideword,
            "scenario_id": scenario_id,
            "scenario_description": scenario_description,
            "hazard_event": hazard_event,
            "asil": asil,
        }
        entry["associations"].append(association)
        return {
            "sg_id": sg_id,
            "safety_goal": str(entry["safety_goal"]),
            "safety_state": str(entry["safety_state"]),
        }

    def to_dict(self) -> dict[str, dict[str, Any]]:
        result: dict[str, dict[str, Any]] = {}
        for sg_id in sorted(self.catalog):
            entry = self.catalog[sg_id]
            result[sg_id] = entry
        return result
