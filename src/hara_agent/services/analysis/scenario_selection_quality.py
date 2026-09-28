"""Deterministic policy and ranking for governed Scenario selection.

The Provider never defines applicability, binding authority, coverage intent,
or Method membership.  This module derives those decisions from structured
project, malfunction, hazard, causal, and compiled MethodContract evidence.
Free-text parsing is deliberately limited to explicit engineering terms.
"""

from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
import re
from typing import Any, Iterable

from hara_agent.contracts import (
    SemanticCompatibility,
    ScenarioAtomCandidateSet, ScenarioBindingAuthority,
    ScenarioBindingDecision, ScenarioCoveragePlan,
    ScenarioDimensionApplicability,
    ScenarioDimensionApplicabilityDecision,
)
from hara_agent.models import ScenarioCandidate


_CATEGORY_MARKERS: dict[str, tuple[str, ...]] = {
    "ACTION_PARK": (
        "parking", "park", "slow driving", "parking maneuver", "parking-in",
        "parking-out", "park-in", "park-out", "泊车", "泊入", "泊出",
    ),
    "LOCATION_PARKING": ("parking", "car park", "garage", "parkhaus", "停车场", "车库"),
    "LOCATION_MOTORWAY": ("motorway", "autobahn", "高速公路"),
    "LOCATION_EXPRESSWAY": ("expressway", "快速路"),
    "LOCATION_CITY": ("city road", "city traffic", "innerstädt", "城市道路"),
    "LOCATION_RURAL": ("rural road", "country road", "landstraße", "乡村道路"),
    "ACTION_REVERSE": ("reverse", "reversing", "backing", "倒车", "后退"),
    "ACTION_HOLD": ("holding", "hold capability", "rollaway", "parking brake", "stands", "standing", "halten", "驻车", "溜车"),
    "ACTION_STOP": ("stopping", "stop", "braking", "deceleration", "制动", "停止", "停滞"),
    "ACTION_ACCELERATE": ("acceleration", "accelerate", "propulsion", "加速", "驱动"),
    "ACTION_TURN": ("steering", "turn", "lateral", "转向", "横向"),
    "ACTION_ABORT": (
        "abort", "cancel", "cancellation", "takeover", "take-over",
        "control handover", "handover", "hand-over",
        "取消", "接管", "交接", "退出",
    ),
    "OBJECT_PEDESTRIAN": ("pedestrian", "person", "vru", "行人"),
    "OBJECT_CYCLIST": ("cyclist", "bicycle", "two-wheel", "骑行", "自行车"),
    "OBJECT_VEHICLE": (
        "passenger_car", "rear vehicle", "front vehicle", "other vehicle",
        "another vehicle", "adjacent vehicle", "neighboring vehicle",
        "后方车辆", "前方车辆", "其他车辆", "相邻车辆", "对向车辆",
    ),
    "OBJECT_STATIC": ("static obstacle", "obstacle", "pillar", "cone", "静态障碍", "障碍物"),
    "OBJECT_OCCUPANT": (
        "occupant", "vehicle occupant", "car occupant", "passenger inside",
        "passengers in", "passenger of", "乘员",
    ),
    "TRAFFIC_FOLLOWING": (
        "following", "rear-end", "rear end", "rear vehicle", "追尾", "跟车",
        "后车追尾", "与后车", "后方来车",
    ),
    "TRAFFIC_ONCOMING": ("oncoming", "head-on", "opposing traffic", "对向", "迎面"),
    "TRAFFIC_CROSSING": ("crossing traffic", "cross traffic", "intersection traffic", "交叉交通", "横穿车辆"),
    "TRAFFIC_CUT_IN": ("cut-in", "cutting-in", "cut in", "切入", "加塞"),
    "TRAFFIC_REVERSE": ("reverse interaction", "reversing into", "backing into", "倒车碰", "倒车驶入"),
    "TRAFFIC_PARKING": ("parking traffic", "parking-lot traffic", "停车场交通"),
    "ROAD_SLOPE": ("slope", "gradient", "incline", "坡道", "坡度", "斜坡"),
    "ROAD_LOW_FRICTION": ("low friction", "reduced friction", "wet", "rain", "低附", "湿滑", "雨"),
}

_ACTION_CATEGORIES = frozenset(name for name in _CATEGORY_MARKERS if name.startswith("ACTION_"))
_OBJECT_CATEGORIES = frozenset(name for name in _CATEGORY_MARKERS if name.startswith("OBJECT_"))
_TRAFFIC_CATEGORIES = frozenset(name for name in _CATEGORY_MARKERS if name.startswith("TRAFFIC_"))
_ROAD_CATEGORIES = frozenset(name for name in _CATEGORY_MARKERS if name.startswith("ROAD_"))


def _json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def _tokens(value: str) -> tuple[str, ...]:
    return tuple(
        token for token in re.findall(r"[a-z0-9][a-z0-9_-]{1,}", value.casefold())
        if token not in {"the", "and", "with", "from", "into", "while", "that", "this"}
    )


def _explicit_categories(text: str) -> set[str]:
    folded = text.casefold()
    return {
        category for category, markers in _CATEGORY_MARKERS.items()
        if any(marker.casefold() in folded for marker in markers)
    }


def _operating_action_categories(text: str) -> set[str]:
    """Read an operation from scenario wording without treating a place as an action."""
    categories = _explicit_categories(text) & _ACTION_CATEGORIES
    folded = text.casefold()
    if "ACTION_PARK" in categories and not any(marker in folded for marker in (
        "parking maneuver", "parking-in", "parking-out", "park-in", "park-out",
        "泊车", "泊入", "泊出",
    )):
        categories.remove("ACTION_PARK")
    return categories


def normalize_object_category(value: Any) -> str:
    folded = str(value).strip().casefold()
    if any(token in folded for token in ("pedestrian", "person", "vru", "行人")):
        return "OBJECT_PEDESTRIAN"
    if any(token in folded for token in ("cycl", "bicycle", "自行车", "骑行")):
        return "OBJECT_CYCLIST"
    if any(token in folded for token in ("vehicle", "car", "truck", "车辆", "汽车")):
        return "OBJECT_VEHICLE"
    if any(token in folded for token in ("obstacle", "pillar", "cone", "障碍", "柱")):
        return "OBJECT_STATIC"
    if any(token in folded for token in ("occupant", "passenger", "乘员")):
        return "OBJECT_OCCUPANT"
    return ""


def _normalize_traffic_relation(value: Any) -> set[str]:
    folded = str(value or "").strip().casefold()
    if not folded:
        return set()
    direct = {
        "following": "TRAFFIC_FOLLOWING",
        "rear_following": "TRAFFIC_FOLLOWING",
        "oncoming": "TRAFFIC_ONCOMING",
        "opposing": "TRAFFIC_ONCOMING",
        "head_on": "TRAFFIC_ONCOMING",
        "crossing": "TRAFFIC_CROSSING",
        "cross_traffic": "TRAFFIC_CROSSING",
        "cut_in": "TRAFFIC_CUT_IN",
        "cut-in": "TRAFFIC_CUT_IN",
        "parking_traffic": "TRAFFIC_PARKING",
    }
    result = {direct[folded]} if folded in direct else set()
    result.update(_explicit_categories(folded) & _TRAFFIC_CATEGORIES)
    return result


class ScenarioSemanticQueryBuilder:
    """Build compact structured features without making engineering inferences."""

    _SPEED_RANGE = re.compile(
        r"(?:(?P<minimum>\d+(?:\.\d+)?)\s*(?:-|–|~|至|到)\s*)?"
        r"(?P<maximum>\d+(?:\.\d+)?)\s*(?:km/h|kph|kmh)",
        re.IGNORECASE,
    )

    @classmethod
    def _function_speed_constraints(
        cls, function: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Extract only explicit speed bounds from typed Function fields.

        The result carries source-field identity and numbers; it does not infer
        an operating phase from product wording.
        """
        records: list[dict[str, Any]] = []
        for field in ("odd_constraints", "preconditions"):
            values = function.get(field, [])
            if not isinstance(values, list):
                continue
            for index, value in enumerate(values):
                text = str(value)
                for match in cls._SPEED_RANGE.finditer(text):
                    minimum = match.group("minimum")
                    records.append({
                        "min_kph": float(minimum) if minimum is not None else 0.0,
                        "max_kph": float(match.group("maximum")),
                        "source_field": f"FUNCTION.{field}[{index}]",
                        "source_value": text,
                    })
        return records

    @staticmethod
    def _declared_operational_contexts(
        *, function: dict[str, Any], parent: ScenarioCandidate,
    ) -> list[dict[str, str]]:
        """Collect explicit context labels without interpreting free text."""
        facts = parent.facts if isinstance(parent.facts, dict) else {}
        instance = (
            parent.analysis_instance
            if isinstance(parent.analysis_instance, dict) else {}
        )
        sources = (
            ("FUNCTION.operational_context", function.get("operational_context")),
            ("FUNCTION.operating_mode", function.get("operating_mode")),
            ("PARENT.facts.operational_context", facts.get("operational_context")),
            ("PARENT.analysis_instance.operational_context", instance.get("operational_context")),
        )
        result: list[dict[str, str]] = []
        for source_field, raw in sources:
            values = raw if isinstance(raw, list) else [raw]
            for value in values:
                normalized = "_".join(str(value or "").strip().upper().split())
                if normalized:
                    result.append({
                        "context": normalized,
                        "source_field": source_field,
                    })
        return result

    @staticmethod
    def build(
        *, malfunction: dict[str, Any], parent: ScenarioCandidate,
        assessment: dict[str, Any], project_context: dict[str, Any],
        fm_template: dict[str, Any], function: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        function = function if isinstance(function, dict) else {}
        causal = assessment.get("causal_assessment", {})
        causal = causal if isinstance(causal, dict) else {}
        facts = parent.facts if isinstance(parent.facts, dict) else {}
        explicit_fields = {
            "MF.description": malfunction.get("description", ""),
            "MF.functional_effect": malfunction.get("functional_effect", ""),
            "MF.vehicle_level_hazard": malfunction.get("vehicle_level_hazard", ""),
            "HE.hazardous_event": assessment.get(
                "hazardous_event", causal.get("hazardous_event", "")
            ),
            "CAUSAL.summary": causal.get("causal_chain", []),
            "PARENT.scenario": {
                "operating_scenario": parent.operating_scenario,
                "operating_mode": parent.operating_mode,
                "vehicle_state": facts.get("vehicle_state", ""),
                "object_type": facts.get("object_type", ""),
                "road_user_type": facts.get("road_user_type", ""),
                "collision_type": facts.get("collision_type", ""),
                "object_position": facts.get("object_position", ""),
            },
        }
        text = " ".join(_json_text(value) for value in explicit_fields.values())
        categories: set[str] = set()
        structured_categories: set[str] = set()
        evidence: dict[str, list[str]] = {}
        traffic_evidence: list[dict[str, str]] = []
        unmapped_object_sources: list[str] = []

        def include(
            fields: dict[str, Any], allowed: frozenset[str], *,
            inference_rule: str = "EXPLICIT_ROLE_SCOPED_MARKER",
        ) -> None:
            for ref, value in fields.items():
                matched = _explicit_categories(_json_text(value)) & allowed
                categories.update(matched)
                for category in sorted(matched):
                    evidence.setdefault(category, []).append(ref)
                    if category in _TRAFFIC_CATEGORIES:
                        traffic_evidence.append({
                            "category": category,
                            "source_field": ref,
                            "source_value": str(value),
                            "inference_rule": inference_rule,
                        })

        action_fields = {
            key: explicit_fields[key] for key in (
                "MF.description", "MF.functional_effect", "MF.vehicle_level_hazard",
                "HE.hazardous_event", "CAUSAL.summary",
            )
        }
        action_fields["PARENT.facts.vehicle_state"] = facts.get("vehicle_state", "")
        include(action_fields, _ACTION_CATEGORIES)
        state_actions = _explicit_categories(_json_text({
            key: explicit_fields[key] for key in (
                "MF.description", "MF.functional_effect", "MF.vehicle_level_hazard",
                "HE.hazardous_event", "CAUSAL.summary",
            )
        })) & {"ACTION_ABORT"}
        operating_action_fields = {
            "PARENT.situational_description": parent.situational_description,
            "PARENT.operating_scenario": parent.operating_scenario,
            "PARENT.facts.operating_scenario": facts.get("operating_scenario", ""),
            "PARENT.facts.vehicle_state": facts.get("vehicle_state", ""),
        }
        operating_actions: set[str] = set()
        for ref, value in operating_action_fields.items():
            matched = _operating_action_categories(str(value))
            operating_actions.update(matched)
            for category in matched:
                evidence.setdefault(category, []).append(ref)
        consequence_actions = _explicit_categories(_json_text({
            key: explicit_fields[key] for key in (
                "MF.functional_effect", "MF.vehicle_level_hazard",
                "HE.hazardous_event", "CAUSAL.summary",
            )
        })) & (_ACTION_CATEGORIES - {"ACTION_ABORT", "ACTION_PARK"})
        if state_actions:
            # A transition/cancellation is the malfunction identity, not a
            # physical Method action. A supported operation or control effect
            # can supply the physical action while the state failure remains
            # visible for causal validation. Without one, retain the gap.
            physical_actions = operating_actions | consequence_actions
            categories.difference_update(_ACTION_CATEGORIES)
            categories.update(physical_actions or state_actions)
        else:
            categories.update(operating_actions)
        include({
            key: explicit_fields[key] for key in (
                "MF.vehicle_level_hazard", "HE.hazardous_event", "CAUSAL.summary",
            )
        }, _OBJECT_CATEGORIES)
        include({
            key: explicit_fields[key] for key in (
                "MF.vehicle_level_hazard", "HE.hazardous_event", "CAUSAL.summary",
            )
        }, _TRAFFIC_CATEGORIES)
        include({
            key: explicit_fields[key] for key in (
                "MF.description", "MF.functional_effect", "MF.vehicle_level_hazard",
                "HE.hazardous_event", "CAUSAL.summary",
            )
        }, _ROAD_CATEGORIES)

        for field in ("object_type", "road_user_type"):
            raw_object = facts.get(field, "")
            if normalized := normalize_object_category(raw_object):
                categories.add(normalized)
                structured_categories.add(normalized)
                evidence.setdefault(normalized, []).append(f"PARENT.facts.{field}")
            elif str(raw_object or "").strip():
                unmapped_object_sources.append(f"PARENT.facts.{field}")

        for field in ("traffic_relation", "interaction_relation", "relative_motion"):
            for relation in _normalize_traffic_relation(facts.get(field, "")):
                categories.add(relation)
                structured_categories.add(relation)
                ref = f"PARENT.facts.{field}"
                evidence.setdefault(relation, []).append(ref)
                traffic_evidence.append({
                    "category": relation,
                    "source_field": ref,
                    "source_value": str(facts.get(field, "")),
                    "inference_rule": "STRUCTURED_TRAFFIC_RELATION",
                })

        option = fm_template.get("active_option", {})
        if isinstance(option, dict):
            if normalized := normalize_object_category(option.get("obj_type", "")):
                categories.add(normalized)
                structured_categories.add(normalized)
                evidence.setdefault(normalized, []).append("FM_TEMPLATE.obj_type")
            elif str(option.get("obj_type", "")).strip():
                unmapped_object_sources.append("FM_TEMPLATE.obj_type")
            include(
                {"FM_TEMPLATE.label": option.get("label", "")},
                _ACTION_CATEGORIES,
            )
            collision = str(option.get("collision_type", "")).strip().casefold()
            if collision in {"head_on", "oncoming", "crossing", "cross_traffic", "cut_in"}:
                for relation in _normalize_traffic_relation(collision):
                    categories.add(relation)
                    structured_categories.add(relation)
                    evidence.setdefault(relation, []).append("FM_TEMPLATE.collision_type")
                    traffic_evidence.append({
                        "category": relation,
                        "source_field": "FM_TEMPLATE.collision_type",
                        "source_value": collision,
                        "inference_rule": "STRUCTURED_COLLISION_RELATION",
                    })

        collision = str(facts.get("collision_type", "")).strip().casefold()
        if collision in {"head_on", "oncoming", "crossing", "cross_traffic", "cut_in"}:
            for relation in _normalize_traffic_relation(collision):
                categories.add(relation)
                structured_categories.add(relation)
                evidence.setdefault(relation, []).append("PARENT.facts.collision_type")
                traffic_evidence.append({
                    "category": relation,
                    "source_field": "PARENT.facts.collision_type",
                    "source_value": collision,
                    "inference_rule": "STRUCTURED_COLLISION_RELATION",
                })

        project_location_text = _json_text({
            "odd_locations": project_context.get("odd_locations", []),
            "odd_road_types": project_context.get("odd_road_types", []),
        })
        parent_location_text = _json_text({
            "parent_operating_scenario": parent.operating_scenario,
            "parent_operating_scenario_fact": facts.get("operating_scenario", ""),
        })
        project_location_categories = {
            category for category in _explicit_categories(project_location_text)
            if category.startswith("LOCATION_")
        }
        parent_location_categories = {
            category for category in _explicit_categories(parent_location_text)
            if category.startswith("LOCATION_")
        }
        odd_road_text = _json_text({
            "odd_weather_conditions": project_context.get("odd_weather_conditions", []),
            "odd_road_surfaces": project_context.get("odd_road_surfaces", []),
        })
        return {
            "failure_type": str(malfunction.get("failure_type", "")),
            "guideword": str(malfunction.get("guideword", "")),
            "component_category": str(malfunction.get("component_category", "")),
            "operating_mode": parent.operating_mode,
            "action_categories": sorted(categories & _ACTION_CATEGORIES),
            "operating_action_categories": sorted(operating_actions),
            "control_consequence_action_categories": sorted(consequence_actions),
            "state_failure_action_categories": sorted(state_actions),
            "object_categories": sorted(categories & _OBJECT_CATEGORIES),
            "unmapped_object_sources": list(dict.fromkeys(unmapped_object_sources)),
            "traffic_relations": sorted(categories & _TRAFFIC_CATEGORIES),
            "road_relations": sorted(categories & _ROAD_CATEGORIES),
            # Project ODD is hard legality authority.  Parent location is kept
            # separately as ranking context and must never widen that authority.
            "project_location_categories": sorted(project_location_categories),
            "parent_location_categories": sorted(parent_location_categories),
            "location_categories": sorted(
                project_location_categories | parent_location_categories
            ),
            "odd_road_categories": sorted(
                category for category in _explicit_categories(odd_road_text)
                if category.startswith("ROAD_")
            ),
            "explicit_category_evidence": {
                key: list(dict.fromkeys(value)) for key, value in sorted(evidence.items())
            },
            "traffic_relation_evidence": traffic_evidence,
            "structured_source_categories": sorted(structured_categories),
            "query_tokens": sorted(set(_tokens(text))),
            "causal_tokens": sorted(set(_tokens(_json_text({
                "hazard": explicit_fields["HE.hazardous_event"],
                "causal": explicit_fields["CAUSAL.summary"],
            })))),
            "source_refs": list(explicit_fields),
            "declared_operational_contexts": (
                ScenarioSemanticQueryBuilder._declared_operational_contexts(
                    function=function, parent=parent,
                )
            ),
            "function_speed_constraints": (
                ScenarioSemanticQueryBuilder._function_speed_constraints(function)
            ),
            "project_odd_present": bool(project_context),
        }


class ScenarioBindingPolicy:
    """Separate binding resolution from exact, non-refinable authority."""

    def __init__(self, by_id: dict[str, dict[str, Any]]):
        self.by_id = by_id

    @staticmethod
    def _atom_id(binding: dict[str, Any]) -> str:
        return str(binding.get("atom_id") or binding.get("canonical_atom_id") or "").split("|", 1)[0].strip()

    @staticmethod
    def _speed_range(atom: dict[str, Any] | None) -> tuple[float | None, float | None] | None:
        raw = (atom or {}).get("speed_range_kph")
        if not isinstance(raw, (list, tuple)) or len(raw) != 2:
            return None
        return tuple(float(value) if value is not None else None for value in raw)  # type: ignore[return-value]

    def decide(
        self, parent: ScenarioCandidate, dimension: str,
        project_speed_envelope: tuple[float | None, float | None],
    ) -> ScenarioBindingDecision:
        bindings = parent.facts.get("method_scenario_dimensions", {})
        binding = bindings.get(dimension, {}) if isinstance(bindings, dict) else {}
        binding = binding if isinstance(binding, dict) else {}
        atom_id = self._atom_id(binding)
        atom = self.by_id.get(atom_id)
        provenance = binding.get("atom_provenance", {})
        provenance = provenance if isinstance(provenance, dict) else {}
        refs = tuple(dict.fromkeys(filter(None, (
            f"PARENT.{dimension}",
            ":".join(filter(None, (
                str(provenance.get("source_asset", "")),
                str(provenance.get("source_rule", "")),
            ))),
            str(binding.get("dimension_source", "")),
        ))))
        if not refs:
            refs = (f"PARENT.{dimension}",)
        resolved = str(binding.get("resolution_status", "")).upper() == "RESOLVED"
        explicit = str(binding.get("binding_authority", "")).upper()
        speed_constraint = binding.get("speed_constraint", {})
        speed_constraint = speed_constraint if isinstance(speed_constraint, dict) else {}
        resolved_by = str(speed_constraint.get("resolved_by", "")).upper()
        origin = str(binding.get("selection_origin", "")).upper()

        if explicit in ScenarioBindingAuthority._value2member_map_:
            authority = ScenarioBindingAuthority(explicit)
        elif resolved_by == ScenarioBindingAuthority.RANGE_CONTAINMENT.value:
            authority = ScenarioBindingAuthority.RANGE_CONTAINMENT
        elif origin == "APPROVED_ALIAS":
            authority = ScenarioBindingAuthority.APPROVED_ALIAS
        elif origin == "METHOD_TEMPLATE":
            authority = ScenarioBindingAuthority.METHOD_TEMPLATE_INFERENCE
        elif origin == "DETERMINISTIC_DERIVATION" or len((atom or {}).get("filled_dimensions", [])) > 1:
            authority = ScenarioBindingAuthority.DERIVED_COMPOUND
        elif origin in {"HAZARD_CAUSAL_SEMANTIC_CANDIDATE", "ODD_CONSTRAINED_CATALOG", "LLM_SELECTED_FROM_APPROVED_CANDIDATES"}:
            authority = ScenarioBindingAuthority.ANALYTICAL_SELECTION
        elif resolved and origin == "DIRECT_PROJECT_BINDING":
            authority = ScenarioBindingAuthority.EXACT_PROJECT_FACT
        elif resolved and atom_id:
            authority = ScenarioBindingAuthority.EXACT_METHOD_MAPPING
        else:
            authority = ScenarioBindingAuthority.ANALYTICAL_SELECTION

        refinable = authority in {
            ScenarioBindingAuthority.RANGE_CONTAINMENT,
            ScenarioBindingAuthority.METHOD_TEMPLATE_INFERENCE,
            ScenarioBindingAuthority.ANALYTICAL_SELECTION,
        }
        basis = {
            ScenarioBindingAuthority.RANGE_CONTAINMENT: "Broad Method range establishes compatibility, not an exact child value.",
            ScenarioBindingAuthority.METHOD_TEMPLATE_INFERENCE: "Template inference is source-governed but may be refined by compatible child semantics.",
            ScenarioBindingAuthority.ANALYTICAL_SELECTION: "Analytical parent selection is not immutable child authority.",
            ScenarioBindingAuthority.EXACT_PROJECT_FACT: "Exact project fact is authoritative and non-refinable.",
            ScenarioBindingAuthority.EXACT_METHOD_MAPPING: "Exact one-to-one Method mapping is authoritative and non-refinable.",
            ScenarioBindingAuthority.APPROVED_ALIAS: "Approved exact alias is authoritative and non-refinable.",
            ScenarioBindingAuthority.DERIVED_COMPOUND: "Source-defined compound membership must remain internally consistent.",
        }[authority]
        return ScenarioBindingDecision(
            dimension=dimension, authority=authority, refinable=refinable,
            parent_atom_id=atom_id if resolved and atom is not None else "",
            source_refs=refs, basis=basis,
            project_speed_envelope_kph=(
                project_speed_envelope if dimension == "EGO_DYNAMICS" else None
            ),
            parent_speed_range_kph=(
                self._speed_range(atom) if dimension == "EGO_DYNAMICS" else None
            ),
        )


class ScenarioDimensionApplicabilityService:
    """Determine per-group dimension applicability before Provider selection."""

    @staticmethod
    def assess(
        dimension: str, query: dict[str, Any], binding: ScenarioBindingDecision,
    ) -> ScenarioDimensionApplicabilityDecision:
        evidence_map = query.get("explicit_category_evidence", {})
        evidence_map = evidence_map if isinstance(evidence_map, dict) else {}
        traffic = tuple(str(item) for item in query.get("traffic_relations", []))
        road = tuple(str(item) for item in query.get("road_relations", []))
        actions = tuple(str(item) for item in query.get("action_categories", []))
        objects = tuple(str(item) for item in query.get("object_categories", []))
        refs: list[str] = []
        for category in (*traffic, *road, *actions, *objects):
            raw = evidence_map.get(category, [])
            if isinstance(raw, list):
                refs.extend(str(item) for item in raw)
        refs.extend(binding.source_refs)
        refs = list(dict.fromkeys(refs or ["PARENT.scenario"]))

        if binding.parent_atom_id and not binding.refinable:
            status = ScenarioDimensionApplicability.REQUIRED
            reason = "An exact authoritative parent binding must be preserved."
            trigger = (f"exact_parent_atom={binding.parent_atom_id}",)
        elif dimension == "TRAFFIC_PATTERN":
            represented = query.get("traffic_relations_represented_elsewhere", {})
            represented = represented if isinstance(represented, dict) else {}
            represented_relations = {
                relation for relation in traffic if represented.get(relation)
            }
            if traffic and represented_relations == set(traffic):
                status = ScenarioDimensionApplicability.NOT_APPLICABLE
                reason = (
                    "Every explicit interaction is already represented by source-defined "
                    "compound atoms in other Method dimensions."
                )
                trigger = tuple(
                    f"REPRESENTED_ELSEWHERE:{relation}"
                    for relation in sorted(represented_relations)
                )
            elif traffic:
                status = ScenarioDimensionApplicability.REQUIRED
                reason = "Explicit traffic-interaction semantics require a Method traffic relation."
                trigger = traffic
            elif "OBJECT_VEHICLE" in objects:
                status = ScenarioDimensionApplicability.OPTIONAL
                reason = "A vehicle object exists, but no explicit inter-vehicle traffic relation is proven."
                trigger = ("OBJECT_VEHICLE",)
            else:
                status = ScenarioDimensionApplicability.NOT_APPLICABLE
                reason = "No explicit traffic relation or interacting vehicle is present."
                trigger = tuple(objects) or ("NO_TRAFFIC_RELATION",)
        elif dimension == "EGO_X_ROAD":
            slope_driven = "ROAD_SLOPE" in road and bool({"ACTION_HOLD", "ACTION_REVERSE"} & set(actions))
            if slope_driven:
                status = ScenarioDimensionApplicability.REQUIRED
                reason = "The active holding/reverse mechanism is explicitly road-slope relative."
                trigger = ("ROAD_SLOPE", *sorted({"ACTION_HOLD", "ACTION_REVERSE"} & set(actions)))
            elif "ROAD_SLOPE" in road:
                status = ScenarioDimensionApplicability.OPTIONAL
                reason = "A slope is explicit, but child road-relative semantics are not required by the mechanism."
                trigger = ("ROAD_SLOPE",)
            else:
                status = ScenarioDimensionApplicability.NOT_APPLICABLE
                reason = "No explicit road-relative vehicle relation is present."
                trigger = ("NO_ROAD_RELATION",)
        elif dimension == "OBJECT":
            if (
                set(objects) == {"OBJECT_STATIC"} and not traffic
                and not query.get("unmapped_object_sources")
            ):
                status = ScenarioDimensionApplicability.NOT_APPLICABLE
                reason = (
                    "The source-grounded static physical object remains in scenario facts; "
                    "no independent Method Exposure object atom is required."
                )
                trigger = ("OBJECT_STATIC", "STATIC_PHYSICAL_FACT_PRESERVED")
            elif objects:
                status = ScenarioDimensionApplicability.REQUIRED
                reason = "Explicit structured or hazard-target evidence requires an object dimension."
                trigger = objects
            elif query.get("unmapped_object_sources"):
                status = ScenarioDimensionApplicability.REQUIRED
                reason = "An explicit object has no supported category; object applicability remains unresolved."
                trigger = tuple(map(str, query["unmapped_object_sources"]))
            else:
                status = ScenarioDimensionApplicability.NOT_APPLICABLE
                reason = "No non-ego object or road-user evidence is present."
                trigger = ("NO_NON_EGO_OBJECT",)
        else:
            status = ScenarioDimensionApplicability.REQUIRED
            reason = "This core Scenario dimension is required by the compiled synthesis contract."
            trigger = ("CORE_SCENARIO_DIMENSION",)
        return ScenarioDimensionApplicabilityDecision(
            dimension=dimension, status=status, reason=reason,
            trigger_evidence=tuple(trigger), source_refs=tuple(refs),
        )


class ScenarioSemanticCompatibilityClassifier:
    """Classify semantics without converting missing support into a rejection."""

    @staticmethod
    def _domain_categories(dimension: str, categories: set[str]) -> set[str]:
        if dimension == "OBJECT":
            return categories & _OBJECT_CATEGORIES
        if dimension == "TRAFFIC_PATTERN":
            return categories & _TRAFFIC_CATEGORIES
        if dimension == "EGO_ACTION":
            return categories & (_ACTION_CATEGORIES | _TRAFFIC_CATEGORIES)
        if dimension == "EGO_DYNAMICS":
            return categories & _ACTION_CATEGORIES
        if dimension == "WHERE":
            return {item for item in categories if item.startswith("LOCATION_")}
        if dimension in {"ROAD", "EGO_X_ROAD"}:
            return categories & _ROAD_CATEGORIES
        return set()

    @staticmethod
    def _query_categories(dimension: str, query: dict[str, Any]) -> set[str]:
        if dimension == "OBJECT":
            return set(map(str, query.get("object_categories", [])))
        if dimension == "TRAFFIC_PATTERN":
            return set(map(str, query.get("traffic_relations", [])))
        if dimension == "EGO_ACTION":
            return set(map(str, query.get("action_categories", []))) | set(
                map(str, query.get("traffic_relations", []))
            )
        if dimension == "EGO_DYNAMICS":
            return set(map(str, query.get("action_categories", [])))
        if dimension == "WHERE":
            return set(map(str, query.get("project_location_categories", [])))
        if dimension in {"ROAD", "EGO_X_ROAD"}:
            return set(map(str, query.get("road_relations", [])))
        return set()

    @classmethod
    def classify(
        cls, *, dimension: str, atom: dict[str, Any], query: dict[str, Any],
        fm_template: dict[str, Any], parent_atom_id: str = "",
    ) -> tuple[SemanticCompatibility, str]:
        atom_id = str(atom.get("atom_id", ""))
        if parent_atom_id and atom_id == parent_atom_id:
            return SemanticCompatibility.SUPPORTED, "EXACT_PARENT_BINDING"

        atom_categories = cls._domain_categories(
            dimension, ScenarioCandidateRanker.atom_categories(atom),
        )
        query_categories = cls._query_categories(dimension, query)

        if dimension == "OBJECT":
            active = fm_template.get("active_option", {})
            expected = normalize_object_category(
                active.get("obj_type", "") if isinstance(active, dict) else ""
            )
            if expected:
                if atom_categories and expected not in atom_categories:
                    return SemanticCompatibility.CONTRADICTED, "FM_TEMPLATE_OBJECT_CONTRADICTION"
                if expected in atom_categories:
                    return SemanticCompatibility.SUPPORTED, "FM_TEMPLATE_OBJECT_MATCH"

        if query_categories:
            if atom_categories & query_categories:
                return SemanticCompatibility.SUPPORTED, "EXPLICIT_FIELD_CATEGORY_MATCH"
            if atom_categories:
                return SemanticCompatibility.CONTRADICTED, "EXPLICIT_FIELD_CATEGORY_CONTRADICTION"
            return SemanticCompatibility.UNKNOWN, "ATOM_HAS_NO_EXPLICIT_DOMAIN_SEMANTICS"

        return SemanticCompatibility.UNKNOWN, "NO_EXPLICIT_QUERY_SEMANTIC"


class ScenarioRefinementEvidencePolicy:
    """One evidence rule shared by refinement generation and validation."""

    _STRUCTURED_SCORE_KEYS = (
        "template_score", "action_score", "object_score",
        "traffic_relation_score", "category_context_score",
        "structured_source_score", "physical_semantics_score",
    )

    @classmethod
    def classify(cls, candidate: Any, decision: Any) -> tuple[str, str]:
        if candidate.atom_id == decision.parent_atom_id:
            return "REFINEMENT_SUPPORTED", "EXACT_PARENT_BINDING"
        if (
            decision.authority is ScenarioBindingAuthority.RANGE_CONTAINMENT
            and candidate.speed_range_kph is not None
        ):
            return "REFINEMENT_SUPPORTED", "RANGE_CONTAINMENT"
        if candidate.template_relationship != "NONE":
            return "REFINEMENT_SUPPORTED", "METHOD_TEMPLATE_EVIDENCE"
        for key in cls._STRUCTURED_SCORE_KEYS:
            if float(candidate.ranking_scores.get(key, 0.0)) > 0:
                return "REFINEMENT_SUPPORTED", f"STRUCTURED_EVIDENCE:{key}"
        return (
            "REFINEMENT_UNSUPPORTED",
            "NO_EXACT_TEMPLATE_STRUCTURED_CATEGORY_OR_PHYSICAL_EVIDENCE",
        )


class ScenarioShortlistPolicy:
    """Bounded family-aware shortlist; evidence authority is never cut by K."""

    PRIMARY_BUDGET = 14
    SECONDARY_BUDGET = 6
    DEFAULT_BUDGET = 8
    MAX_ADAPTIVE_BUDGET = 18

    _INTENSITY_TERMS = frozenset({
        "normal", "medium", "strong", "emergency", "light", "heavy",
        "low", "high", "maximum", "minimum", "hard", "soft",
    })

    @staticmethod
    def _speed_regime(candidate: Any) -> str:
        speed = candidate.speed_range_kph
        if speed is None:
            return "NO_SPEED_RANGE"
        lower, upper = speed
        if upper == 0:
            return "STANDSTILL"
        if upper is not None and upper <= 15:
            return "LOW_SPEED"
        if upper is not None and upper <= 50:
            return "URBAN_SPEED"
        if lower is not None and lower >= 100:
            return "HIGH_SPEED"
        return "BROAD_SPEED"

    @classmethod
    def semantic_family(cls, dimension: str, candidate: Any) -> str:
        categories = ScenarioCandidateRanker.atom_categories({
            "label": candidate.label,
            "physical_semantics": candidate.method_semantics,
        })
        domain = ScenarioSemanticCompatibilityClassifier._domain_categories(
            dimension, categories,
        )
        semantics = candidate.method_semantics or {}

        def shape(value: Any) -> Any:
            if isinstance(value, dict):
                return {
                    key: shape(item) for key, item in sorted(value.items())
                    if not isinstance(item, (int, float))
                    and not any(term in key.casefold() for term in (
                        "accel", "decel", "intensity", "magnitude", "rate",
                    ))
                }
            if isinstance(value, (list, tuple)):
                return [shape(item) for item in value if not isinstance(item, (int, float))]
            if isinstance(value, str) and value.casefold() in cls._INTENSITY_TERMS:
                return "INTENSITY"
            return value

        label_tokens = [
            token for token in _tokens(candidate.label)
            if token not in cls._INTENSITY_TERMS and not token.isdigit()
        ]
        motion = "reverse" if "ACTION_REVERSE" in categories else (
            "lateral" if "ACTION_TURN" in categories else
            "longitudinal" if categories & {"ACTION_STOP", "ACTION_ACCELERATE"} else
            "unspecified"
        )
        payload = (
            dimension,
            tuple(sorted(domain)) or tuple(label_tokens[:4]),
            motion,
            _json_text(shape(semantics)),
            candidate.template_relationship,
            cls._speed_regime(candidate),
        )
        return hashlib.sha256(_json_text(payload).encode("utf-8")).hexdigest()[:16]

    @classmethod
    def _intensity_driven(cls, query: dict[str, Any]) -> bool:
        text = " ".join(map(str, (
            query.get("failure_type", ""), query.get("guideword", ""),
            query.get("semantic_text", ""),
        ))).casefold()
        return any(term in text for term in (
            "too much", "too little", "excess", "insufficient", "intensity",
            "magnitude", "stronger", "weaker", "more than", "less than",
        ))

    @classmethod
    def select(
        cls, *, dimension: str, ranked: Iterable[Any], applicability: Any,
        exact: bool, primary_dimensions: Iterable[str],
        secondary_dimensions: Iterable[str], query: dict[str, Any],
    ) -> tuple[tuple[Any, ...], int, str]:
        ordered = tuple(ranked)
        if applicability.status is ScenarioDimensionApplicability.NOT_APPLICABLE:
            return (), 0, "NOT_APPLICABLE"
        if exact:
            return ordered[:1], 1, "EXACT_FIXED"
        primary = dimension in set(primary_dimensions)
        secondary = dimension in set(secondary_dimensions)
        base = (
            cls.PRIMARY_BUDGET if primary else
            cls.SECONDARY_BUDGET if secondary else cls.DEFAULT_BUDGET
        )
        policy = "PRIMARY" if primary else "SECONDARY" if secondary else "DEFAULT"
        if not ordered:
            return (), base, policy

        families: dict[str, list[Any]] = {}
        for candidate in ordered:
            family = candidate.semantic_family or cls.semantic_family(dimension, candidate)
            families.setdefault(family, []).append(candidate)

        mandatory = [
            item for item in ordered
            if float(item.ranking_scores.get("source_evidence_tier", 0.0)) >= 3
            or item.template_relationship != "NONE"
            or item.binding_authority != ScenarioBindingAuthority.ANALYTICAL_SELECTION.value
        ]
        high_family_representatives = [
            values[0] for values in families.values()
            if float(values[0].ranking_scores.get("source_evidence_tier", 0.0)) >= 2
        ]
        required = []
        required_ids: set[str] = set()
        for item in (*mandatory, *high_family_representatives):
            if item.atom_id not in required_ids:
                required.append(item)
                required_ids.add(item.atom_id)
        budget = max(base, len(required))
        budget = min(cls.MAX_ADAPTIVE_BUDGET, budget)
        # Authority wins over the nominal maximum if the Method catalog itself
        # contains more authoritative candidates than the prompt budget.
        budget = max(budget, len(mandatory))

        selected = list(required)
        for values in families.values():
            if len(selected) >= budget:
                break
            if values[0] not in selected:
                selected.append(values[0])

        intensity_driven = cls._intensity_driven(query)
        family_counts = Counter(item.semantic_family for item in selected)
        for item in ordered:
            if len(selected) >= budget:
                break
            if item in selected:
                continue
            if not intensity_driven and family_counts[item.semantic_family] >= 2:
                continue
            selected.append(item)
            family_counts[item.semantic_family] += 1
        selected.sort(key=lambda item: ordered.index(item))
        return tuple(selected), budget, f"{policy}_FAMILY_AWARE"


class ScenarioCoveragePlanner:
    """Create evidence-driven sibling coverage intent for one semantic group."""

    @staticmethod
    def _meaningful_signature(candidate: Any) -> tuple[Any, ...] | None:
        """Collapse atoms that carry the same coverage evidence.

        Lexical/BM25 differences are deliberately excluded: a small text-score
        change does not by itself establish a distinct engineering scenario.
        """
        scores = candidate.ranking_scores
        structured = tuple(
            bool(float(scores.get(name, 0.0))) for name in (
                "template_score", "mechanism_score", "action_score",
                "object_score", "traffic_relation_score", "causal_score",
            )
        )
        physical = candidate.method_semantics or {}
        speed_range = candidate.speed_range_kph
        supported = any(structured) or bool(physical) or speed_range is not None
        if not supported:
            return None
        return (
            structured,
            candidate.template_relationship,
            _json_text(physical) if physical else "",
            tuple(speed_range) if speed_range is not None else (),
        )

    @classmethod
    def axis_priority(
        cls, *, query: dict[str, Any], dimensions: Iterable[str],
        fixed_dimensions: Iterable[str] = (),
    ) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
        applicable = set(dimensions) - set(fixed_dimensions)
        actions = set(map(str, query.get("action_categories", [])))
        traffic = set(map(str, query.get("traffic_relations", [])))
        road = set(map(str, query.get("road_relations", [])))
        objects = set(map(str, query.get("object_categories", [])))
        if "ROAD_SLOPE" in road and "ACTION_HOLD" in actions:
            ordered_primary = ("ROAD", "EGO_X_ROAD", "EGO_ACTION")
        elif "ACTION_REVERSE" in actions and "OBJECT_PEDESTRIAN" in objects:
            ordered_primary = ("EGO_ACTION", "OBJECT", "TRAFFIC_PATTERN")
        elif traffic:
            ordered_primary = ("TRAFFIC_PATTERN", "OBJECT", "EGO_ACTION")
        elif objects:
            ordered_primary = ("OBJECT", "EGO_ACTION", "EGO_DYNAMICS")
        else:
            ordered_primary = ("EGO_ACTION", "EGO_DYNAMICS", "OBJECT")
        primary = tuple(item for item in ordered_primary if item in applicable)
        if not primary:
            primary = tuple(item for item in dimensions if item in applicable)[:1]
        secondary = tuple(
            item for item in ("WHERE", "ROAD", "EGO_X_ROAD", "EGO_DYNAMICS", "OBJECT")
            if item in applicable and item not in primary
        )
        prohibited = tuple(item for item in ("WHERE", "ROAD") if item in secondary)
        return primary, secondary, prohibited

    @classmethod
    def plan(
        cls, *, query: dict[str, Any], candidate_sets: Iterable[ScenarioAtomCandidateSet],
    ) -> ScenarioCoveragePlan:
        sets = tuple(candidate_sets)
        fixed = tuple(
            item.dimension for item in sets
            if item.binding_decision.parent_atom_id and not item.binding_decision.refinable
        )
        applicable = {
            item.dimension for item in sets
            if item.applicability.status is not ScenarioDimensionApplicability.NOT_APPLICABLE
            and item.dimension not in fixed
        }
        primary, secondary, prohibited = cls.axis_priority(
            query=query,
            dimensions=tuple(item.dimension for item in sets if item.dimension in applicable),
        )
        supported_counts = {
            item.dimension: len({
                signature for candidate in item.candidates
                if (signature := cls._meaningful_signature(candidate)) is not None
            })
            for item in sets if item.dimension in primary
        }
        # Every additional sibling must be supported along a primary mechanism
        # axis. Using the strongest single axis prevents unrelated shortlist
        # multiplicity from manufacturing a third scenario.
        desired = max(1, min(3, max(supported_counts.values(), default=1)))
        labels = ("typical", "boundary", "extreme")[:desired]
        definitions = {
            "typical": "Most representative valid scenario for the active failure mechanism.",
            "boundary": "Project-valid scenario near a relevant Method, ODD, or interaction boundary.",
            "extreme": "More demanding but still project-valid scenario for the same mechanism.",
        }
        intents = []
        for index, label in enumerate(labels, start=1):
            required_axes = (
                () if label == "typical" else tuple(
                    dimension for dimension in primary
                    if supported_counts.get(dimension, 0) >= index
                )
            )
            intents.append({
                "coverage_label": label,
                "engineering_semantics": definitions[label],
                "required_variation_dimensions": list(required_axes),
                "supported_primary_signature_counts": supported_counts,
                "risk_classification_objective": False,
            })
        actions = set(map(str, query.get("action_categories", [])))
        traffic = set(map(str, query.get("traffic_relations", [])))
        road = set(map(str, query.get("road_relations", [])))
        objects = set(map(str, query.get("object_categories", [])))
        mechanism_parts = [
            *sorted(actions), *sorted(traffic), *sorted(road), *sorted(objects),
            str(query.get("failure_type", "")), str(query.get("guideword", "")),
        ]
        mechanism = " | ".join(item for item in mechanism_parts if item) or "SOURCE_BOUNDED_SCENARIO_MECHANISM"
        return ScenarioCoveragePlan(
            active_mechanism=mechanism, fixed_dimensions=fixed,
            primary_variation_dimensions=primary,
            secondary_variation_dimensions=secondary,
            prohibited_trivial_only_dimensions=prohibited,
            desired_variant_count=desired, variant_intents=tuple(intents),
            source_refs=tuple(dict.fromkeys(map(str, query.get("source_refs", [])))) or ("PARENT.scenario",),
        )


class ScenarioCandidateRanker:
    """Structured scoring plus a compact Okapi BM25 lexical component."""

    @staticmethod
    def _bm25(query_tokens: Iterable[str], document: str, corpus: Iterable[str]) -> float:
        query = set(query_tokens)
        docs = [list(_tokens(value)) for value in corpus]
        target = list(_tokens(document))
        if not query or not target or not docs:
            return 0.0
        average_length = sum(map(len, docs)) / len(docs) or 1.0
        counts = Counter(target)
        score = 0.0
        for token in query:
            frequency = counts[token]
            if not frequency:
                continue
            containing = sum(token in item for item in docs)
            inverse = math.log(1.0 + (len(docs) - containing + 0.5) / (containing + 0.5))
            score += inverse * (frequency * 2.2) / (
                frequency + 1.2 * (0.25 + 0.75 * len(target) / average_length)
            )
        return round(score, 6)

    @staticmethod
    def atom_category_sources(atom: dict[str, Any]) -> dict[str, str]:
        sources = {
            category: "label"
            for category in _explicit_categories(str(atom.get("label", "")))
        }
        semantics = atom.get("physical_semantics", {})
        semantics = semantics if isinstance(semantics, dict) else {}
        obj = semantics.get("object", {})
        obj = obj if isinstance(obj, dict) else {}
        if normalized := normalize_object_category(obj.get("type", "")):
            for category in _OBJECT_CATEGORIES:
                sources.pop(category, None)
            sources[normalized] = "physical_semantics.object.type"
        dynamics = semantics.get("ego_dynamics", {})
        dynamics = dynamics if isinstance(dynamics, dict) else {}
        if str(dynamics.get("direction", "")).strip().casefold() == "reverse":
            for category in _ACTION_CATEGORIES:
                sources.pop(category, None)
            sources["ACTION_REVERSE"] = "physical_semantics.ego_dynamics.direction"
        traffic_value = semantics.get("traffic_relation", "")
        if traffic_value:
            for category in _TRAFFIC_CATEGORIES:
                sources.pop(category, None)
            for category in _normalize_traffic_relation(traffic_value):
                sources[category] = "physical_semantics.traffic_relation"
        slope = semantics.get("slope", {})
        if isinstance(slope, dict) and slope:
            sources["ROAD_SLOPE"] = "physical_semantics.slope"
        return sources

    @classmethod
    def atom_categories(cls, atom: dict[str, Any]) -> set[str]:
        return set(cls.atom_category_sources(atom))

    @staticmethod
    def rank_key(atom_id: str, scores: dict[str, float]) -> tuple[float, float, str]:
        """Rank authoritative evidence tiers before scalar/BM25 tie-breaking."""
        return (
            -float(scores.get("source_evidence_tier", 0.0)),
            -float(scores.get("final_rank_score", 0.0)),
            atom_id,
        )

    def score(
        self, *, dimension: str, atom: dict[str, Any], query: dict[str, Any],
        fm_template: dict[str, Any], corpus_labels: Iterable[str], odd_passed: bool,
    ) -> tuple[dict[str, float], str]:
        category_sources = self.atom_category_sources(atom)
        categories = set(category_sources)
        actions = set(map(str, query.get("action_categories", [])))
        objects = set(map(str, query.get("object_categories", [])))
        traffic = set(map(str, query.get("traffic_relations", [])))
        road = set(map(str, query.get("road_relations", [])))
        road.update(map(str, query.get("odd_road_categories", [])))
        locations = set(map(str, query.get("location_categories", [])))
        filled_dimensions = set(map(str, atom.get("filled_dimensions", [])))
        action_score = float(bool(
            categories & actions
            and ("EGO_ACTION" in filled_dimensions or "EGO_DYNAMICS" in filled_dimensions)
        ))
        object_score = float(bool(
            categories & objects and "OBJECT" in filled_dimensions
        ))
        traffic_score = float(bool(categories & traffic))
        category_context_score = float(
            bool(categories & locations) if dimension == "WHERE" else
            bool(categories & road) if dimension in {"ROAD", "EGO_X_ROAD"} else
            False
        )
        label = str(atom.get("label", ""))
        failure_tokens = set(_tokens(str(query.get("failure_type", ""))))
        guideword_tokens = set(_tokens(str(query.get("guideword", ""))))
        mechanism_score = float(bool((failure_tokens | guideword_tokens) & set(_tokens(label))))
        lexical_score = self._bm25(query.get("query_tokens", []), label, corpus_labels)
        causal_score = self._bm25(query.get("causal_tokens", []), label, corpus_labels)

        template_score = 0.0
        active_option = fm_template.get("active_option", {})
        options = [active_option] if isinstance(active_option, dict) and active_option else fm_template.get("source_governed_constraints", [])
        for option in options if isinstance(options, list) else []:
            if not isinstance(option, dict):
                continue
            expected = normalize_object_category(option.get("obj_type", ""))
            if (
                expected and expected in categories
                and "OBJECT" in filled_dimensions
                and dimension in {"OBJECT", "EGO_ACTION"}
            ):
                template_score = 1.0
                break
        odd_score = 1.0 if odd_passed and dimension in {"WHERE", "ROAD", "EGO_X_ROAD", "EGO_DYNAMICS"} else 0.0
        relevant_categories = (
            locations if dimension == "WHERE" else
            road if dimension in {"ROAD", "EGO_X_ROAD"} else
            objects if dimension == "OBJECT" else
            traffic if dimension == "TRAFFIC_PATTERN" else
            actions | traffic if dimension == "EGO_ACTION" else
            actions if dimension == "EGO_DYNAMICS" else
            set()
        )
        structured_source_score = float(bool(
            categories & relevant_categories
            & set(map(str, query.get("structured_source_categories", [])))
        ))
        physical_semantics_score = float(any(
            category in relevant_categories and source.startswith("physical_semantics.")
            for category, source in category_sources.items()
        ))
        source_evidence_tier = float(
            4 if template_score else
            3 if structured_source_score or physical_semantics_score else
            2 if any((action_score, object_score, traffic_score, category_context_score)) else
            1 if odd_score else
            0
        )
        scores = {
            "template_score": template_score,
            "mechanism_score": mechanism_score,
            "action_score": action_score,
            "object_score": object_score,
            "traffic_relation_score": traffic_score,
            "odd_score": odd_score,
            "causal_score": causal_score,
            "lexical_score": lexical_score,
            "category_context_score": category_context_score,
            "structured_source_score": structured_source_score,
            "physical_semantics_score": physical_semantics_score,
            "source_evidence_tier": source_evidence_tier,
        }
        scores["final_rank_score"] = round(
            5.0 * template_score + 4.0 * traffic_score + 3.0 * action_score
            + 3.0 * object_score + 3.0 * category_context_score
            + 3.0 * physical_semantics_score
            + 2.0 * mechanism_score
            + odd_score + causal_score + lexical_score,
            6,
        )
        evidence = ",".join(
            key for key, value in scores.items()
            if key not in {"final_rank_score", "source_evidence_tier"} and value
        )
        return scores, (
            f"source tier={int(source_evidence_tier)}; structured/BM25 rank "
            f"evidence={evidence or 'hard_filter_only'}"
        )


class ScenarioVariantDiversityValidator:
    """Reject sibling sets whose differences do not satisfy their coverage plan."""

    @staticmethod
    def reasons(
        plan: ScenarioCoveragePlan, selections: Iterable[dict[str, tuple[str, ...]]],
    ) -> list[str]:
        variants = tuple(selections)
        if len(variants) <= 1:
            return []
        dimensions = sorted({key for item in variants for key in item})
        primary = set(plan.primary_variation_dimensions)
        prohibited = set(plan.prohibited_trivial_only_dimensions)
        reasons = []
        for left_index, left in enumerate(variants):
            for right in variants[left_index + 1:]:
                pair_varying = {
                    dimension for dimension in dimensions
                    if tuple(left.get(dimension, ())) != tuple(right.get(dimension, ()))
                }
                if not pair_varying & primary or (
                    pair_varying and pair_varying <= prohibited
                ):
                    reasons.append("TRIVIAL_VARIANT_DIVERSITY")
        baseline = variants[0]
        for variant, intent in zip(variants[1:], plan.variant_intents[1:]):
            required = set(map(str, intent.get("required_variation_dimensions", [])))
            if required and not any(
                tuple(baseline.get(dimension, ()))
                != tuple(variant.get(dimension, ()))
                for dimension in required
            ):
                reasons.append("TRIVIAL_VARIANT_DIVERSITY")
        return sorted(set(reasons))


__all__ = [
    "normalize_object_category",
    "ScenarioBindingPolicy", "ScenarioCandidateRanker", "ScenarioCoveragePlanner",
    "ScenarioDimensionApplicabilityService", "ScenarioSemanticQueryBuilder",
    "ScenarioRefinementEvidencePolicy", "ScenarioSemanticCompatibilityClassifier",
    "ScenarioShortlistPolicy",
    "ScenarioVariantDiversityValidator",
]
