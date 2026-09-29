"""Source-linked analytical branches for an allowed driver-seat configuration."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from typing import Any, Iterable

from hara_agent.models import ReviewStatus, ScenarioCandidate, SourceRef


_POSITIONS = ("in_driver_seat", "outside_driver_seat")
_SOURCE_SEAT_TERMS = {
    "在驾驶位": "in_driver_seat",
    "不在驾驶位": "outside_driver_seat",
}


class DriverConfigurationBrancher:
    """Expand only the driver positions allowed by applicable project facts."""

    def __init__(self, *, policy_id: str = ""):
        self.policy_id = policy_id

    @staticmethod
    def _source_refs(fact: dict[str, Any]) -> tuple[SourceRef, ...]:
        result = []
        for raw in fact.get("source_refs", []):
            if not isinstance(raw, dict):
                continue
            try:
                result.append(SourceRef(**raw))
            except TypeError:
                continue
        return tuple(dict.fromkeys(result))

    @classmethod
    def allowed_position(cls, fact: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        """Resolve an exact source seat term without inferring vehicle location."""
        context = fact.get("context", {})
        if not isinstance(context, dict) or not cls._source_refs(fact):
            return "", {}
        explicit = str(context.get("allowed_driver_position", "")).strip().casefold()
        if explicit in _POSITIONS:
            return explicit, context
        value = str(fact.get("value", "")).strip()
        position = _SOURCE_SEAT_TERMS.get(value, "")
        if (
            position
            and context.get("位姿状态") == value
            and all(value in source.excerpt for source in cls._source_refs(fact))
        ):
            return position, {
                **{key: item for key, item in context.items() if key != "位姿状态"},
                "allowed_driver_position": position,
            }
        return "", {}

    @staticmethod
    def _scope_matches(context: dict[str, Any], scenario: ScenarioCandidate) -> bool:
        candidate = {
            **scenario.facts,
            "operating_mode": scenario.operating_mode,
            "scenario_id": scenario.scenario_id,
            "parent_scenario_id": scenario.source_scenario_id,
            "malfunction_id": scenario.analysis_instance.get("malfunction_id", ""),
        }
        return all(
            str(candidate.get(key, "")).strip().casefold()
            == str(value).strip().casefold()
            for key, value in context.items() if key != "allowed_driver_position"
        )

    @staticmethod
    def _parent_position(scenario: ScenarioCandidate) -> str:
        for value in (
            scenario.facts.get("driver_position"),
            scenario.facts.get("allowed_driver_position"),
        ):
            normalized = str(value or "").strip().casefold()
            if normalized in _POSITIONS:
                return normalized
        return ""

    @staticmethod
    def _independent_vehicle_location(metadata: Any) -> bool:
        if not isinstance(metadata, dict):
            return False
        refs = metadata.get("source_refs", [])
        if not isinstance(refs, list):
            return False
        text = " ".join(
            str(item.get("excerpt", "")) for item in refs if isinstance(item, dict)
        ).casefold()
        return any(term in text for term in (
            "车外", "车内", "outside the vehicle", "outside vehicle",
            "inside the vehicle", "inside vehicle", "not in the vehicle",
        ))

    @classmethod
    def _rescope_inherited_metadata(
        cls, value: Any, *, old_scenario_id: str, new_scenario_id: str,
    ) -> Any:
        """Keep validated scenario assumptions valid for one narrower child."""
        if isinstance(value, list):
            return [
                cls._rescope_inherited_metadata(
                    item, old_scenario_id=old_scenario_id,
                    new_scenario_id=new_scenario_id,
                ) for item in value
            ]
        if not isinstance(value, dict):
            return value
        result = {
            key: cls._rescope_inherited_metadata(
                item, old_scenario_id=old_scenario_id,
                new_scenario_id=new_scenario_id,
            ) for key, item in value.items()
        }
        for key in ("applicable_scope", "analysis_assumption_scope"):
            scope = result.get(key)
            if (
                isinstance(scope, dict)
                and str(scope.get("scenario_id", "")) == old_scenario_id
            ):
                scope["scenario_id"] = new_scenario_id
                scope["parent_scenario_id"] = old_scenario_id
                result["inherited_as_driver_branch_subset"] = True
        return result

    def expand(
        self, scenario: ScenarioCandidate, risk_facts: Iterable[dict[str, Any]],
    ) -> tuple[ScenarioCandidate, ...]:
        if not self.policy_id:
            return (scenario,)
        if scenario.analysis_instance.get("driver_configuration_branch"):
            return (scenario,)
        by_position: dict[str, list[tuple[str, SourceRef]]] = {}
        for fact in risk_facts:
            if not isinstance(fact, dict):
                continue
            if str(fact.get("parameter", "")).upper() != "DRIVER_IN_VEHICLE":
                continue
            if str(fact.get("approval", "")).upper() != "FINALIZED":
                continue
            position, context = self.allowed_position(fact)
            if not position or not self._scope_matches(context, scenario):
                continue
            for source in self._source_refs(fact):
                by_position.setdefault(position, []).append((
                    str(fact.get("fact_id", "")), source,
                ))
        if not by_position:
            return (scenario,)

        fixed = self._parent_position(scenario)
        if fixed:
            by_position = {key: value for key, value in by_position.items() if key == fixed}
            if not by_position:
                context = deepcopy(scenario.context_resolution)
                context["driver_configuration"] = {
                    "status": "SOURCE_CONFLICT",
                    "reason": "Scenario driver position conflicts with the project allowed set.",
                }
                facts = deepcopy(scenario.facts)
                provenance = deepcopy(scenario.fact_provenance)
                facts.pop("driver_in_vehicle", None)
                provenance.pop("driver_in_vehicle", None)
                facts["driver_configuration_source_conflict"] = True
                return (replace(
                    scenario, facts=facts, fact_provenance=provenance,
                    context_resolution=context,
                ),)
            # The existing scenario is already a single position. Reuse its
            # identity, while keeping the distinct vehicle-location predicate
            # grounded in what the source actually proves.
            entries = by_position[fixed]
            source_payload = [
                source.__dict__.copy() for source in dict.fromkeys(
                    source for _, source in entries
                )
            ]
            facts = deepcopy(scenario.facts)
            provenance = deepcopy(scenario.fact_provenance)
            context = deepcopy(scenario.context_resolution)
            context["driver_configuration"] = {
                "status": "SINGLE_POSITION_APPLIED",
                "driver_position": fixed,
                "source_fact_ids": sorted({fact_id for fact_id, _ in entries if fact_id}),
                "source_refs": source_payload,
                "policy_id": self.policy_id,
            }
            existing_vehicle_location = facts.get("driver_in_vehicle")
            independent = self._independent_vehicle_location(
                provenance.get("driver_in_vehicle")
            )
            if fixed == "in_driver_seat":
                if existing_vehicle_location is False and independent:
                    facts.pop("driver_in_vehicle", None)
                    provenance.pop("driver_in_vehicle", None)
                    facts["driver_configuration_source_conflict"] = True
                    context["driver_configuration"]["status"] = "SOURCE_CONFLICT"
                else:
                    facts["driver_in_vehicle"] = True
                    provenance["driver_in_vehicle"] = {
                        "provenance": "DERIVED", "origin": "SCENARIO_DEFINED",
                        "approval": "FINALIZED", "validation_status": "VALIDATED",
                        "source_refs": source_payload,
                        "source_fact_ids": sorted({fact_id for fact_id, _ in entries if fact_id}),
                        "applicable_scope": {
                            "malfunction_id": str(scenario.analysis_instance.get("malfunction_id", "")),
                            "scenario_id": scenario.scenario_id,
                            "parent_scenario_id": scenario.source_scenario_id,
                        },
                        "selection_basis": "IN_DRIVER_SEAT_ENTAILS_IN_VEHICLE",
                        "policy_id": self.policy_id,
                    }
            elif not independent:
                facts.pop("driver_in_vehicle", None)
                provenance.pop("driver_in_vehicle", None)
            return (replace(
                scenario, facts=facts, fact_provenance=provenance,
                context_resolution=context,
            ),)

        branches = []
        for position in _POSITIONS:
            entries = by_position.get(position, [])
            if not entries:
                continue
            if (
                position == "in_driver_seat"
                and scenario.facts.get("driver_in_vehicle") is False
                and self._independent_vehicle_location(
                    scenario.fact_provenance.get("driver_in_vehicle")
                )
            ):
                # A separately sourced "outside vehicle" condition narrows
                # this scenario. It cannot be combined with an in-seat branch.
                continue
            fact_ids = sorted({fact_id for fact_id, _ in entries if fact_id})
            sources = list(dict.fromkeys(source for _, source in entries))
            material = {
                "parent_scenario_id": scenario.scenario_id,
                "malfunction_id": str(scenario.analysis_instance.get("malfunction_id", "")),
                "driver_position": position,
                "source_fact_ids": fact_ids,
                "policy_id": self.policy_id,
            }
            digest = hashlib.sha256(json.dumps(
                material, ensure_ascii=False, sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")).hexdigest()
            child_id = f"SCN-ANALYTICAL-{digest[:16].upper()}"
            branch_id = f"DRIVER-{digest[:12].upper()}"
            scope = {
                "malfunction_id": str(scenario.analysis_instance.get("malfunction_id", "")),
                "scenario_id": child_id,
                "parent_scenario_id": scenario.scenario_id,
            }
            source_payload = [source.__dict__.copy() for source in sources]
            facts = deepcopy(scenario.facts)
            provenance = {
                field: self._rescope_inherited_metadata(
                    metadata, old_scenario_id=scenario.scenario_id,
                    new_scenario_id=child_id,
                )
                for field, metadata in scenario.fact_provenance.items()
            }
            metadata = {
                "provenance": "DERIVED",
                "origin": "SCENARIO_DEFINED",
                "approval": "FINALIZED",
                "validation_status": "VALIDATED",
                "source_refs": source_payload,
                "source_fact_ids": fact_ids,
                "applicable_scope": scope,
                "selection_basis": "PROJECT_ALLOWED_DRIVER_SEAT_CONFIGURATION",
                "policy_id": self.policy_id,
            }
            facts["driver_position"] = position
            facts["allowed_driver_position"] = position
            provenance["driver_position"] = deepcopy(metadata)
            provenance["allowed_driver_position"] = deepcopy(metadata)
            independent_location = self._independent_vehicle_location(
                provenance.get("driver_in_vehicle")
            )
            location_conflict = False
            if position == "in_driver_seat":
                # Occupying the driver seat entails being in the vehicle.
                # The converse is not true for outside_driver_seat.
                if facts.get("driver_in_vehicle") is False and independent_location:
                    facts.pop("driver_in_vehicle", None)
                    provenance.pop("driver_in_vehicle", None)
                    facts["driver_configuration_source_conflict"] = True
                    location_conflict = True
                else:
                    facts["driver_in_vehicle"] = True
                    provenance["driver_in_vehicle"] = {
                        **deepcopy(metadata),
                        "selection_basis": "IN_DRIVER_SEAT_ENTAILS_IN_VEHICLE",
                    }
            else:
                if not independent_location:
                    facts.pop("driver_in_vehicle", None)
                    provenance.pop("driver_in_vehicle", None)
            vehicle_location_resolution = (
                "SOURCE_CONFLICT" if location_conflict else
                "TRUE_ENTAILED" if position == "in_driver_seat" else
                "EXPLICIT_VEHICLE_LOCATION" if independent_location
                else "UNKNOWN_OUTSIDE_SEAT_IS_NOT_OUTSIDE_VEHICLE"
            )
            branch = {
                "branch_id": branch_id,
                "parent_scenario_id": scenario.scenario_id,
                "driver_position": position,
                "source_fact_ids": fact_ids,
                "source_refs": source_payload,
                "policy_id": self.policy_id,
                "driver_in_vehicle_resolution": vehicle_location_resolution,
            }
            instance = self._rescope_inherited_metadata(
                scenario.analysis_instance,
                old_scenario_id=scenario.scenario_id,
                new_scenario_id=child_id,
            )
            instance["instance_id"] = child_id
            instance["driver_configuration_branch"] = branch
            instance["applicable_scope"] = scope
            context = deepcopy(scenario.context_resolution)
            context["driver_configuration"] = {
                "status": "SOURCE_CONFLICT" if location_conflict else "BRANCHED",
                **branch,
            }
            label = "在驾驶位" if position == "in_driver_seat" else "不在驾驶位"
            branches.append(replace(
                scenario, scenario_id=child_id, facts=facts,
                fact_provenance=provenance, analysis_instance=instance,
                context_resolution=context,
                status=ReviewStatus.PENDING,
                situational_description=f"{scenario.situational_description} | 驾驶员配置：{label}",
                situational_detailing=f"{scenario.situational_detailing} | 驾驶员配置：{label}",
                sources=list(dict.fromkeys([*scenario.sources, *sources])),
                source_scenario_id=scenario.source_scenario_id or scenario.scenario_id,
                atomic_variant=f"{scenario.atomic_variant}:driver:{position}",
                semantic_fingerprint=digest,
            ))
        if not branches:
            context = deepcopy(scenario.context_resolution)
            context["driver_configuration"] = {
                "status": "SOURCE_CONFLICT",
                "reason": "No allowed driver-seat position is compatible with the scenario vehicle-location fact.",
            }
            facts = deepcopy(scenario.facts)
            provenance = deepcopy(scenario.fact_provenance)
            facts.pop("driver_in_vehicle", None)
            provenance.pop("driver_in_vehicle", None)
            facts["driver_configuration_source_conflict"] = True
            return (replace(
                scenario, facts=facts, fact_provenance=provenance,
                context_resolution=context,
            ),)
        return tuple(branches)


__all__ = ["DriverConfigurationBrancher"]
