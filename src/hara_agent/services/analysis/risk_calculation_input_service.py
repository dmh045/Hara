from __future__ import annotations

from typing import Any
import math

from hara_agent.contracts import (
    CalculationStatus, ControllabilityAssessmentInput, ExposureAssessment,
    ExposureDimensionAssessment, ExposureMethodDomain, FTTIAssessmentInput,
    PhysicalConsequence, SeverityAssessmentInput, SpeedSemantic,
)


class RiskCalculationInputService:
    """Assemble typed S/E/C/FTTI inputs without choosing engineering values."""

    @staticmethod
    def _number(value: Any) -> float | None:
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or value < 0):
            return None
        return float(value)

    def severity(
        self, malfunction_id: str, scenario_id: str, scenario: dict[str, Any],
        speed_semantic: SpeedSemantic | None = None,
    ) -> SeverityAssessmentInput:
        semantic = speed_semantic or SpeedSemantic.UNRESOLVED
        if speed_semantic is None:
            if self._number(scenario.get("delta_v_kph")) is not None:
                semantic = SpeedSemantic.DELTA_V
            elif self._number(scenario.get("impact_speed_kph")) is not None:
                semantic = SpeedSemantic.IMPACT_SPEED
            elif self._number(scenario.get("relative_speed_kph")) is not None:
                semantic = SpeedSemantic.RELATIVE_SPEED
        return SeverityAssessmentInput(
            malfunction_id=malfunction_id,
            scenario_id=scenario_id,
            consequence=PhysicalConsequence(
                collision_object=str(scenario.get("collision_object", "")),
                collision_configuration=str(scenario.get("collision_configuration", "")),
                ego_speed_kph=self._number(scenario.get("ego_speed_kph")),
                collision_type=str(scenario.get("collision_type", "")),
                road_user_type=str(scenario.get("road_user_type", "")),
                relative_speed_kph=self._number(scenario.get("relative_speed_kph")),
                impact_speed_kph=self._number(scenario.get("impact_speed_kph")),
                delta_v_kph=self._number(scenario.get("delta_v_kph")),
            ),
            speed_semantic=semantic,
            status=CalculationStatus.PENDING_METHOD_SEMANTICS,
            reason=(
                "Physical consequence inputs are assembled; the active MethodContract "
                "must select the approved speed semantic and severity table."
            ),
        )

    def exposure(self, scenario_id: str, scenario: dict[str, Any]) -> ExposureAssessment:
        bindings = scenario.get("method_scenario_dimensions", {})
        dimensions = []
        if isinstance(bindings, dict):
            for name, binding in sorted(bindings.items()):
                if not isinstance(binding, dict):
                    continue
                value = str(
                    binding.get("method_value") or binding.get("project_value") or ""
                ).strip()
                if not value:
                    continue
                dimensions.append(ExposureDimensionAssessment(
                    dimension=str(name), scenario_value=value,
                    matched_entry_ids=(),
                    selected_domain=ExposureMethodDomain.UNRESOLVED,
                    status=CalculationStatus.PENDING_METHOD_SEMANTICS,
                    reason=(
                        "The dimension value is bound, but no normative atom-to-E and "
                        "cross-dimension combination rule is compiled."
                    ),
                ))
        return ExposureAssessment(
            scenario_id=scenario_id,
            dimensions=tuple(dimensions),
            status=CalculationStatus.PENDING_METHOD_SEMANTICS,
            reason=(
                "Dimension-level exposure inputs are assembled; Z/F selection, "
                "dependency and combination rules remain MethodContract decisions."
            ),
        )

    def controllability(
        self, malfunction_id: str, scenario_id: str, scenario: dict[str, Any],
    ) -> ControllabilityAssessmentInput:
        input_keys = (
            "ttc_s", "driver_in_vehicle", "has_remote_app",
            "direct_control_available",
            "remote_intervention_available", "other_road_user_avoidance_possible",
            "vehicle_stability", "function_type",
        )
        return ControllabilityAssessmentInput(
            malfunction_id=malfunction_id,
            scenario_id=scenario_id,
            ttc_s=self._number(scenario.get("ttc_s")),
            driver_position=str(scenario.get("driver_position", "")),
            driver_in_vehicle=scenario.get("driver_in_vehicle"),
            has_remote_app=scenario.get("has_remote_app"),
            function_type=str(scenario.get("function_type", "")),
            direct_control_available=scenario.get("direct_control_available"),
            remote_intervention_available=scenario.get(
                "remote_intervention_available"
            ),
            other_road_user_avoidance_possible=scenario.get(
                "other_road_user_avoidance_possible"
            ),
            vehicle_stability=str(scenario.get("vehicle_stability", "")),
            inputs_used=tuple(
                key for key in input_keys
                if scenario.get(key) is not None and scenario.get(key) != ""
            ),
            status=CalculationStatus.PENDING_METHOD_SEMANTICS,
            reason=(
                "TTC and intervention inputs are assembled; profile routing and "
                "TTC-to-C bands must be compiled from the approved MethodContract."
            ),
        )

    @staticmethod
    def ftti(
        malfunction_id: str, scenario_id: str, asil: str,
    ) -> FTTIAssessmentInput:
        return FTTIAssessmentInput(
            malfunction_id=malfunction_id,
            scenario_id=scenario_id,
            asil=asil,
            status=CalculationStatus.PENDING_METHOD_SEMANTICS,
            reason=(
                "ASIL is available as an upstream result, but the active MethodContract "
                "does not yet compile FTTI inputs, budgets and formulas."
            ),
        )
