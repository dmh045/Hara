from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from hara_agent.method_sources import MethodSourceResolver
from hara_agent.models import MalfunctionCandidate, ScenarioCandidate
from hara_agent.services.analysis.scenario_method_service import ScenarioMethodService


ROOT = Path(__file__).resolve().parents[2]


def _current():
    method = MethodSourceResolver().resolve(
        template_path=None,
        baseline_manifest_path=ROOT / "method_assets/fusa_baseline_v1/manifest.yaml",
        report_template_path=ROOT / "references/HARA_Template_AI_20260327.xlsx",
    ).method
    malfunction = MalfunctionCandidate(
        malfunction_id="MF-F01-002", function_id="F01", guideword="More",
        guideword_id="GW-MORE",
        description="MDC输出的制动减速度控制指令幅值超出泊车场景需求值",
        functional_effect="IPB执行过大的制动扭矩，车辆减速度超过预期",
        vehicle_level_hazard="泊车时车辆急减速，易引发后车追尾或车内乘员前倾磕碰",
        causal_chain=["制动指令超出需求", "车辆急减速"],
        component_category="computing", failure_type="excessive",
    )
    scenario = ScenarioCandidate(
        scenario_id="SCN-TEST-BRAKING", operating_scenario="泊车",
        situational_description="泊车", situational_detailing="泊车",
        facts={"ego_speed_kph": 7.0, "scenario_atom_ids": ["PH005"]},
        semantic_fingerprint="bounded-test-parent",
    )
    return method, scenario, malfunction


def test_exact_current_brake_overcommand_uses_governed_braking_template():
    method, scenario, malfunction = _current()
    service = ScenarioMethodService(method)
    match = service.match_fm_template(malfunction)
    assert match.injectable
    assert match.template.template_id == "FM_TEMPLATE_003"
    assert match.matched_by == ("ENGINEERING_DECISION",)
    assert match.decision_source_ref["source_type"] == "project_analysis_policy"
    variants, audit = service.instantiate_analytical_candidates(
        malfunction, [scenario],
    )
    assert audit["selection_mode"] == "STRONG_TEMPLATE_ANALYTICAL_INSTANCES"
    assert {item.analysis_instance["source_option_id"] for item in variants} == {
        "FM_TEMPLATE_003:OPTION:1", "FM_TEMPLATE_003:OPTION:2",
    }
    assert {item.facts["object_speed_kph"] for item in variants} == {0.0, 10.0}
    assert {item.facts["relative_distance_m"] for item in variants} == {3.0}
    assert {item.facts["object_position"] for item in variants} == {"rear"}
    assert {item.facts["collision_type"] for item in variants} == {"REAR_END"}
    assert all(item.facts["scenario_atom_ids"] == scenario.facts["scenario_atom_ids"]
               for item in variants)


def test_governed_mapping_does_not_expand_to_changed_malfunction():
    method, _, malfunction = _current()
    changed = replace(malfunction, description=malfunction.description + "（不同故障）")
    match = ScenarioMethodService(method).match_fm_template(changed)
    assert not match.injectable
    assert match.template is None
    assert match.reason == "GOVERNED_DECISION_SCOPE_MISMATCH"
