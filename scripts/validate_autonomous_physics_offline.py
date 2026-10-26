"""Real Item context -> ordinary application -> scripted JSON -> draft report.

OFFLINE ONLY. This client has no transport and cannot consume Provider budget.
The scripted proposals/causal answers test plumbing, not LLM engineering quality.
No collision branch, rating, derived speed, TTC or approval is injected.
"""
import argparse
from collections import Counter
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

from hara_agent.application import HARAApplication
from hara_agent.config import RunConfig
from hara_agent.infrastructure.llm.protocol import LLMResponse
from hara_agent.method_sources import MethodSourceResolver
from hara_agent.services.extraction import DocumentReader, select_source_blocks
from hara_agent.workflow import HARAState, WorkflowStage, CheckpointRepository

ROOT = Path(__file__).resolve().parents[1]


class OfflinePhysicalClient:
    def __init__(self, output, *, range_proposals=False, symbolic_drive=False, interaction_coverage=False):
        self.output = output
        self.config = SimpleNamespace(provider="OFFLINE_SCRIPTED", model="offline-physical-fixtures-v1", max_tokens=8192)
        self.calls = Counter()
        self.range_proposals = range_proposals
        self.symbolic_drive = symbolic_drive
        self.interaction_coverage = interaction_coverage

    def proposal(self, request):
        from hara_agent.services.semantic.physical_scenario_agent import expand_physical_prompt_view
        payload = json.loads(request.user_prompt)
        context = expand_physical_prompt_view(payload["context"], payload.get("source_texts", {}))
        if context.get('task_scope') == 'CONTROL_INPUTS_ONLY':
            return {'hypotheses': [], 'uncovered_conditions': ['Offline control assumptions require engineering review'],
                'control_assessment': {'rationale': 'OFFLINE: independent driver braking remains outside the automatic command fault',
                    'basis_refs': ['MF.description'], 'conditions': ['Independent manual braking path remains effective in this test'],
                    'uncovered_conditions': [], 'failed_controls': ['automatic_braking_command'], 'remaining_controls': ['driver_braking'],
                    'intervention_assumptions': {key: {'value': False, 'rationale': 'Explicit offline no independent evasion representative',
                        'basis_refs': ['SCENARIO.conditions'], 'conditions': ['No timely independent intervention in this test world']}
                        for key in ('remote_intervention_available', 'other_road_user_avoidance_possible')}}}
        # Fixture dispatch is confined to this test harness, not runtime rules.
        function = context["malfunction"]["function_id"]
        label = {"F02": "drive", "F03": "lateral", "F08": "rollaway"}.get(function, "drive")
        raw = json.loads((ROOT / f"tests/fixtures/autonomous_physics/{label}.json").read_text())
        h = raw["hypotheses"][0]
        if label == "drive" and self.range_proposals:
            parameter = next(p for p in h["parameters"] if p["field"] == "longitudinal_acceleration_mps2")
            parameter.update(kind="RANGE", lower=.2, upper=4,
                rationale="Offline estimated interval for sensitivity, not a measured capability or true upper bound")
        if label == "drive" and self.symbolic_drive:
            parameter = next(p for p in h["parameters"] if p["field"] == "longitudinal_acceleration_mps2")
            parameter.pop("value")
            parameter.update(kind="SYMBOLIC", symbol="a_fault", rationale="Unspecified fault acceleration; retain parametric relationship")
            h["conditions"].append("For the conditional contact claim: a_fault > 0 and the model remains valid until contact")
        facts = context["conditions"]["facts"]
        h["geometry"]["road_user_type"] = facts.get("road_user_type", "VEHICLE")
        initial = facts.get("ego_speed_kph", 0)
        desired_initial = 0 if label == "rollaway" else max(initial, 3.6) if label == "lateral" else initial
        h["parameters"][0]["value"] = desired_initial
        h["condition_changes"] = {"initial_state": initial != desired_initial, "target": True,
            "rationale": "Explicit separate stationary target; initial condition changed only where required by this candidate model, with Atom rebinding"}
        h["geometry"]["object_id"] = "analytical-stationary-target"
        if label != "lateral":
            direction = facts.get("ego_longitudinal_direction", "FORWARD")
            if direction not in {"FORWARD", "REVERSE"}:
                direction = "FORWARD"
            h["geometry"].update(ego_longitudinal_direction=direction,
                object_position="REAR" if direction == "REVERSE" else "FRONT")
        if label == "rollaway":
            tag = facts.get("source_slope_direction", facts.get("source_operating_conditions", {}).get("source_slope_direction"))
            # This is an explicit test proposal, not a rule mapping all source
            # 'uphill' tags to reverse. Source orientation remains unconfirmed.
            dynamics = facts.get("method_scenario_dimensions", {}).get("EGO_DYNAMICS", {}).get("method_semantics", {}).get("ego_dynamics", {})
            # Two test subsets exercise both admissible orientation assumptions
            # without duplicating any parent. This fixture choice is not an
            # inference from the Item tag or an executable engineering rule.
            uphill = tag == "上坡" and dynamics.get("motion_state") != "STATIONARY"
            h["grade_frame"] = {"coordinate_frame": "BODY_FORWARD_POSITIVE", "slope_reference": "UNSPECIFIED",
                "heading_relative_to_downhill": "UPHILL" if uphill else "DOWNHILL",
                "downhill_body_direction": "REAR" if uphill else "FRONT", "orientation_basis": "ANALYSIS_ASSUMPTION",
                "rationale": "Offline coverage fixture explicitly assumes this heading; the source label does not establish it. The two test subsets exercise opposite orientations without doubling parents"}
            h["geometry"].update(ego_longitudinal_direction="REVERSE" if uphill else "FORWARD", object_position="REAR" if uphill else "FRONT")
            road = facts.get("method_scenario_dimensions", {}).get("ROAD", {}).get("method_semantics", {}).get("slope", {}).get("range_pct", [0, 15])
            for p in h["parameters"]:
                if p["field"] == "slope_percent":
                    p["value"] = min(15, road[-1])
                    p["rationale"] = "Offline representative grade within supplied project 15% and selected ROAD range; not a measurement"
        h["rationale"] = "OFFLINE scripted analytical proposal for " + context["malfunction"]["description"]
        if self.interaction_coverage and label == "drive":
            raw = json.loads((ROOT / "tests/fixtures/autonomous_physics/moving_contact.json").read_text())
            original = raw['hypotheses'][0]
            original['condition_changes'] = {"initial_state": False, "target": True,
                "rationale": "Offline independent moving participants inside the same source ODD"}
            original['geometry'].update(ego_longitudinal_direction=direction)
            raw['hypotheses'] = []
            for index, interaction in enumerate(('oncoming', 'following', 'crossing')):
                proposal = deepcopy(original)
                g = proposal['geometry']
                g.update(object_id='offline-' + interaction, object_position='FRONT' if direction == 'FORWARD' else 'REAR',
                    object_longitudinal_direction=('REVERSE' if direction == 'FORWARD' else 'FORWARD') if interaction == 'oncoming' else direction,
                    collision_type='FRONTAL' if interaction == 'oncoming' else 'REAR_END')
                if interaction == 'crossing':
                    g.update(object_position='LEFT', object_longitudinal_direction='RIGHT', collision_type='SIDE', road_user_type='PEDESTRIAN')
                proposal['hypothesis_id'] = interaction
                proposal['rationale'] = 'OFFLINE: excessive drive reaches a ' + interaction + ' participant before intervention; conditional finite trajectories'
                for p in proposal['parameters']:
                    if p['field'] == 'ego_speed_kph': p['value'] = initial
                    if p['field'] == 'object_offset_x_m': p['value'] = 8 if direction == 'FORWARD' else -8
                    if p['field'] == 'object_offset_y_m': p['value'] = 4 if interaction == 'crossing' else 0
                raw['hypotheses'].append(proposal)
        if self.interaction_coverage and context.get('source_condition_focus'):
            for proposal in raw['hypotheses']:
                # Test responses select actual source alternatives and explain
                # their connection to the chosen interaction. This mapping is
                # a frozen offline fixture, never a runtime selection policy.
                wanted, effect = {
                    'oncoming': ('水平车位', '平行车位旁的直行通道中，自车与对向车辆沿同一纵向走廊接近'),
                    'following': ('单侧障碍物', '单侧边界限制绕行空间；本代表保持沿通道追近前方移动车辆，不据此宣称所有参与者均无法规避'),
                    'crossing': ('垂直车位', '自车出库方向与通道行人横穿方向正交，有限包络及到达时刻共同决定接触'),
                }.get(proposal['hypothesis_id'], ('', ''))
                if wanted not in context['project_context'].get('odd_road_types', []):
                    proposal['uncovered_conditions'].append('本离线回复未实例化额外的车位或天气条件；坡向/侧向轨迹仅覆盖已列条件')
                    continue
                focus = {'source_field': 'odd_road_types', 'value': wanted}
                proposal['operating_condition'] = {**focus,
                    'physical_effect': effect + '（离线条件性代表）'}
        return raw

    @staticmethod
    def causal(request):
        raw, _ = json.JSONDecoder().raw_decode(request.user_prompt.split("Scenarios=", 1)[1])
        shared = raw.get("shared_context", {}) if isinstance(raw, dict) else {}
        scenarios = raw["scenarios"] if isinstance(raw, dict) else raw
        result = []
        for sc in scenarios:
            interaction = {**shared.get("physical_interaction", {}), **sc.get("physical_interaction", {})}
            derived = interaction.get("derived_physics_evidence", {})
            impact = next((ref for ref in ("DERIVED.ego_impact_speed_kph", "DERIVED.relative_speed_kph") if ref in derived), None)
            analysis = {**shared.get("conditional_physical_analysis", {}), **sc.get("conditional_physical_analysis", {})}
            symbolic = any(p["kind"] == "SYMBOLIC" for p in analysis.get("hypothesis", {}).get("parameters", []))
            outcome = analysis.get("physics", {}).get("motion_model_outcome", {})
            positive = bool(impact) or (symbolic and outcome.get("contact_possible_in_model") is not False)
            contact_ref = impact or ("SCN.fault_motion_model" if symbolic else None)
            contact_basis = "DERIVED_PHYSICS" if impact else "DIRECT_FACT" if symbolic else "ASSUMPTION"
            chain = {"m_to_b": {"claim": "在给定失效定义下发生所列运动偏离（离线模拟判断）", "basis_type": "DIRECT_FACT", "evidence_refs": ["MF.description"]},
                "b_to_i": {"claim": "仅在所列参数关系成立、目标位于接近路径且模型持续有效至首次接触的条件下，参数化轨迹到达所列目标" if symbolic and positive else "在该模型与假设下到达所列目标" if positive else "该候选未形成可计算的接触轨迹",
                    "basis_type": contact_basis, "evidence_refs": [contact_ref] if positive else []}}
            if positive:
                chain["i_to_h"] = {"claim": "在所列条件下与该目标发生接触", "basis_type": contact_basis, "evidence_refs": [contact_ref]}
            result.append(dict(scenario_id=sc["scenario_id"], physically_feasible=True, functionally_relevant=True,
                causally_relevant=positive, breakpoint="NONE" if positive else "B_TO_I", causal_chain=chain,
                risk_dimension_changes=[], hazardous_event="条件性故障运动导致与所列目标接触（离线模拟因果判断）" if positive else "",
                physical_interaction={"target": "OTHER"}, confidence=.9, status="FINALIZED"))
        return {"assessments": result}

    def complete_json(self, request):
        self.calls[request.task] += 1
        if request.task == "construct_post_fault_physical_scenario":
            data = self.proposal(request)
        elif request.task == "assess_scenario_feasibility":
            data = self.causal(request)
        elif request.task == "interpret_scenario_risk_facts":
            requests, _ = json.JSONDecoder().raw_decode(request.user_prompt.split("fact_requests=", 1)[1])
            data = {"results": [{k: r[k] for k in ("malfunction_id", "scenario_id", "fact_type")} |
                {"status": "NOT_FOUND", "evidence_ids": [], "reason": "Offline fixture does not invent extracted facts"} for r in requests]}
        else:
            raise AssertionError(f"Unexpected offline task: {request.task}")
        response = LLMResponse(data, self.config.model, f"offline-{sum(self.calls.values())}",
            {"offline_scripted": True, "prompt_tokens": 0, "completion_tokens": 0})
        with (self.output / "scripted_requests_responses.jsonl").open("a") as stream:
            stream.write(json.dumps({"request": asdict(request), "response": asdict(response)}, ensure_ascii=False) + "\n")
        return response


def source_state(saved, method, item, run_id):
    from hara_agent.models import ItemDefinitionFacts
    from hara_agent.services.semantic.item_definition_agent import ItemDefinitionNormalizer
    document = DocumentReader().read(item)
    selection = select_source_blocks(document.blocks, method.metadata.get("project_analysis_policy", {}))
    chosen = {"MF-F01-001", "MF-F02-002", "MF-F03-002", "MF-F08-001"}
    malfunctions = [m for m in saved["malfunctions"] if m["malfunction_id"] in chosen]
    function_ids = {m["function_id"] for m in malfunctions}
    typed = deepcopy(saved["item_definition"]["typed"])
    typed["risk_facts"] = [f for f in typed["risk_facts"] if "scenario_id" not in f.get("context", {})]
    fact_ids = {f["fact_id"] for f in typed["risk_facts"]}
    typed["method_risk_fact_bindings"] = [b for b in typed["method_risk_fact_bindings"] if b["source_fact_id"] in fact_ids]
    typed = asdict(ItemDefinitionNormalizer.normalize_source_scope(
        ItemDefinitionFacts.from_dict(typed), document.blocks, document.source_id))
    state = HARAState(run_id, stage=WorkflowStage.MALFUNCTIONS,
        item_definition={"typed": typed, "source_id": document.source_id, "source_path": str(item),
                         "source_selection": selection.to_dict(),
                         "blocks": [asdict(block) for block in selection.selected_blocks]},
        functions=[f for f in saved["functions"] if f["function_id"] in function_ids], malfunctions=malfunctions,
        method_contract={"template_hash": method.metadata["template_hash"]})
    return state


def run(output: Path, source_checkpoint: Path, *, range_proposals=False):
    output.mkdir(parents=True, exist_ok=False)
    original = source_checkpoint.read_bytes()
    saved = json.loads(original)
    item = ROOT / "input/ItemDef.docx"
    resolution = MethodSourceResolver().resolve(template_path=None,
        baseline_manifest_path=ROOT / "method_assets/fusa_baseline_v1/manifest.yaml",
        report_template_path=ROOT / "references/HARA_Template_AI_20260327.xlsx")
    method = resolution.method
    run_id = "physical-offline-20261018-" + output.name
    state = source_state(saved, method, item, run_id)
    state.record("offline_source_context_replay", source_checkpoint=str(source_checkpoint),
        available_malfunction_count=len(saved["malfunctions"]),
        source_checkpoint_sha256=hashlib.sha256(original).hexdigest(), item_sha256=hashlib.sha256(item.read_bytes()).hexdigest(),
        note="Existing extracted source context; simulated raw Provider JSON; no historical physical branches or risk conclusions reused")
    repository = CheckpointRepository(output / "checkpoints")
    repository.save(state)
    config = RunConfig(item, None, output / "HARA_Autonomous_Physics_Offline.xlsx", output / "checkpoints",
        allow_draft=True, resume=True, run_id=run_id, operating_mode="parking_vehicle_control", max_workers=1)
    client = OfflinePhysicalClient(output, range_proposals=range_proposals)
    old_review = os.environ.get("HARA_REVIEW_ARTIFACT_DIR")
    os.environ["HARA_REVIEW_ARTIFACT_DIR"] = str(output / "review")
    try:
        result = HARAApplication(config, client).run()
    finally:
        if old_review is None:
            os.environ.pop("HARA_REVIEW_ARTIFACT_DIR", None)
        else:
            os.environ["HARA_REVIEW_ARTIFACT_DIR"] = old_review
    state = result.state
    summary = {"run_id": run_id, "validation_mode": "OFFLINE_SCRIPTED_PRODUCTION_APPLICATION",
        "live_provider_requests": 0, "reason": result.reason, "scripted_calls": dict(client.calls),
        "physical_metrics": state.item_definition.get("physical_analysis_metrics"),
        "scenarios": len(state.scenarios), "risk_results": len(state.risk_results),
        "models": dict(Counter(s.facts.get("fault_motion_model", "missing") for s in state.scenarios)),
        "conditional_s": sum(bool(r.severity.value) and r.severity.status.value == "PENDING" for r in state.risk_results),
        "conditional_asil": sum(bool(r.asil.value) and r.asil.status.value == "PENDING" for r in state.risk_results),
        "existing_rule_asil_offline": sum(r.asil.status.value == "FINALIZED" for r in state.risk_results),
        "live_validated_new_events": 0,
        "historical_checkpoint_unchanged": source_checkpoint.read_bytes() == original,
        "new_full_run_ready": False, "live_gap": "No valid new paid budget; engineering plausibility and transport stability untested"}
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "physical_metrics"}, ensure_ascii=False))
    return summary


def inventory(output: Path, source_checkpoint: Path):
    """Enumerate current candidates and pending plans; do not request any plan."""
    from hara_agent.models import ItemDefinitionFacts, MalfunctionCandidate, ReviewStatus, SourceRef
    from hara_agent.services.analysis import MethodScenarioCandidateService, ProjectFactResolver
    from hara_agent.services.analysis.malfunction_situation_selection import selection_function
    from hara_agent.services.analysis.scenario_equivalence import scenario_conditions, signature
    from hara_agent.services.analysis.physical_hypothesis_instantiation_service import PhysicalHypothesisInstantiationService

    output.mkdir(parents=True, exist_ok=False)
    original = source_checkpoint.read_bytes()
    saved = json.loads(original)
    method = MethodSourceResolver().resolve(template_path=None,
        baseline_manifest_path=ROOT / "method_assets/fusa_baseline_v1/manifest.yaml",
        report_template_path=ROOT / "references/HARA_Template_AI_20260327.xlsx").method
    typed = deepcopy(saved["item_definition"]["typed"])
    typed["risk_facts"] = [f for f in typed["risk_facts"] if "scenario_id" not in f.get("context", {})]
    ids = {f["fact_id"] for f in typed["risk_facts"]}
    typed["method_risk_fact_bindings"] = [b for b in typed["method_risk_fact_bindings"] if b["source_fact_id"] in ids]
    project = ItemDefinitionFacts.from_dict(typed)
    functions = {f["function_id"]: selection_function(f) for f in saved["functions"]}
    service = MethodScenarioCandidateService(method)
    producer = PhysicalHypothesisInstantiationService(method)
    mode = "parking_vehicle_control"
    parents, _ = service.generate(project_facts=project, operating_mode=mode,
        speed_resolution=ProjectFactResolver().resolve_speed_context(project, mode), functions=list(functions.values()))
    rows, totals = [], Counter()
    for raw in saved["malfunctions"]:
        mf = MalfunctionCandidate(**{**raw, "status": ReviewStatus(raw["status"]),
            "sources": [SourceRef(**s) for s in raw["sources"]]})
        children, audit = service.generate_for_malfunction(mf, parents, project_facts=project,
            function=functions.get(mf.function_id))
        waiting = [c for c in children if producer.task_scope(c, mf.malfunction_id)]
        groups = {}
        for candidate in waiting:
            key = signature(scenario_conditions(candidate, without_driver=True))
            groups.setdefault(key, []).append(candidate.scenario_id)
        generation = audit["atom_generation"]
        row = {"malfunction_id": mf.malfunction_id, "description": mf.description,
            "current_candidates": len(children), "existing_physics_sufficient": len(children) - len(waiting),
            "pending_physics_candidates": len(waiting), "pending_plan_groups": len(groups),
            "potential_plan_reuse": len(waiting) - len(groups),
            "exact_candidate_coalescing": generation["coalesced_candidate_count"],
            "plan_groups": groups, "status": "NOT_REQUESTED", "new_hypotheses": 0, "live_calls": 0}
        rows.append(row)
        totals.update({k: row[k] for k in ("current_candidates", "existing_physics_sufficient", "pending_physics_candidates",
            "pending_plan_groups", "potential_plan_reuse", "exact_candidate_coalescing")})
    summary = {"mode": "CURRENT_GENERATION_INVENTORY_NO_PROVIDER", "source_checkpoint_sha256": hashlib.sha256(original).hexdigest(),
        "method_source_hash": method.metadata["method_source_hash"], "malfunctions": len(rows), "base_contexts": len(parents),
        **totals, "live_calls": 0, "new_hypotheses": 0, "source_unchanged": source_checkpoint.read_bytes() == original,
        "budget_note": "pending_plan_groups is one first proposal per unchanged non-driver condition group. Local schema repair can add one each; transport retries and downstream causal/risk requests are extra and must share an explicit budget. At most 3 instantiations per input candidate. No full-run request count is promised before actual proposals."}
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    (output / "per_malfunction.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-checkpoint", type=Path, default=ROOT / "runtime/agent/full-p6-20261012-002.checkpoint.json")
    parser.add_argument("--inventory-only", action="store_true")
    parser.add_argument("--range-proposals", action="store_true")
    args = parser.parse_args()
    if args.inventory_only:
        inventory(args.output, args.source_checkpoint)
    else:
        run(args.output, args.source_checkpoint, range_proposals=args.range_proposals)
