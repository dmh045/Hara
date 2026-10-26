from hara_agent.contracts import FactOrigin, FactType, RequiredFactSpec
from hara_agent.models import ProjectFactOutputType
from hara_agent.services.extraction import (
    ProjectFactNormalizer, build_project_fact_spec_batches,
)
from hara_agent.workflow.nodes.item_artifacts import _speed_extraction_modes


def _required(fact_type, origin, unit="", constraints=()):
    return RequiredFactSpec(
        fact_type=fact_type,
        required_for=("EXPOSURE",),
        unit=unit,
        constraints=tuple(constraints),
        condition="",
        origin=origin,
        source_rule_ids=(),
        source_refs=(),
    )


def test_batches_are_compiled_from_modes_and_method_contract_only():
    batches = build_project_fact_spec_batches(
        ("Garage: automated operation", "Highway", "garage"),
        (
            _required(FactType.OCCURRENCE_FREQUENCY, FactOrigin.HUMAN_EVIDENCE),
            _required(FactType.COLLISION_TYPE, FactOrigin.SCENARIO_FACT),
            _required(FactType.FUNCTION, FactOrigin.PROJECT_FACT),
        ),
    )

    speed = batches["mode_speed_envelopes"]
    assert [item.context_hints for item in speed] == [("Garage",), ("Highway",), ()]
    assert speed[-1].structural_kind == "OPERATIONAL_SPEED_CANDIDATE"
    assert speed[-1].exclusion_aliases == ("Garage", "Highway")
    risk = batches["method_risk_facts"]
    assert [item.fact_type for item in risk] == ["OCCURRENCE_FREQUENCY"]
    assert all("parking" not in repr(item).casefold() for item in (*speed, *risk))


def test_explicit_requested_mode_limits_speed_extraction_scope():
    assert _speed_extraction_modes(
        ("Active",),
        ("OFF", "Standby", "Active", "Override", "Abort", "Finish", "Error"),
    ) == ("Active",)


def test_requested_mode_routes_source_speed_even_when_identifier_is_not_literal():
    from hara_agent.services.extraction.evidence_retrieval import DeterministicEvidenceRetriever
    from hara_agent.services.analysis.project_fact_resolver import ProjectFactResolver
    from hara_agent.models import ItemDefinitionFacts
    spec = build_project_fact_spec_batches(("parking_vehicle_control",), ())["mode_speed_envelopes"][0]
    block = {"block_id": "B1", "location": "table[2].row[3]", "kind": "table_row",
             "text": "速度范围 | 搜索车位0-40kph，控车范围0-12kph"}
    ranked = DeterministicEvidenceRetriever().rank([block], [spec.retrieval_spec()])
    assert [b.block_id for b in ranked[spec.fact_type]] == ['B1']
    raw = {"fact_type": spec.fact_type, "status": "FOUND", "operator": "RANGE",
           "value": 0, "value_max": 12, "unit": "kph", "operating_mode": "parking_vehicle_control",
           "semantic_scope": "OPERATIONAL_SPEED", "source_block_id": "B1", "source_excerpt": "控车范围0-12kph"}
    normalized = ProjectFactNormalizer().normalize([spec], [raw], [block], 'item', {spec.fact_type: {'B1'}})
    assert not normalized.failures
    facts = ItemDefinitionFacts(system_description='AVP', item_boundary='Vehicle',
        speed_envelopes=normalized.speed_envelopes, sources=normalized.speed_envelopes[0].sources)
    resolved = ProjectFactResolver().resolve_speed_context(facts, 'parking_vehicle_control')
    assert resolved.resolved_value == 12 and not resolved.fallback_used
    assert resolved.source_refs[0].excerpt == '控车范围0-12kph'
    for mutation in ({'semantic_scope': 'CAPABILITY'}, {'source_excerpt': '控车范围0-7kph'}, {'source_block_id': 'UNKNOWN'}):
        rejected = ProjectFactNormalizer().normalize([spec], [{**raw, **mutation}], [block], 'item', {spec.fact_type: {'B1'}})
        assert rejected.failures and not rejected.speed_envelopes


def test_declared_modes_are_used_when_no_mode_was_requested():
    assert _speed_extraction_modes((), ("search", "parking")) == (
        "search", "parking",
    )


def test_method_risk_fact_normalization_is_neutral_and_source_grounded():
    spec = build_project_fact_spec_batches(
        (),
        (_required(FactType.OCCURRENCE_FREQUENCY, FactOrigin.PROJECT_FACT),),
    )["method_risk_facts"][0]
    assert spec.output_type is ProjectFactOutputType.RISK_FACT
    result = ProjectFactNormalizer().normalize(
        (spec,),
        [{
            "fact_type": "OCCURRENCE_FREQUENCY",
            "status": "FOUND",
            "value": "F",
            "unit": "",
            "context": {},
            "source_block_id": "B1",
        }],
        [{
            "block_id": "B1",
            "location": "table[1].row[2]",
            "text": "OCCURRENCE_FREQUENCY = F",
        }],
        "ItemDef.docx",
    )

    assert not result.failures
    assert result.risk_facts[0].parameter == "OCCURRENCE_FREQUENCY"
    assert result.risk_facts[0].value == "F"
    assert result.risk_facts[0].fact_id.startswith("RF-")
    assert result.risk_facts[0].source_refs[0].location == "table[1].row[2]"


def test_method_decisions_and_executor_outputs_are_not_requested_from_item_definition():
    batches = build_project_fact_spec_batches((), (
        _required(FactType.EXPOSURE, FactOrigin.HUMAN_EVIDENCE),
        _required(
            FactType.AVOIDABILITY_PERCENT,
            FactOrigin.HUMAN_EVIDENCE,
            unit="%",
        ),
    ))

    assert "method_risk_facts" not in batches


def test_driver_allowed_set_is_preserved_as_two_scoped_atomic_facts():
    spec = build_project_fact_spec_batches((), (
        _required(FactType.DRIVER_IN_VEHICLE, FactOrigin.PROJECT_FACT),
    ))["method_risk_facts"][0]
    assert spec.structural_kind == "CATEGORICAL_ALLOWED_SET_CANDIDATE"
    block = {
        "block_id": "D1", "location": "table[14].row[13]",
        "text": "位姿状态 | 在驾驶位/不在驾驶位",
    }
    raw = [
        {
            "fact_type": spec.fact_type, "status": "FOUND", "value": "true",
            "unit": "", "context": {"allowed_driver_position": "in_driver_seat"},
            "semantic_scope": "DRIVER_CONFIGURATION", "source_block_id": "D1",
            "source_excerpt": "在驾驶位",
        },
        {
            "fact_type": spec.fact_type, "status": "FOUND", "value": "false",
            "unit": "", "context": {"allowed_driver_position": "outside_driver_seat"},
            "semantic_scope": "DRIVER_CONFIGURATION", "source_block_id": "D1",
            "source_excerpt": "不在驾驶位",
        },
    ]

    result = ProjectFactNormalizer().normalize(
        (spec,), raw, (block,), "ItemDef.docx", {spec.fact_type: {"D1"}},
    )

    assert not result.failures
    assert [(item.value, item.context["allowed_driver_position"]) for item in result.risk_facts] == [
        ("true", "in_driver_seat"), ("false", "outside_driver_seat"),
    ]
