from __future__ import annotations

import pytest

from hara_agent.models import SourceRef
from hara_agent.services.extraction import (
    DocumentBlock,
    FactRetrievalSpec,
    ProjectFactNormalizer,
    build_project_fact_spec_batches,
    retrieval_blocks,
)
from hara_agent.services.semantic import ItemEvidenceRouter
from hara_agent.services.semantic.item_definition_agent import ItemDefinitionNormalizer


def _page(text: str, number: int = 5) -> DocumentBlock:
    return DocumentBlock(f"PAGE-{number:04d}", "page", f"page[{number}]", text)


def test_pdf_routing_uses_exact_page_passages_without_replacing_source_page():
    original = _page(
        ("Unrelated introductory text.\n" * 35)
        + "►The user may position the parking space only when TopView is installed.\n"
        + ("Unrelated appendix text.\n" * 20)
    )

    fragments = retrieval_blocks([original])
    routed = ItemEvidenceRouter().route(
        [vars(original)], "project_evidence",
        required_specs=(FactRetrievalSpec("topview", ("TopView",)),),
        max_characters=1500,
    )

    assert original.block_id == "PAGE-0005"
    assert "TopView" in original.text
    assert len(fragments) > 1
    assert "".join(part["text"] for part in fragments) == original.text
    assert all(len(part["text"]) <= 600 for part in fragments)
    assert all(part["block_id"] != original.block_id for part in fragments)
    assert routed is not None
    assert any("TopView is installed" in part["text"] for part in routed.source_blocks)
    assert all(part["location"] == "page[5]" for part in routed.source_blocks)
    assert all(part["text"] in original.text for part in routed.source_blocks)
    assert all(part["parent_block_id"] == original.block_id for part in routed.source_blocks)
    assert len(routed.text) < len(original.text)


def test_pdf_negative_condition_stays_with_selected_passage():
    original = _page(
        ("Other feature.\n" * 45)
        + "►The function controls longitudinal and lateral motion.\n"
        + "►Keine Degradation auf Park-Lenk-Assistenz.\n"
        + ("Other feature.\n" * 20)
    )
    result = ItemEvidenceRouter().route(
        [vars(original)], "project_evidence",
        required_specs=(FactRetrievalSpec("fallback", ("Park-Lenk-Assistenz",)),),
        max_characters=1200,
    )

    assert result is not None
    assert "Keine Degradation" in result.text
    assert "Other feature.\n" * 45 not in result.text


def test_training_speed_cannot_become_execution_speed_but_training_mode_can_use_it():
    original = _page(
        ("Background.\n" * 60)
        + "►Training:\n"
        + "►The function records a trajectory at speed ≤40 km/h. Challenge: 50 km/h.\n"
        + "►Parkvorgang:\n"
        + "►Execution speed is configurable.\n"
    )
    fragments = retrieval_blocks([original])
    training = next(part for part in fragments if "≤40 km/h" in part["text"])
    execution = next(part for part in fragments if "Execution speed" in part["text"])
    assert training["source_phase"] == "TRAINING"
    assert "source_phase" not in execution
    routed = ItemEvidenceRouter().route(
        [vars(original)], "project_evidence",
        required_specs=(FactRetrievalSpec("speed", ("≤40 km/h",)),),
        max_characters=1500,
    )
    assert routed is not None
    assert "source_phase=TRAINING" in routed.text
    assert all("source_phase=" not in block["text"] for block in routed.source_blocks)
    odd_route = ItemEvidenceRouter().route([vars(original)], "odd_repair", max_characters=1500)
    assert odd_route is not None
    assert "source_phase=TRAINING" in odd_route.text

    for mode, expected_count in (("parking_vehicle_control", 0), ("training", 1), ("Trainingsphase", 1)):
        spec = build_project_fact_spec_batches((mode,), ())["mode_speed_envelopes"][0]
        candidate = {
            "fact_type": spec.fact_type, "status": "FOUND", "operator": "LE",
            "value": 40, "unit": "km/h", "operating_mode": mode,
            "semantic_scope": "OPERATIONAL_SPEED",
            "source_block_id": training["block_id"],
            "source_excerpt": "speed ≤40 km/h",
        }
        result = ProjectFactNormalizer().normalize(
            [spec], [candidate], [training], "rTPA.pdf",
            {spec.fact_type: {training["block_id"]}},
        )
        assert len(result.speed_envelopes) == expected_count
        assert ("SOURCE_PHASE_MISMATCH" in {failure.code for failure in result.failures}) == (expected_count == 0)
        if result.speed_envelopes:
            source = result.speed_envelopes[0].sources[0]
            assert source.location == original.location
            assert source.excerpt in original.text


def test_docx_blocks_are_unchanged_by_pdf_routing_view():
    block = DocumentBlock("P-0001", "paragraph", "paragraph[1]", "Parking speed ≤7 km/h", ("ODD",))
    assert retrieval_blocks([block]) == [{
        "block_id": block.block_id, "kind": block.kind, "location": block.location,
        "text": block.text, "section_path": ["ODD"],
    }]


def test_explicit_word_training_section_has_the_same_speed_scope_guard():
    block = DocumentBlock(
        "P-0002", "paragraph", "paragraph[2]", "Training speed ≤40 km/h",
        ("Training",),
    )
    spec = build_project_fact_spec_batches(("parking_vehicle_control",), ())["mode_speed_envelopes"][0]
    candidate = {
        "fact_type": spec.fact_type, "status": "FOUND", "operator": "LE",
        "value": 40, "unit": "km/h", "operating_mode": "parking_vehicle_control",
        "semantic_scope": "OPERATIONAL_SPEED", "source_block_id": block.block_id,
        "source_excerpt": "speed ≤40 km/h",
    }
    result = ProjectFactNormalizer().normalize(
        [spec], [candidate], [retrieval_blocks([block])[0]], "Item.docx",
        {spec.fact_type: {block.block_id}},
    )
    assert not result.speed_envelopes
    assert [failure.code for failure in result.failures] == ["SOURCE_PHASE_MISMATCH"]


@pytest.mark.parametrize("background_lines", [0, 60])
def test_core_pdf_speed_envelope_keeps_training_value_out_of_execution_mode(background_lines):
    page = _page(
        ("Background.\n" * background_lines)
        + "►Training:\n►Recording speed ≤40 km/h. Challenge: 50 km/h.\n"
        + "►Parkvorgang:\n►Execution speed is configurable.\n"
    )
    for mode, expected_count in (("parking_vehicle_control", 0), ("training", 1), ("Trainingsphase", 1)):
        raw = {
            "system_description": "Test", "item_boundary": "Vehicle",
            "operating_modes": [mode],
            "odd": {"locations": [], "road_types": [], "weather_conditions": [],
                    "road_surfaces": [], "speed_range_kph": None},
            "source_location": "page[5]", "source_excerpt": "Background.",
            "confidence": 0.8, "status": "PENDING",
            "speed_envelopes": [{
                "operating_mode": mode, "speed_min_kph": 0, "speed_max_kph": 40,
                "status": "PENDING", "sources": [{
                    "source_type": "item_definition", "source_id": "test.pdf",
                    "location": "page[5]", "excerpt": "Recording speed ≤40 km/h",
                }],
            }],
        }
        facts, warnings = ItemDefinitionNormalizer._parse_with_warnings(
            raw, "test.pdf", [page],
        )
        assert len(facts.speed_envelopes) == expected_count
        assert bool(warnings) == (expected_count == 0)
        if warnings:
            assert "SOURCE_PHASE_MISMATCH" in warnings[0]


def test_core_pdf_phase_guard_locates_cross_passage_and_whitespace_citations():
    page = _page(
        "►Training:\n►" + ("recording context " * 35)
        + "speed ≤40 km/h.\n"
        + "►Parkvorgang:\n►Execution  speed ≤7 km/h.\n"
    )
    passages = retrieval_blocks([page])
    training_parts = [part for part in passages if part.get("source_phase") == "TRAINING"]
    assert len(training_parts) >= 2
    boundary = training_parts[-2]["span_end"]
    crossing_excerpt = page.text[boundary - 12:boundary + 12]
    assert ItemDefinitionNormalizer._training_speed_source(
        [SourceRef("item_definition", "test.pdf", "page[5]", crossing_excerpt)],
        [page], passages,
    )
    assert not ItemDefinitionNormalizer._training_speed_source(
        [SourceRef("item_definition", "test.pdf", "page[5]", "Execution speed ≤7 km/h")],
        [page], passages,
    )
