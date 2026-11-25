from dataclasses import replace

import pytest

from hara_agent.models import FunctionDefinition, SourceRef
from hara_agent.services.validation.function_validator import FunctionValidator


@pytest.mark.parametrize("name", [
    "Bestimmung der Preselection beim Ausparken",
    "Bedienung über Tastendruck/Softkey und Sprachsteuerung",
    "Visualisierung von Parklücken und Trajektorie",
    "Determine the proposed direction for leaving a parking space",
    "输出制动扭矩",
])
def test_concise_multilingual_function_labels_preserve_traceability(name):
    item = FunctionDefinition(
        function_id="F1", name=name, output="output signal",
        sources=[SourceRef("item_definition", "input.pdf", "page[5]", name)],
    )
    validator = FunctionValidator()
    assert validator.validate([item]) == []
    assert {f.code for f in validator.validate([replace(item, sources=[])])} == {"missing_source"}
    assert "duplicate_function" in {f.code for f in validator.validate([item, item])}


@pytest.mark.parametrize("name", ["车辆控制" * 11, "This is a complete requirement paragraph " * 4])
def test_paragraph_length_labels_still_fail(name):
    item = FunctionDefinition(
        function_id="F1", name=name, output="signal",
        sources=[SourceRef("item_definition", "input.pdf", "page[5]", name)],
    )
    assert "function_name_too_long" in {f.code for f in FunctionValidator().validate([item])}


@pytest.mark.parametrize("padding", ["", "  "])
def test_output_copied_from_full_description_is_rejected(padding):
    # Preserve the concrete counterexample from the retired P001 checker.
    description = "MDC发送制动请求并控制车辆减速"
    item = FunctionDefinition(
        "F1", "输出制动扭矩", padding + description + padding,
        description=description,
        sources=[SourceRef("item_definition", "item.docx", "table[6].row[2]", description)],
    )
    findings = FunctionValidator().validate([item])
    assert {finding.code for finding in findings} == {"output_repeats_description"}
    with pytest.raises(ValueError, match="F1:output_repeats_description"):
        FunctionValidator().ensure_valid([item])


@pytest.mark.parametrize("description,output", [
    ("", "制动扭矩"),
    ("MDC发送制动请求并控制车辆减速", "制动请求"),
    ("Send brake torque command", "brake torque command"),
])
def test_supported_output_can_share_terms_with_description(description, output):
    item = FunctionDefinition(
        "F1", "输出制动扭矩", output, description=description,
        sources=[SourceRef("item_definition", "item.docx", "table[6].row[2]", "输出制动扭矩")],
    )
    assert FunctionValidator().validate([item]) == []
