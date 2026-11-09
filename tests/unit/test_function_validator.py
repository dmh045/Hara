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
