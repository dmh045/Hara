from __future__ import annotations

from pathlib import Path

import pytest
from hara_agent.contracts import StructuredRiskMethod
from hara_agent.method_sources import (
    MethodSourceKind, MethodSourceResolver, YamlBaselineCompileError,
    YamlBaselineCompiler,
)
from hara_agent.template import TemplateRoleCompiler


ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "method_assets/fusa_baseline_v1/manifest.yaml"
REPORT_TEMPLATE = ROOT / "references/HARA_Template_AI_20260327.xlsx"


def _report_contract():
    return TemplateRoleCompiler().compile_method(REPORT_TEMPLATE).report_contract


def test_yaml_baseline_compiles_to_one_method_contract():
    method = YamlBaselineCompiler().compile(
        MANIFEST, report_contract=_report_contract(),
    )
    assert method.metadata["source_kind"] == "YAML_BASELINE"
    assert method.metadata["method_source_hash"] == method.metadata["template_hash"]
    assert len(method.guidewords.guidewords) == 8
    assert len(method.scenario_model.dimensions) == 7
    assert len(method.asil.mappings) == 80
    assert isinstance(method.structured_risk_method, StructuredRiskMethod)
    assert len(method.structured_risk_method.exposure.atoms) == 197
    assert method.compiler_version == "yaml-baseline-compiler-v3"


def test_yaml_baseline_rejects_duplicate_canonical_role():
    target = ROOT / "tests/fixtures/yaml_baseline/duplicate_role_manifest.yaml"
    with pytest.raises(YamlBaselineCompileError, match="Duplicate canonical role"):
        YamlBaselineCompiler().compile(target, report_contract=_report_contract())


def test_yaml_baseline_asil_zero_short_circuit_and_iso_body():
    method = YamlBaselineCompiler().compile(
        MANIFEST, report_contract=_report_contract(),
    )
    matrix = {
        (item.severity, item.exposure, item.controllability): item.result
        for item in method.asil.mappings
    }
    assert matrix[("S0", "E4", "C3")] == "QM"
    assert matrix[("S3", "E0", "C3")] == "QM"
    assert matrix[("S3", "E4", "C0")] == "QM"
    assert matrix[("S3", "E4", "C3")] == "D"


def test_method_source_resolver_preserves_explicit_template_route():
    direct = TemplateRoleCompiler().compile_method(REPORT_TEMPLATE)
    resolution = MethodSourceResolver().resolve(
        template_path=REPORT_TEMPLATE,
        baseline_manifest_path=MANIFEST,
        report_template_path=REPORT_TEMPLATE,
    )
    assert resolution.source_kind is MethodSourceKind.TEMPLATE
    assert resolution.method.audit_snapshot() == direct.audit_snapshot()
    assert resolution.report_template_hash == direct.metadata["template_hash"]


def test_method_source_resolver_separates_yaml_and_report_hashes():
    resolution = MethodSourceResolver().resolve(
        template_path=None,
        baseline_manifest_path=MANIFEST,
        report_template_path=REPORT_TEMPLATE,
    )
    assert resolution.source_kind is MethodSourceKind.YAML_BASELINE
    assert resolution.method.metadata["method_source_hash"] != resolution.report_template_hash
