from __future__ import annotations

from pathlib import Path
from typing import Any

from .yaml_loader import load_yaml_mapping, sha256_file


class YamlBaselineValidationError(ValueError):
    pass


def validate_manifest(manifest_path: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]], dict[str, str]]:
    manifest = load_yaml_mapping(manifest_path)
    base = manifest_path.parent
    for key in ("method_id", "method_version", "sources", "normalized_sources", "policies"):
        if not manifest.get(key):
            raise YamlBaselineValidationError(f"Baseline manifest missing {key}")
    paths: dict[str, str] = {}
    for section in ("sources", "normalized_sources"):
        values = manifest.get(section)
        if not isinstance(values, dict):
            raise YamlBaselineValidationError(f"Baseline manifest {section} must be a mapping")
        for role, relative in values.items():
            if role in paths:
                raise YamlBaselineValidationError(f"Duplicate canonical role: {role}")
            paths[str(role)] = str(relative)

    excluded = {str(value) for value in manifest.get("excluded_numeric_authorities", [])}
    canonical_numeric = {
        paths.get("asil", ""), paths.get("severity", ""),
        paths.get("controllability_profile", ""),
    }
    overlap = sorted(excluded.intersection(canonical_numeric))
    if overlap:
        raise YamlBaselineValidationError(
            f"An excluded numeric authority is also canonical: {overlap}"
        )

    assets: dict[str, dict[str, Any]] = {}
    hashes: dict[str, str] = {}
    expected = manifest.get("asset_hashes", {})
    if not isinstance(expected, dict):
        raise YamlBaselineValidationError("asset_hashes must be a mapping")
    for relative, expected_hash in expected.items():
        path = (base / str(relative)).resolve()
        try:
            path.relative_to(base.resolve())
        except ValueError as error:
            raise YamlBaselineValidationError(
                f"Baseline asset escapes bundle root: {relative}"
            ) from error
        if not path.is_file():
            raise YamlBaselineValidationError(f"Hashed baseline asset does not exist: {relative}")
        actual = sha256_file(path)
        if str(expected_hash).lower() != actual:
            raise YamlBaselineValidationError(f"Baseline asset hash mismatch: {relative}")
        hashes[str(relative)] = actual

    for role, relative in paths.items():
        path = (base / relative).resolve()
        try:
            path.relative_to(base.resolve())
        except ValueError as error:
            raise YamlBaselineValidationError(
                f"Baseline asset escapes bundle root: {relative}"
            ) from error
        if not path.is_file():
            raise YamlBaselineValidationError(f"Baseline asset does not exist: {relative}")
        actual = sha256_file(path)
        if relative not in expected:
            hashes[relative] = actual
        assets[role] = load_yaml_mapping(path)
    controllability = assets.get("controllability_profile", {})
    applicability = controllability.get("ttc_applicability", {})
    if applicability:
        conditions = applicability.get("all_of", {})
        if (set(applicability) != {"all_of"} or not conditions
                or any(field not in {"driver_in_vehicle", "emergency_braking_available"}
                       or not isinstance(value, bool) for field, value in conditions.items())):
            raise YamlBaselineValidationError("Unsupported controllability TTC applicability conditions")
    policy = controllability.get("unknown_override_policy")
    if policy is not None and policy not in {
        "BLOCK_TTC", "SKIP_TO_TTC", "UNSPECIFIED",
    }:
        raise YamlBaselineValidationError(
            "controllability_profile.unknown_override_policy must be "
            "BLOCK_TTC, SKIP_TO_TTC, or UNSPECIFIED"
        )
    return manifest, assets, hashes
