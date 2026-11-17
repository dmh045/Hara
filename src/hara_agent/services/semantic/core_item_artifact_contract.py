from __future__ import annotations

from copy import deepcopy
from typing import Any


ODD_SCOPE_EVIDENCE_SCHEMA = {
    "type": "array",
    "items": {"type": "object", "additionalProperties": False,
        "required": ["field", "values", "function_ids", "operating_modes", "source_location", "source_excerpt"],
        "properties": {
            "field": {"enum": ["odd_locations", "odd_road_types", "odd_weather_conditions", "odd_road_surfaces"]},
            "values": {"type": "array", "minItems": 1, "items": {"type": "string"}},
            "function_ids": {"type": "array", "items": {"type": "string"}},
            "operating_modes": {"type": "array", "items": {"type": "string"}},
            "source_location": {"type": "string"}, "source_excerpt": {"type": "string"},
        }},
}


CORE_ITEM_ARTIFACTS_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "CoreItemArtifacts",
    "type": "object",
    "additionalProperties": False,
    "required": ["item_definition", "functions"],
    "properties": {
        "item_definition": {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "system_description", "item_boundary", "operating_modes", "odd",
                "source_location", "source_excerpt", "confidence", "status",
            ],
            "properties": {
                "system_description": {"type": "string"},
                "item_boundary": {"type": "string"},
                "operating_modes": {
                    "type": ["array", "null"],
                    "items": {"type": "string"},
                },
                "odd": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": [
                        "locations", "road_types", "weather_conditions",
                        "road_surfaces", "speed_range_kph",
                    ],
                    "properties": {
                        "locations": {"type": ["array", "null"], "items": {"type": "string"}},
                        "road_types": {"type": ["array", "null"], "items": {"type": "string"}},
                        "weather_conditions": {"type": ["array", "null"], "items": {"type": "string"}},
                        "road_surfaces": {"type": ["array", "null"], "items": {"type": "string"}},
                        "scope_evidence": ODD_SCOPE_EVIDENCE_SCHEMA,
                        "speed_range_kph": {
                            "oneOf": [
                                {"type": "array", "minItems": 2, "maxItems": 2},
                                {"type": "object"},
                                {"type": "string"},
                                {"type": "null"},
                            ],
                        },
                    },
                },
                "source_location": {"type": "string"},
                "source_excerpt": {"type": "string"},
                "confidence": {"type": ["number", "string"]},
                "status": {"type": "string"},
            },
        },
        "functions": {
            "type": "array",
            "maxItems": 20,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "function_id", "name", "output", "description",
                    "preconditions", "triggers", "odd_constraints",
                    "fallback_behavior", "consequences", "source_location",
                    "source_excerpt", "confidence", "status",
                ],
                "properties": {
                    "function_id": {"type": "string"},
                    "name": {"type": "string"},
                    "output": {"type": "string"},
                    "description": {"type": "string"},
                    "preconditions": {"type": ["array", "null"], "items": {"type": "string"}},
                    "triggers": {"type": ["array", "null"], "items": {"type": "string"}},
                    "odd_constraints": {"type": ["array", "null"], "items": {"type": "string"}},
                    "fallback_behavior": {"type": ["string", "null"]},
                    "consequences": {"type": ["array", "null"], "items": {"type": "string"}},
                    "source_location": {"type": "string"},
                    "source_excerpt": {"type": "string"},
                    "confidence": {"type": ["number", "string"]},
                    "status": {"type": "string"},
                },
            },
        },
    },
}


class CoreItemArtifactsContractError(ValueError):
    def __init__(self, errors: list[str]):
        self.errors = list(errors)
        super().__init__(
            "CoreItemArtifacts contract failed: " + "; ".join(self.errors[:20])
        )


_ITEM_FIELDS = {
    "system_description", "item_boundary", "operating_modes", "odd",
    "source_location", "source_excerpt", "confidence", "status",
}
_ODD_FIELDS = {
    "locations", "road_types", "weather_conditions", "road_surfaces",
    "speed_range_kph", "scope_evidence",
}
_FUNCTION_FIELDS = {
    "function_id", "name", "output", "description", "preconditions",
    "triggers", "odd_constraints", "fallback_behavior", "consequences",
    "source_location", "source_excerpt", "confidence", "status",
}
_FUNCTION_STRING_FIELDS = (
    "function_id", "name", "output", "description", "source_location",
    "source_excerpt", "status",
)
_FUNCTION_LIST_FIELDS = (
    "preconditions", "triggers", "odd_constraints", "consequences",
)


def _string_list(
    value: Any,
    path: str,
    normalizations: list[dict[str, Any]],
    errors: list[str],
) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        stripped = value.strip()
        normalizations.append({
            "path": path,
            "from_type": "string",
            "to_type": "array",
        })
        return [stripped] if stripped else []
    if not isinstance(value, list):
        errors.append(f"{path} must be an array, string, or null; got {type(value).__name__}")
        return []
    result = []
    for index, item in enumerate(value):
        if not isinstance(item, str):
            errors.append(f"{path}[{index}] must be a string; got {type(item).__name__}")
            continue
        stripped = item.strip()
        if stripped:
            result.append(stripped)
    return result


def _required_string(value: Any, path: str, errors: list[str]) -> str:
    if not isinstance(value, str):
        errors.append(f"{path} must be a string; got {type(value).__name__}")
        return ""
    return value.strip()


def normalize_core_item_artifacts(
    payload: Any,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Validate and mechanically normalize the CoreItemArtifacts contract.

    Only representation-preserving conversions are allowed here.  In
    particular, one string in a string-list field becomes a one-element list;
    no engineering value is inferred or synthesized.
    """

    if not isinstance(payload, dict):
        raise CoreItemArtifactsContractError([
            f"top-level must be an object; got {type(payload).__name__}"
        ])
    errors: list[str] = []
    normalizations: list[dict[str, Any]] = []
    item_raw = payload.get("item_definition")
    functions_raw = payload.get("functions")
    if not isinstance(item_raw, dict):
        errors.append(
            f"item_definition must be an object; got {type(item_raw).__name__}"
        )
        item_raw = {}
    if not isinstance(functions_raw, list):
        errors.append(f"functions must be an array; got {type(functions_raw).__name__}")
        functions_raw = []

    item = {key: deepcopy(value) for key, value in item_raw.items() if key in _ITEM_FIELDS}
    unknown_item = sorted(set(item_raw) - _ITEM_FIELDS)
    if unknown_item:
        normalizations.append({
            "path": "item_definition",
            "ignored_fields": unknown_item,
        })
    for field in ("system_description", "item_boundary", "source_location", "source_excerpt", "status"):
        item[field] = _required_string(item_raw.get(field), f"item_definition.{field}", errors)
    item["operating_modes"] = _string_list(
        item_raw.get("operating_modes"), "item_definition.operating_modes",
        normalizations, errors,
    )
    odd_raw = item_raw.get("odd")
    if not isinstance(odd_raw, dict):
        errors.append(f"item_definition.odd must be an object; got {type(odd_raw).__name__}")
        odd_raw = {}
    odd = {key: deepcopy(value) for key, value in odd_raw.items() if key in _ODD_FIELDS}
    for field in ("locations", "road_types", "weather_conditions", "road_surfaces"):
        odd[field] = _string_list(
            odd_raw.get(field), f"item_definition.odd.{field}",
            normalizations, errors,
        )
    odd["speed_range_kph"] = deepcopy(odd_raw.get("speed_range_kph"))
    item["odd"] = odd
    item["confidence"] = deepcopy(item_raw.get("confidence"))

    functions = []
    for index, raw in enumerate(functions_raw):
        path = f"functions[{index}]"
        if not isinstance(raw, dict):
            errors.append(f"{path} must be an object; got {type(raw).__name__}")
            continue
        function = {
            key: deepcopy(value) for key, value in raw.items()
            if key in _FUNCTION_FIELDS
        }
        unknown = sorted(set(raw) - _FUNCTION_FIELDS)
        if unknown:
            normalizations.append({"path": path, "ignored_fields": unknown})
        for field in _FUNCTION_STRING_FIELDS:
            function[field] = _required_string(raw.get(field), f"{path}.{field}", errors)
        for field in _FUNCTION_LIST_FIELDS:
            function[field] = _string_list(
                raw.get(field), f"{path}.{field}", normalizations, errors,
            )
        fallback = raw.get("fallback_behavior")
        if fallback is None:
            function["fallback_behavior"] = ""
        elif isinstance(fallback, str):
            function["fallback_behavior"] = fallback.strip()
        elif isinstance(fallback, list) and all(isinstance(value, str) for value in fallback):
            function["fallback_behavior"] = "; ".join(
                value.strip() for value in fallback if value.strip()
            )
            normalizations.append({
                "path": f"{path}.fallback_behavior",
                "from_type": "array",
                "to_type": "string",
            })
        else:
            errors.append(
                f"{path}.fallback_behavior must be a string, string array, or null; "
                f"got {type(fallback).__name__}"
            )
            function["fallback_behavior"] = ""
        function["confidence"] = deepcopy(raw.get("confidence"))
        functions.append(function)

    if len(functions_raw) > 20:
        errors.append(f"functions must contain at most 20 items; got {len(functions_raw)}")
    if errors:
        raise CoreItemArtifactsContractError(errors)
    return {"item_definition": item, "functions": functions}, normalizations
