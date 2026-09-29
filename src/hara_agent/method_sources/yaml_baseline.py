from __future__ import annotations

from pathlib import Path
import re
from typing import Any

from hara_agent.contracts import (
    ASILMapping, ASILMatrix, CompileStatus, CompilerDiagnostic,
    CompilerDiagnosticCode, CompilerDiagnosticSeverity, ControllabilityBand,
    ControllabilityCondition, ControllabilityContract, ControllabilityOverride,
    ControllabilityProfile, ControllabilityBranchPolicy, DerivationMethod,
    ExposureAggregationPolicy,
    ExposureAtom, ExposureContract, ExposureDomainRule, ExposureMethod,
    ExposureMethodDomain, Guideword,
    GuidewordContract, FMTemplateMatch, FailureModeTaxonomyValue,
    FailureModeSelectorTaxonomy, FMTemplateScenario,
    FMScenarioTemplate, FMScenarioTemplateCatalog,
    FMTemplateSelectorAdapter, FMTemplateSelectorMapping,
    DomainTriggeringStateMapping, DomainKinematicDefaults,
    ConfirmedFallbackDimension, ScenarioDomainKnowledge,
    ScenarioMethodContract,
    MethodContract, NormativeStrength, ReportContract, RequiredFactSpec,
    RoleBinding, ScaleLevel, ScenarioConstraintDisposition,
    ScenarioConstraintPredicate, ScenarioConstraintRule, ScenarioDimension,
    ScenarioModel, SeverityContract,
    SeverityMethod, SeverityMethodBand, SeverityMethodSemantic,
    SeveritySemanticResolution, SeveritySemanticSource, SeveritySourceAuthority,
    SeveritySourceRole, SeverityScale, SourceRef, SpeedSemantic,
    StructuredRiskMethod, TemplateRole, UnknownOverridePolicy, WorkflowContract,
)
from hara_agent.contracts.method_contract import FactOrigin, FactType
from hara_agent.models import ReviewStatus

from .yaml_loader import bundle_hash, canonical_node_text
from .yaml_validation import YamlBaselineValidationError, validate_manifest


class YamlBaselineCompileError(ValueError):
    pass


class YamlBaselineCompiler:
    """Compile a governed YAML bundle into the production MethodContract."""

    compiler_version = "yaml-baseline-compiler-v1"

    @staticmethod
    def _binding(role: TemplateRole, source_hash: str, location: str) -> RoleBinding:
        return RoleBinding(
            role=role, sheet="yaml", region=location,
            detection_method="BASELINE_MANIFEST", structural_signature=(location,),
            semantic_signature=(role.value,), confidence=1.0,
            source_hash=source_hash,
        )

    @staticmethod
    def _source(bundle: str, asset: str, location: str, value: Any) -> SourceRef:
        return SourceRef.create(
            workbook=asset, template_hash=bundle, sheet="yaml", range=location,
            raw_text=canonical_node_text(value),
        )

    @staticmethod
    def _level_scales(
        bundle: str, asset: str, values: dict[str, Any], prefix: str,
    ) -> tuple[ScaleLevel, ...]:
        result = []
        for level, raw in values.items():
            description = raw.get("description", "") if isinstance(raw, dict) else str(raw)
            source = YamlBaselineCompiler._source(bundle, asset, f"{prefix}.{level}", raw)
            result.append(ScaleLevel(str(level), description, description, asset, source))
        return tuple(result)

    def _compile_scenario_method(
        self,
        *,
        bundle_hash_value: str,
        assets: dict[str, dict[str, Any]],
        paths: dict[str, str],
    ) -> ScenarioMethodContract:
        """Compile selected confirmed Scenario assets; never expose raw YAML at runtime."""
        def compile_taxonomy_values(
            *, asset_key: str, collection_key: str,
        ) -> tuple[FailureModeTaxonomyValue, ...]:
            raw_values = assets[asset_key].get(collection_key, [])
            if not isinstance(raw_values, list) or not raw_values:
                raise YamlBaselineCompileError(
                    f"{asset_key}.{collection_key} must be a non-empty list"
                )
            compiled: list[FailureModeTaxonomyValue] = []
            canonical_ids: set[str] = set()
            declared_aliases: set[str] = set()
            for index, raw_value in enumerate(raw_values):
                if not isinstance(raw_value, dict):
                    raise YamlBaselineCompileError(
                        f"{asset_key}.{collection_key}[{index}] must be a mapping"
                    )
                canonical_id = str(raw_value.get("id", "")).strip()
                aliases = raw_value.get("aliases", [])
                if not canonical_id or not isinstance(aliases, list):
                    raise YamlBaselineCompileError(
                        f"{asset_key}.{collection_key}[{index}] has invalid id or aliases"
                    )
                normalized_aliases = tuple(str(value).strip() for value in aliases)
                if any(not value for value in normalized_aliases):
                    raise YamlBaselineCompileError(
                        f"{asset_key}.{collection_key}[{index}] has blank alias"
                    )
                if canonical_id in canonical_ids or canonical_id in declared_aliases:
                    raise YamlBaselineCompileError(
                        f"{asset_key}.{collection_key} has duplicate selector value {canonical_id!r}"
                    )
                if len(set(normalized_aliases)) != len(normalized_aliases):
                    raise YamlBaselineCompileError(
                        f"{asset_key}.{collection_key}[{index}] has duplicate aliases"
                    )
                if any(alias in canonical_ids or alias in declared_aliases or alias == canonical_id
                       for alias in normalized_aliases):
                    raise YamlBaselineCompileError(
                        f"{asset_key}.{collection_key}[{index}] has colliding alias"
                    )
                canonical_ids.add(canonical_id)
                declared_aliases.update(normalized_aliases)
                compiled.append(FailureModeTaxonomyValue(
                    canonical_id=canonical_id,
                    aliases=normalized_aliases,
                    description=str(raw_value.get("description", "")),
                    source_ref=self._source(
                        bundle_hash_value, paths[asset_key],
                        f"{collection_key}[{index}]", raw_value,
                    ),
                ))
            return tuple(compiled)

        component_values = compile_taxonomy_values(
            asset_key="component_taxonomy", collection_key="categories",
        )
        failure_type_values = compile_taxonomy_values(
            asset_key="failure_type_taxonomy", collection_key="types",
        )
        selector_taxonomy = FailureModeSelectorTaxonomy(
            component_categories=component_values,
            failure_types=failure_type_values,
            component_source_ref=self._source(
                bundle_hash_value, paths["component_taxonomy"], "categories",
                assets["component_taxonomy"].get("categories", []),
            ),
            failure_type_source_ref=self._source(
                bundle_hash_value, paths["failure_type_taxonomy"], "types",
                assets["failure_type_taxonomy"].get("types", []),
            ),
        )

        template_asset = assets["scenario_templates"]
        raw_templates = template_asset.get("templates", [])
        if not isinstance(raw_templates, list):
            raise YamlBaselineCompileError("fm_scenario_templates.templates must be a list")
        geometry = template_asset.get("avp_odd", {})
        if not isinstance(geometry, dict):
            raise YamlBaselineCompileError("fm_scenario_templates.avp_odd must be a mapping")
        geometry_pairs: list[tuple[str, float]] = []
        for key in ("gap_front_m", "gap_side_m", "gap_rear_m"):
            value = geometry.get(key)
            if not isinstance(value, (int, float)):
                raise YamlBaselineCompileError(f"fm_scenario_templates.{key} must be numeric")
            geometry_pairs.append((key.removeprefix("gap_").removesuffix("_m"), float(value)))
        template_source = self._source(
            bundle_hash_value, paths["scenario_templates"], "templates", raw_templates,
        )
        templates: list[FMScenarioTemplate] = []
        template_selector_refs: dict[tuple[str, str], SourceRef] = {}
        for index, raw_template in enumerate(raw_templates, start=1):
            if not isinstance(raw_template, dict):
                raise YamlBaselineCompileError(f"fm_scenario_templates template {index} must be a mapping")
            raw_match = raw_template.get("fm_match", {})
            scenarios = raw_template.get("required_scenarios", [])
            if not isinstance(raw_match, dict) or not isinstance(scenarios, list) or not scenarios:
                raise YamlBaselineCompileError(f"fm_scenario_templates template {index} is incomplete")
            selectors = {
                "keywords": raw_match.get("keywords", []),
                "component_categories": raw_match.get("component_category", []),
                "failure_types": raw_match.get("failure_type", []),
            }
            if any(not isinstance(value, list) or not all(str(item).strip() for item in value)
                   for value in selectors.values()):
                raise YamlBaselineCompileError(f"fm_scenario_templates template {index} has invalid selectors")
            match = FMTemplateMatch(**{
                name: tuple(str(item).strip() for item in value)
                for name, value in selectors.items()
            })
            for selector_type, values in (
                ("COMPONENT_CATEGORY", match.component_categories),
                ("FAILURE_TYPE", match.failure_types),
            ):
                field = (
                    "component_category" if selector_type == "COMPONENT_CATEGORY"
                    else "failure_type"
                )
                for value in values:
                    template_selector_refs.setdefault(
                        (selector_type, value),
                        self._source(
                            bundle_hash_value, paths["scenario_templates"],
                            f"templates[{index - 1}].fm_match.{field}", raw_match,
                        ),
                    )
            compiled_scenarios: list[FMTemplateScenario] = []
            for scenario_index, raw_scenario in enumerate(scenarios):
                if not isinstance(raw_scenario, dict):
                    raise YamlBaselineCompileError(
                        f"fm_scenario_templates template {index} scenario {scenario_index} must be a mapping"
                    )
                required = ("label", "obj_type", "obj_position", "obj_distance_m", "obj_v_kph", "collision_type")
                if any(key not in raw_scenario or not str(raw_scenario[key]).strip() for key in required):
                    raise YamlBaselineCompileError(
                        f"fm_scenario_templates template {index} scenario {scenario_index} is incomplete"
                    )
                compiled_scenarios.append(FMTemplateScenario(
                    label=str(raw_scenario["label"]), obj_type=str(raw_scenario["obj_type"]),
                    obj_position=str(raw_scenario["obj_position"]),
                    obj_distance_m=float(raw_scenario["obj_distance_m"]),
                    obj_v_kph=float(raw_scenario["obj_v_kph"]),
                    collision_type=str(raw_scenario["collision_type"]),
                    source_ref=self._source(
                        bundle_hash_value, paths["scenario_templates"],
                        f"templates[{index - 1}].required_scenarios[{scenario_index}]", raw_scenario,
                    ),
                ))
            templates.append(FMScenarioTemplate(
                template_id=f"FM_TEMPLATE_{index:03d}", match=match,
                required_scenarios=tuple(compiled_scenarios),
                source_ref=self._source(
                    bundle_hash_value, paths["scenario_templates"], f"templates[{index - 1}]", raw_template,
                ), original_precedence=index,
            ))

        taxonomy_values = {
            "COMPONENT_CATEGORY": selector_taxonomy.component_categories,
            "FAILURE_TYPE": selector_taxonomy.failure_types,
        }
        taxonomy_by_id = {
            selector_type: {item.canonical_id: item for item in values}
            for selector_type, values in taxonomy_values.items()
        }
        raw_template_values = {
            selector_type: {
                value for (kind, value) in template_selector_refs if kind == selector_type
            }
            for selector_type in taxonomy_values
        }
        adapter_asset = assets["fm_template_selector_map"]
        raw_mappings = adapter_asset.get("mappings", [])
        if not isinstance(raw_mappings, list):
            raise YamlBaselineCompileError("fm_template_selector_map.mappings must be a list")
        mappings: list[FMTemplateSelectorMapping] = []
        seen_mapping_ids: set[str] = set()
        seen_mapping_sources: set[tuple[str, str]] = set()
        for index, raw_mapping in enumerate(raw_mappings):
            if not isinstance(raw_mapping, dict):
                raise YamlBaselineCompileError(
                    f"fm_template_selector_map.mappings[{index}] must be a mapping"
                )
            mapping_id = str(raw_mapping.get("mapping_id", "")).strip()
            selector_type = str(raw_mapping.get("selector_type", "")).strip()
            source_value = str(raw_mapping.get("source_template_value", "")).strip()
            canonical_target = str(raw_mapping.get("canonical_target", "")).strip()
            semantics = str(raw_mapping.get("mapping_semantics", "")).strip()
            runtime_status = str(raw_mapping.get("runtime_status", "")).strip()
            raw_candidates = raw_mapping.get("candidate_targets", [])
            candidate_targets = tuple(str(item).strip() for item in raw_candidates)
            if (
                not mapping_id
                or selector_type not in taxonomy_values
                or not source_value
                or semantics not in {"EXACT_EQUIVALENT", "FUNCTIONAL_PARENT", "AMBIGUOUS"}
                or runtime_status not in {"ACTIVE", "UNRESOLVED"}
                or not isinstance(raw_candidates, list)
            ):
                raise YamlBaselineCompileError(
                    f"fm_template_selector_map.mappings[{index}] has invalid governance fields"
                )
            key = (selector_type, source_value)
            if mapping_id in seen_mapping_ids or key in seen_mapping_sources:
                raise YamlBaselineCompileError(
                    "METHOD_SOURCE_CONFLICT: duplicate FM template selector mapping "
                    f"{mapping_id!r}/{key!r}"
                )
            if source_value not in raw_template_values[selector_type]:
                raise YamlBaselineCompileError(
                    f"fm_template_selector_map {mapping_id} references unknown template selector {source_value!r}"
                )
            if source_value in taxonomy_by_id[selector_type]:
                raise YamlBaselineCompileError(
                    f"fm_template_selector_map {mapping_id} must not remap exact canonical selector {source_value!r}"
                )
            if runtime_status == "ACTIVE":
                if (
                    semantics == "AMBIGUOUS"
                    or not canonical_target
                    or candidate_targets
                    or canonical_target not in taxonomy_by_id[selector_type]
                ):
                    raise YamlBaselineCompileError(
                        f"fm_template_selector_map {mapping_id} has invalid ACTIVE mapping"
                    )
            else:
                if (
                    semantics != "AMBIGUOUS"
                    or canonical_target
                    or len(candidate_targets) < 2
                    or len(set(candidate_targets)) != len(candidate_targets)
                    or any(value not in taxonomy_by_id[selector_type] for value in candidate_targets)
                ):
                    raise YamlBaselineCompileError(
                        f"fm_template_selector_map {mapping_id} has invalid UNRESOLVED mapping"
                    )
            seen_mapping_ids.add(mapping_id)
            seen_mapping_sources.add(key)
            mappings.append(FMTemplateSelectorMapping(
                mapping_id=mapping_id,
                selector_type=selector_type,
                source_template_value=source_value,
                canonical_target=canonical_target,
                mapping_semantics=semantics,
                runtime_status=runtime_status,
                template_source_ref=template_selector_refs[key],
                taxonomy_source_ref=(
                    taxonomy_by_id[selector_type][canonical_target].source_ref
                    if canonical_target else None
                ),
                candidate_targets=candidate_targets,
            ))
        required_mappings = {
            (selector_type, value)
            for selector_type, values in raw_template_values.items()
            for value in values
            if value not in taxonomy_by_id[selector_type]
        }
        if seen_mapping_sources != required_mappings:
            missing = sorted(required_mappings - seen_mapping_sources)
            extra = sorted(seen_mapping_sources - required_mappings)
            raise YamlBaselineCompileError(
                "METHOD_SOURCE_CONFLICT: FM template selector adapter coverage mismatch "
                f"missing={missing} extra={extra}"
            )
        selector_adapter = FMTemplateSelectorAdapter(
            mappings=tuple(mappings),
            source_ref=self._source(
                bundle_hash_value, paths["fm_template_selector_map"], "mappings", raw_mappings,
            ),
        )

        domain_asset = assets["scenario_domain"]
        raw_mapping = domain_asset.get("triggering_state_mapping", {}).get("rules", [])
        if not isinstance(raw_mapping, list):
            raise YamlBaselineCompileError("avp_low_speed.triggering_state_mapping.rules must be a list")
        trigger_mappings: list[DomainTriggeringStateMapping] = []
        for index, raw_rule in enumerate(raw_mapping, start=1):
            if not isinstance(raw_rule, dict):
                raise YamlBaselineCompileError(f"triggering state mapping {index} must be a mapping")
            speed = raw_rule.get("v_other_kmh", {})
            if not isinstance(speed, dict) or not all(str(raw_rule.get(key, "")).strip()
                                                       for key in ("collision_type", "target", "key")):
                raise YamlBaselineCompileError(f"triggering state mapping {index} is incomplete")
            trigger_mappings.append(DomainTriggeringStateMapping(
                rule_id=f"DOMAIN_TRIGGER_{index:03d}",
                collision_type=str(raw_rule["collision_type"]), target=str(raw_rule["target"]),
                v_other_min_kmh=(float(speed["min"]) if speed.get("min") is not None else None),
                v_other_max_kmh=(float(speed["max"]) if speed.get("max") is not None else None),
                triggering_state_key=str(raw_rule["key"]),
                source_ref=self._source(
                    bundle_hash_value, paths["scenario_domain"],
                    f"triggering_state_mapping.rules[{index - 1}]", raw_rule,
                ),
            ))
        kinematics = domain_asset.get("standard_kinematic_values", {})
        if not isinstance(kinematics, dict):
            raise YamlBaselineCompileError("avp_low_speed.standard_kinematic_values must be a mapping")
        gaps = kinematics.get("gap_m", {})
        values = kinematics.get("v_other_kmh", [])
        if not isinstance(gaps, dict) or not isinstance(values, list) or not all(
            isinstance(value, (int, float)) for value in values
        ):
            raise YamlBaselineCompileError("avp_low_speed standard kinematic values are invalid")
        kinematic_defaults = DomainKinematicDefaults(
            v_other_kmh=tuple(float(value) for value in values),
            gap_m=tuple((str(key), float(value)) for key, value in sorted(gaps.items())),
            reaction_delay_s=(
                float(kinematics["default_reaction_delay_s"])
                if kinematics.get("default_reaction_delay_s") is not None else None
            ),
            source_ref=self._source(
                bundle_hash_value, paths["scenario_domain"], "standard_kinematic_values", kinematics,
            ),
        )
        template_gaps = dict(geometry_pairs)
        domain_gaps = dict(kinematic_defaults.gap_m)
        for key in sorted(set(template_gaps).intersection(domain_gaps)):
            if template_gaps[key] != domain_gaps[key]:
                raise YamlBaselineCompileError(
                    "METHOD_SOURCE_CONFLICT: FM template AVP ODD geometry "
                    f"{key}={template_gaps[key]} conflicts with domain default "
                    f"{key}={domain_gaps[key]}"
                )
        seen_trigger_keys: dict[tuple[str, str, float | None, float | None], str] = {}
        for item in trigger_mappings:
            key = (item.collision_type, item.target, item.v_other_min_kmh, item.v_other_max_kmh)
            prior = seen_trigger_keys.setdefault(key, item.triggering_state_key)
            if prior != item.triggering_state_key:
                raise YamlBaselineCompileError(
                    "METHOD_SOURCE_CONFLICT: overlapping triggering-state mappings "
                    f"for {key} resolve to both {prior!r} and {item.triggering_state_key!r}"
                )
        raw_fallback = domain_asset.get("fallback_scenario_dimensions", {})
        if not isinstance(raw_fallback, dict):
            raise YamlBaselineCompileError("avp_low_speed.fallback_scenario_dimensions must be a mapping")
        fallback_dimensions = tuple(
            ConfirmedFallbackDimension(
                source_dimension=str(name), terms=tuple(str(term) for term in terms),
                target_dimension="",
                source_ref=self._source(
                    bundle_hash_value, paths["scenario_domain"],
                    f"fallback_scenario_dimensions.{name}", terms,
                ),
            )
            for name, terms in sorted(raw_fallback.items())
            if isinstance(terms, list) and all(str(term).strip() for term in terms)
        )
        if len(fallback_dimensions) != len(raw_fallback):
            raise YamlBaselineCompileError("avp_low_speed fallback dimensions must contain non-empty term lists")
        domain_source = self._source(bundle_hash_value, paths["scenario_domain"], "domain", domain_asset)
        return ScenarioMethodContract(
            fm_template_catalog=FMScenarioTemplateCatalog(
                templates=tuple(templates), odd_geometry_m=tuple(geometry_pairs),
                source_ref=template_source,
            ),
            failure_mode_selector_taxonomy=selector_taxonomy,
            fm_template_selector_adapter=selector_adapter,
            domain_knowledge=ScenarioDomainKnowledge(
                triggering_state_mappings=tuple(trigger_mappings),
                kinematic_defaults=kinematic_defaults,
                fallback_dimensions=fallback_dimensions,
                numeric_sections=("exposure", "controllability", "severity"),
                source_ref=domain_source,
            ),
            example_catalogs=tuple(
                self._source(bundle_hash_value, paths[role], "examples", assets[role])
                for role in ("coupling_examples", "infeasible_examples")
            ),
        )

    def compile(
        self, manifest_path: str | Path, *, report_contract: ReportContract,
    ) -> MethodContract:
        path = Path(manifest_path).expanduser().resolve()
        try:
            manifest, assets, hashes = validate_manifest(path)
            source_hash = bundle_hash(manifest, hashes)
            return self._compile_valid(manifest, assets, hashes, source_hash, report_contract)
        except (KeyError, TypeError, ValueError, YamlBaselineValidationError) as error:
            raise YamlBaselineCompileError(str(error)) from error

    def _compile_valid(
        self, manifest: dict[str, Any], assets: dict[str, dict[str, Any]],
        hashes: dict[str, str], source_hash: str, report_contract: ReportContract,
    ) -> MethodContract:
        source_paths = {**manifest["sources"], **manifest["normalized_sources"]}
        refs: list[SourceRef] = []
        project_policy = assets.get("project_analysis_policy", {})
        compiled_project_policy: dict[str, Any] = {}
        if project_policy:
            selection = project_policy.get("input_selection", {})
            speed = project_policy.get("ego_speed_point_selection", {})
            controllability = project_policy.get("controllability", {})
            decisions = project_policy.get("decisions", {})
            if (
                not str(project_policy.get("policy_id", "")).strip()
                or not str(project_policy.get("version", "")).strip()
                or not isinstance(selection, dict)
                or not str(selection.get("policy_id", "")).strip()
                or not isinstance(selection.get("excluded_sections_for_scoring"), list)
                or not isinstance(speed, dict)
                or speed.get("choice") != "UPPER_CLOSED_BOUND_OF_EFFECTIVE_OPERATION_RANGE"
                or speed.get("require_finite_closed_upper_bound") is not True
                or speed.get("require_source_and_scope") is not True
                or not str(speed.get("rule_id", "")).strip()
                or not isinstance(decisions, dict)
                or set(decisions) != {"A1", "A2", "A3", "A4", "A5"}
                or not isinstance(controllability, dict)
                or (controllability and (
                    controllability.get("unknown_override_policy") not in {
                        "BLOCK_TTC", "SKIP_TO_TTC", "UNSPECIFIED"
                    }
                    or controllability.get("status") != "CONFIRMED_FOR_CURRENT_PROJECT"
                    or not str(controllability.get("basis", "")).strip()
                ))
            ):
                raise YamlBaselineCompileError("Invalid governed project analysis policy")
            compiled_project_policy = {
                "policy_id": str(project_policy["policy_id"]),
                "version": str(project_policy["version"]),
                "project_scope": str(project_policy.get("project_scope", "")),
                "source_kind": str(project_policy.get("source_kind", "")),
                "source_note": str(project_policy.get("source_note", "")),
                "input_selection": dict(selection),
                "ego_speed_point_selection": dict(speed),
                "controllability": dict(controllability),
                "decisions": dict(decisions),
                "candidate_defaults": list(project_policy.get("candidate_defaults", [])),
                "release_approval": str(project_policy.get("release_approval", "PENDING")),
                "source_ref": {
                    "source_type": "project_analysis_policy",
                    "source_id": f"{project_policy['policy_id']}@{source_hash}",
                    "location": source_paths["project_analysis_policy"],
                    "excerpt": str(project_policy["policy_id"]),
                },
            }
            refs.append(self._source(
                source_hash, source_paths["project_analysis_policy"],
                "project_analysis_policy", project_policy,
            ))
        scenario_method = self._compile_scenario_method(
            bundle_hash_value=source_hash, assets=assets, paths=source_paths,
        )

        guide_asset = assets["guidewords"]
        guide_ref = self._source(source_hash, source_paths["guidewords"], "guidewords", guide_asset["guidewords"])
        refs.append(guide_ref)
        guidewords = tuple(
            Guideword(
                guideword_id=str(item["id"]), name=str(item["name"]),
                description=str(item["description"]), order=index,
                source_ref=self._source(
                    source_hash, source_paths["guidewords"], f"guidewords[{index - 1}]", item,
                ),
            )
            for index, item in enumerate(guide_asset["guidewords"], start=1)
        )
        if len({item.guideword_id for item in guidewords}) != len(guidewords):
            raise YamlBaselineCompileError("Guideword IDs must be unique")

        atom_asset = assets["exposure_atoms"]
        raw_atoms = atom_asset.get("atoms", [])
        atom_ids = [str(item.get("id", "")) for item in raw_atoms]
        if not atom_ids or len(set(atom_ids)) != len(atom_ids):
            raise YamlBaselineCompileError("VDA atom IDs must be present and unique")
        allowed_dimensions = {
            "WHERE", "WHERE_MICRO", "ROAD", "EGO_ACTION", "EGO_X_ROAD",
            "TRAFFIC_PATTERN", "EGO_DYNAMICS", "OBJECT",
        }
        exposure_atoms = []
        scenario_atom_catalog = []
        grouped: dict[str, list[str]] = {key: [] for key in allowed_dimensions}
        for index, item in enumerate(raw_atoms):
            dimensions = tuple(str(item["dim"]).split("+"))
            if any(value not in allowed_dimensions for value in dimensions):
                raise YamlBaselineCompileError(
                    f"Atom {item['id']} has an unsupported dimension: {dimensions}"
                )
            source = self._source(
                source_hash, source_paths["exposure_atoms"], f"atoms[{index}]", item,
            )
            refs.append(source)
            duration = str(item.get("e_t") or "")
            frequency = str(item.get("e_f") or "")
            duration = duration if duration in {"E0", "E1", "E2", "E3", "E4"} else ""
            frequency = frequency if frequency in {"E0", "E1", "E2", "E3", "E4"} else ""
            exposure_atoms.append(ExposureAtom(
                atom_id=str(item["id"]), dimensions=dimensions,
                label=str(item["label"]), duration_level=duration,
                frequency_level=frequency, source_ref=source,
            ))
            compound = item.get("compound", {})
            compound = compound if isinstance(compound, dict) else {}
            fills = compound.get("fills", dimensions)
            fills = tuple(map(str, fills)) if isinstance(fills, list) else dimensions
            spec = assets["atom_spec"].get("specs", {}).get(
                str(item.get("actual_id") or item["id"]),
                assets["atom_spec"].get("specs", {}).get(str(item["id"]), {}),
            )
            spec = spec if isinstance(spec, dict) else {}
            dynamics = compound.get("ego_dynamics", {})
            dynamics = dynamics if isinstance(dynamics, dict) else {}
            dynamics = dynamics or (
                spec.get("ego_dynamics", {})
                if isinstance(spec.get("ego_dynamics", {}), dict) else {}
            )
            slope = compound.get("slope", {})
            slope = slope if isinstance(slope, dict) else {}
            slope = slope or (
                spec.get("slope", {})
                if isinstance(spec.get("slope", {}), dict) else {}
            )
            slope = dict(slope)
            if "pct_min" in slope and "min_inclusive" not in slope:
                minimum = str(slope["pct_min"])
                label = str(item.get("label", ""))
                slope["min_inclusive"] = not bool(re.search(
                    rf"(?:slope\s*>\s*{re.escape(minimum)}|"
                    rf"{re.escape(minimum)}\s*%?\s*<\s*slope)",
                    label, re.IGNORECASE,
                ))
            physical_semantics = {
                key: value for key, value in {
                    "ego_dynamics": dynamics,
                    "slope": slope,
                    "object": compound.get("object", spec.get("object", {})),
                    "traffic_pattern": compound.get(
                        "traffic_pattern", spec.get("traffic_pattern", {})
                    ),
                }.items() if isinstance(value, dict) and value
            }
            speed_range = dynamics.get("v_range_kph")
            if not isinstance(speed_range, list) or len(speed_range) != 2:
                lower = dynamics.get("v_min_kph")
                upper = dynamics.get("v_max_kph")
                speed_range = [lower, upper] if lower is not None or upper is not None else []
            scenario_atom_catalog.append({
                "atom_id": str(item["id"]),
                "actual_id": str(item.get("actual_id") or item["id"]),
                "label": str(item["label"]),
                "aliases": [str(value) for value in item.get("aliases", [])],
                "filled_dimensions": list(fills),
                "v1": str(item.get("v1", "")),
                "v2": str(item.get("v2", "")),
                "v2_proper": str(item.get("v2_proper", "")),
                "compound": compound,
                "physical_semantics": physical_semantics,
                "speed_range_kph": speed_range,
                "source_asset": source_paths["exposure_atoms"],
                "source_rule": f"atoms[{index}]",
                "source_tag": str(item.get("src", "")),
            })
            display = f"{item['id']} | {item['label']}"
            for dimension in dimensions:
                grouped[dimension].append(display)

        execution_order = [
            "WHERE", "ROAD", "EGO_ACTION", "EGO_X_ROAD",
            "TRAFFIC_PATTERN", "EGO_DYNAMICS", "OBJECT",
        ]
        scenario_dimensions = []
        for dimension in execution_order:
            values = tuple(dict.fromkeys(grouped.get(str(dimension), [])))
            if not values:
                raise YamlBaselineCompileError(f"Scenario dimension has no atoms: {dimension}")
            source = self._source(
                source_hash, source_paths["scenario_structure"],
                f"dimensions.{dimension}", assets["scenario_structure"].get("dimensions", {}),
            )
            scenario_dimensions.append(ScenarioDimension(
                dimension_id=f"DIM-{dimension}", canonical_name=str(dimension),
                display_name=str(dimension), values=values, unit="", source_ref=source,
                semantics="VDA702_ATOM_ID_LABEL",
            ))

        alias_asset = assets["scenario_aliases"]
        raw_aliases = alias_asset.get("aliases", [])
        if not isinstance(raw_aliases, list):
            raise YamlBaselineCompileError("scenario_aliases.aliases must be a list")
        valid_dimensions = {item.canonical_name for item in scenario_dimensions}
        approved_scenario_aliases: list[dict[str, Any]] = []
        seen_alias_ids: set[str] = set()
        for index, raw_alias in enumerate(raw_aliases):
            if not isinstance(raw_alias, dict):
                raise YamlBaselineCompileError(
                    f"scenario alias {index} must be a mapping"
                )
            alias_id = str(raw_alias.get("alias_id", "")).strip()
            dimension = str(raw_alias.get("dimension", "")).strip()
            source_term = str(raw_alias.get("source_term", "")).strip()
            status = str(raw_alias.get("status", "")).strip()
            if not alias_id or alias_id in seen_alias_ids:
                raise YamlBaselineCompileError(
                    f"scenario alias {index} has a missing or duplicate alias_id"
                )
            seen_alias_ids.add(alias_id)
            if dimension not in valid_dimensions:
                raise YamlBaselineCompileError(
                    f"scenario alias {alias_id} has an unknown dimension {dimension!r}"
                )
            if not source_term:
                raise YamlBaselineCompileError(
                    f"scenario alias {alias_id} has no source_term"
                )
            if status not in {"PROPOSED", "APPROVED", "REJECTED"}:
                raise YamlBaselineCompileError(
                    f"scenario alias {alias_id} has an invalid status {status!r}"
                )
            if status != "APPROVED":
                continue
            target = raw_alias.get("canonical_target")
            if not isinstance(target, dict):
                raise YamlBaselineCompileError(
                    f"APPROVED scenario alias {alias_id} has no canonical_target"
                )
            atom_id = str(target.get("atom_id", "")).strip()
            atom = next(
                (item for item in scenario_atom_catalog if item["atom_id"] == atom_id),
                None,
            )
            if atom is None:
                raise YamlBaselineCompileError(
                    f"APPROVED scenario alias {alias_id} references unknown atom {atom_id!r}"
                )
            if dimension not in atom["filled_dimensions"]:
                raise YamlBaselineCompileError(
                    f"APPROVED scenario alias {alias_id} has DIMENSION_MISMATCH: "
                    f"{atom_id} does not fill {dimension}"
                )
            label = str(target.get("canonical_label", "")).strip()
            if label and label != atom["label"]:
                raise YamlBaselineCompileError(
                    f"APPROVED scenario alias {alias_id} canonical_label does not match {atom_id}"
                )
            approval = raw_alias.get("approval")
            if not isinstance(approval, dict) or not str(
                approval.get("reviewer", "")
            ).strip() or not str(approval.get("reviewed_at", "")).strip():
                raise YamlBaselineCompileError(
                    f"APPROVED scenario alias {alias_id} requires reviewer and reviewed_at"
                )
            approved_scenario_aliases.append({
                "alias_id": alias_id,
                "dimension": dimension,
                "source_term": source_term,
                "target_atom_id": atom_id,
                "canonical_target": {
                    "atom_id": atom_id,
                    "canonical_label": atom["label"],
                },
                "mapping_type": str(raw_alias.get("mapping_type", "")),
                "provenance": dict(raw_alias.get("provenance", {})),
                "rationale": str(raw_alias.get("rationale", "")),
                "approval": dict(approval),
                "source_asset": source_paths["scenario_aliases"],
                "source_rule": f"aliases[{index}]",
            })

        coverage_asset = assets["scenario_coverage_rules"]
        raw_knowledge_sources = coverage_asset.get("knowledge_sources", [])
        if not isinstance(raw_knowledge_sources, list):
            raise YamlBaselineCompileError(
                "scenario_coverage_rules.knowledge_sources must be a list"
            )
        allowed_knowledge_classes = {
            "NORMATIVE_METHOD_RULE", "PROJECT_DERIVED_ENGINEERING_RULE",
            "EXAMPLE_TEMPLATE", "SEMANTIC_HINT_ONLY", "NUMERIC_AUTHORITY",
            "SCENARIO_TEMPLATE_CONSTRAINT", "DOMAIN_RULE",
        }
        coverage_knowledge_sources: list[dict[str, str]] = []
        for index, raw_source in enumerate(raw_knowledge_sources):
            if not isinstance(raw_source, dict):
                raise YamlBaselineCompileError(
                    f"scenario coverage knowledge source {index} must be a mapping"
                )
            asset = str(raw_source.get("source_asset", "")).strip()
            classification = str(raw_source.get("classification", "")).strip()
            rationale = str(raw_source.get("rationale", "")).strip()
            if not asset or classification not in allowed_knowledge_classes or not rationale:
                raise YamlBaselineCompileError(
                    f"scenario coverage knowledge source {index} is incomplete or invalid"
                )
            coverage_knowledge_sources.append({
                "source_asset": asset,
                "classification": classification,
                "rationale": rationale,
            })

        raw_coverage_rules = coverage_asset.get("rules", [])
        if not isinstance(raw_coverage_rules, list):
            raise YamlBaselineCompileError("scenario_coverage_rules.rules must be a list")
        approved_coverage_rules: list[dict[str, Any]] = []
        coverage_status_counts = {"PROPOSED": 0, "APPROVED": 0, "REJECTED": 0}
        seen_coverage_rule_ids: set[str] = set()
        selector_fields = {
            "name", "output", "description", "preconditions", "triggers", "odd_constraints",
        }
        for index, raw_rule in enumerate(raw_coverage_rules):
            if not isinstance(raw_rule, dict):
                raise YamlBaselineCompileError(
                    f"scenario coverage rule {index} must be a mapping"
                )
            rule_id = str(raw_rule.get("coverage_rule_id", "")).strip()
            status = str(raw_rule.get("status", "")).strip()
            scope = raw_rule.get("scope")
            provenance = raw_rule.get("provenance")
            rationale = str(raw_rule.get("rationale", "")).strip()
            if not rule_id or rule_id in seen_coverage_rule_ids:
                raise YamlBaselineCompileError(
                    f"scenario coverage rule {index} has a missing or duplicate coverage_rule_id"
                )
            seen_coverage_rule_ids.add(rule_id)
            if status not in coverage_status_counts:
                raise YamlBaselineCompileError(
                    f"scenario coverage rule {rule_id} has an invalid status {status!r}"
                )
            coverage_status_counts[status] += 1
            if not isinstance(scope, dict) or not isinstance(provenance, dict) or not rationale:
                raise YamlBaselineCompileError(
                    f"scenario coverage rule {rule_id} requires scope, provenance and rationale"
                )
            selector = scope.get("function_selector", {})
            modes = scope.get("operating_modes", [])
            if not isinstance(selector, dict) or not isinstance(modes, list):
                raise YamlBaselineCompileError(
                    f"scenario coverage rule {rule_id} has an invalid scope"
                )
            if any(field not in selector_fields for field in selector):
                raise YamlBaselineCompileError(
                    f"scenario coverage rule {rule_id} has an unknown function selector"
                )
            normalized_selector: dict[str, list[str]] = {}
            for field, terms in selector.items():
                if not isinstance(terms, list) or not all(str(term).strip() for term in terms):
                    raise YamlBaselineCompileError(
                        f"scenario coverage rule {rule_id} selector {field} must be non-empty strings"
                    )
                normalized_selector[field] = [str(term).strip() for term in terms]
            if not all(str(mode).strip() for mode in modes):
                raise YamlBaselineCompileError(
                    f"scenario coverage rule {rule_id} operating_modes must be non-empty strings"
                )
            dimension_sets: dict[str, list[str]] = {}
            for field in (
                "required_dimensions", "optional_dimensions", "not_applicable_dimensions",
            ):
                values = raw_rule.get(field, [])
                if not isinstance(values, list) or any(
                    str(value).strip() not in valid_dimensions for value in values
                ):
                    raise YamlBaselineCompileError(
                        f"scenario coverage rule {rule_id} has an invalid {field}"
                    )
                dimension_sets[field] = [str(value).strip() for value in values]
                if len(set(dimension_sets[field])) != len(dimension_sets[field]):
                    raise YamlBaselineCompileError(
                        f"scenario coverage rule {rule_id} duplicates {field}"
                    )
            all_dimensions = [
                item for values in dimension_sets.values() for item in values
            ]
            if len(set(all_dimensions)) != len(all_dimensions):
                raise YamlBaselineCompileError(
                    f"scenario coverage rule {rule_id} assigns a dimension to multiple statuses"
                )
            provenance_class = str(provenance.get("classification", "")).strip()
            source_asset = str(provenance.get("source_asset", "")).strip()
            source_rule = str(provenance.get("source_rule", "")).strip()
            if not source_asset or not source_rule or provenance_class not in {
                "NORMATIVE_METHOD_RULE", "PROJECT_DERIVED_ENGINEERING_RULE",
                "EXAMPLE_TEMPLATE", "SEMANTIC_HINT_ONLY", "NUMERIC_AUTHORITY",
                "SCENARIO_TEMPLATE_CONSTRAINT", "DOMAIN_RULE",
            }:
                raise YamlBaselineCompileError(
                    f"scenario coverage rule {rule_id} has incomplete provenance"
                )
            if status != "APPROVED":
                continue
            if not normalized_selector and not modes:
                raise YamlBaselineCompileError(
                    f"APPROVED scenario coverage rule {rule_id} needs a function or context selector"
                )
            if not all_dimensions:
                raise YamlBaselineCompileError(
                    f"APPROVED scenario coverage rule {rule_id} declares no dimensions"
                )
            if provenance_class not in {
                "NORMATIVE_METHOD_RULE", "PROJECT_DERIVED_ENGINEERING_RULE",
            }:
                raise YamlBaselineCompileError(
                    f"APPROVED scenario coverage rule {rule_id} has non-executable provenance"
                )
            approval = raw_rule.get("approval")
            if not isinstance(approval, dict) or not str(
                approval.get("reviewer", "")
            ).strip() or not str(approval.get("reviewed_at", "")).strip():
                raise YamlBaselineCompileError(
                    f"APPROVED scenario coverage rule {rule_id} requires reviewer and reviewed_at"
                )
            approved_coverage_rules.append({
                "coverage_rule_id": rule_id,
                "scope": {
                    "function_selector": normalized_selector,
                    "operating_modes": [str(mode).strip() for mode in modes],
                },
                **dimension_sets,
                "provenance": dict(provenance),
                "rationale": rationale,
                "approval": dict(approval),
                "source_asset": source_paths["scenario_coverage_rules"],
                "source_rule": f"rules[{index}]",
            })
        refs.append(self._source(
            source_hash, source_paths["scenario_coverage_rules"],
            "scenario_coverage_rules", coverage_asset,
        ))

        constraint_rules = []
        raw_constraint_rules = assets["scenario_constraints"].get("rules", [])
        if not isinstance(raw_constraint_rules, list):
            raise YamlBaselineCompileError("scenario_constraints.rules must be a list")
        for index, raw_rule in enumerate(raw_constraint_rules):
            if not isinstance(raw_rule, dict):
                raise YamlBaselineCompileError(
                    f"scenario_constraints.rules[{index}] must be a mapping"
                )
            raw_predicates = raw_rule.get("predicates", [])
            if not isinstance(raw_predicates, list) or not raw_predicates:
                raise YamlBaselineCompileError(
                    f"scenario constraint {raw_rule.get('id', index)!r} has no predicates"
                )
            predicates = []
            for predicate in raw_predicates:
                if not isinstance(predicate, dict):
                    raise YamlBaselineCompileError("scenario constraint predicate must be a mapping")
                values = predicate.get("values", [])
                if not isinstance(values, list):
                    raise YamlBaselineCompileError("scenario constraint predicate values must be a list")
                predicates.append(ScenarioConstraintPredicate(
                    dimension=str(predicate.get("dimension", "")),
                    values=tuple(map(str, values)),
                ))
            try:
                disposition = ScenarioConstraintDisposition(
                    str(raw_rule.get("disposition", ""))
                )
                normative_strength = NormativeStrength(
                    str(raw_rule.get("normative_strength", "NORMATIVE"))
                )
            except ValueError as error:
                raise YamlBaselineCompileError(
                    f"scenario constraint {raw_rule.get('id', index)!r} is invalid: {error}"
                ) from error
            constraint_rules.append(ScenarioConstraintRule(
                rule_id=str(raw_rule.get("id", "")),
                predicates=tuple(predicates),
                disposition=disposition,
                reason=str(raw_rule.get("reason", "")),
                normative_strength=normative_strength,
                source_ref=self._source(
                    source_hash, source_paths["scenario_constraints"],
                    f"rules[{index}]", raw_rule,
                ),
                executable=bool(raw_rule.get("executable", True)),
            ))

        severity_asset = assets["severity"]
        severity_ref = self._source(source_hash, source_paths["severity"], "s_by_relative_velocity", severity_asset["s_by_relative_velocity"])
        manifest_severity_ref = self._source(
            source_hash, "manifest.yaml", "policies.severity_primary_method",
            manifest["policies"]["severity_primary_method"],
        )
        delta_v_annotation_ref = self._source(
            source_hash, source_paths["severity"],
            "comments.preceding_s_by_relative_velocity",
            "ΔV threshold examples and collision annotations",
        )
        aeb_delta_v_ref = self._source(
            source_hash, source_paths["severity"], "s_by_relative_velocity.aeb_delta_v",
            severity_asset["s_by_relative_velocity"]["aeb_delta_v"],
        )
        severity_semantic = SeverityMethodSemantic(
            source_status="CONFIRMED",
            source_named_semantic="RELATIVE_VELOCITY",
            compiled_semantic=SpeedSemantic.RELATIVE_SPEED,
            semantic_resolution=SeveritySemanticResolution.CONFIRMED_RELATIVE_VELOCITY,
            source_elements=(
                SeveritySemanticSource(
                    source_ref=manifest_severity_ref,
                    field_name="severity_primary_method",
                    field_type="YAML scalar",
                    role=SeveritySourceRole.MACHINE_READABLE_CONFIGURATION,
                    authority=SeveritySourceAuthority.PRIMARY,
                    semantic="RELATIVE_VELOCITY",
                    compiler_consumer="YamlBaselineCompiler",
                ),
                SeveritySemanticSource(
                    source_ref=severity_ref,
                    field_name="s_by_relative_velocity",
                    field_type="YAML mapping key",
                    role=SeveritySourceRole.MACHINE_READABLE_NORMATIVE,
                    authority=SeveritySourceAuthority.PRIMARY,
                    semantic="RELATIVE_VELOCITY",
                    compiler_consumer="YamlBaselineCompiler",
                    runtime_consumer="SeverityMethodExecutor",
                ),
                SeveritySemanticSource(
                    source_ref=aeb_delta_v_ref,
                    field_name="s_by_relative_velocity.aeb_delta_v",
                    field_type="YAML mapping key",
                    role=SeveritySourceRole.MACHINE_READABLE_NORMATIVE,
                    authority=SeveritySourceAuthority.SUPPORTING,
                    semantic="DELTA_V",
                    compiler_consumer="",
                    runtime_consumer="",
                ),
                SeveritySemanticSource(
                    source_ref=delta_v_annotation_ref,
                    field_name="ΔV threshold annotations",
                    field_type="YAML comment",
                    role=SeveritySourceRole.COMMENT_ONLY,
                    authority=SeveritySourceAuthority.NON_NORMATIVE,
                    semantic="DELTA_V",
                    compiler_consumer="",
                ),
            ),
            diagnostic_codes=(
                "SEVERITY_SEMANTIC_ROLE_PRESERVING_TRANSLATION",
                "SPECIALIZED_AEB_DELTA_V_TABLE_NOT_SELECTED",
            ),
            annotation_semantics=("DELTA_V",),
        )
        refs.append(severity_ref)
        severity_bands: list[SeverityMethodBand] = []
        relative_tables = severity_asset["s_by_relative_velocity"]
        table_items: list[tuple[str, str, list[dict[str, Any]]]] = []
        for collision_type, rows in relative_tables["vehicle_collision"].items():
            table_items.append(("vehicle", str(collision_type), rows))
        for group in ("pedestrian_collision", "cyclist_collision", "motorcycle_collision"):
            table_items.append((group.removesuffix("_collision"), "any", relative_tables[group]))
        for group, collision_type, rows in table_items:
            previous: float | None = None
            for index, row in enumerate(rows):
                lower = float(row["min"]) if "min" in row else previous
                upper = float(row["max"]) if "max" in row else None
                source = self._source(
                    source_hash, source_paths["severity"],
                    f"s_by_relative_velocity.{group}.{collision_type}[{index}]", row,
                )
                severity_bands.append(SeverityMethodBand(
                    rule_id=f"S-{group}-{collision_type}-{index + 1}",
                    collision_group=group, collision_type=collision_type,
                    result=str(row["s"]), lower_kph=lower, upper_kph=upper,
                    lower_inclusive=True, upper_inclusive=False, source_ref=source,
                ))
                previous = upper

        domain_rules = []
        for index, item in enumerate(assets["exposure_domain"]["rules"]):
            source = self._source(
                source_hash, source_paths["exposure_domain"], f"rules[{index}]", item,
            )
            domain_rules.append(ExposureDomainRule(
                rule_id=str(item["rule_id"]),
                component_categories=tuple(map(str, item["match_component_category"])),
                domain=ExposureMethodDomain(str(item["e_dimension"])),
                source_ref=source,
            ))

        c_asset = assets["controllability_profile"]
        c_ref = self._source(source_hash, source_paths["controllability_profile"], "ttc_thresholds", c_asset["ttc_thresholds"])
        refs.append(c_ref)
        bands = []
        lower: float | None = None
        for index, item in enumerate(c_asset["ttc_thresholds"]["bands"]):
            upper = item.get("upper_bound")
            upper_value = float(upper) if upper is not None else None
            bands.append(ControllabilityBand(
                rule_id=f"C-TTC-{index + 1}", result=str(item["result"]),
                source_ref=self._source(
                    source_hash, source_paths["controllability_profile"],
                    f"ttc_thresholds.bands[{index}]", item,
                ),
                lower_ttc_s=lower, upper_ttc_s=upper_value,
                lower_inclusive=False if lower is not None else True,
                upper_inclusive=bool(item.get("upper_inclusive", False)),
                review_status=ReviewStatus.FINALIZED,
            ))
            lower = upper_value
        self._validate_c_bands(bands)

        aliases = {
            str(key): str(value)
            for key, value in assets["controllability_aliases"]["aliases"].items()
        }
        overrides = []
        for priority, item in enumerate(c_asset.get("override_rules", []), start=1):
            expression = str(item["when"])
            separator = " OR " if " OR " in expression else " AND "
            conditions = []
            for clause in expression.split(separator):
                field, operator, raw = clause.strip().split()
                if operator != "==" or raw.casefold() not in {"true", "false"}:
                    raise YamlBaselineCompileError(f"Unsupported C override expression: {expression}")
                conditions.append(ControllabilityCondition(
                    field=aliases.get(field, field), expected=raw.casefold() == "true",
                ))
            source = self._source(
                source_hash, source_paths["controllability_profile"],
                f"override_rules[{priority - 1}]", item,
            )
            overrides.append(ControllabilityOverride(
                rule_id=str(item["id"]), any_of=tuple(conditions) if separator == " OR " else (),
                all_of=tuple(conditions) if separator == " AND " else (),
                result=str(item["result"]), priority=priority, source_ref=source,
            ))

        iso = assets["asil"]
        asil_mappings = []
        for severity in ("S0", "S1", "S2", "S3"):
            for exposure in ("E0", "E1", "E2", "E3", "E4"):
                for controllability in ("C0", "C1", "C2", "C3"):
                    if "0" in (severity[1:], exposure[1:], controllability[1:]):
                        value = str(manifest["policies"]["asil_zero_short_circuit"])
                        location = "policies.asil_zero_short_circuit"
                    else:
                        value = str(iso["asil_matrix"][severity][exposure][controllability])
                        location = f"asil_matrix.{severity}.{exposure}.{controllability}"
                    asil_mappings.append(ASILMapping(
                        severity, exposure, controllability, value,
                        self._source(source_hash, source_paths["asil"], location, value),
                    ))
        if len(asil_mappings) != 80:
            raise YamlBaselineCompileError("ASIL matrix expansion must contain 80 cells")

        scale_sources = self._level_scales(source_hash, source_paths["severity"], severity_asset["s_levels"], "s_levels")
        exposure_levels = tuple(
            ScaleLevel(level, str(text), str(text), source_paths["exposure_atoms"],
                       self._source(source_hash, source_paths["exposure_atoms"], f"e_value_levels.{level}", text))
            for level, text in atom_asset["e_value_levels"].items()
        )
        c_levels = tuple(
            ScaleLevel(value, value, value, source_paths["controllability_profile"], c_ref)
            for value in ("C0", "C1", "C2", "C3")
        )
        policy_asset = assets["exposure_policy"]
        policy_source = self._source(
            source_hash, source_paths["exposure_policy"], "policy", policy_asset,
        )
        aggregation_policy = ExposureAggregationPolicy(
            policy_id=str(policy_asset["policy_id"]),
            minimum_level=int(policy_asset["minimum_level"]),
            all_highest_operand=str(policy_asset["all_highest"]["operand"]),
            all_highest_result=str(policy_asset["all_highest"]["result"]),
            mixed_high_operands=tuple(map(str, policy_asset["mixed_high"]["operands"])),
            mixed_high_result=str(policy_asset["mixed_high"]["result"]),
            mixed_strategy=str(policy_asset["mixed_levels"]["strategy"]),
            independent_decrement=int(policy_asset["equal_levels"]["independent_decrement"]),
            dependent_decrement=int(policy_asset["equal_levels"]["dependent_decrement"]),
            source_ref=policy_source,
        )
        exposure_method = ExposureMethod(
            method_id="fusa_exposure_v1", atoms=tuple(exposure_atoms),
            domain_rules=tuple(domain_rules),
            strong_couplings=tuple(
                tuple(map(str, pair))
                for pair in assets["scenario_structure"]["dimension_dependency"]["strong_coupling"]
            ),
            aggregation_policy=aggregation_policy,
            dimension_fallback_policy=str(manifest["policies"]["exposure_dimension_fallback"]),
            source_refs=(domain_rules[0].source_ref, exposure_atoms[0].source_ref),
        )
        project_c_policy = compiled_project_policy.get("controllability", {})
        raw_unknown_override_policy = project_c_policy.get(
            "unknown_override_policy", c_asset.get("unknown_override_policy"),
        )
        if (
            project_c_policy and c_asset.get("unknown_override_policy") is not None
            and c_asset["unknown_override_policy"] != raw_unknown_override_policy
        ):
            raise YamlBaselineCompileError(
                "Project controllability policy conflicts with Method profile"
            )
        compiled_unknown_override_policy = (
            UnknownOverridePolicy.UNSPECIFIED
            if raw_unknown_override_policy is None
            else UnknownOverridePolicy(str(raw_unknown_override_policy))
        )
        structured = StructuredRiskMethod(
            severity=SeverityMethod(
                method_id="relative_velocity_v2",
                speed_semantic=SpeedSemantic.RELATIVE_SPEED,
                semantic=severity_semantic,
                bands=tuple(severity_bands), source_ref=severity_ref,
                road_user_groups=tuple(
                    (str(key), str(value))
                    for key, value in assets["severity_semantics"]["road_user_groups"].items()
                ),
                collision_types=tuple(
                    (str(key), str(value))
                    for key, value in assets["severity_semantics"]["collision_types"].items()
                ),
                fallback_policy=str(manifest["policies"]["severity_fallback_method"]),
            ),
            exposure=exposure_method,
            controllability_profile=ControllabilityProfile(
                profile_id="iav_avp_v1", bands=tuple(bands), source_ref=c_ref,
                review_status=ReviewStatus.FINALIZED,
            ),
            controllability_overrides=tuple(overrides),
            controllability_branch_policy=ControllabilityBranchPolicy(
                profile_id="iav_avp_v1", source_status="CONFIRMED",
                unknown_override_policy=compiled_unknown_override_policy,
                source_ref=self._source(
                    source_hash, (
                        source_paths["project_analysis_policy"]
                        if project_c_policy else source_paths["controllability_profile"]
                    ),
                    "unknown_override_policy", {
                        "compiled_value": compiled_unknown_override_policy.value,
                        "asset_field_present": bool(project_c_policy)
                        or "unknown_override_policy" in c_asset,
                    },
                ),
                method_hash=source_hash,
            ),
            asil_zero_short_circuit=str(manifest["policies"]["asil_zero_short_circuit"]),
            method_source_hash=source_hash,
        )
        raw_risk_vocabulary_mappings = assets["risk_vocabulary_mappings"].get(
            "mappings", [],
        )
        if not isinstance(raw_risk_vocabulary_mappings, list):
            raise YamlBaselineCompileError(
                "risk_vocabulary_mappings.mappings must be a list"
            )
        risk_vocabulary_physics = assets["risk_vocabulary_mappings"].get("physics", {})
        if not isinstance(risk_vocabulary_physics, dict):
            raise YamlBaselineCompileError("risk_vocabulary_mappings.physics must be a mapping")
        severity_targets = {
            "road_user_type": {
                canonical for canonical, _ in structured.severity.road_user_groups
            },
            "collision_type": {
                canonical for canonical, _ in structured.severity.collision_types
            },
        }
        longitudinal_collision_types = risk_vocabulary_physics.get(
            "longitudinal_collision_types", [],
        )
        if (
            not isinstance(longitudinal_collision_types, list)
            or not longitudinal_collision_types
            or any(str(value) not in severity_targets["collision_type"] for value in longitudinal_collision_types)
        ):
            raise YamlBaselineCompileError(
                "risk_vocabulary_mappings.physics.longitudinal_collision_types "
                "must contain active collision vocabulary"
            )
        physics_rule_id = str(risk_vocabulary_physics.get("derivation_rule_id", "")).strip()
        ego_directions = risk_vocabulary_physics.get("ego_longitudinal_directions", [])
        object_directions = risk_vocabulary_physics.get("object_longitudinal_directions", [])
        if (
            not physics_rule_id
            or not isinstance(ego_directions, list) or not ego_directions
            or not isinstance(object_directions, list) or not object_directions
            or any(not str(value).strip() for value in ego_directions + object_directions)
        ):
            raise YamlBaselineCompileError(
                "risk_vocabulary_mappings.physics direction semantics are incomplete"
            )
        risk_vocabulary_mappings: list[dict[str, Any]] = []
        seen_risk_vocabulary_values: set[tuple[str, str]] = set()
        for index, raw_mapping in enumerate(raw_risk_vocabulary_mappings):
            if not isinstance(raw_mapping, dict):
                raise YamlBaselineCompileError(
                    f"risk_vocabulary_mappings.mappings[{index}] must be a mapping"
                )
            field = str(raw_mapping.get("field", "")).strip()
            raw_value = str(raw_mapping.get("raw_value", "")).strip()
            canonical_value = str(raw_mapping.get("canonical_value", "")).strip()
            mapping_rule_id = str(raw_mapping.get("mapping_rule_id", "")).strip()
            if (
                field not in severity_targets
                or not raw_value
                or not canonical_value
                or not mapping_rule_id
            ):
                raise YamlBaselineCompileError(
                    f"risk_vocabulary_mappings.mappings[{index}] is incomplete"
                )
            if canonical_value not in severity_targets[field]:
                raise YamlBaselineCompileError(
                    "risk_vocabulary_mappings target is not active Severity vocabulary: "
                    f"field={field!r} target={canonical_value!r}"
                )
            key = (field, raw_value.casefold())
            if key in seen_risk_vocabulary_values:
                raise YamlBaselineCompileError(
                    "risk_vocabulary_mappings has duplicate exact mapping: "
                    f"field={field!r} raw_value={raw_value!r}"
                )
            seen_risk_vocabulary_values.add(key)
            source_ref = self._source(
                source_hash,
                source_paths["risk_vocabulary_mappings"],
                f"mappings[{index}]",
                raw_mapping,
            )
            risk_vocabulary_mappings.append({
                "field": field,
                "raw_value": raw_value,
                "canonical_value": canonical_value,
                "mapping_rule_id": mapping_rule_id,
                "mapping_source": str(
                    raw_mapping.get(
                        "mapping_source",
                        assets["risk_vocabulary_mappings"].get("mapping_source", ""),
                    )
                ).strip(),
                "source_ref": {
                    "workbook": source_ref.workbook,
                    "template_hash": source_ref.template_hash,
                    "sheet": source_ref.sheet,
                    "range": source_ref.range,
                    "raw_text": source_ref.raw_text,
                    "source_hash": source_ref.source_hash,
                },
            })

        workflow_binding = self._binding(TemplateRole.WORKFLOW, source_hash, "runtime_workflow")
        empty_derivation = lambda role, pattern: DerivationMethod(
            inputs_required=(), derivation_pattern=pattern,
            qm_handling="QM does not generate a Safety Goal",
            aggregation_instructions="Aggregate by canonical safety intent",
            semantic_derivation_required=True, assumptions=(), instructions=(),
            source_binding=self._binding(role, source_hash, "normalized/derivation_rules.yaml"),
        )
        required = self._required_facts(source_hash, source_paths, severity_ref, c_ref)
        diagnostics = (
            CompilerDiagnostic(
                severity=CompilerDiagnosticSeverity.WARNING,
                code=CompilerDiagnosticCode.BASELINE_SCENARIO_BINDING_INCOMPLETE,
                message=(
                    "The seven-dimension atom ontology is compiled, but the active baseline "
                    "contains no normative cross-dimension compatibility rules. Candidate-level "
                    "unresolved dimensions remain PENDING with machine-readable reasons."
                ),
                role=TemplateRole.SCENARIO_MODEL,
                source_refs=(exposure_atoms[0].source_ref,),
            ),
            CompilerDiagnostic(
                severity=CompilerDiagnosticSeverity.WARNING,
                code=CompilerDiagnosticCode.SEVERITY_FALLBACK_UNCOMPILED,
                message=(
                    "Relative-speed severity bands are executable; the traffic-domain "
                    "fallback remains pending contract compilation."
                ),
                role=TemplateRole.SEVERITY_RULES,
                source_refs=(severity_ref,),
            ),
            CompilerDiagnostic(
                severity=CompilerDiagnosticSeverity.WARNING,
                code=CompilerDiagnosticCode.FTTI_METHOD_UNCOMPILED,
                message=(
                    "FTTI YAML assets are hash-bound but formula routing and the audited "
                    "formula registry are not yet connected to runtime scoring."
                ),
                source_refs=(self._source(
                    source_hash, source_paths["ftti_formulas"], "formulas",
                    assets["ftti_formulas"].get("formulas", []),
                ),),
            ),
        )
        source_refs = tuple(dict.fromkeys([
            *refs,
            self._source(
                source_hash, source_paths["risk_vocabulary_mappings"], "mappings",
                raw_risk_vocabulary_mappings,
            ),
            self._source(
                source_hash, source_paths["risk_vocabulary_mappings"], "physics",
                risk_vocabulary_physics,
            ),
            scenario_method.fm_template_catalog.source_ref
            if scenario_method.fm_template_catalog is not None else guide_ref,
            scenario_method.failure_mode_selector_taxonomy.component_source_ref
            if scenario_method.failure_mode_selector_taxonomy is not None else guide_ref,
            scenario_method.failure_mode_selector_taxonomy.failure_type_source_ref
            if scenario_method.failure_mode_selector_taxonomy is not None else guide_ref,
            scenario_method.fm_template_selector_adapter.source_ref
            if scenario_method.fm_template_selector_adapter is not None else guide_ref,
            scenario_method.domain_knowledge.source_ref
            if scenario_method.domain_knowledge is not None else guide_ref,
            *scenario_method.example_catalogs,
            *(item.source_ref for item in guidewords),
            *(item.source_ref for item in severity_bands),
            *(item.source_ref for item in bands),
            *(item.source_ref for item in overrides),
            *(item.source_ref for item in asil_mappings),
        ]))
        return MethodContract(
            metadata={
                "template_hash": source_hash,
                "method_source_hash": source_hash,
                "source_kind": "YAML_BASELINE",
                "method_id": str(manifest["method_id"]),
                "method_version": str(manifest["method_version"]),
                "project_analysis_policy": compiled_project_policy,
                "asset_hashes": dict(sorted(hashes.items())),
                "scenario_atom_catalog": scenario_atom_catalog,
                "scenario_aliases": approved_scenario_aliases,
                "scenario_coverage_rules": approved_coverage_rules,
                "scenario_coverage_governance": {
                    "approved_count": coverage_status_counts["APPROVED"],
                    "proposed_count": coverage_status_counts["PROPOSED"],
                    "rejected_count": coverage_status_counts["REJECTED"],
                    "source_asset": source_paths["scenario_coverage_rules"],
                },
                "scenario_coverage_knowledge_sources": coverage_knowledge_sources,
                "risk_vocabulary_mappings": risk_vocabulary_mappings,
                "risk_vocabulary_physics": {
                    "derivation_rule_id": physics_rule_id,
                    "longitudinal_collision_types": [
                        str(value) for value in longitudinal_collision_types
                    ],
                    "ego_longitudinal_directions": [
                        str(value) for value in ego_directions
                    ],
                    "object_longitudinal_directions": [
                        str(value) for value in object_directions
                    ],
                },
                "severity_source_inventory": [
                    {
                        "source": source_paths["severity"],
                        "role": "ACTIVE_SEVERITY_TABLE",
                        "confirmed": True,
                        "contains_thresholds": True,
                        "contains_delta_v_derivation": False,
                        "contains_collision_lookup": True,
                        "runtime_consumer": "SeverityMethodExecutor",
                        "method_authority": "ACTIVE_COMPILED_TABLE",
                    },
                    {
                        "source": source_paths["asil"],
                        "role": "ISO_SEVERITY_REFERENCE",
                        "confirmed": True,
                        "contains_thresholds": True,
                        "contains_delta_v_derivation": False,
                        "contains_collision_lookup": True,
                        "runtime_consumer": "MethodContractASILService",
                        "method_authority": "NOT_ACTIVE_SEVERITY_TABLE",
                    },
                    {
                        "source": source_paths["scenario_domain"],
                        "role": "DOMAIN_SEVERITY_REFERENCE",
                        "confirmed": True,
                        "contains_thresholds": True,
                        "contains_delta_v_derivation": False,
                        "contains_collision_lookup": False,
                        "runtime_consumer": "NONE",
                        "method_authority": "EXCLUDED_NUMERIC_SECTION",
                    },
                    {
                        "source": source_paths["atom_spec"],
                        "role": "SCENARIO_PHYSICS_SPEC",
                        "confirmed": True,
                        "contains_thresholds": False,
                        "contains_delta_v_derivation": False,
                        "contains_collision_lookup": False,
                        "runtime_consumer": "MethodAtomResolver",
                        "method_authority": "NOT_SEVERITY_AUTHORITY",
                    },
                    {
                        "source": source_paths["scenario_templates"],
                        "role": "SCENARIO_TEMPLATE_CONSTRAINT",
                        "confirmed": True,
                        "contains_thresholds": False,
                        "contains_delta_v_derivation": False,
                        "contains_collision_lookup": True,
                        "runtime_consumer": "ScenarioMethodService",
                        "method_authority": "NOT_SEVERITY_AUTHORITY",
                    },
                    {
                        "source": source_paths["severity_semantics"],
                        "role": "SEVERITY_CATEGORY_NORMALIZATION",
                        "confirmed": True,
                        "contains_thresholds": False,
                        "contains_delta_v_derivation": False,
                        "contains_collision_lookup": True,
                        "runtime_consumer": "SeverityMethodExecutor",
                        "method_authority": "ACTIVE_CATEGORY_MAPPING",
                    },
                    {
                        "source": source_paths["risk_vocabulary_mappings"],
                        "role": "TEMPLATE_TO_SEVERITY_VOCABULARY_MAPPING",
                        "confirmed": True,
                        "contains_thresholds": False,
                        "contains_delta_v_derivation": False,
                        "contains_collision_lookup": True,
                        "runtime_consumer": "RiskVocabularyAdapter",
                        "method_authority": "ACTIVE_CATEGORY_MAPPING",
                    },
                ],
            },
            workflow=WorkflowContract(steps=(), source_binding=workflow_binding),
            guidewords=GuidewordContract(
                guidewords, self._binding(TemplateRole.GUIDEWORD_TABLE, source_hash, source_paths["guidewords"]),
            ),
            scenario_model=ScenarioModel(
                dimensions=tuple(scenario_dimensions),
                structural_constraints=("bounded_atom_selection_required",),
                source_binding=self._binding(TemplateRole.SCENARIO_MODEL, source_hash, source_paths["scenario_structure"]),
                source_type="YAML_BASELINE_SCENARIO_ONTOLOGY",
                constraint_rules=tuple(constraint_rules),
                scenario_method=scenario_method,
            ),
            severity=SeverityContract(
                SeverityScale(scale_sources, self._binding(TemplateRole.SEVERITY_LEVELS, source_hash, source_paths["severity"])),
                rules=(), diagnostics=(),
            ),
            exposure=ExposureContract(exposure_levels, (), (), (), (), ()),
            controllability=ControllabilityContract(c_levels, (), (), (), (), ()),
            asil=ASILMatrix(
                ("S0", "S1", "S2", "S3"), ("E0", "E1", "E2", "E3", "E4"),
                ("C0", "C1", "C2", "C3"), tuple(asil_mappings), "QM",
                self._binding(TemplateRole.ASIL_MATRIX, source_hash, source_paths["asil"]), (),
            ),
            safety_goal_method=empty_derivation(TemplateRole.SAFETY_GOAL_METHOD, "baseline_safety_intent_semantic_derivation"),
            safe_state_method=empty_derivation(TemplateRole.SAFE_STATE_METHOD, "baseline_project_capability_safe_state_derivation"),
            report_contract=report_contract,
            required_fact_specs=required,
            diagnostics=diagnostics, sources=source_refs,
            compile_status=CompileStatus.READY_WITH_WARNINGS,
            engineering_rules_compiled=True,
            structured_risk_method=structured,
            compiler_version=self.compiler_version,
        )

    @staticmethod
    def _validate_c_bands(bands: list[ControllabilityBand]) -> None:
        probes = (0.0, 3.0, 3.000001, 4.0, 4.000001, 5.0, 5.000001, 100.0)
        for value in probes:
            matches = []
            for band in bands:
                lower_ok = band.lower_ttc_s is None or value > band.lower_ttc_s or (
                    value == band.lower_ttc_s and band.lower_inclusive
                )
                upper_ok = band.upper_ttc_s is None or value < band.upper_ttc_s or (
                    value == band.upper_ttc_s and band.upper_inclusive
                )
                if lower_ok and upper_ok:
                    matches.append(band)
            if len(matches) != 1:
                raise YamlBaselineCompileError(f"C profile gap/overlap at TTC={value}")

    def _required_facts(
        self, source_hash: str, paths: dict[str, str], severity: SourceRef,
        controllability: SourceRef,
    ) -> tuple[RequiredFactSpec, ...]:
        def spec(
            fact: FactType, required_for: tuple[str, ...], origin: FactOrigin,
            source: SourceRef, unit: str = "", constraints: tuple[str, ...] = (),
        ):
            return RequiredFactSpec(
                fact, required_for, unit, constraints, "when applicable",
                origin, (), (source,),
            )
        return (
            spec(
                FactType.COLLISION_TYPE, ("SEVERITY", "FTTI"),
                FactOrigin.SCENARIO_FACT, severity,
                constraints=(
                    "IN ('FRONTAL','REAR_END','SIDE','VEHICLE_TO_ROAD_USER')",
                ),
            ),
            spec(
                FactType.ROAD_USER_TYPE, ("SEVERITY",),
                FactOrigin.SCENARIO_FACT, severity,
                constraints=(
                    "IN ('VEHICLE','PEDESTRIAN','CYCLIST','MOTORCYCLIST')",
                ),
            ),
            spec(FactType.DELTA_V, ("SEVERITY",), FactOrigin.DERIVED_FACT, severity, "km/h"),
            spec(FactType.TTC, ("CONTROLLABILITY", "FTTI"), FactOrigin.DERIVED_FACT, controllability, "s"),
            spec(FactType.DRIVER_IN_VEHICLE, ("CONTROLLABILITY",), FactOrigin.PROJECT_FACT, controllability),
            spec(FactType.REMOTE_INTERVENTION_AVAILABLE, ("CONTROLLABILITY",), FactOrigin.PROJECT_FACT, controllability),
            spec(FactType.INTERVENTION_AVAILABLE, ("CONTROLLABILITY",), FactOrigin.SCENARIO_FACT, controllability),
        )
