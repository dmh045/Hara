from __future__ import annotations

from collections.abc import Mapping
import re
from typing import Any

from .quantity_format import quantity, quantity_range, narrative_quantities, FIELD_UNITS


def _status(evidence: Any) -> str:
    value = getattr(evidence, "status", "")
    return str(getattr(value, "value", value))


class EngineeringReportTextMapper:
    """Map committed runtime status to compact Chinese review text.

    This layer owns presentation wording only.  It neither supplements project
    facts nor makes a risk-method decision.
    """

    _PENDING_VALUE = "待评定"
    _COLLISION_TEXT = {
        "FRONTAL": "正面碰撞", "REAR_END": "追尾碰撞",
        "SIDE": "侧向碰撞", "LATERAL": "侧向碰撞",
    }

    def __init__(self, language: Mapping[str, Any] | None = None):
        self.language = language or {}
    _CATEGORY_TEXT = {
        "LOCATION_PARKING": "停车场内",
        "LOCATION_MOTORWAY": "高速公路",
        "LOCATION_EXPRESSWAY": "城市快速路",
        "LOCATION_CITY": "城市道路",
        "LOCATION_RURAL": "乡村道路",
        "ACTION_PARK": "执行泊车",
        "ACTION_REVERSE": "执行低速倒车",
        "ACTION_STOP": "执行停车或制动",
        "ACTION_HOLD": "执行驻车保持",
        "ACTION_ACCELERATE": "执行加速",
        "ACTION_TURN": "执行转向",
        "OBJECT_PEDESTRIAN": "行人",
        "OBJECT_CYCLIST": "骑行者",
        "OBJECT_VEHICLE": "其他车辆",
        "OBJECT_STATIC": "静态障碍物",
        "OBJECT_TRAM": "有轨电车",
        "OBJECT_OCCUPANT": "车辆乘员",
        "TRAFFIC_FOLLOWING": "后方跟随车辆",
        "TRAFFIC_ONCOMING": "对向交通参与者",
        "TRAFFIC_CROSSING": "横穿交通参与者",
        "TRAFFIC_CUT_IN": "切入车辆",
        "TRAFFIC_REVERSE": "倒车路径内交通参与者",
        "TRAFFIC_PARKING": "停车场内交通参与者",
        "ROAD_SLOPE": "坡道路段",
        "ROAD_LOW_FRICTION": "低附着路面",
    }
    _MODE_TEXT = {
        "parking vehicle control": "泊车控车",
        "parking_vehicle_control": "泊车控车",
        "active": "激活",
        "standby": "待机",
        "override": "驾驶员接管",
        "abort": "中止",
        "finish": "完成",
        "error": "故障",
        "off": "关闭",
    }
    _VARIANT_TEXT = {
        "typical": "代表场景",
        "representative": "代表场景",
        "boundary": "边界场景",
        "extreme": "高要求场景",
        "demanding": "高要求场景",
    }
    _CONTEXT_TEXT = {
        "ENTRY": "进入功能阶段",
        "CRUISE": "巡航阶段",
        "MAXIMUM_SPEED_DURING_PARKING": "泊车阶段",
        "PARKING": "泊车阶段",
        "CONTROL": "车辆控制阶段",
    }
    _POSITION_TEXT = {
        "front": "前方",
        "rear": "后方",
        "left": "左侧",
        "right": "右侧",
    }
    _EXPOSURE_DIMENSION_LABELS = {
        "WHERE": "场所",
        "ROAD": "道路条件",
        "EGO_ACTION": "自车动作",
        "EGO_X_ROAD": "自车与道路关系",
        "TRAFFIC_PATTERN": "交通关系/交通模式",
        "EGO_DYNAMICS": "自车动态",
        "OBJECT": "对象/交通参与者",
    }
    _CONDITION_TEXT = {
        "stationary target": "静止目标",
        "offline proposal; no live validation": "离线方案，未做真实模型验证",
        "moving targets; real capability bound unknown": "移动目标未覆盖，车辆真实能力上界未知",
    }

    @staticmethod
    def _has_reason(value: Any, token: str) -> bool:
        return token in str(value or "").upper()

    @classmethod
    def _display_mode(cls, mode: Any) -> str:
        return cls._MODE_TEXT.get(str(mode or "").strip().casefold(), "")

    @staticmethod
    def _number(value: Any, unit: str = "km/h") -> str:
        return quantity(value, unit)

    @classmethod
    def speed_text(cls, scenario: Any) -> str:
        facts = getattr(scenario, "facts", {}) or {}
        context = facts.get("speed_context_resolution", {})
        if isinstance(context, Mapping) and str(context.get("status", "")).upper() == "SOURCE_CONFLICT":
            expressions = [
                str(item) for item in context.get("source_expressions", [])
                if str(item).strip()
            ]
            if expressions:
                return f"车速约束：来源存在冲突（{' / '.join(expressions)}），待确认"

        point = facts.get("ego_speed_kph")
        provenance = getattr(scenario, "fact_provenance", {}) or {}
        point_provenance = provenance.get("ego_speed_kph", {})
        if (
            isinstance(point, (int, float)) and not isinstance(point, bool)
            and isinstance(point_provenance, Mapping)
            and (
                str(point_provenance.get("approval", "")).upper() == "FINALIZED"
                or (
                    bool(facts.get("physical_analysis"))
                    and point_provenance.get("validation_status") == "VALIDATED"
                    and point_provenance.get("analysis_value_semantic") == "PRE_FAULT_OPERATING_SPEED"
                )
            )
        ):
            if point_provenance.get("analysis_value_semantic") == "PRE_FAULT_OPERATING_SPEED":
                impact = facts.get("ego_impact_speed_kph")
                suffix = f"；碰撞时本车速度：{cls._number(impact)} km/h" if isinstance(impact, (int, float)) and not isinstance(impact, bool) else "；碰撞速度另行确定"
                return f"故障前/初始车速：{cls._number(point)} km/h" + suffix
            return f"分析车速：{cls._number(point)} km/h"

        constraint = facts.get("ego_speed_constraint")
        if not isinstance(constraint, Mapping):
            return ""
        minimum = constraint.get("min_kph", constraint.get("speed_min_kph"))
        maximum = constraint.get("max_kph", constraint.get("speed_max_kph"))
        if not all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in (minimum, maximum)):
            return ""
        minimum_value = float(minimum)
        maximum_value = float(maximum)
        if minimum_value == maximum_value:
            return f"适用车速范围：{cls._number(maximum_value)} km/h"
        display_kind = str(context.get("display_kind", "")) if isinstance(context, Mapping) else ""
        if minimum_value == 0 and display_kind == "UPPER_BOUND":
            return f"适用车速范围：不高于{cls._number(maximum_value)} km/h"
        return (
            f"适用车速范围：{cls._number(minimum_value)}–"
            f"{cls._number(maximum_value)} km/h"
        )

    @staticmethod
    def _dimensions(scenario: Any) -> Mapping[str, Any]:
        facts = getattr(scenario, "facts", {}) or {}
        value = facts.get("method_scenario_dimensions", {})
        return value if isinstance(value, Mapping) else {}

    @staticmethod
    def _categories(scenario: Any) -> set[str]:
        instance = getattr(scenario, "analysis_instance", {}) or {}
        query = instance.get("structured_semantic_query", {})
        query = query if isinstance(query, Mapping) else {}
        categories = {
            str(item)
            for field in (
                "action_categories", "object_categories", "traffic_relations",
                "road_relations", "location_categories",
            )
            for item in query.get(field, [])
        }
        facts = getattr(scenario, "facts", {}) or {}
        if facts.get("physical_analysis"):
            # The parent query is a search space, not the instantiated target's
            # traffic relation. Its actual position/direction is rendered below.
            categories = {category for category in categories if not category.startswith("TRAFFIC_")}
        scoped_object = {
            "VEHICLE": "OBJECT_VEHICLE", "PASSENGER_CAR": "OBJECT_VEHICLE",
            "PEDESTRIAN": "OBJECT_PEDESTRIAN", "CYCLIST": "OBJECT_CYCLIST",
            "FIXED_OBJECT": "OBJECT_STATIC",
        }.get(str(facts.get("road_user_type", "")).upper())
        if scoped_object:
            # The synthesis query describes the parent search space. A bound
            # child's concrete object must not retain another parent's class.
            categories = {category for category in categories if not category.startswith("OBJECT_")}
            categories.add(scoped_object)
            return categories
        object_value = str(
            facts.get("object_type") or facts.get("scenario_object_atom") or facts.get("road_user_type", "") or ""
        ).casefold()
        if any(token in object_value for token in ("tram", "有轨电车")):
            categories.discard("OBJECT_STATIC")
            categories.add("OBJECT_TRAM")
        elif any(token in object_value for token in ("pedestrian", "person", "行人")):
            categories.add("OBJECT_PEDESTRIAN")
        elif any(token in object_value for token in ("cycl", "bicycle", "骑行", "自行车")):
            categories.add("OBJECT_CYCLIST")
        elif any(token in object_value for token in ("vehicle", "car", "truck", "车辆", "汽车")):
            categories.add("OBJECT_VEHICLE")
        elif any(token in object_value for token in ("obstacle", "pillar", "cone", "障碍", "柱")):
            categories.add("OBJECT_STATIC")
        return categories

    @staticmethod
    def _chinese_source(value: Any) -> str:
        text = str(value or "").strip().rstrip("。；;")
        return text if re.search(r"[\u3400-\u9fff]", text) else ""

    @classmethod
    def _first_category(cls, categories: set[str], names: tuple[str, ...]) -> str:
        return next((cls._CATEGORY_TEXT[name] for name in names if name in categories), "")

    @classmethod
    def variant_text(cls, scenario: Any) -> str:
        context = getattr(scenario, "context_resolution", {}) or {}
        synthesis = context.get("scenario_synthesis", {})
        raw = str(synthesis.get("coverage_label", "") if isinstance(synthesis, Mapping) else "")
        planned = (
            isinstance(synthesis, Mapping)
            and synthesis.get("coverage_status") == "PLANNED_NOT_INSTANTIATED"
        )
        if not raw and not planned:
            raw = str(getattr(scenario, "atomic_variant", "") or "").rsplit(":", 1)[-1]
        if raw.casefold() in cls._VARIANT_TEXT:
            return cls._VARIANT_TEXT[raw.casefold()]
        intents = synthesis.get("coverage_intents", []) if isinstance(synthesis, Mapping) else []
        labels = list(dict.fromkeys(
            cls._VARIANT_TEXT.get(str(item).casefold(), "")
            for item in intents
            if cls._VARIANT_TEXT.get(str(item).casefold(), "")
        ))
        return f"计划覆盖：{' / '.join(labels)}" if labels else "受控分析场景"

    @staticmethod
    def coverage_variant_count(scenario: Any) -> int:
        context = getattr(scenario, "context_resolution", {}) or {}
        synthesis = context.get("scenario_synthesis", {})
        value = synthesis.get("desired_variant_count", 1) if isinstance(synthesis, Mapping) else 1
        return int(value) if isinstance(value, int) and value > 0 else 1

    @classmethod
    def _context_text(cls, scenario: Any) -> str:
        facts = getattr(scenario, "facts", {}) or {}
        speed = facts.get("speed_context_resolution", {})
        selected = str(speed.get("selected_context", "")) if isinstance(speed, Mapping) else ""
        return cls._CONTEXT_TEXT.get(selected, "")

    @classmethod
    def _object_position(cls, scenario: Any) -> str:
        facts = getattr(scenario, "facts", {}) or {}
        return cls._POSITION_TEXT.get(
            str(facts.get("object_position", "")).strip().casefold(), ""
        )

    @classmethod
    def _object_motion(cls, scenario: Any) -> str:
        facts = getattr(scenario, "facts", {}) or {}
        speed = facts.get("object_speed_kph")
        direction = {"FORWARD": "沿车体前向运动", "REVERSE": "沿车体后向运动",
                     "LEFT": "向车体左侧横穿", "RIGHT": "向车体右侧横穿"}.get(facts.get("object_longitudinal_direction"))
        return (f"目标以{quantity(speed, 'km/h')} km/h{direction}"
                if direction and isinstance(speed, (int, float)) and not isinstance(speed, bool) and speed > 0 else "")

    def object_interaction_summary(self, scenario: Any) -> str:
        """Render only structured object and interaction facts."""
        facts = getattr(scenario, "facts", {}) or {}
        categories = self._categories(scenario)
        object_value = self._first_category(categories, (
            "OBJECT_TRAM", "OBJECT_PEDESTRIAN", "OBJECT_CYCLIST", "OBJECT_VEHICLE",
            "OBJECT_STATIC", "OBJECT_OCCUPANT",
        ))
        if object_value and facts.get("object_speed_kph") == 0 and str(facts.get("object_longitudinal_direction", "")).upper() == "STATIONARY":
            object_value = "静止" + object_value
        traffic = self._first_category(categories, (
            "TRAFFIC_FOLLOWING", "TRAFFIC_ONCOMING", "TRAFFIC_CROSSING",
            "TRAFFIC_CUT_IN", "TRAFFIC_REVERSE", "TRAFFIC_PARKING",
        ))
        values = []
        if object_value:
            values.append(f"对象：{object_value}")
        position = self._object_position(scenario)
        if position:
            values.append(f"位置：{position}")
        if traffic:
            values.append(f"交互：{traffic}")
        if motion := self._object_motion(scenario):
            values.append(motion)
        return "；".join(values)

    @classmethod
    def physical_inputs(
        cls, scenario: Any, trace: Mapping[str, Any],
    ) -> str:
        """Distinguish accepted risk inputs from unscored scenario candidates."""

        facts = getattr(scenario, "facts", {}) or {}
        context = trace.get("hazardous_event_risk_context", {})
        context = context if isinstance(context, Mapping) else {}
        candidate_only = trace.get("risk_scoring_invoked") is False or not context
        control = trace.get("controllability", {})
        control = control if isinstance(control, Mapping) else {}
        derived = control.get("derived_ttc", {})
        derived = derived if isinstance(derived, Mapping) else {}

        def value(field: str) -> Any:
            if field in context:
                record = context[field]
                if not isinstance(record, Mapping):
                    return None
                return record.get("value") if str(record.get("status", "")) == "AVAILABLE" else None
            return facts.get(field) if candidate_only else None

        closing_speed = derived.get("closing_speed_kph")
        if closing_speed is None and candidate_only:
            closing_speed = facts.get("closing_speed_kph")
        ttc = value("ttc_s") if "ttc_s" in context else derived.get("ttc_s")
        if ttc is None and candidate_only:
            ttc = facts.get("ttc_s")
        quantities = (
            ("初始自车速度" if facts.get("physical_analysis") else "自车速度", value("ego_speed_kph"), "km/h"),
            ("目标速度", value("object_speed_kph"), "km/h"),
            ("碰撞相对速度" if facts.get("physical_analysis") else "相对速度", value("relative_speed_kph"), "km/h"),
            ("接近速度", closing_speed if not facts.get("physical_analysis") else None, "km/h"),
            ("相对距离", value("relative_distance_m"), "m"),
            ("TTC", ttc, "s"),
        )
        parts = [
            f"{label}：{cls._number(number, unit)} {unit}"
            for label, number, unit in quantities
            if isinstance(number, (int, float)) and not isinstance(number, bool)
        ]
        model = facts.get("fault_motion_model")
        analysis = facts.get("physical_analysis", {})
        control_hypothesis = (facts.get("control_analysis") or analysis).get("hypothesis", {})
        if reason := control_hypothesis.get("unresolved_controls", {}).get("braking"):
            parts.append("制动能力未决依据：" + reason)
        if controls := facts.get("control_analysis", {}).get("hypothesis"):
            parts.append("干预能力假设：" + "；".join(controls["conditions"]))
            for proposal in controls.get("intervention_assumptions", {}).values():
                parts.append("干预条件：" + "；".join(proposal["conditions"]))
        if analysis:
            if facts.get("post_stop_motion") == "HOLD_POSITION":
                parts.append("停车后保持原位置至所列模型时限（分析假设），继续检查移动目标是否接触")
            labels = {"ego_speed_kph": "初速(km/h)", "relative_distance_m": "行程(m)",
                "longitudinal_acceleration_mps2": "纵向加速度(m/s²)", "slope_percent": "坡度(%)",
                "residual_resistance_mps2": "残余阻力减速度(m/s²)", "lateral_clearance_m": "侧向净距(m)",
                "lateral_speed_mps": "初始侧速(m/s)", "lateral_acceleration_mps2": "侧向加速度(m/s²)",
                "turn_radius_m": "条件性转弯半径(m)",
                "model_valid_until_s": "模型时域(s)", "post_fault_speed_cap_kph": "残余限速(km/h)",
                "object_offset_x_m": "目标初始纵向中心偏移(m，车前为正)",
                "object_offset_y_m": "目标初始横向中心偏移(m，车左为正)",
                "contact_half_length_m": "双方合计接触包络半长(m)",
                "contact_half_width_m": "双方合计接触包络半宽(m)"}
            parts.extend(f"{labels[key]}：{quantity(number, FIELD_UNITS.get(key, ''))}"
                for key, number in analysis.get("sample_parameters", {}).items()
                if key in labels and key not in {"ego_speed_kph", "relative_distance_m"})
            for parameter in analysis.get("hypothesis", {}).get("parameters", []):
                if parameter["kind"] == "SYMBOLIC":
                    parts.append(f"符号条件：{labels.get(parameter['field'], parameter['field'])}={parameter['symbol']}")
                elif parameter["kind"] == "RANGE":
                    parts.append(f"假设范围：{labels.get(parameter['field'], parameter['field'])}={quantity_range(parameter['lower'], parameter['upper'], FIELD_UNITS.get(parameter['field'], ''))}；有限采样未证明连续不变性")
            if any(p["kind"] == "SYMBOLIC" for p in analysis.get("hypothesis", {}).get("parameters", [])):
                boundaries = analysis.get("sensitivity", {}).get("method_boundaries", [])
                relations = analysis.get("sensitivity", {}).get("parameter_relations", [])
                if relations:
                    parts.extend("参数关系：" + text for relation in relations for text in relation["relations"])
                else:
                    parts.append("参数关系：v²=v₀²+2ad；d=v₀t+at²/2（首次接触且模型持续有效）" if model == "CONSTANT_ACCELERATION_TO_IMPACT_DISTANCE" else "参数关系见物理 trace")
                parts.extend(f"边界条件：{labels.get(row['parameter'], row['parameter'])}≈{quantity(row['boundary'], FIELD_UNITS.get(row['parameter'], ''))}；精确关系见物理追溯"
                    for row in boundaries[:3] if isinstance(row.get("boundary"), (float, int)))
            grade = analysis.get("grade_mapping", {})
            if grade:
                parts.append(f"坡向来源标签：{grade.get('source_tag') or '未注明'}；原参考方向：{ {'EGO_FORWARD': '车头前向', 'ROAD_REFERENCE': '道路参考向', 'UNSPECIFIED': '未注明'}.get(grade.get('source_reference'), '见来源')}")
                parts.append("朝向假设：" + ("车头朝上坡，重力向车后" if grade.get("ego_heading_relative_to_downhill") == "UPHILL" else "车头朝下坡，重力向车前"))
            # Full conditions, parameter objects and exclusions remain in the
            # existing risk trace. Keep intervention assumptions visible here.
            parts.extend(text for text in analysis.get("hypothesis", {}).get("conditions", [])
                if re.search(r"接管|介入|干预|制动通道|制动能力|interven|driver braking|braking path", text, re.IGNORECASE))
            if facts.get("first_contact_time_s") is not None and facts.get("ttc_s") is None:
                parts.append(f"首次接触：{quantity(facts['first_contact_time_s'], 's')} s（不作为纵向 TTC）")
            if condition := facts.get("selected_source_condition"):
                parts.append(f"代表运行条件：{condition['value']}；物理影响：{condition['physical_effect']}")
        if model == "CONSTANT_SPEED_REPRESENTATIVE":
            parts.append("恒速代表工况；用于所列碰撞条件，不是故障最大速度包络")
        elif model:
            names = {"GRADE_ROLLAWAY_TO_IMPACT_DISTANCE": "坡道溜车（重力减残余阻力）",
                "CONSTANT_ACCELERATION_TO_IMPACT_DISTANCE": "恒加速度至首次接触",
                "CONSTANT_LATERAL_ACCELERATION_TO_CONTACT": "独立横向加速至接触（不等同于等速转向）",
                "CONSTANT_SPEED_CURVATURE_TO_CONTACT": "等速圆弧至侧向首次接触（无侧滑假设）",
                "RECTILINEAR_MOVING_TARGET_CONTACT": "移动目标相对轨迹与有限包络首次接触"}
            parts.append(f"运动模型：{names.get(model, model)}")
        if not any(part.startswith("TTC：") for part in parts) and (
            str(derived.get("closing_speed_status", "")) == "TTC_NOT_CLOSING"
        ):
            parts.append("TTC：未接近，无有限值")
        position = cls._POSITION_TEXT.get(str(facts.get("object_position", "")).casefold(), "")
        if position:
            parts.append(f"目标位置：{position}")
        directions = {"FORWARD": "前进", "REVERSE": "倒车", "STATIONARY": "静止", "LEFT": "向左", "RIGHT": "向右"}
        for field, label in (
            ("ego_longitudinal_direction", "自车方向"),
            ("ego_lateral_direction", "自车侧向位移方向"),
            ("object_longitudinal_direction", "目标方向"),
        ):
            direction = directions.get(str(facts.get(field, "")).upper(), "")
            if direction:
                parts.append(f"{label}：{direction}")
        if not parts:
            return ""
        return (
            "场景候选（未核实用于评分）：" if candidate_only else ""
        ) + "；".join(parts)

    @staticmethod
    def driver_branch(scenario: Any, trace: Mapping[str, Any]) -> str:
        facts = getattr(scenario, "facts", {}) or {}
        instance = getattr(scenario, "analysis_instance", {}) or {}
        branch = instance.get("driver_configuration_branch", {}) if isinstance(instance, Mapping) else {}
        branch = branch if isinstance(branch, Mapping) else {}
        position = str(facts.get("driver_position", branch.get("driver_position", "")))
        label = {
            "in_driver_seat": "在驾驶位",
            "outside_driver_seat": "不在驾驶位",
        }.get(position, "")
        context = trace.get("hazardous_event_risk_context", {})
        context = context if isinstance(context, Mapping) else {}
        vehicle = context.get("driver_in_vehicle", {})
        if isinstance(vehicle, Mapping) and str(vehicle.get("status", "")) == "AVAILABLE":
            in_vehicle = vehicle.get("value")
        elif "driver_in_vehicle" not in context:
            in_vehicle = facts.get("driver_in_vehicle")
        else:
            in_vehicle = None
        parts = [f"驾驶员：{label}"] if label else []
        provenance = getattr(scenario, "fact_provenance", {}) or {}
        convention = provenance.get("driver_in_vehicle", {})
        if convention.get("project_rule_id") == "P7_C4_OUTSIDE_SEAT_AS_OUTSIDE_VEHICLE":
            return "；".join(parts)
        if in_vehicle is True:
            parts.append("车内状态：在车内")
        elif in_vehicle is False:
            parts.append("车内状态：在车外")
        elif label:
            parts.append("车内状态：未确定")
        return "；".join(parts)

    @staticmethod
    def _ttc_intervention_basis(control: Mapping[str, Any]) -> str:
        matched = {row.get("field"): row.get("actual")
            for row in control.get("profile_applicability", [])
            if isinstance(row.get("actual"), bool) and row.get("actual") == row.get("expected")}
        parts = [yes if matched[key] else no for key, yes, no in (
            ("driver_in_vehicle", "驾驶员在车内", "驾驶员不在车内"),
            ("emergency_braking_available", "故障后制动能力条件成立", "故障后制动不可用"),
        ) if key in matched]
        if control.get("unknown_policy_action") == "SKIP_TO_TTC":
            inputs = control.get("inputs", control.get("override_inputs", {}))
            unknown = [label for key, label in (
                ("remote_intervention_available", "远程干预"),
                ("other_road_user_avoidance_possible", "他方避让"),
            ) if key in inputs and inputs[key] is None]
            parts.append(("、".join(unknown) + "未确认；" if unknown else "") +
                         "未计入尚未确认的额外干预收益")
        return "；".join(parts)

    @classmethod
    def controllability_branch(cls, trace: Mapping[str, Any]) -> str:
        control = trace.get("controllability", {})
        if not isinstance(control, Mapping) or not control:
            return ""
        if control.get("decision_status") == "CONDITIONAL_REFERENCE_MATCH":
            return control.get("reference_rule", {}).get("rationale", "按参考工况进行条件性 C 评定；适用性待审核")
        if control.get("decision_status") == "OVERRIDE_MATCH":
            return "C 分支：干预覆盖；依据已命中的干预条件评分，未使用 TTC 分档"
        if control.get("decision_status") == "REFERENCE_APPLICABILITY_UNRESOLVED":
            labels = {"emergency_braking_available": "独立制动能力", "driver_in_vehicle": "驾驶员位置",
                "remote_intervention_available": "远程干预能力", "other_road_user_avoidance_possible": "他方避让条件"}
            missing = dict.fromkeys(key for rule in control.get("unresolved_reference_inputs", [])
                for key in rule.get("unknown_conditions", {}))
            detail = "、".join(labels.get(key, "干预条件") for key in missing)
            return f"已有适用场景类型的条件性 C 规则，但{detail or '必要干预条件'}尚未明确；不能仅凭接触时间给出等级"
        if control.get("decision_status") == "MODEL_NOT_APPLICABLE":
            reason = str(control.get("pending_reason", ""))
            if reason.startswith("REAR_APPROACH_BRAKING_APPLICABILITY_UNRESOLVED"):
                return "后车从本车行进方向的后方接近；本车制动不能直接作为避免被追尾的路径，当前避险方法适用性未确定"
            if reason.startswith("LATERAL_C_METHOD_NOT_AVAILABLE"):
                return "侧向接触模型；当前方法无适用 C 规则，局部待评定"
            if reason.startswith("POST_STOP_BRAKING_APPLICABILITY_UNRESOLVED"):
                return "本车停止后被移动目标接触；继续制动不是已成立的避险路径，当前方法无适用 C 规则"
            return "当前交互不满足所选适用条件；当前方法无适用 C 规则"
        if control.get("decision_status") == "MODEL_DEPENDENCY_UNRESOLVED":
            return "故障影响规避控制；C 方法依赖尚未成立"
        if control.get("decision_status") == "PHYSICAL_INPUT_UNRESOLVED":
            return "当前轨迹在模型有效范围内未建立接触，或几何不适用；不能据此判定范围外无危险，也不能直接套用碰撞避险分档"
        if control.get("decision_status") == "PROFILE_APPLICABILITY_UNRESOLVED":
            inputs = control.get("override_inputs", control.get("inputs", control.get("physical_inputs", {})))
            if inputs.get("driver_in_vehicle") is False:
                return "驾驶员在车外，车内制动反应分档不适用；远程干预或他方避让尚无适用结论"
            reason = control.get("pending_reason", "")
            missing = [label for key, label in (("driver_in_vehicle", "驾驶员可在车内操作"),
                ("emergency_braking_available", "故障后仍有可用制动能力")) if key in reason]
            return "制动反应分档不适用或依据未齐：" + "、".join(missing or ["驾驶员及制动路径条件未成立"])
        stage = {
            "OVERRIDE": "干预覆盖",
            "TTC": "TTC 分档",
            "UNRESOLVED": "分支未确定",
        }.get(str(control.get("decision_tree_stage", "")), "")
        parts = [f"C 分支：{stage}"] if stage else []
        ttc = control.get("derived_ttc", {}).get("ttc_s")
        if isinstance(ttc, (float, int)):
            parts.append(f"预计首次接触：{quantity(ttc, 's')} s")
        action = {"SKIP_TO_TTC": cls._ttc_intervention_basis(control),
            "BLOCK_TTC": "干预条件未明确，保留待评定",
            "UNSPECIFIED_BLOCKED": "未知干预条件的处理方法尚未明确",
            "CONFLICT_BLOCKED": "干预条件冲突，保留待评定"}.get(control.get("unknown_policy_action"))
        if action:
            parts.append(action)
        elif not control.get("unknown_policy_action") and control.get("unknown_override_policy") == "UNSPECIFIED":
            parts.append("未知干预条件的处理方法尚未明确")  # Legacy trace without an action field.
        return "；".join(parts)

    @classmethod
    def analysis_basis(cls, scenario: Any) -> str:
        provenance = getattr(scenario, "fact_provenance", {}) or {}
        if not isinstance(provenance, Mapping):
            return ""
        physical = (getattr(scenario, "facts", {}) or {}).get("physical_analysis")
        if physical:
            origins = physical.get("input_origins", {})
            labels = {"PROJECT_INPUT": "项目事实", "HUMAN_CONFIRMATION": "工程师输入",
                "SCENARIO_DEFINED": "方法场景条件", "METHOD_CONTRACT": "方法规则",
                "LLM_ANALYSIS_HYPOTHESIS": "LLM 分析假设", "DERIVED": "程序推导"}
            fields = {"ego_speed_kph": "初速", "relative_distance_m": "距离",
                "longitudinal_acceleration_mps2": "纵向加速度", "slope_percent": "坡度",
                "lateral_clearance_m": "侧向净距", "lateral_acceleration_mps2": "侧向加速度", "turn_radius_m": "转弯半径",
                "lateral_speed_mps": "初始侧速", "residual_resistance_mps2": "残余阻力"}
            groups = {}
            for field in fields:
                if field in origins:
                    groups.setdefault(labels.get(origins[field], "既有条件"), []).append(fields[field])
            symbolic = physical.get("sensitivity", {}).get("parameter_relations")
            return ("；".join(f"{label}：{'、'.join(names)}" for label, names in groups.items())
                + ("；符号参数关系及分档边界见物理输入；未生成唯一碰撞速度或接触时间。" if symbolic else
                   "；碰撞速度/首次接触时间由同一轨迹计算。")
                + "条件性分析，非车辆实测能力；完整假设、未覆盖范围及来源见本行审计关联的风险追溯。")
        labels = {
            "ego_speed_kph": "自车速度", "object_speed_kph": "目标速度",
            "relative_distance_m": "相对距离", "ttc_s": "TTC",
            "driver_position": "驾驶员配置",
            "fault_motion_model": "运动分析条件", "ego_impact_speed_kph": "碰撞速度",
        }
        parts = []
        if (getattr(scenario, "facts", {}) or {}).get("control_analysis"):
            parts.append("沿用既有轨迹；干预能力为条件性假设，C/ASIL 数值不代表已获工程批准")
        all_source_locations = []
        for field, label in labels.items():
            record = provenance.get(field, {})
            if not isinstance(record, Mapping) or not record:
                continue
            rule = str(
                record.get("policy_id") or record.get("selection_basis")
                or record.get("derivation_rule_id") or ""
            )
            approval = str(record.get("approval", ""))
            sources = record.get("source_refs", [])
            source_locations = [
                str(item.get("location", ""))
                for item in sources if isinstance(item, Mapping) and item.get("location")
            ] if isinstance(sources, (list, tuple)) else []
            description = ["按已有来源或方法条件设定"] if rule else []
            if approval and approval != "FINALIZED":
                description.append("待审设定")
            all_source_locations.extend(source_locations)
            if description:
                parts.append(f"{label}：{'；'.join(description)}")
        if all_source_locations:
            parts.append("完整来源位置及规则见本行审计关联")
        diagnostics = (getattr(scenario, "analysis_instance", {}) or {}).get("physical_plan_diagnostics", [])
        gap_labels = {
            "PHYSICAL_PLAN_INVALID": "物理方案生成或有限纠正未通过，尚未形成有效工况；不能视为危险已排除",
            "PHYSICAL_MODEL_NOT_APPLICABLE": "当前可用物理模型不覆盖所提出的交互条件，该方案未生成碰撞数值",
            "TARGET_NOT_ON_POST_FAULT_TRAVEL_PATH": "所提目标不在故障后的运动路径上，该方案未生成碰撞数值",
            "FIXED_MOVING_TARGET_OUTSIDE_MODEL_CATALOG": "当前目标已确定处于运动中，现有自主模型仅支持静止目标；保留该危险场景，尚未计算故障轨迹",
            "REQUIRED_MODEL_INPUT_OMITTED": "物理方案遗漏必需输入且有限纠正后仍未补齐，未形成完整数值工况",
        }
        gaps = dict.fromkeys(gap_labels[item.get("reason")] for item in diagnostics
                            if isinstance(item, Mapping) and item.get("reason") in gap_labels)
        if gaps:
            parts.extend(gaps)
            parts.append("完整模型缺口及未覆盖范围见场景物理方案诊断")
        return "；".join(parts)

    def scenario(self, scenario: Any, *, variant_count: int = 1) -> tuple[str, str]:
        """Render validated structured facts without exposing Method vocabulary."""
        facts = getattr(scenario, "facts", {}) or {}
        categories = self._categories(scenario)
        source_conditions = facts.get("source_operating_conditions", {})
        location = self._chinese_source(self._dimensions(scenario).get("WHERE", {}).get("project_value")) or self._chinese_source(source_conditions.get("operating_scenario", "")) or self._chinese_source(
            getattr(scenario, "operating_scenario", "")
        ) or self._first_category(categories, (
            "LOCATION_PARKING", "LOCATION_CITY", "LOCATION_EXPRESSWAY",
            "LOCATION_MOTORWAY", "LOCATION_RURAL",
        ))
        if not location:
            raw_location = str(getattr(scenario, "operating_scenario", "") or "").casefold()
            location = "停车场内" if any(token in raw_location for token in ("parking", "garage", "parkhaus")) else "项目运行区域内"
        mode = self._display_mode(
            getattr(scenario, "operating_mode", "") or facts.get("operating_mode", "")
        )
        action = self._first_category(categories, (
            "ACTION_PARK", "ACTION_REVERSE", "ACTION_STOP", "ACTION_HOLD",
            "ACTION_ACCELERATE", "ACTION_TURN",
        ))
        traffic = self._first_category(categories, (
            "TRAFFIC_FOLLOWING", "TRAFFIC_ONCOMING", "TRAFFIC_CROSSING",
            "TRAFFIC_CUT_IN", "TRAFFIC_REVERSE", "TRAFFIC_PARKING",
        ))
        object_value = self._first_category(categories, (
            "OBJECT_TRAM", "OBJECT_PEDESTRIAN", "OBJECT_CYCLIST", "OBJECT_VEHICLE",
            "OBJECT_STATIC", "OBJECT_OCCUPANT",
        ))
        speed = self.speed_text(scenario)
        if object_value and facts.get("object_speed_kph") == 0 and str(facts.get("object_longitudinal_direction", "")).upper() == "STATIONARY":
            object_value = "静止" + object_value
        context_text = self._context_text(scenario)
        position = self._object_position(scenario)

        clauses = [location]
        if condition := facts.get("selected_source_condition"):
            clauses.append(f"代表运行条件为{condition['value']}")
        if mode:
            clauses.append(f"AVP处于{mode}状态")
        driver_position = {
            "in_driver_seat": "驾驶员在驾驶位",
            "outside_driver_seat": "驾驶员不在驾驶位",
        }.get(str(facts.get("driver_position", "")).strip().casefold(), "")
        if driver_position:
            clauses.append(driver_position)
        if context_text:
            clauses.append(context_text)
        if action:
            clauses.append(f"车辆{action}")
        if traffic:
            clauses.append(f"周边存在{traffic}")
        elif object_value:
            clauses.append(f"周边{position}存在{object_value}" if position else f"周边存在{object_value}")
        if motion := self._object_motion(scenario):
            clauses.append(motion)
        if speed:
            clauses.append(speed.replace("：", "为", 1))
        operational = "，".join(clauses) + "。"

        details = []
        slope_direction = facts.get("source_slope_direction", source_conditions.get("source_slope_direction", ""))
        slope_pct = facts.get("source_slope_pct", source_conditions.get("source_slope_pct"))
        if slope_direction and isinstance(slope_pct, (int, float)):
            details.append(f"道路坡度：{slope_direction}{self._number(slope_pct)}%")
        road = self._first_category(categories, ("ROAD_LOW_FRICTION", "ROAD_SLOPE"))
        if road:
            details.append(f"道路条件：{road}")
        if object_value:
            details.append(f"交通对象：{object_value}")
        if position:
            details.append(f"对象位置：{position}")
        if traffic:
            details.append(f"交通关系：{traffic}")
        if motion:
            details.append(motion)
        if condition:
            details.append(f"条件影响：{condition['physical_effect']}")
        if speed:
            details.append(speed)
        for field, label in (("relative_distance_m", "目标距离"), ("longitudinal_acceleration_mps2", "故障后纵向加速度")):
            if isinstance(facts.get(field), (int, float)) and not isinstance(facts[field], bool):
                unit = FIELD_UNITS[field]
                details.append(f"{label}：{quantity(facts[field], unit)} {unit}")
        dimensions = self._dimensions(scenario)
        selected_conditions = [self._chinese_source(binding.get("label") or binding.get("atom_label"))
            for binding in dimensions.values() if isinstance(binding, Mapping)
            and binding.get("resolution_status") == "RESOLVED"]
        selected_conditions = [label for label in selected_conditions if label]
        if selected_conditions:
            details.append("方法条件：" + " / ".join(selected_conditions))
        return operational, "；".join(details) + "。" if details else "运行细节未确定。"

    @classmethod
    def hazardous_event(cls, value: Any, scenario: Any = None) -> str:
        text = narrative_quantities(str(value or "").strip())
        if scenario is not None:
            identity = str((getattr(scenario, "facts", {}) or {}).get("object_id") or "")
            label = cls._first_category(cls._categories(scenario), (
                "OBJECT_PEDESTRIAN", "OBJECT_CYCLIST", "OBJECT_VEHICLE", "OBJECT_STATIC", "OBJECT_TRAM"))
            if identity and label:
                text = re.sub(rf"(?<![A-Za-z0-9_]){re.escape(identity)}(?![A-Za-z0-9_])", lambda _: label, text)
        for raw in ("VEHICLE", "PEDESTRIAN", "CYCLIST", "TRAM"):
            display = cls._CATEGORY_TEXT["OBJECT_" + raw]
            # Translate standalone type tokens, including an already appended
            # Chinese label, without changing identifiers such as VEHICLE_01.
            text = re.sub(rf"(?<![A-Za-z0-9_-]){raw}(?![A-Za-z0-9_-])(?:\s*{re.escape(display)})?",
                          lambda _match, label=display: label, text)
        text = re.sub(r"(?<![A-Za-z])EPB(?![A-Za-z])", "电子驻车制动", text, flags=re.IGNORECASE)
        text = re.sub(r"(?<![A-Za-z])System\s+OK(?![A-Za-z])", "系统正常", text, flags=re.IGNORECASE)
        for raw, display in cls._MODE_TEXT.items():
            text = re.sub(rf"(?<![A-Za-z]){re.escape(raw)}(?![A-Za-z])", display, text, flags=re.IGNORECASE)
        for raw, display in cls._COLLISION_TEXT.items():
            text = re.sub(rf"(?<![A-Za-z0-9_-]){raw}\s*(?:接触|碰撞)", display, text)
        clauses = [item.strip() for item in re.split(r"[；;]", text) if item.strip()]
        return "；".join(dict.fromkeys(clauses)).rstrip("。") + "。" if clauses else "待危险事件确认。"

    def pending_value(self, evidence: Any) -> str:
        return self._PENDING_VALUE

    @classmethod
    def known_inputs(cls, facts: Mapping[str, Any]) -> str:
        labels = {"ego_speed_kph": "初始自车速度", "object_speed_kph": "目标速度",
                  "relative_speed_kph": "相对速度", "relative_distance_m": "相对距离",
                  "ttc_s": "TTC", "longitudinal_acceleration_mps2": "纵向加速度"}
        parts = [f"{label} {quantity(facts[field], FIELD_UNITS[field])} {FIELD_UNITS[field]}"
                 for field, label in labels.items() if facts.get(field) is not None]
        position = cls._POSITION_TEXT.get(str(facts.get("object_position", "")).lower())
        if position:
            parts.append(position + "目标")
        directions = {"FORWARD": "前进", "REVERSE": "倒车", "STATIONARY": "静止", "LATERAL": "侧向运动"}
        for field, label in (("ego_longitudinal_direction", "自车"), ("object_longitudinal_direction", "目标")):
            if facts.get(field) in directions:
                parts.append(label + directions[facts[field]])
        if facts.get("fault_motion_model"):
            parts.append("所选运动模型见场景明细")
        return "、".join(parts)

    @classmethod
    def condition_combination(cls, scenario: Any) -> str:
        """One observed combination, never a product of independent value lists."""
        facts = scenario.facts
        grade = facts.get("physical_analysis", {}).get("grade_mapping", {})
        values = [
            {"FORWARD": "前进", "REVERSE": "倒车", "STATIONARY": "初始静止"}.get(
                str(facts.get("ego_longitudinal_direction", "")).upper(), "运动方向待定"),
            cls._POSITION_TEXT.get(str(facts.get("object_position", "")).lower(), "目标方位待定") + "目标",
            {"UPHILL": "车头朝上坡", "DOWNHILL": "车头朝下坡"}.get(
                grade.get("ego_heading_relative_to_downhill"), ""),
            {"in_driver_seat": "在驾驶位", "outside_driver_seat": "不在驾驶位"}.get(
                facts.get("driver_position"), ""),
        ]
        return "＋".join(value for value in values if value)

    def group_differences(self, scenarios: list[Any]) -> str:
        """Name varying raw conditions; keep their actual combinations in detail."""
        fields = {"ego_speed_kph": "初速", "relative_distance_m": "目标距离",
            "longitudinal_acceleration_mps2": "加速度", "object_position": "目标方位",
            "object_id": "目标身份", "road_user_type": "目标类型", "driver_position": "驾驶员位置",
            "ego_longitudinal_direction": "自车方向", "object_speed_kph": "目标速度",
            "source_slope_direction": "坡向", "source_slope_pct": "坡度", "fault_motion_model": "运动模型"}
        varied = [label for key, label in fields.items()
            if any(s.facts.get(key) != scenarios[0].facts.get(key) for s in scenarios[1:])]
        for dimension, label in self._EXPOSURE_DIMENSION_LABELS.items():
            if any(self._dimensions(s).get(dimension) != self._dimensions(scenarios[0]).get(dimension)
                   for s in scenarios[1:]):
                varied.append(label)
        if any(s.operating_mode != scenarios[0].operating_mode for s in scenarios[1:]):
            varied.append("运行模式")
        return ("、".join(dict.fromkeys(varied)) + "不同，实际组合及参数见场景明细"
                if varied else "显示的运行条件相同；身份与结果分别保留在场景明细")

    @staticmethod
    def safety_intent(malfunctions: list[str], original: str) -> str:
        """Restate the existing prohibited output; never add a control measure."""
        intents = list(dict.fromkeys(text.strip().rstrip("。；") for text in malfunctions if text.strip()))
        if len(intents) == 1:
            return "防止" + intents[0] + "。"
        # Distinct intents must remain visible, not collapse to the first member.
        return "；".join("防止" + text for text in intents) + "。" if intents else original

    def safety_goal_presentation(self, members: list[dict], original: str) -> dict:
        """Reword supported existing intents, keeping every member's evidence.

        This report-language catalog is deliberately not a new SG generator.
        Exact source matches cannot silently broaden an unknown member's scope.
        """
        records = []
        for member in members:
            matches = [rule for rule in self.language.get("safety_goal_rewordings", [])
                       if rule.get("source") and all(member.get(field) == value
                           for field, value in rule["source"].items())]
            unambiguous = len({rule["text"] for rule in matches}) == 1
            text, basis = self._grounded_safety_intent(member, original)
            if unambiguous:
                text, basis = matches[0]["text"], matches[0]["source"]
            records.append({**member, "display_text": text,
                "wording_id": matches[0]["id"] if unambiguous else "grounded_behavior_excerpt",
                "basis": basis})
        texts = list(dict.fromkeys(record["display_text"].rstrip("。；") for record in records))
        # A supported complete short intent subsumes its repeated elaboration.
        # Do not use a common character prefix (it may cut away a negation).
        texts = [text for text in texts if not any(
            text != other and text.startswith(other) for other in texts)]
        return {"text": "；".join(texts) + "。" if texts else original,
                "members": records, "nature": "DISPLAY_ONLY_NO_ENGINEERING_APPROVAL"}

    @staticmethod
    def _grounded_safety_intent(member: Mapping, original: str) -> tuple[str, dict]:
        """Extract the stated unsafe behavior, not a new safety measure/outcome.

        This is source-clause compression. Unknown wording stays as the original
        goal rather than manufacturing an intent from a guideword or parameters.
        """
        hazard = str(member.get("vehicle_hazard", ""))
        # An "or" in the parent hazard names different mechanisms. The concrete
        # event, rather than the first parent alternative, determines this member.
        alternatives = bool(re.search(r"[，；。]\s*或|或者|\bor\b", hazard))
        fields = ("hazardous_event",) if alternatives else (
            "vehicle_hazard", "functional_effect", "hazardous_event")
        for field in fields:
            source = str(member.get(field, "")).strip()
            clauses = [c.strip(" 。；，") for c in re.split(r"[，；。]|(?:从而|导致|造成|引发)", source) if c.strip()]
            for clause in clauses:
                # Do not turn a negated or uncertain harm claim into an intent.
                if re.search(r"未发生|未出现|未产生|不会|不导致|不造成|不引发|无碰撞|无接触|不能确认|待确定|未知|可能受伤|存在.*风险", clause):
                    continue
                if field == "hazardous_event":
                    # Drop an explicit scene prefix, not arbitrary numbers: a
                    # parameter-bearing sentence otherwise remains in evidence.
                    clause = re.sub(r"^.*(?:行驶时|过程中|场景下)[，、]?", "", clause)
                    clause = re.sub(r"^.*?因(?=.+(?:失效|失去|无法|卡死|异常))", "", clause)
                if re.search(r"\d", clause):
                    continue
                if re.search(r"非预期|意外|失控|失去|失效|偏离|异常|丧失|溜车|过大|过小|不足|无法|不能|过度|延迟|提前|滞后|卡死|停滞|持续运行|固定不变|不随.*变化|反向|反而|虚假.*干扰|突发.*制动|与.*相反|未.*(?:退出|关闭|激活|驻车)", clause):
                    clause = re.sub(r"^(?:发生|出现|产生)", "", clause)
                    clause = re.sub(r"^(?:车辆|本车)(?:发生|出现)", "车辆", clause)
                    return "防止" + clause + "。", {field: source, "selected_behavior": clause,
                        "function": member.get("function", ""), "malfunction": member.get("malfunction", ""),
                        "hazardous_event": member.get("hazardous_event", "")}
        return original, {"original_goal": original, "reason": "No independently stated unsafe behavior to compress"}

    @classmethod
    def _collision_description(cls, road_user_type: Any, collision_type: Any) -> str:
        target = {"VEHICLE": "车辆", "PASSENGER_CAR": "乘用车", "PEDESTRIAN": "行人", "CYCLIST": "骑行者", "FIXED_OBJECT": "固定障碍物"}.get(road_user_type, "")
        collision = cls._COLLISION_TEXT.get(collision_type, "")
        if (collision or collision_type == "VEHICLE_TO_ROAD_USER") and road_user_type in {"PEDESTRIAN", "CYCLIST", "FIXED_OBJECT"}:
            side = {"FRONTAL": "前部", "REAR_END": "后部", "SIDE": "侧面", "LATERAL": "侧面"}.get(collision_type, "")
            return f"车辆{side}与{target}碰撞"
        return target + collision

    @classmethod
    def branch_result(cls, branch: Mapping[str, Any]) -> str:
        scores = " / ".join(str(branch.get(key) or "待评定") for key in ("S", "E", "C", "ASIL"))
        parts = [f"分支 {branch.get('branch_id', '')}：{scores}"]
        context = branch.get("risk_context", {})
        def value(field):
            evidence = context.get(field, {})
            return evidence.get("value") if isinstance(evidence, Mapping) else None
        description = cls._collision_description(value("road_user_type"), value("collision_type"))
        if description:
            parts.append(description)
        if branch.get("relative_speed_kph") is not None:
            parts.append("碰撞相对速度 " + quantity(branch["relative_speed_kph"], "km/h") + " km/h")
        if branch.get("ttc_s") is not None:
            parts.append("TTC " + quantity(branch["ttc_s"], "s") + " s")
        elif branch.get("first_contact_time_s") is not None:
            parts.append("首次接触时间 " + quantity(branch["first_contact_time_s"], "s") + " s")
        else:
            parts.append("接触时间待定")
        return "；".join(parts)

    def potential_harm(self, value: Any, *, severity: Any, method_description: str = "") -> str:
        if str(value or "").strip():
            return self.language.get("potential_harm_translations", {}).get(str(value), str(value))
        if getattr(severity, "value", None):
            if method_description:
                description = self.language.get("potential_harm_translations", {}).get(
                    method_description, method_description)
                return f"按当前 S 分档：{description}"
            return "伤害描述待确定"
        if _status(severity) != "FINALIZED":
            return "待S评定完成后确定"
        return self._PENDING_VALUE

    def severity_rationale(self, evidence: Any, trace: Mapping[str, Any]) -> str:
        if getattr(evidence, "value", None):
            inputs = trace.get("physical_inputs", {})
            speed = inputs.get("relative_speed_kph", trace.get("inputs", {}).get("relative_speed_kph"))
            description = self._collision_description(inputs.get("road_user_type"), inputs.get("collision_type"))
            parts = [description] if description else []
            if isinstance(speed, (int, float)):
                parts.append("碰撞相对速度 " + quantity(speed, "km/h") + " km/h")
            if parts:
                return "；".join(parts) + f"，对应伤害等级 {evidence.value}。"
        if _status(evidence) == "FINALIZED":
            production = trace.get("input_production", {})
            if production.get("rule_id"):
                return "已按来源支持的代表碰撞工况完成 S 评定；未证明所有故障轨迹的速度上界。"
            return "已按当前方法完成 S 评定。"
        production = trace.get("input_production", {})
        reason = str(trace.get("pending_reason", "") or getattr(evidence, "review_reason", ""))
        missing = next((text for code, text in (
            ("MISSING_RELATIVE_SPEED", "缺少该危险事件可用于评定的碰撞相对速度"),
            ("MISSING_ROAD_USER_TYPE", "缺少交通参与者类型"),
            ("MISSING_COLLISION_TYPE", "缺少碰撞类型信息"),
        ) if self._has_reason(reason, code)), "")
        if isinstance(production, Mapping) and production.get("pending_reason"):
            reasons = {
                "TEMPLATE_MECHANISM_NOT_SUPPORTED": "原模板与当前失效机制不匹配，所供对象/几何条件不能沿用",
                "MOTION_MODEL_OR_ENVELOPE_NOT_SOURCED": "缺故障后的适用运动模型或速度包络",
                "TARGET_OR_PHYSICAL_SOURCE_UNRESOLVED": "碰撞对象、构型或其物理条件尚未由适用来源确定",
            }
            known = production.get("after", {})
            available = self.known_inputs(known)
            cause = reasons.get(production['pending_reason'], "")
            explanation = "；".join(part for part in (missing, cause) if part)
            return f"{explanation or '输入生产尚有未决项，具体原因见风险追溯'}；已有 {available or '来源场景条件'}；S 暂未定。"
        if missing:
            return missing + "，S 暂不评定。"
        return "缺少 S 评定所需项目事实，S 暂不评定。"

    def exposure_rationale(self, evidence: Any, trace: Mapping[str, Any], *, calculated: bool = False) -> str:
        if _status(evidence) == "FINALIZED" or calculated:
            bindings = trace.get("atom_details", trace.get("atom_bindings", []))
            atoms = []
            if isinstance(bindings, list):
                for item in bindings:
                    if not isinstance(item, Mapping) or not item.get("used"):
                        continue
                    level = str(item.get("e_rank", item.get("E_class", ""))).strip()
                    dimensions = item.get("dimensions", item.get("dimension", []))
                    if not isinstance(dimensions, (list, tuple)):
                        dimensions = [dimensions]
                    labels = []
                    for dimension in dimensions:
                        term = trace.get("scenario_terms", {}).get(str(dimension), {})
                        atom_label = str(term.get("atom_label") or "")
                        if atom_label and item.get("atom_id") in {term.get("atom_id"), term.get("canonical_atom_id")}:
                            # The rank belongs to the selected method condition,
                            # not to the source phrase used to search for it.
                            labels.append(self.language.get("scenario_condition_translations", {}).get(atom_label, atom_label))
                        else:
                            labels.append(self._EXPOSURE_DIMENSION_LABELS.get(str(dimension).strip().upper(), "其他场景维度"))
                    if level:
                        labels = labels or ["其他场景维度"]
                        atoms.append(f"{'/'.join(dict.fromkeys(labels))}={level}")
            coupling = trace.get("dependency_coupling", {})
            coupling = coupling if isinstance(coupling, Mapping) else {}
            branch = str(trace.get("aggregation_rule", coupling.get("policy_branch", "")) or "")
            branch_text = {
                "all_e4": "各条件暴露等级均为 E4",
                "e3_e4_mix": "E3 与 E4 条件共同出现",
                "min_when_unequal": "不同暴露等级组合取较低等级",
                "same_independent_minus_one": "同等级条件相互独立，组合降低一级",
                "same_coupled_no_change": "同等级条件存在耦合，组合保持原等级",
            }.get(branch, "依据所列运行条件的暴露等级组合")
            atom_text = "、".join(dict.fromkeys(atoms)) if atoms else "各运行条件的暴露等级见明细"
            final_value = str(trace.get("result", getattr(evidence, "value", "")) or "")
            skipped = [item for item in bindings if item.get("skipped")]
            gap = "；无可用暴露值的条件未参与本次组合，见追溯" if skipped else ""
            return f"{atom_text}；{branch_text}，最终 {final_value}{gap}。"
        readiness = trace.get("input_readiness", {})
        readiness = readiness if isinstance(readiness, Mapping) else {}
        readiness_status = str(readiness.get("status", ""))
        relevant = readiness.get("unresolved_relevant_dimensions", [])
        relevant_text = "、".join(
            self._EXPOSURE_DIMENSION_LABELS.get(
                str(item).strip().upper(), "其他场景维度"
            )
            for item in relevant
        ) if isinstance(relevant, list) else ""
        if readiness_status == "PENDING_RELEVANT_DIMENSION":
            return (
                f"当前方法要素集合不足以确定 E；{relevant_text or '仍有'}相关场景维度未解析，"
                "其源定义方法要素可能改变当前结果，E 暂不评定。"
            )
        if readiness_status == "PENDING_ATOM_BINDING":
            return "已解析场景维度的方法要素绑定不完整或未纳入当前集合，E 暂不评定。"
        if readiness_status == "PENDING_AMBIGUOUS_ATOM_SET":
            return "暴露度相关场景维度存在多个未消解的方法要素候选，E 暂不评定。"
        if readiness_status == "SOURCE_CONFLICT":
            if readiness.get("reason_code") == "EXPOSURE_SCENARIO_CONSTRAINT_CONFLICT":
                return "已选道路与自车道路关系的坡度区间互不相容，不能组成同一工况；需重新生成相容条件，E 暂不评定。"
            return "暴露度的方法契约来源或组件域存在冲突，E 暂不评定。"
        reason = trace.get("missing_method_semantics", "") or getattr(evidence, "review_reason", "")
        if self._has_reason(reason, "EXPOSURE_DIMENSION_COVERAGE"):
            return "暴露度场景维度覆盖规则未定义，E 暂不评定。"
        return "缺少 E 评定所需方法语义或项目事实，E 暂不评定。"

    def controllability_rationale(self, evidence: Any, trace: Mapping[str, Any]) -> str:
        if getattr(evidence, "value", None):
            if trace.get("decision_status") == "CONDITIONAL_REFERENCE_MATCH":
                rule = trace.get("reference_rule", {})
                speed = rule.get("trajectory_max_speed_kph")
                prefix = f"接触前轨迹速度上界 {quantity(speed, 'km/h')} km/h；" if isinstance(speed, (int, float)) else ""
                return prefix + rule.get("rationale", "按参考工况进行条件性 C 评定；适用性待审核")
            override = trace.get("override_inputs", trace.get("inputs", trace.get("physical_inputs", {})))
            if trace.get("decision_status") == "OVERRIDE_MATCH" or trace.get("override", {}).get("matched") is True:
                conditions = [yes if override[key] else no for key, yes, no in (
                    ("driver_in_vehicle", "驾驶员在车内", "驾驶员不在车内"),
                    ("remote_intervention_available", "可远程干预", "无远程干预"),
                    ("other_road_user_avoidance_possible", "其他交通参与者可避让", "其他交通参与者不能避让"),
                ) if isinstance(override.get(key), bool)]
                return ("、".join(conditions) + "；" if conditions else "") + f"按已命中的优先干预规则评为 {evidence.value}，未使用 TTC 分档。"
            inputs = trace.get("physical_inputs", {})
            ttc = inputs.get("ttc_s", trace.get("derived_ttc", {}).get("ttc_s"))
            if isinstance(ttc, (int, float)):
                basis = self._ttc_intervention_basis(trace)
                if trace.get("unknown_policy_action") == "SKIP_TO_TTC" and override.get("driver_in_vehicle") is False:
                    basis += "；驾驶员在车外，当前时间分档未证明实际可用的干预路径"
                return ((basis + "；") if basis else "") + f"预计 {quantity(ttc, 's')} s 后首次接触，按当前纵向可控性时间分档为 {evidence.value}。"
            available = [label for key, label in (("driver_in_vehicle", "驾驶员在车内"),
                ("remote_intervention_available", "可远程干预"), ("other_road_user_avoidance_possible", "其他交通参与者可避让"))
                if override.get(key) is True]
            if available:
                return "、".join(available) + f"；按适用干预条件评为 {evidence.value}。"
        if _status(evidence) == "FINALIZED":
            return "已按当前方法完成 C 评定。"
        if trace.get("decision_status") in {"MODEL_NOT_APPLICABLE", "MODEL_DEPENDENCY_UNRESOLVED", "PROFILE_APPLICABILITY_UNRESOLVED"}:
            control = {**trace, "pending_reason": trace.get("pending_reason") or getattr(evidence, "review_reason", "")}
            reason = self.controllability_branch({"controllability": control})
            available = self.known_inputs(trace.get("known_physical_inputs", {}))
            return reason + (f"；已有 {available}" if available else "") + "；C 暂未定。"
        diagnostic = trace.get("input_diagnostic", {})
        if isinstance(diagnostic, Mapping) and diagnostic.get("reason"):
            labels = {
                "MISSING_DISTANCE": "缺相对距离", "MISSING_GEOMETRY": "缺运动方向或几何关系",
                "MODEL_NOT_APPLICABLE": "当前交互不满足现有 TTC 模型条件",
                "NOT_CLOSING": "当前匀速模型不接近，未证明后续过程，不能判 C0",
                "TTC_NOT_CLOSING": "当前匀速模型不接近，未证明后续过程，不能判 C0",
                "SOURCE_NOT_FINALIZED": "已有数值的来源未通过校验",
            }
            code = str(diagnostic["reason"])
            known = trace.get("known_physical_inputs", {})
            if code == "MODEL_NOT_APPLICABLE" and known.get("fault_motion_model") in {
                    "CONSTANT_LATERAL_ACCELERATION_TO_CONTACT", "CONSTANT_SPEED_CURVATURE_TO_CONTACT"}:
                labels[code] = "侧向交互：当前纵向 TTC 模型不适用"
            available = self.known_inputs(known)
            return f"{labels.get(code, '可控性输入尚有未决项，具体原因见风险追溯')}；已有 {available or '场景方法要素'}；C 暂未定。"
        reason = trace.get("decision_status", "") or getattr(evidence, "review_reason", "")
        if self._has_reason(reason, "METHOD_BRANCH_UNRESOLVED") or self._has_reason(reason, "UNKNOWN_BRANCH_POLICY_UNSPECIFIED"):
            return "未知分支策略未定义，C 暂不评定。"
        return "缺少 C 评定所需控制上下文或方法分支，C 暂不评定。"

    def asil_rationale(self, evidence: Any, trace: Mapping[str, Any]) -> str:
        grades = trace.get("branch_grades", trace)
        if getattr(evidence, "value", None) and all(grades.get(k) for k in ("S", "E", "C")):
            return "同一危险分支的 " + " / ".join(str(grades[k]) for k in ("S", "E", "C")) + f" 对应 ASIL {evidence.value}。"
        if getattr(evidence, "value", None):
            known = " / ".join(str(grades[k]) for k in ("S", "E", "C") if grades.get(k))
            missing = "/".join(k for k in ("S", "E", "C") if not grades.get(k))
            return (f"已记录的分支 ASIL 为 {evidence.value}；已有 {known or '分支结果'}，"
                    f"{missing} 仍未评定；该结果不表示所有维度均已完成，方法依据见追溯。")
        if _status(evidence) == "FINALIZED":
            return "已按当前 ASIL 矩阵完成评定。"
        return "S/E/C 未全部确定，ASIL 暂不评定。"

    @staticmethod
    def safe_state_path(candidate: Mapping[str, Any]) -> tuple[tuple, str] | None:
        """A display group for a response path, never risk/parameter equivalence."""
        response = candidate.get("transition", {})
        if response.get("model") != "CONSTANT_NET_DECELERATION":
            return None
        labels = {"driver_braking": "驾驶员制动踏板", "braking": "制动执行器",
                  "automatic_braking_command": "自动制动命令链", "remote_control": "远程控制链",
                  "automatic_drive_command": "自动驱动命令链", "automatic_steering_command": "自动转向命令链",
                  "automatic_parking_command": "自动驻车命令链",
                  "parking_hold": "驻车保持", "steering": "转向", "drive": "驱动",
                  "speed_limiter": "限速控制"}
        controls = tuple(sorted(candidate.get("required_controls", [])))
        if not controls or any(c not in labels for c in controls):
            return None
        held = response.get("post_stop_motion") == "HOLD_POSITION"
        positions = tuple(sorted(candidate.get("driver_positions", [])))
        scope = {("in_driver_seat",): "驾驶员在驾驶位", ("outside_driver_seat",): "驾驶员不在驾驶位"}.get(positions, "按成员适用条件")
        label = f"{scope}，通过{'、'.join(labels[c] for c in controls)}减速至停止" + ("并保持静止" if held else "")
        return (controls, positions, held), label

    @classmethod
    def safe_state_member(cls, candidate: Mapping[str, Any], scenario: Any) -> str:
        text = "安全状态候选：" + cls.hazardous_event(candidate["description"], scenario)
        response = candidate.get("transition", {})
        if response.get("net_deceleration_mps2") is not None:
            text += "；处置净减速度 " + quantity(response["net_deceleration_mps2"], "m/s2") + " m/s²"
        return text

    @staticmethod
    def ftti_value(timing_inputs: list[Mapping[str, Any]]) -> str:
        lost = [x for x in timing_inputs if x.get("timing_trace", {}).get("status") == "AVOIDABILITY_ALREADY_LOST"]
        values = [x["hazard_time_limit_s"] for x in timing_inputs
                  if x.get("hazard_time_limit_s") is not None and x not in lost]
        if not values:
            return "初始已无制动避碰余量" + ("；部分成员待确定" if len(lost) != len(timing_inputs) else "") if lost else "待确定"
        label = quantity(min(values), "s") + " s"
        return label + ("；含初始无制动避碰余量成员" if lost else "") + (
            "；部分成员待确定" if len(values) + len(lost) != len(timing_inputs) else "")

    @staticmethod
    def ftti_rationale(timing_inputs: list[Mapping[str, Any]] | None = None) -> str:
        boundary_reasons = {
            "NO_UNMITIGATED_CONTACT_BOUNDARY": "当前未干预轨迹未形成接触，不能为该接触判据给出时间边界。",
            "NO_CONTACT_UNDER_THIS_MOTION": "该运动假设下不会发生接触，没有该接触判据的时间边界。",
            "RESPONSE_NOT_STRONGER_THAN_FAULT_DECELERATION": "假设的处置净减速度弱于故障轨迹的既有减速度，当前制动边界算法不适用。",
            "RESPONSE_BOUNDARY_UNRESOLVED": "该处置轨迹尚未求得有效的时间边界。",
            "RESPONSE_INPUTS_UNRESOLVED": "响应计算所需参数仍不完整或不满足物理条件。",
        }
        explanations = []
        for item in timing_inputs or []:
            reason = str(item.get("reason", ""))
            if reason.startswith("未形成该危险判据的数值边界："):
                code = reason.split("：", 1)[1]
                reason = boundary_reasons.get(code, "当前轨迹尚未形成有效时间边界，具体原因见时间追溯。")
            if any("\u4e00" <= ch <= "\u9fff" for ch in reason):
                explanations.append(reason)
        reasons = "；".join(dict.fromkeys(explanations))
        lost = [item for item in timing_inputs or [] if item.get("timing_trace", {}).get("status") == "AVOIDABILITY_ALREADY_LOST"]
        lost_note = (f"{len(lost)}个成员在模型初始状态已无制动避碰余量：即使立即按假设净减速度制动，也不能在目标前停下。"
                     "原始0秒表示没有可用的响应延迟窗口，不是可供设计采用的正常FTTI，也不证明所有其他处置均无效。" if lost else "")
        calculated = [item for item in timing_inputs or [] if item.get("hazard_time_limit_s") is not None and item not in lost]
        if lost and not calculated:
            return lost_note + ("其余成员的具体时间缺口见逐成员追溯。" if len(lost) != len(timing_inputs or []) else "")
        if calculated:
            missing = len(timing_inputs or []) - len(calculated) - len(lost)
            moving = any(item.get("timing_trace", {}).get("response_model_family") == "MOVING_TARGET_BRAKING" for item in calculated)
            release = any(item.get("timing_trace", {}).get("response_model_family") == "MOVING_TARGET_FAULT_RELEASE" for item in calculated)
            return ("条件性时间边界：从故障发生到失去所选响应的避碰能力；"
                    + ("主表取已计算有效成员的最小值，逐成员数值见时间追溯。" if len(calculated) > 1 else "")
                    + ("解除异常减速的成员保持响应开始时的速度，在声明时域内检查与移动目标的间隔，不表示停车或瞬间恢复故障前速度。" if release else "")
                    + ("移动目标制动成员按原轨迹继续运动，检查故障行驶、制动及停车保持三段，在模型有效时段内确定首次失去避碰余量的边界。" if moving else
                       "依据同一轨迹、静止目标及剩余净减速假设计算。" if not release else "") +
                    "检测、决策和处置建立必须在该边界前完成；后续完成耗时不与此边界相加。"
                    "尚未经工程批准，不是完整ODD的FTTI；假设与逐成员计算见时间追溯。"
                    + lost_note + (f"仍有{missing}个成员尚无数值时间边界，具体控制、轨迹或输入缺口见逐成员追溯；已知最小值不代表全组结论。" if missing else ""))
        timings = [row for item in timing_inputs or [] for row in item.get("branch_timings") or [item]]
        available = [item for item in timings if item.get("contact_time_s") is not None]
        if not available:
            return "缺少可用的接触时间；" + (reasons or "故障起点、危险判据及安全处置时间预算尚未确定，FTTI待定。")
        kinds = {item.get("contact_time_kind") for item in available}
        label = "侧向首次接触" if kinds == {"first_contact_time_s"} else "轨迹接触"
        values = [item["contact_time_s"] for item in available]
        extent = (f"最小 {quantity(min(values), 's')} s、最大 {quantity(max(values), 's')} s"
                  "（仅为已计算样本统计，不代表连续可达范围）" if min(values) != max(values)
                  else quantity(values[0], "s") + " s")
        origin = ("从各模型初始状态计时" if all(
            item.get("time_origin") == "MODEL_INITIAL_STATE" for item in available)
            else "计时起点以各成员来源为准")
        return (f"已有{label}时间 {extent}（{origin}）；"
                + (reasons or "尚未绑定故障起点及危险判据") + "接触时间不能作为 FTTI。")


__all__ = ["EngineeringReportTextMapper"]
