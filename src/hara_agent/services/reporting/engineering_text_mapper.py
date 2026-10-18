from __future__ import annotations

from collections.abc import Mapping
import re
from typing import Any

from .quantity_format import quantity, quantity_range, FIELD_UNITS


def _status(evidence: Any) -> str:
    value = getattr(evidence, "status", "")
    return str(getattr(value, "value", value))


class EngineeringReportTextMapper:
    """Map committed runtime status to compact Chinese review text.

    This layer owns presentation wording only.  It neither supplements project
    facts nor makes a risk-method decision.
    """

    _PENDING_VALUE = "待评定"

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
            and str(point_provenance.get("approval", "")).upper() == "FINALIZED"
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

    def object_interaction_summary(self, scenario: Any) -> str:
        """Render only structured object and interaction facts."""
        categories = self._categories(scenario)
        object_value = self._first_category(categories, (
            "OBJECT_TRAM", "OBJECT_PEDESTRIAN", "OBJECT_CYCLIST", "OBJECT_VEHICLE",
            "OBJECT_STATIC", "OBJECT_OCCUPANT",
        ))
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
            ("相对速度", value("relative_speed_kph"), "km/h"),
            ("初始接近速度" if facts.get("physical_analysis") else "接近速度", closing_speed, "km/h"),
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
        if analysis:
            labels = {"ego_speed_kph": "初速(km/h)", "relative_distance_m": "行程(m)",
                "longitudinal_acceleration_mps2": "纵向加速度(m/s²)", "slope_percent": "坡度(%)",
                "residual_resistance_mps2": "残余阻力减速度(m/s²)", "lateral_clearance_m": "侧向净距(m)",
                "lateral_speed_mps": "初始侧速(m/s)", "lateral_acceleration_mps2": "侧向加速度(m/s²)",
                "model_valid_until_s": "模型时域(s)", "post_fault_speed_cap_kph": "残余限速(km/h)"}
            parts.append("假设实例：" + ", ".join(f"{labels.get(key, key)}={quantity(value, FIELD_UNITS.get(key, ""))}" for key, value in analysis.get("sample_parameters", {}).items()))
            for parameter in analysis.get("hypothesis", {}).get("parameters", []):
                if parameter["kind"] == "SYMBOLIC":
                    parts.append(f"符号条件：{labels.get(parameter['field'], parameter['field'])}={parameter['symbol']}")
                elif parameter["kind"] == "RANGE":
                    parts.append(f"假设范围：{labels.get(parameter['field'], parameter['field'])}={quantity_range(parameter['lower'], parameter['upper'], FIELD_UNITS.get(parameter['field'], ''))}；有限采样未证明连续不变性")
            if any(p["kind"] == "SYMBOLIC" for p in analysis.get("hypothesis", {}).get("parameters", [])):
                boundaries = analysis.get("sensitivity", {}).get("method_boundaries", [])
                parts.append("参数关系：v²=v₀²+2ad；d=v₀t+at²/2（首次接触且模型持续有效）" if model == "CONSTANT_ACCELERATION_TO_IMPACT_DISTANCE" else "参数关系见物理 trace")
                parts.extend(f"边界条件：{labels.get(row['parameter'], row['parameter'])}≈{quantity(row['boundary'], FIELD_UNITS.get(row['parameter'], ''))}；精确关系见物理追溯"
                    for row in boundaries[:3] if isinstance(row.get("boundary"), (float, int)))
            grade = analysis.get("grade_mapping", {})
            if grade:
                parts.append(f"坡向来源标签：{grade.get('source_tag') or '未注明'}；原参考方向：{ {'EGO_FORWARD': '车头前向', 'ROAD_REFERENCE': '道路参考向', 'UNSPECIFIED': '未注明'}.get(grade.get('source_reference'), '见来源')}")
                parts.append("朝向假设：" + ("车头朝上坡，重力向车后" if grade.get("ego_heading_relative_to_downhill") == "UPHILL" else "车头朝下坡，重力向车前"))
            inherited = [labels.get(field, field) for field, record in analysis.get("model_input_view", {}).items() if record.get("action") == "INHERITED"]
            if inherited:
                parts.append("相容父输入继承：" + "、".join(inherited))
            parts.append("适用条件：" + "；".join(cls._CONDITION_TEXT.get(text, text)
                for text in analysis.get("hypothesis", {}).get("conditions", [])))
            parts.append("未覆盖：" + "；".join(cls._CONDITION_TEXT.get(text, text)
                for text in analysis.get("sensitivity", {}).get("uncovered_conditions", [])))
            if facts.get("first_contact_time_s") is not None:
                parts.append(f"侧向首次接触：{quantity(facts['first_contact_time_s'], 's')} s（不作为纵向 TTC）")
            parts.append("假设追溯：" + str(analysis.get("plan_key", ""))[:12] + ":" + str(analysis.get("hypothesis_id", "")).split(":", 1)[-1])
        if model == "CONSTANT_SPEED_REPRESENTATIVE":
            parts.append("恒速代表工况；用于所列碰撞条件，不是故障最大速度包络")
        elif model:
            names = {"GRADE_ROLLAWAY_TO_IMPACT_DISTANCE": "坡道溜车（重力减残余阻力）",
                "CONSTANT_ACCELERATION_TO_IMPACT_DISTANCE": "恒加速度至首次接触",
                "CONSTANT_LATERAL_ACCELERATION_TO_CONTACT": "侧向恒加速度至首次接触"}
            parts.append(f"运动模型：{names.get(model, model)}")
        if not any(part.startswith("TTC：") for part in parts) and (
            str(derived.get("closing_speed_status", "")) == "TTC_NOT_CLOSING"
        ):
            parts.append("TTC：未接近，无有限值")
        position = cls._POSITION_TEXT.get(str(facts.get("object_position", "")).casefold(), "")
        if position:
            parts.append(f"目标位置：{position}")
        directions = {"FORWARD": "前进", "REVERSE": "倒车", "STATIONARY": "静止"}
        for field, label in (
            ("ego_longitudinal_direction", "自车方向"),
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
    def controllability_branch(trace: Mapping[str, Any]) -> str:
        control = trace.get("controllability", {})
        if not isinstance(control, Mapping) or not control:
            return ""
        if control.get("decision_status") == "MODEL_NOT_APPLICABLE":
            return "侧向接触模型；当前方法无适用 C 规则，局部待评定"
        if control.get("decision_status") == "MODEL_DEPENDENCY_UNRESOLVED":
            return "故障影响规避控制；C 方法依赖尚未成立"
        stage = {
            "OVERRIDE": "干预覆盖",
            "TTC": "TTC 分档",
            "UNRESOLVED": "分支未确定",
        }.get(str(control.get("decision_tree_stage", "")), "")
        parts = [f"C 分支：{stage}"] if stage else []
        rule_ids = control.get("rule_ids", [])
        if isinstance(rule_ids, (list, tuple)):
            parts.extend(f"命中规则：{item}" for item in rule_ids if str(item))
        policy = str(control.get("unknown_override_policy", ""))
        if policy:
            parts.append("未知干预条件的处理：" + {"SKIP_TO_TTC": "进入 TTC 分档",
                "UNSPECIFIED": "方法尚未明确", "PENDING": "保留待评定"}.get(policy, "见方法依据"))
        action = str(control.get("unknown_policy_action", ""))
        if action and action not in {"NOT_INVOKED", ""}:
            if action != policy:
                parts.append("处理结果：" + {"SKIP_TO_TTC": "进入 TTC 分档"}.get(action, "见方法依据"))
        return "；".join(parts)

    @staticmethod
    def analysis_basis(scenario: Any) -> str:
        provenance = getattr(scenario, "fact_provenance", {}) or {}
        if not isinstance(provenance, Mapping):
            return ""
        physical = (getattr(scenario, "facts", {}) or {}).get("physical_analysis")
        if physical:
            origins = physical.get("input_origins", {})
            labels = {"PROJECT_INPUT": "项目事实", "HUMAN_CONFIRMATION": "工程师输入",
                "SCENARIO_DEFINED": "方法场景条件", "METHOD_CONTRACT": "方法规则",
                "LLM_ANALYSIS_HYPOTHESIS": "LLM 分析假设", "DERIVED": "程序推导"}
            counts = {}
            for origin in origins.values():
                label = labels.get(origin, "既有条件")
                counts[label] = counts.get(label, 0) + 1
            return ("；".join(f"{label} {count} 项" for label, count in counts.items())
                + "；碰撞速度/首次接触时间由同一轨迹计算。条件性结果不代表车辆实测能力。"
                + "方案：" + physical["plan_key"][:12] + "；完整输入及来源见该运行的物理追踪。")
        labels = {
            "ego_speed_kph": "自车速度", "object_speed_kph": "目标速度",
            "relative_distance_m": "相对距离", "ttc_s": "TTC",
            "driver_position": "驾驶员配置",
            "fault_motion_model": "运动分析条件", "ego_impact_speed_kph": "碰撞速度",
        }
        parts = []
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
            description = [rule] if rule else []
            if approval and approval != "FINALIZED":
                description.append("待审设定")
            all_source_locations.extend(source_locations)
            if description:
                parts.append(f"{label}：{'；'.join(description)}")
        if all_source_locations:
            parts.append("来源：" + ", ".join(dict.fromkeys(all_source_locations)))
        return "；".join(parts)

    def scenario(self, scenario: Any, *, variant_count: int = 1) -> tuple[str, str]:
        """Render validated structured facts without exposing Method vocabulary."""
        facts = getattr(scenario, "facts", {}) or {}
        categories = self._categories(scenario)
        source_conditions = facts.get("source_operating_conditions", {})
        location = self._chinese_source(source_conditions.get("operating_scenario", "")) or self._chinese_source(
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
        context_text = self._context_text(scenario)
        position = self._object_position(scenario)

        clauses = [location]
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
        if speed:
            clauses.append(speed.replace("：", "为", 1))
        operational = "，".join(clauses) + "。"

        details = []
        slope_direction = facts.get("source_slope_direction", source_conditions.get("source_slope_direction", ""))
        slope_pct = facts.get("source_slope_pct", source_conditions.get("source_slope_pct"))
        if slope_direction and isinstance(slope_pct, (int, float)):
            details.append(f"道路坡度：{slope_direction}{self._number(slope_pct)}%")
        allowed_roads = source_conditions.get("permitted_road_types", [])
        if allowed_roads:
            details.append("允许道路/车位范围见 Item Definition；本行未逐项验证几何覆盖")
        road = self._first_category(categories, ("ROAD_LOW_FRICTION", "ROAD_SLOPE"))
        if road:
            details.append(f"道路条件：{road}")
        if object_value:
            details.append(f"交通对象：{object_value}")
        if position:
            details.append(f"对象位置：{position}")
        if traffic:
            details.append(f"交通关系：{traffic}")
        if speed:
            details.append(speed)
        dimensions = self._dimensions(scenario)
        selected_conditions = [self._chinese_source(binding.get("label") or binding.get("atom_label"))
            for binding in dimensions.values() if isinstance(binding, Mapping)
            and binding.get("resolution_status") == "RESOLVED"]
        selected_conditions = [label for label in selected_conditions if label]
        variant = self.variant_text(scenario)
        details.append(f"分析变体：{variant}" + ("（标签不单独证明新增风险边界）" if variant != "受控分析场景" else ""))
        if selected_conditions:
            details.append("方法条件：" + " / ".join(selected_conditions))
        if variant_count > 1:
            details.append(
                f"本逻辑组计划覆盖{variant_count}个分析变体；当前为综合前审阅投影"
            )
        return operational, "；".join(details) + "。"

    @classmethod
    def hazardous_event(cls, value: Any) -> str:
        text = str(value or "").strip()
        text = re.sub(r"(?<![A-Za-z])EPB(?![A-Za-z])", "电子驻车制动", text, flags=re.IGNORECASE)
        text = re.sub(r"(?<![A-Za-z])System\s+OK(?![A-Za-z])", "系统正常", text, flags=re.IGNORECASE)
        for raw, display in cls._MODE_TEXT.items():
            text = re.sub(rf"(?<![A-Za-z]){re.escape(raw)}(?![A-Za-z])", display, text, flags=re.IGNORECASE)
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
            text = matches[0]["text"] if unambiguous else self.safety_intent([member["malfunction"]], original)
            records.append({**member, "display_text": text,
                "wording_id": matches[0]["id"] if unambiguous else "original_intent",
                "basis": matches[0]["source"] if unambiguous else {"malfunction": member["malfunction"]}})
        texts = list(dict.fromkeys(record["display_text"].rstrip("。；") for record in records))
        return {"text": "；".join(texts) + "。" if texts else original,
                "members": records, "nature": "DISPLAY_ONLY_NO_ENGINEERING_APPROVAL"}

    @staticmethod
    def branch_result(branch: Mapping[str, Any]) -> str:
        scores = " / ".join(str(branch.get(key) or "待评定") for key in ("S", "E", "C", "ASIL"))
        parts = [f"分支 {branch.get('branch_id', '')}：{scores}"]
        context = branch.get("risk_context", {})
        for field, labels in (
            ("road_user_type", {"PEDESTRIAN": "行人", "CYCLIST": "骑行者", "VEHICLE": "车辆", "PASSENGER_CAR": "乘用车", "FIXED_OBJECT": "固定障碍物"}),
            ("collision_type", {"FRONTAL": "正面碰撞", "REAR_END": "追尾碰撞", "SIDE": "侧向碰撞", "LATERAL": "侧向碰撞"}),
        ):
            value = context.get(field, {})
            if isinstance(value, Mapping) and value.get("value") in labels:
                parts.append(labels[value["value"]])
        if branch.get("relative_speed_kph") is not None:
            parts.append("碰撞相对速度 " + quantity(branch["relative_speed_kph"], "km/h") + " km/h")
        if branch.get("ttc_s") is not None:
            parts.append("TTC " + quantity(branch["ttc_s"], "s") + " s")
        elif branch.get("first_contact_time_s") is not None:
            parts.append("侧向首次接触 " + quantity(branch["first_contact_time_s"], "s") + " s（非纵向 TTC）")
        else:
            parts.append("接触时间待定")
        return "；".join(parts)

    def potential_harm(self, value: Any, *, severity: Any) -> str:
        if str(value or "").strip():
            return self.language.get("potential_harm_translations", {}).get(str(value), str(value))
        if getattr(severity, "value", None):
            return "伤害描述待确定"
        if _status(severity) != "FINALIZED":
            return "待S评定完成后确定"
        return self._PENDING_VALUE

    def severity_rationale(self, evidence: Any, trace: Mapping[str, Any]) -> str:
        if "条件性" in str(getattr(evidence, "review_reason", "")):
            return f"条件性 S 试算：{getattr(evidence, 'value', '') or '尚缺适用输入'}；使用所列假设与当前方法表，不代表车辆实测能力。"
        if _status(evidence) == "FINALIZED":
            production = trace.get("input_production", {})
            if production.get("rule_id"):
                return "已按来源支持的代表碰撞工况完成 S 评定；未证明所有故障轨迹的速度上界。"
            return "已按当前方法完成 S 评定。"
        production = trace.get("input_production", {})
        if isinstance(production, Mapping) and production.get("pending_reason"):
            reasons = {
                "TEMPLATE_MECHANISM_NOT_SUPPORTED": "原模板与当前失效机制不匹配，所供对象/几何条件不能沿用",
                "MOTION_MODEL_OR_ENVELOPE_NOT_SOURCED": "缺故障后的适用运动模型或速度包络",
                "TARGET_OR_PHYSICAL_SOURCE_UNRESOLVED": "碰撞对象、构型或其物理条件尚未由适用来源确定",
            }
            known = production.get("after", {})
            available = self.known_inputs(known)
            return f"{reasons.get(production['pending_reason'], '输入生产尚有未决项，具体原因见风险追溯')}；已有 {available or '来源场景条件'}；S 暂未定。"
        reason = str(trace.get("pending_reason", "") or getattr(evidence, "review_reason", ""))
        if self._has_reason(reason, "MISSING_RELATIVE_SPEED"):
            return "缺少该危险事件的实际相对速度，S 暂不评定。"
        if self._has_reason(reason, "MISSING_ROAD_USER_TYPE"):
            return "缺少交通参与者类型，S 暂不评定。"
        if self._has_reason(reason, "MISSING_COLLISION_TYPE"):
            return "缺少碰撞类型信息，S 暂不评定。"
        return "缺少 S 评定所需项目事实，S 暂不评定。"

    def exposure_rationale(self, evidence: Any, trace: Mapping[str, Any], *, calculated: bool = False) -> str:
        if _status(evidence) == "FINALIZED" or calculated:
            requested = str(trace.get("requested_domain", "") or "")
            actual = str(trace.get("domain", trace.get("actual_domain", "")) or "")
            bindings = trace.get("atom_bindings", [])
            atoms = []
            if isinstance(bindings, list):
                for item in bindings:
                    if not isinstance(item, Mapping) or not item.get("used"):
                        continue
                    level = str(item.get("E_class", "")).strip()
                    dimensions = item.get("dimension", [])
                    if not isinstance(dimensions, (list, tuple)):
                        dimensions = [dimensions]
                    labels = [
                        self._EXPOSURE_DIMENSION_LABELS.get(
                            str(dimension).strip().upper(), "其他场景维度"
                        )
                        for dimension in dimensions if str(dimension).strip()
                    ]
                    if level:
                        labels = labels or ["其他场景维度"]
                        atoms.append(f"{'/'.join(labels)}={level}")
            coupling = trace.get("dependency_coupling", {})
            coupling = coupling if isinstance(coupling, Mapping) else {}
            branch = str(coupling.get("policy_branch", "") or "")
            branch_text = {
                "all_e4": "有效方法要素均为 E4，按 FUSA v1 全 E4 规则",
                "e3_e4_mix": "有效方法要素包含 E3 与 E4，按 FUSA v1 混合高等级规则",
                "min_when_unequal": "有效方法要素等级不相同，按 FUSA v1 取较低等级",
                "same_independent_minus_one": "同等级方法要素相互独立，按 FUSA v1 降一级",
                "same_coupled_no_change": "同等级方法要素存在耦合，按 FUSA v1 不降级",
            }.get(branch, "按已记录的 FUSA v1 聚合规则")
            domain_text = (
                f"请求 {requested} 域，实际使用 {actual} 域"
                if requested and actual else "已记录 Exposure 域"
            )
            fallback = "；请求域无可用值，已回退到实际域" if trace.get("scenario_level_fallback") else ""
            atom_text = "、".join(atoms) if atoms else "方法要素与聚合步骤见风险追溯"
            final_value = str(trace.get("result", getattr(evidence, "value", "")) or "")
            return f"{domain_text}；{atom_text}；{branch_text}{fallback}，最终 {final_value}。"
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
            return "暴露度的方法契约来源或组件域存在冲突，E 暂不评定。"
        reason = trace.get("missing_method_semantics", "") or getattr(evidence, "review_reason", "")
        if self._has_reason(reason, "EXPOSURE_DIMENSION_COVERAGE"):
            return "暴露度场景维度覆盖规则未定义，E 暂不评定。"
        return "缺少 E 评定所需方法语义或项目事实，E 暂不评定。"

    def controllability_rationale(self, evidence: Any, trace: Mapping[str, Any]) -> str:
        if _status(evidence) == "FINALIZED":
            return "已按当前方法完成 C 评定。"
        diagnostic = trace.get("input_diagnostic", {})
        if isinstance(diagnostic, Mapping) and diagnostic.get("reason"):
            labels = {
                "MISSING_DISTANCE": "缺相对距离", "MISSING_GEOMETRY": "缺运动方向或几何关系",
                "MODEL_NOT_APPLICABLE": "侧向交互：当前纵向 TTC 模型不适用",
                "NOT_CLOSING": "当前匀速模型不接近，未证明后续过程，不能判 C0",
                "TTC_NOT_CLOSING": "当前匀速模型不接近，未证明后续过程，不能判 C0",
                "SOURCE_NOT_FINALIZED": "已有数值的来源未通过校验",
            }
            code = str(diagnostic["reason"])
            known = trace.get("known_physical_inputs", {})
            available = self.known_inputs(known)
            return f"{labels.get(code, '可控性输入尚有未决项，具体原因见风险追溯')}；已有 {available or '场景方法要素'}；C 暂未定。"
        reason = trace.get("decision_status", "") or getattr(evidence, "review_reason", "")
        if self._has_reason(reason, "METHOD_BRANCH_UNRESOLVED") or self._has_reason(reason, "UNKNOWN_BRANCH_POLICY_UNSPECIFIED"):
            return "未知分支策略未定义，C 暂不评定。"
        return "缺少 C 评定所需控制上下文或方法分支，C 暂不评定。"

    def asil_rationale(self, evidence: Any, trace: Mapping[str, Any]) -> str:
        if "条件性" in str(getattr(evidence, "review_reason", "")):
            return "条件性分支 ASIL 试算；每个分支按自身 S/E/C 查询当前矩阵。未确认工程结论。"
        if _status(evidence) == "FINALIZED":
            return "已按当前 ASIL 矩阵完成评定。"
        return "S/E/C 未全部确定，ASIL 暂不评定。"

    @staticmethod
    def ftti_rationale() -> str:
        return "当前运行未启用 FTTI 计算。"


__all__ = ["EngineeringReportTextMapper"]
