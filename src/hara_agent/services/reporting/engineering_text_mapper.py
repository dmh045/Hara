from __future__ import annotations

from collections import Counter
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
    _CONTACT_OBJECT_TEXT = {"VEHICLE": "其他车辆", "PASSENGER_CAR": "乘用车", "PEDESTRIAN": "行人",
                            "CYCLIST": "骑行者", "FIXED_OBJECT": "固定障碍物"}
    _PHYSICAL_MODEL_TEXT = {
        "CONSTANT_SPEED_REPRESENTATIVE": "恒速代表运动",
        "GRADE_ROLLAWAY_TO_IMPACT_DISTANCE": "坡道溜车（重力减残余阻力）",
        "CONSTANT_ACCELERATION_TO_IMPACT_DISTANCE": "恒加速度至首次接触",
        "CONSTANT_LATERAL_ACCELERATION_TO_CONTACT": "独立横向加速至接触（不等同于等速转向）",
        "CONSTANT_SPEED_CURVATURE_TO_CONTACT": "等速圆弧至侧向首次接触（无侧滑假设）",
        "RECTILINEAR_MOVING_TARGET_CONTACT": "移动目标相对轨迹与有限包络首次接触",
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
    _OBJECT_DIRECTION_TEXT = {
        "FORWARD": "沿车体前向运动", "REVERSE": "沿车体后向运动",
        "LEFT": "向车体左侧横穿", "RIGHT": "向车体右侧横穿",
        "STATIONARY": "静止", "LATERAL": "沿车体侧向运动",
    }
    _PHYSICAL_MODEL_GAP_TEXT = "当前可用物理模型不覆盖所提出的交互条件，该方案未生成碰撞数值"
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
    _CONTROL_TEXT = {
        "driver_braking": "驾驶员制动踏板", "braking": "制动执行器",
        "automatic_braking_command": "自动制动命令链", "remote_control": "远程控制链",
        "automatic_drive_command": "自动驱动命令链", "automatic_steering_command": "自动转向命令链",
        "automatic_parking_command": "自动驻车命令链", "parking_hold": "驻车保持",
        "steering": "转向", "drive": "驱动", "speed_limiter": "限速控制",
    }
    _SAFE_STATE_GAP_LABELS = {
        "RESPONSE_INSUFFICIENT": "即刻制动也无法在接触前停止",
        "RESPONSE_INTERACTION_UNSAFE": "所选响应仍会与目标接触",
        "DRIVER_CONTROL_UNAVAILABLE": "未满足驾驶员在驾驶位操作踏板的条件",
        "DRIVER_POSITION_OUT_OF_SCOPE": "处置方案不适用当前驾驶员位置",
        "NO_ANALYSIS": "尚未建立处置模型",
        "NO_PROPOSAL": "尚未形成适用处置方案",
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
        diagnostics = (getattr(scenario, "analysis_instance", {}) or {}).get("physical_plan_diagnostics", [])
        if not facts.get("physical_analysis") and any(str(d.get("reason", "")).startswith(
                "ROLLAWAY_REQUIRES_STATIONARY_INITIAL_STATE") for d in diagnostics):
            return "静止后溜车初态尚未建立；父场景运行速度代表值不作为本阶段初速"
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

    @staticmethod
    def has_foreign_prose(value: Any, *, include_slashes: bool = False) -> bool:
        """Detect narrative prose, not units, IDs or accepted abbreviations."""
        if not include_slashes:
            return bool(re.search(r"\b[A-Za-z]{3,}(?:[ -]+[A-Za-z]{2,}){2,}\b", str(value or "")))
        return any(re.search(r"[a-z]", match[0]) and re.search(r"\s", match[0]) for match in re.finditer(
            r"\b[A-Za-z]{2,}(?:[ /-]+[A-Za-z]{2,}){2,}\b", str(value or "")))

    @classmethod
    def analysis_note(cls, value: Any, fallback: str, *, include_slashes: bool = False) -> str:
        # This is a structured presentation fallback, not an alleged translation.
        # The full source language is retained in the existing audit table.
        return fallback if cls.has_foreign_prose(value, include_slashes=include_slashes) else cls.engineering_terms(narrative_quantities(str(value or "")))

    @classmethod
    def engineering_terms(cls, text: str) -> str:
        """Localize known contract tokens in prose, never inside identifiers."""
        labels = {**cls._MODE_TEXT, **cls._CONTROL_TEXT, **cls._PHYSICAL_MODEL_TEXT,
            "Vehicle ready to start": "车辆准备起步",
            "Rear-、Front-或TopView": "后视、前视或俯视画面",
            "Rear-, Front-或TopView": "后视、前视或俯视画面",
            "RearView": "后视画面", "FrontView": "前视画面", "TopView": "俯视画面",
            "profile": "配置方案", "maneuver": "操作", "manoeuvre": "操作",
            "Deceleration normal": "正常减速",
            "Normal lateral acceleration": "正常侧向加速度", "Normal lateral accel": "正常侧向加速度",
            "Increased lateral acceleration": "较高侧向加速度", "Increased lateral accel": "较高侧向加速度",
            "High lateral acceleration": "高侧向加速度", "High lateral accel": "高侧向加速度",
            "Very high lateral acceleration": "很高侧向加速度", "Very high lateral accel": "很高侧向加速度",
            "Parkplatz": "停车场", "Parkhaus": "停车楼", "Garage": "车库",
            "activation in motion": "行驶中激活",
            "FORWARD": "前向", "REVERSE": "后向", "LEFT": "左", "RIGHT": "右",
            "FRONT": "前方", "REAR": "后方", "STATIONARY": "静止",
            "CONTACT": "发生接触", "REAR_END": "追尾",
            "NO_CONTACT_WITHIN_MODEL": "模型时域内未接触", "HELD_AFTER_STOP": "停止后保持",
            "HOLD_POSITION": "保持当前位置", "CONSTANT_NET_DECELERATION": "恒定净减速度",
            "LOSS_OF_BRAKING_AVOIDABILITY": "失去制动避碰能力边界",
            "fault_release": "故障作用解除", "exposure": "暴露",
            "FIRST_CONTACT_BOUND": "首次接触边界", "FAULT_ONSET": "故障发生时刻",
            "FRONTAL": "正面", "PROJECT": "项目",
            "REQUIRED_NOT_ESTABLISHED": "所需通道尚未建立",
            "VEHICLE": "其他车辆", "PEDESTRIAN": "行人", "CYCLIST": "骑行者",
            "RETAINED_FOR_RISK": "保留用于风险分析", "EXCLUDED_FROM_RISK": "不进入风险分析",
            "PENDING_CAUSAL": "因果待定", "NOT_ASSESSED": "尚未分析",
            "severity": "严重度", "controllability": "可控性", "asil": "ASIL",
            "in_driver_seat": "在驾驶位", "outside_driver_seat": "不在驾驶位",
            "intervention_assumptions": "干预能力假设", "post_stop_motion": "停止后运动",
            "control_assessment": "控制能力分析",
            "driver_configurations": "驾驶员配置", "driver_positions": "驾驶员位置范围",
            "remaining_controls": "保留的控制能力", "failed_controls": "失效的控制能力",
            "unresolved_controls": "未明确的控制能力",
            "emergency_braking_available": "紧急制动可用性",
            "other_road_user_avoidance_possible": "他方避让能力",
            "remote_intervention_available": "远程干预可用性",
            "output_contract": "输出约束", "transition": "处置过程", "safe_state": "安全状态",
            "object_speed_kph": "目标速度", "operating_context": "运行上下文",
            "odd_road_types": "允许道路类型",
            "uncovered_conditions": "未覆盖条件", "driver_control_assessments": "驾驶员控制能力分析",
            "safe_state_proposal": "安全处置方案", "vehicle_control": "泊车控车",
            "pedestrian": "行人", "clearance": "净距", "hold": "保持", "brake": "制动",
            "µ-split": "左右分离附着", "Splitt / gravel": "碎石或砾石", "gravel": "砾石", "Splitt": "碎石",
            "passenger car": "乘用车", "passenger_car": "乘用车", "多故事车库": "多层车库",
            "remaining": "保留的", "failed": "失效", "unresolved": "未明确", "intervention": "干预",
            "true": "是", "false": "否",
            "kph": "km/h"}
        lateral_method_label = re.fullmatch(
            r"(?:Normal|Increased|High|Very high) lateral accel(?:eration)?\s*(?:\([^)]*\))?", text, re.IGNORECASE)
        if cls.has_foreign_prose(text) and not re.search(r"[\u3400-\u9fff]", text) and not lateral_method_label:
            return text
        text = re.sub(r"向(FORWARD|REVERSE)(?:方向)?(?![A-Za-z0-9_-])",
                      lambda match: "向前" if match[1] == "FORWARD" else "向后", text)
        for raw, display in sorted(labels.items(), key=lambda item: -len(item[0])):
            text = re.sub(rf"(?<![A-Za-z0-9_-]){re.escape(raw)}(?![A-Za-z0-9_-])",
                          lambda _match, label=display: label, text, flags=re.IGNORECASE)
        return text

    @classmethod
    def item_fact_text(cls, field: str, value: Any, project: Mapping[str, Any],
                       functions: Mapping[str, Mapping[str, Any]]) -> str:
        """Keep source function/mode qualifiers visible on the Item overview."""
        if not isinstance(value, list):
            return cls.engineering_terms(str(value))
        records = [r for r in project.get("odd_scope_evidence", []) if r.get("field") == field]
        values = []
        for item in value:
            matches = [r for r in records if item in r.get("values", [])]
            scoped = [r for r in matches if r.get("function_ids") or r.get("operating_modes")]
            if scoped and not any(not r.get("function_ids") and not r.get("operating_modes") for r in matches):
                qualifications = []
                for record in scoped:
                    parts = []
                    if record.get("function_ids"):
                        parts.append("、".join(str(functions.get(identity, {}).get("name") or identity)
                                               for identity in record["function_ids"]))
                    if record.get("operating_modes"):
                        parts.append("模式：" + "、".join(record["operating_modes"]))
                    qualifications.append("，".join(parts))
                values.append(str(item) + "（仅适用" + "；或".join(dict.fromkeys(qualifications)) + "）")
            else:
                values.append(str(item))
        return cls.engineering_terms("、".join(values))

    @staticmethod
    def causal_status(value: str) -> str:
        return {
            "ELIGIBLE — RISK SCORING INVOKED": "因果已通过校验，已执行风险计算",
            "METHOD_VALID — CAUSAL_REVALIDATION_REQUIRED": "方法条件有效，因果需重新验证",
            "PENDING — RISK SCORING NOT EXECUTED": "因果待定，尚未执行风险计算",
            "CAUSAL_GAP — EXCLUDED_FROM_RISK": "因果不成立，未进入风险计算",
        }.get(value, value)

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
        direction = cls._OBJECT_DIRECTION_TEXT.get(facts.get("object_longitudinal_direction"))
        return (f"目标以{quantity(speed, 'km/h')} km/h{direction}"
                if direction and facts.get("object_longitudinal_direction") != "STATIONARY"
                and isinstance(speed, (int, float)) and not isinstance(speed, bool) and speed > 0 else "")

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
        if layout := cls.layout_text(scenario):
            parts.append(layout)
        analysis = facts.get("physical_analysis", {})
        control_hypothesis = (facts.get("control_analysis") or analysis).get("hypothesis", {})
        if reason := control_hypothesis.get("unresolved_controls", {}).get("braking"):
            parts.append("制动能力未决依据：" + cls.analysis_note(reason,
                "当前人员配置下尚未建立可用的制动响应结论；命令通道、执行能力及独立性限定见审计表原始分析说明"))
        if controls := facts.get("control_analysis", {}).get("hypothesis"):
            conditions = controls["conditions"]
            parts.append("干预能力假设：" + "；".join(dict.fromkeys(cls.analysis_note(text,
                "本场景干预能力附有适用条件，完整外文条件见审计表原始控制方案", include_slashes=True) for text in conditions)))
            if any(cls.has_foreign_prose(text, include_slashes=True) for text in conditions):
                for field, label in (("remaining_controls", "方案中保留的控制"), ("failed_controls", "方案中失效的控制")):
                    if identities := controls.get(field):
                        parts.append(label + "：" + "、".join(cls._CONTROL_TEXT.get(c, c) for c in identities))
            for proposal in controls.get("intervention_assumptions", {}).values():
                parts.append("干预条件：" + "；".join(dict.fromkeys(cls.analysis_note(text,
                    "该干预判断附有适用条件，完整外文条件见审计表原始控制方案", include_slashes=True) for text in proposal["conditions"])))
        if analysis:
            if facts.get("post_stop_motion") == "HOLD_POSITION":
                parts.append("停车后保持原位置至所列模型时限（分析假设），继续检查移动目标是否接触")
            labels = {"ego_speed_kph": "初速(km/h)", "relative_distance_m": "行程(m)",
                "longitudinal_acceleration_mps2": "纵向加速度(m/s²)", "slope_percent": "坡度(%)",
                "residual_resistance_mps2": "残余阻力减速度(m/s²)", "lateral_clearance_m": "侧向净距(m)",
                "lateral_speed_mps": "初始侧速(m/s)", "lateral_acceleration_mps2": "侧向加速度(m/s²)",
                "turn_radius_m": "条件性转弯半径(m)",
                "ego_heading_angle_deg": "自车速度方向角(°，车前向左为正)",
                "object_heading_angle_deg": "目标速度方向角(°，车前向左为正)",
                "object_reaction_time_s": "目标开始制动前的反应时间(s，模型起点计)",
                "object_braking_deceleration_mps2": "目标条件性制动减速度(m/s²)",
                "model_valid_until_s": "模型时域(s)", "post_fault_speed_cap_kph": "残余限速(km/h)",
                "object_offset_x_m": "目标初始纵向中心偏移(m，车前为正)",
                "object_offset_y_m": "目标初始横向中心偏移(m，车左为正)",
                "contact_half_length_m": "双方合计接触包络半长(m)",
                "contact_half_width_m": "双方合计接触包络半宽(m)"}
            parts.extend(f"{labels[key]}：{quantity(number, FIELD_UNITS.get(key, ''))}"
                for key, number in analysis.get("sample_parameters", {}).items()
                if key in labels and key not in {"ego_speed_kph", "relative_distance_m"})
            if "object_reaction_time_s" in facts:
                parts.append("目标按所列反应时间先匀速、后制动、停止后保持；属于条件性轨迹，不自动证明人员可控性")
            if "ego_heading_angle_deg" in facts or "object_heading_angle_deg" in facts:
                parts.append("角度描述固定速度方向，不是方向盘或车轮转角，不表示主动转向避碰")
            if facts.get("contact_normal_speed_kph") is not None:
                parts.append(f"接触方向闭合速度：{quantity(facts['contact_normal_speed_kph'], 'km/h')} km/h；与完整相对速度分别记录，S沿用所列方法输入")
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
            intervention_notes = [text for text in analysis.get("hypothesis", {}).get("conditions", [])
                if re.search(r"接管|介入|干预|制动通道|制动能力|interven|driver braking|braking path", text, re.IGNORECASE)]
            parts.extend(text for text in intervention_notes if not cls.has_foreign_prose(text))
            if any(cls.has_foreign_prose(text) for text in intervention_notes):
                for field, label in (("remaining_controls", "方案中保留的控制"), ("failed_controls", "方案中失效的控制")):
                    controls = control_hypothesis.get(field, [])
                    if controls:
                        parts.append(label + "：" + "、".join(cls._CONTROL_TEXT.get(c, c) for c in controls))
                parts.append("以上为控制能力分析假设；通道独立性、触发条件及限定原文见审计表，实际用于C评定的条件见本行可控性依据")
            if facts.get("first_contact_time_s") is not None and facts.get("ttc_s") is None:
                parts.append(f"首次接触：{quantity(facts['first_contact_time_s'], 's')} s（不作为纵向 TTC）")
            if condition := facts.get("selected_source_condition"):
                parts.append(f"代表运行条件：{condition['value']}；物理影响：" + cls.analysis_note(
                    condition['physical_effect'], "本行列示该条件下的对象、几何和运动参数；完整作用说明见审计表原始分析说明"))
        if model == "CONSTANT_SPEED_REPRESENTATIVE":
            parts.append("恒速代表工况；用于所列碰撞条件，不是故障最大速度包络")
        elif model:
            parts.append(f"运动模型：{cls._PHYSICAL_MODEL_TEXT.get(model, model)}")
        coverage = facts.get("motion_model_outcome", {}).get("target_longitudinal_coverage", {})
        if coverage:
            contact_x = coverage.get("contact_x_m")
            interval = coverage.get("effective_contact_interval_m")
            if isinstance(contact_x, (int, float)):
                parts.append(f"侧方接触点纵向坐标：{quantity(contact_x, 'm')} m（车头前向为正）")
            if isinstance(interval, (list, tuple)) and len(interval) == 2:
                parts.append("目标有效纵向接触区间：" + quantity_range(interval[0], interval[1], 'm') + " m")
            if coverage.get("limitation"):
                parts.append("目标覆盖条件：" + coverage["limitation"])
        elif model == "CONSTANT_SPEED_CURVATURE_TO_CONTACT":
            parts.append("圆弧模型将目标近似为沿初始行驶方向延展的侧边界；本历史结果未单独核验有限对象的纵向覆盖，不能等同于任意单辆车辆的接触。")
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
            labels = cls._OBJECT_DIRECTION_TEXT if field == "object_longitudinal_direction" else directions
            direction = labels.get(str(facts.get(field, "")).upper(), "")
            if direction:
                parts.append(f"{label}：{direction}")
        if not parts:
            return ""
        return (
            "场景候选（未核实用于评分）：" if candidate_only else ""
        ) + cls.engineering_terms("；".join(parts))

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
        if label and facts.get("physical_analysis", {}).get("driver_configuration_assumption"):
            label += "（分析假设）"
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
        if control.get("physical_plan_gap"):
            return control["physical_plan_gap"] + "；对应模型范围与未覆盖条件见物理方案诊断"
        if control.get("decision_status") == "CONDITIONAL_REFERENCE_MATCH":
            rule = control.get("reference_rule", {})
            if (control.get("intervention_applicability", {}).get("manual", {}).get("status") == "MANUAL_BRAKING_PATH_UNAVAILABLE"
                    and rule.get("when", {}).get("emergency_braking_available") is False):
                value = control.get("result", "")
                return ("驾驶员人工制动路径已明确不可用；按当前参考方法" +
                    (f"条件性评为 {value}" if value else "作条件性评定") +
                    "；自动制动路径及响应边界另见安全状态与 FTTI。")
            return rule.get("rationale", "按参考工况进行条件性 C 评定；适用性待审核")
        if control.get("decision_status") == "OVERRIDE_MATCH":
            return "C 分支：干预覆盖；依据已命中的干预条件评分，未使用 TTC 分档"
        if control.get("decision_status") == "REFERENCE_APPLICABILITY_UNRESOLVED":
            if control.get("intervention_applicability", {}).get("manual", {}).get("status") == "MANUAL_BRAKING_PATH_UNRESOLVED":
                return "已有适用场景类型的条件性 C 规则，但驾驶员人工制动路径尚未明确；自动制动执行能力不能替代该条件"
            labels = {"emergency_braking_available": "独立制动能力", "driver_in_vehicle": "驾驶员位置",
                "required_controls": "驾驶员人工制动路径",
                "steering_control_available": "转向控制能力",
                "remote_intervention_available": "远程干预能力", "other_road_user_avoidance_possible": "他方避让条件"}
            missing = dict.fromkeys(key for rule in control.get("unresolved_reference_inputs", [])
                for key in rule.get("unknown_conditions", {}))
            detail = "、".join(labels.get(key, "干预条件") for key in missing)
            return f"已有适用场景类型的条件性 C 规则，但{detail or '必要干预条件'}尚未明确；不能仅凭接触时间给出等级"
        # Old checkpoints used MODEL_NOT_APPLICABLE for a C-method scope gap.
        if control.get("decision_status") in {"METHOD_NOT_APPLICABLE", "MODEL_NOT_APPLICABLE"}:
            reason = str(control.get("pending_reason", ""))
            if reason.startswith("REAR_APPROACH_BRAKING_APPLICABILITY_UNRESOLVED"):
                return "后车从本车行进方向的后方接近；本车制动不能直接作为避免被追尾的路径，当前避险方法适用性未确定"
            if reason.startswith("LATERAL_C_METHOD_NOT_AVAILABLE"):
                conditions = dict.fromkeys(item.get("reason_code")
                    for item in control.get("reference_scope_exclusions", []) if item.get("condition"))
                labels = {"CONFLICT_VISIBILITY_OCCLUDED": "目标可见性受遮挡影响",
                          "CONFLICT_RECOGNITION_DELAYED": "驾驶员或目标的危险识别存在延迟"}
                reasons = [labels[key] for key in conditions if key in labels]
                if reasons:
                    return "所选参考规则排除当前遮挡或延迟识别条件：" + "；".join(reasons) + "；C 待评定"
                limits = dict.fromkeys((item["actual"], item["upper_exclusive"])
                    for item in control.get("reference_scope_exclusions", [])
                    if item.get("field") == "trajectory_max_speed_kph")
                if limits:
                    return "；".join(f"当前轨迹最高速度 {cls._number(speed)} km/h，现有侧向参考规则要求低于 {cls._number(limit)} km/h"
                        for speed, limit in limits) + "；不满足该适用条件，C 待评定"
                return "侧向接触已建模；当前方法无适用 C 规则，局部待评定"
            if reason.startswith("POST_STOP_BRAKING_APPLICABILITY_UNRESOLVED"):
                return "本车停止后被移动目标接触；继续制动不是已成立的避险路径，当前方法无适用 C 规则"
            if reason.startswith("STATIONARY_EGO_BRAKING_APPLICABILITY_UNRESOLVED"):
                return "本车在所计算轨迹中始终静止；目标接触时间不能作为本车制动避让的时窗，其他避险路径尚未成立"
            return "当前交互不满足所选适用条件；当前方法无适用 C 规则"
        if control.get("decision_status") == "MODEL_DEPENDENCY_UNRESOLVED":
            return "故障影响规避控制；C 方法依赖尚未成立"
        if control.get("decision_status") == "PHYSICAL_INPUT_UNRESOLVED":
            return "当前轨迹在模型有效范围内未建立接触，或几何不适用；不能据此判定范围外无危险，也不能直接套用碰撞避险分档"
        if control.get("decision_status") == "PROFILE_APPLICABILITY_UNRESOLVED":
            if control.get("intervention_applicability", {}).get("manual", {}).get("status") == "MANUAL_BRAKING_PATH_UNRESOLVED":
                return "尚未建立驾驶员人工制动路径；自动制动执行能力不能替代车内驾驶员的制动反应分档条件"
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
    def physical_model_gap(cls, scenario: Any) -> str:
        """Current uninstantiated model gap, not a superseded plan diagnostic."""
        facts = getattr(scenario, "facts", {}) or {}
        if facts.get("physical_analysis") or facts.get("fault_motion_model"):
            return ""
        diagnostics = (getattr(scenario, "analysis_instance", {}) or {}).get("physical_plan_diagnostics", [])
        return cls._PHYSICAL_MODEL_GAP_TEXT if any(
            isinstance(item, Mapping) and item.get("reason") == "PHYSICAL_MODEL_NOT_APPLICABLE"
            for item in diagnostics
        ) else ""

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
                + ("保持本车静止时，同一目标也会接触；需结合故障增量危险依据审核，不能仅把接触本身归因于故障。"
                   if physical.get("stationary_ego_counterfactual", {}).get("status") == "CONTACT" else "")
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
            "PHYSICAL_INSTANTIATION_FAILED": "物理工况实例化未通过；保留来源候选，尚未形成可评定的具体危险事件",
            "PHYSICAL_MODEL_NOT_APPLICABLE": cls._PHYSICAL_MODEL_GAP_TEXT,
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

    @classmethod
    def layout_text(cls, scenario: Any, *, compact: bool = False) -> str:
        relation = (getattr(scenario, "facts", {}) or {}).get("layout_relation", {})
        label = {"PARALLEL": "平行车位", "ANGLED": "斜列车位", "PERPENDICULAR": "垂直车位"}.get(relation.get("kind"))
        if not label:
            return ""
        title = f"布局假设：{label}"
        if compact:
            return title
        axes = relation.get("axes_deg", {})
        parts = [title]
        if all(isinstance(axes.get(key), (int, float)) and not isinstance(axes[key], bool) for key in ("bay", "aisle")):
            parts.append(f"初始车体坐标下，车位轴线 {quantity(axes['bay'], 'deg')}°、通道轴线 {quantity(axes['aisle'], 'deg')}°（车头前向为0°）")
        labels = {"BAY_AXIS": "车位轴线", "AISLE_AXIS": "通道轴线"}
        for actor, name in (("ego", "自车"), ("object", "目标")):
            check = relation.get("path_checks", {}).get(actor, {})
            if check.get("status") == "INITIAL_DIRECTION_MATCH":
                parts.append(name + "初始运动沿" + labels.get(check.get("path"), "所声明轴线"))
            elif check.get("status") == "SYMBOLIC_OR_UNRESOLVED":
                parts.append(name + "运动与布局轴线的关系尚未确定")
        if relation.get("trajectory_scope") == "INITIAL_TANGENT_ONLY":
            parts.append("仅核对圆弧初始切向，后续运动沿所列圆弧计算")
        if relation.get("geometry_scope") == "LOCAL_INITIAL_RELATION_ONLY":
            parts.append("仅建立局部初始关系，不代表已验证双方沿两条布局轴线交互")
        return "；".join(parts)

    def scenario(self, scenario: Any, *, variant_count: int = 1) -> tuple[str, str]:
        """Render validated structured facts without exposing Method vocabulary."""
        facts = getattr(scenario, "facts", {}) or {}
        categories = self._categories(scenario)
        source_conditions = facts.get("source_operating_conditions", {})
        condition = facts.get("selected_source_condition")
        location_candidates = [self._dimensions(scenario).get("WHERE", {}).get("project_value"),
            source_conditions.get("operating_scenario", ""), getattr(scenario, "operating_scenario", "")]
        source_selection = condition or source_conditions.get("selected_condition", {})
        if source_selection.get("source_field") == "odd_road_types":
            # A parent candidate may use a road/bay type as its context label.
            # The instantiated child selects its own allowed type below; that
            # inherited label is neither its location nor a second condition.
            road_types = source_conditions.get("permitted_road_types", facts.get("source_road_type_scope", []))
            location_candidates = [value for value in location_candidates if value not in road_types]
        location = next((text for value in location_candidates if (text := self._chinese_source(value))), "") or self._first_category(categories, (
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
        if layout := self.layout_text(scenario, compact=True):
            clauses.append(layout)
        if condition:
            clauses.append(f"代表运行条件为{condition['value']}")
        if mode:
            clauses.append(f"系统处于{mode}状态")
        driver_position = {
            "in_driver_seat": "驾驶员在驾驶位",
            "outside_driver_seat": "驾驶员不在驾驶位",
        }.get(str(facts.get("driver_position", "")).strip().casefold(), "")
        if driver_position:
            clauses.append(("分析假设为" if facts.get("physical_analysis", {}).get("driver_configuration_assumption")
                            else "") + driver_position)
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
        if layout := self.layout_text(scenario):
            details.append(layout)
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
            details.append("条件影响：" + self.analysis_note(condition['physical_effect'],
                "本行列示当前对象、位置和运动条件；完整作用说明见关联场景明细及审计表"))
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
        return operational, ("；".join(details) + "。" if details else
            self.physical_model_gap(scenario) or "尚未形成可用于数值分析的结构化运行条件；具体来源和缺口见场景明细。")

    @classmethod
    def hazardous_event(cls, value: Any, scenario: Any = None) -> str:
        text = narrative_quantities(str(value or "").strip())
        facts = (getattr(scenario, "facts", {}) or {}) if scenario is not None else {}
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
        text = re.sub(r"(?<![A-Za-z0-9_])BRAKE_AND_HOLD(?![A-Za-z0-9_])", "制动并停车保持", text)
        text = re.sub(r"(?<![A-Za-z])EPB(?![A-Za-z])", "电子驻车制动", text, flags=re.IGNORECASE)
        text = re.sub(r"(?<![A-Za-z])System\s+OK(?![A-Za-z])", "系统正常", text, flags=re.IGNORECASE)
        reverse_contact = cls._reversing_stationary_rear_vehicle(
            facts.get("road_user_type"), facts.get("collision_type"), facts)
        for raw, display in cls._COLLISION_TEXT.items():
            if raw == "REAR_END" and reverse_contact:
                display = "后向碰撞"
            text = re.sub(rf"(?<![A-Za-z0-9_-]){raw}\s*(?:接触|碰撞)", display, text)
        text = cls.engineering_terms(text)
        if reverse_contact:
            text = text.replace("追尾碰撞", "后向碰撞")
        clauses = [item.strip() for item in re.split(r"[；;]", text) if item.strip()]
        return "；".join(dict.fromkeys(clauses)).rstrip("。") + "。" if clauses else "待危险事件确认。"

    def review_event(self, value: Any, scenario: Any, malfunction: Mapping) -> str:
        """Present a foreign narrative using its own structured world only.

        No machine translation, new causal approval or collision claim is made.
        The original statement is kept verbatim in AuditReferenceView.
        """
        if not self.has_foreign_prose(value):
            return self.hazardous_event(value, scenario)
        facts = getattr(scenario, "facts", {}) or {}
        fault = self._chinese_source(malfunction.get("description"))
        interaction = self.object_interaction_summary(scenario)
        if not fault or not interaction:
            return self.hazardous_event(value, scenario)
        parts = ["结构化摘要", "失效表现：" + fault, interaction]
        outcome = facts.get("motion_model_outcome", {}).get("status")
        if outcome == "CONTACT":
            time = facts.get("first_contact_time_s", facts.get("ttc_s"))
            parts.append("当前条件轨迹" + (f"在{quantity(time, 's')} s" if time is not None else "") + "与所列对象接触")
        elif outcome in {"NO_CONTACT_WITHIN_MODEL", "NO_POSITIVE_CONTACT_ROOT"}:
            parts.append("当前条件轨迹未确认接触；保留危险状态不等于已发生碰撞")
        else:
            parts.append("该条件下的危险交互尚不能由当前物理模型完整表达")
        parts.append("完整因果声明及限定见审计表原始分析说明")
        return self.engineering_terms("；".join(parts) + "。")

    def pending_value(self, evidence: Any) -> str:
        return self._PENDING_VALUE

    @classmethod
    def known_inputs(cls, facts: Mapping[str, Any]) -> str:
        labels = {"ego_speed_kph": "自车速度输入", "object_speed_kph": "目标速度",
                  "relative_speed_kph": "相对速度", "relative_distance_m": "相对距离",
                  "ttc_s": "TTC", "longitudinal_acceleration_mps2": "纵向加速度"}
        parts = [f"{label} {quantity(facts[field], FIELD_UNITS[field])} {FIELD_UNITS[field]}"
                 for field, label in labels.items() if facts.get(field) is not None]
        position = cls._POSITION_TEXT.get(str(facts.get("object_position", "")).lower())
        if position:
            parts.append(position + "目标")
        directions = {"FORWARD": "前进", "REVERSE": "倒车", "STATIONARY": "静止", "LATERAL": "侧向运动"}
        for field, label in (("ego_longitudinal_direction", "自车"), ("object_longitudinal_direction", "目标")):
            labels = cls._OBJECT_DIRECTION_TEXT if field == "object_longitudinal_direction" else directions
            if facts.get(field) in labels:
                parts.append(label + labels[facts[field]])
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
            records.append({**member, "display_text": self.engineering_terms(text),
                "wording_id": matches[0]["id"] if unambiguous else "grounded_behavior_excerpt",
                "basis": basis})
        texts = list(dict.fromkeys(record["display_text"].rstrip("。；") for record in records))
        # A supported complete short intent subsumes its repeated elaboration.
        # Do not use a common character prefix (it may cut away a negation).
        texts = [text for text in texts if not any(
            text != other and text.startswith(other) for other in texts)]
        # The SG identity already belongs to these members. A shared contact
        # intent must name their actual objects, not a parent's example object.
        objects = list(dict.fromkeys(record.get("road_user_type") for record in records))
        labels = self._CONTACT_OBJECT_TEXT
        if (len(objects) > 1 and all(obj in labels for obj in objects)
                and all(record["wording_id"] == "grounded_behavior_excerpt"
                    and record["basis"].get("member_contact_grounded") for record in records)):
            texts = ["防止车辆与" + "、".join(labels[obj] for obj in objects) + "发生接触"]
        return {"text": "；".join(texts) + "。" if texts else self.engineering_terms(original),
                "members": records, "nature": "DISPLAY_ONLY_NO_ENGINEERING_APPROVAL"}

    @staticmethod
    def _grounded_safety_intent(member: Mapping, original: str) -> tuple[str, dict]:
        """Extract the stated unsafe behavior, not a new safety measure/outcome.

        This is source-clause compression. Unknown wording stays as the original
        goal rather than manufacturing an intent from a guideword or parameters.
        """
        hazard = str(member.get("vehicle_hazard", ""))
        event = str(member.get("hazardous_event", ""))
        negative_claim = r"未发生|未出现|未产生|不会|不导致|不造成|不引发|无碰撞|无(?:物理)?接触|不能确认|待确定|未知|可能受伤|存在.*风险|(?:未|没有)(?:发生)?(?:碰撞|相撞|擦碰|刮碰|碾压|物理接触|驶入|停入|进入)|未与.+(?:接触|碰撞)"
        member_contact = (member.get("contact_status") == "CONTACT"
            and member.get("road_user_type") in EngineeringReportTextMapper._CONTACT_OBJECT_TEXT
            and bool(re.search(r"碰撞|相撞|擦碰|刮碰|碾压|接触", event))
            and bool(re.search(r"碰撞|相撞|擦碰|刮碰|碾压|接触", hazard)))
        # Parent hazards may contain an example collision object. Only replace
        # that excerpt when it explicitly names a different object class; a
        # correct shared behavior or an object-neutral intent stays intact.
        object_terms = {"VEHICLE": r"车辆|邻车|后车|前车|乘用车|汽车",
                        "PEDESTRIAN": r"行人|人员|交通参与者|道路使用者",
                        "CYCLIST": r"骑行|骑车|自行车|交通参与者|道路使用者",
                        "FIXED_OBJECT": r"障碍物|路缘|充电桩|立柱|墙|边界物体"}
        target = re.search(r"与(.+?)(?:碰撞|相撞|擦碰|刮碰|碾压|接触)", hazard)
        target_text = target.group(1) if target else ""
        actual_type = "VEHICLE" if member.get("road_user_type") == "PASSENGER_CAR" else member.get("road_user_type")
        parent_object_mismatch = (bool(target_text) and actual_type in object_terms
            and any(re.search(pattern, target_text) for pattern in object_terms.values())
            and not re.search(object_terms[actual_type], target_text))
        prefer_member_event = member_contact and parent_object_mismatch
        # An "or" in the parent hazard names different mechanisms. The concrete
        # event, rather than the first parent alternative, determines this member.
        alternatives = bool(re.search(r"[，；。]\s*或|或者|\bor\b", hazard))
        fields = ("hazardous_event",) if alternatives or prefer_member_event else (
            "vehicle_hazard", "functional_effect", "hazardous_event")
        for field in fields:
            source = str(member.get(field, "")).strip()
            clauses = re.split(r"([，；。]|(?:从而|导致|造成|引发))", source)
            candidates = []
            shared_subject = ""
            for index in range(0, len(clauses), 2):
                clause = clauses[index].strip(" 。；，")
                if not clause:
                    continue
                if not index or clauses[index - 1] != "，":
                    shared_subject = ""
                # Do not turn a negated or uncertain harm claim into an intent.
                if (re.search(negative_claim, clause)
                        or re.search(r"(?:并非|并不|不是)(?:无法|不能)", clause)):
                    shared_subject = ""
                    continue
                subject = re.match(r"^(车辆|本车|自车|驾驶员|乘员)", clause)
                if subject:
                    shared_subject = subject.group()
                elif re.match(r"^(?:无法|不能)", clause):
                    # A comma may omit the preceding explicit actor. Keep that
                    # actor, never infer a driver/occupant from scenario facts.
                    if not shared_subject:
                        continue
                    clause = shared_subject + clause
                else:
                    shared_subject = ""
                if field == "hazardous_event":
                    # Drop an explicit scene prefix, not arbitrary numbers: a
                    # parameter-bearing sentence otherwise remains in evidence.
                    clause = re.sub(r"^.*(?:行驶时|过程中|场景下)[，、]?", "", clause)
                    clause = re.sub(r"^.*?因(?=.+(?:失效|失去|无法|卡死|异常))", "", clause)
                if re.search(r"\d", clause):
                    continue
                # A causal prefix or prepositional noun phrase is not a safety
                # intent. Keep searching supported clauses instead of emitting
                # fragments such as "防止因..." or "防止车辆基于...目标".
                if clause.startswith(("因", "由于", "基于", "当", "在")) and not re.search(
                        r"车辆|本车|自车|驾驶员|执行器|制动|转向|功能", clause):
                    continue
                if re.search(r"^(?:车辆|本车|自车)?(?:因|由于|基于).*(?:目标|输出|信息|数据|请求)$", clause):
                    continue
                if (re.search(r"非预期|(?:高于|低于)预期|意外|失控|失去|失效|偏离|异常|丧失|溜车|过大|过小|不足|无法|不能|过度|延迟|提前|滞后|卡死|停滞|持续运行|固定不变|不随.*变化|反向|反而|虚假.*干扰|突发.*制动|与.*相反|未.*(?:退出|关闭|激活|驻车)|(?:尚未|未经).*(?:确认|验证).*(?:泊车|运动|进入|驶入)|碰撞|相撞|擦碰|刮碰|碾压|与.*物理接触|禁停|禁止停车", clause)
                        or member_contact and re.search(r"(?:高于|低于|超过)目标|过快|过慢|与.*接触", clause)):
                    clause = re.sub(r"^(?:发生|出现|产生)", "", clause)
                    clause = re.sub(r"^(?:车辆|本车)(?:发生|出现)", "车辆", clause)
                    # Keep the sentence's explicit subject when a later clause
                    # only says "与...距离不足". Never supply an unrelated actor.
                    if clause.startswith("与") and re.match(r"^(车辆|本车|自车)", source):
                        clause = re.match(r"^(车辆|本车|自车)", source).group() + clause
                    vehicle_subject = bool(re.match(r"^(车辆|本车|自车)", clause))
                    if member_contact and field == "hazardous_event" and not vehicle_subject:
                        continue
                    if vehicle_subject:
                        clause = re.sub(r"的(?:车辆级危险|风险)状态$", "", clause)
                        clause = re.sub(r"因[^，；。]*?而", "", clause)
                        # An intent prevents a stated possible event; this is
                        # not an assertion that the event occurred or is certain.
                        clause = clause.replace("可能", "")
                    candidates.append((vehicle_subject, clause))
            if candidates:
                clause = max(enumerate(candidates), key=lambda pair: (pair[1][0], -pair[0]))[1][1]
                return "防止" + clause + "。", {field: source, "selected_behavior": clause,
                    "function": member.get("function", ""), "malfunction": member.get("malfunction", ""),
                    "hazardous_event": member.get("hazardous_event", ""),
                    **({"member_contact_grounded": not bool(re.search(negative_claim, event)),
                        "road_user_type": member["road_user_type"], "contact_status": "CONTACT"}
                       if member_contact else {})}
        if member_contact and (prefer_member_event or alternatives) and not re.search(negative_claim, event):
            # A parameter-bearing event may have no compressible behavior
            # clause. Its explicit, calculated contact still names this member's
            # object without importing the parent's different example object.
            return "防止车辆与" + EngineeringReportTextMapper._CONTACT_OBJECT_TEXT[member["road_user_type"]] + "发生接触。", {
                "hazardous_event": event, "road_user_type": member["road_user_type"],
                "contact_status": "CONTACT", "member_contact_grounded": True,
                "original_goal": original, "reason": "Explicit member contact; no complete behavior clause to compress"}
        return original, {"original_goal": original, "reason": "No independently stated unsafe behavior to compress"}

    @staticmethod
    def _reversing_stationary_rear_vehicle(road_user_type: Any, collision_type: Any, facts: Mapping) -> bool:
        speed = facts.get("object_speed_kph")
        return (road_user_type in {"VEHICLE", "PASSENGER_CAR"} and collision_type == "REAR_END"
            and facts.get("ego_longitudinal_direction") == "REVERSE" and facts.get("object_position") == "REAR"
            and isinstance(speed, (int, float)) and not isinstance(speed, bool) and speed == 0
            and facts.get("object_longitudinal_direction") == "STATIONARY")

    @classmethod
    def _collision_description(cls, road_user_type: Any, collision_type: Any, scenario: Any = None) -> str:
        target = {"VEHICLE": "车辆", "PASSENGER_CAR": "乘用车", "PEDESTRIAN": "行人", "CYCLIST": "骑行者", "FIXED_OBJECT": "固定障碍物"}.get(road_user_type, "")
        if cls._reversing_stationary_rear_vehicle(road_user_type, collision_type,
                (getattr(scenario, "facts", {}) or {}) if scenario is not None else {}):
            return "倒车碰撞后方" + target
        collision = cls._COLLISION_TEXT.get(collision_type, "")
        if (collision or collision_type == "VEHICLE_TO_ROAD_USER") and road_user_type in {"PEDESTRIAN", "CYCLIST", "FIXED_OBJECT"}:
            side = {"FRONTAL": "前部", "REAR_END": "后部", "SIDE": "侧面", "LATERAL": "侧面"}.get(collision_type, "")
            return f"车辆{side}与{target}碰撞"
        return target + collision

    @classmethod
    def branch_result(cls, branch: Mapping[str, Any], scenario: Any = None) -> str:
        scores = " / ".join(str(branch.get(key) or "待评定") for key in ("S", "E", "C", "ASIL"))
        parts = [f"分支 {branch.get('branch_id', '')}：{scores}"]
        context = branch.get("risk_context", {})
        def value(field):
            evidence = context.get(field, {})
            return evidence.get("value") if isinstance(evidence, Mapping) else None
        declared = (getattr(scenario, "facts", {}) or {}).get("collision_branches", []) if scenario is not None else []
        same_branch = len(declared) == 1 and declared[0].get("branch_id") == branch.get("branch_id")
        description = cls._collision_description(value("road_user_type"), value("collision_type"),
            scenario if same_branch else None)
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

    def severity_rationale(self, evidence: Any, trace: Mapping[str, Any], *, scenario: Any = None) -> str:
        if getattr(evidence, "value", None):
            inputs = trace.get("physical_inputs", {})
            speed = inputs.get("relative_speed_kph", trace.get("inputs", {}).get("relative_speed_kph"))
            description = self._collision_description(inputs.get("road_user_type"), inputs.get("collision_type"), scenario)
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
            return self.engineering_terms(f"{atom_text}；{branch_text}，最终 {final_value}{gap}。")
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
                if (trace.get("intervention_applicability", {}).get("manual", {}).get("status") == "MANUAL_BRAKING_PATH_UNAVAILABLE"
                        and rule.get("when", {}).get("emergency_braking_available") is False):
                    return self.controllability_branch({"controllability": {**trace, "result": evidence.value}})
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
        if not getattr(evidence, "value", None) and trace.get("physical_plan_gap"):
            return self.controllability_branch({"controllability": trace}) + "；C 暂未定。"
        if trace.get("decision_status") in {"METHOD_NOT_APPLICABLE", "MODEL_NOT_APPLICABLE",
                "MODEL_DEPENDENCY_UNRESOLVED",
                "PROFILE_APPLICABILITY_UNRESOLVED", "REFERENCE_APPLICABILITY_UNRESOLVED",
                "PHYSICAL_INPUT_UNRESOLVED"}:
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
    def safe_state_outcome(candidate: Mapping[str, Any], timing_input: Mapping[str, Any] | None = None) -> tuple[str, str] | None:
        """Name the modelled end state, not the actuator used to reach it.

        Display grouping only: avoid-contact wording requires the same scoped
        proposal's calculation trace; a response model alone states a target.
        Member controls and qualifications remain in the detail.
        """
        response = candidate.get("transition", {})
        timing_input = timing_input or {}
        trace = timing_input.get("timing_trace", {})
        checked_proposal = trace.get("hypothesis", {}).get("safe_state_proposal", {})
        if (timing_input.get("scenario_id") != candidate.get("scenario_id") or not checked_proposal
                or any(candidate.get(field) != checked_proposal.get(field) for field in (
                    "description", "required_controls", "conditions", "transition", "required_command_path"))):
            trace = {}
        status = trace.get("status")
        if status in {"NO_IMMEDIATE_SAFE_BRAKING_WINDOW", "AVOIDABILITY_ALREADY_LOST"}:
            return "UNSAFE_RESPONSE", "所选响应仍会与目标接触；该提案不能作为安全状态"
        validated = status == "CONDITIONAL_BOUNDARY"
        immediate_only = (status == "NO_UNMITIGATED_CONTACT_BOUNDARY"
                          and trace.get("response_contact", {}).get("status") == "NO_CONTACT_WITHIN_MODEL")
        outcome = None
        if response.get("model") == "CONSTANT_NET_DECELERATION":
            if response.get("post_stop_motion") == "HOLD_POSITION":
                outcome = ("STOP_AND_HOLD", "车辆停止并保持静止")
            else:
                outcome = ("STOP", "车辆停止；停车后的保持条件见成员明细")
        elif response.get("model") == "RELEASE_DECELERATION_TO_CONSTANT_SPEED":
            outcome = ("CONTINUE", "解除异常减速，保持响应开始时的速度行驶")
        if outcome:
            if validated:
                outcome = outcome[0] + "_CHECKED", outcome[1] + "；在已检查范围内避免与目标接触"
                if trace.get("pending_reason") == "FAULT_CLOCK_UNRESOLVED":
                    outcome = outcome[0] + "_CLOCK_UNBOUND", outcome[1] + "；故障起点或危险判据待绑定，尚不能作为FTTI"
            elif immediate_only:
                outcome = outcome[0] + "_IMMEDIATE_ONLY", outcome[1] + "；即刻响应在已检查时域内未接触目标，响应延迟边界未确定"
            else:
                outcome = outcome[0] + "_UNVERIFIED", "目标状态：" + outcome[1] + "；响应避碰效果尚待验证"
        if outcome and candidate.get("required_command_path"):
            return outcome[0] + "_REQUIRED_PATH", "目标状态（需建立独立响应通道）：" + outcome[1].removeprefix("目标状态：")
        return outcome

    @staticmethod
    def safe_state_path(candidate: Mapping[str, Any]) -> tuple[tuple, str] | None:
        """A display group for a response path, never risk/parameter equivalence."""
        response = candidate.get("transition", {})
        if response.get("model") != "CONSTANT_NET_DECELERATION":
            return None
        labels = EngineeringReportTextMapper._CONTROL_TEXT
        controls = tuple(sorted(candidate.get("required_controls", [])))
        if not controls or any(c not in labels for c in controls):
            return None
        held = response.get("post_stop_motion") == "HOLD_POSITION"
        positions = tuple(sorted(candidate.get("driver_positions", [])))
        scope = {("in_driver_seat",): "驾驶员在驾驶位", ("outside_driver_seat",): "驾驶员不在驾驶位"}.get(positions, "按成员适用条件")
        label = f"{scope}，通过{'、'.join(labels[c] for c in controls)}减速至停止" + ("并保持静止" if held else "")
        return (controls, positions, held), label

    @classmethod
    def safe_state_member(cls, candidate: Mapping[str, Any], scenario: Any,
                          timing_input: Mapping[str, Any] | None = None) -> str:
        outcome = cls.safe_state_outcome(candidate, timing_input)
        original = cls.hazardous_event(candidate["description"], scenario)
        text = "期望安全状态：" + (outcome[1] if outcome else original + "；响应避碰效果尚待验证")
        if outcome and not cls.has_foreign_prose(candidate["description"]):
            text += "；原提案及限定：" + original
        required = candidate.get("required_command_path")
        if required:
            text += "；尚需建立的系统响应：" + cls.analysis_note(required["description"],
                "独立系统命令通道尚未建立，完整依赖和原始要求见审计表原始分析说明")
        path = cls.safe_state_path(candidate)
        if path:
            text += "；处置路径：" + path[1]
        response = candidate.get("transition", {})
        if response.get("net_deceleration_mps2") is not None:
            text += "；处置净减速度 " + quantity(response["net_deceleration_mps2"], "m/s2") + " m/s²"
        return text

    @staticmethod
    def ftti_value(timing_inputs: list[Mapping[str, Any]]) -> str:
        lost = [x for x in timing_inputs if x.get("timing_trace", {}).get("status") == "AVOIDABILITY_ALREADY_LOST"]
        unsafe = [x for x in timing_inputs if x.get("timing_trace", {}).get("status") == "NO_IMMEDIATE_SAFE_BRAKING_WINDOW"]
        values = [x["hazard_time_limit_s"] for x in timing_inputs
                  if x.get("hazard_time_limit_s") is not None and x not in lost]
        if not values:
            parts = (["初始已无制动避碰余量"] if lost else []) + (["所选响应不能避碰"] if unsafe else [])
            pending = [x for x in timing_inputs if x not in lost and x not in unsafe]
            if pending:
                labels = dict.fromkeys(EngineeringReportTextMapper.ftti_gap(x) for x in pending)
                parts.extend(labels)
            return "；".join(parts) or "待确定"
        label = quantity(min(values), "s") + " s"
        gaps = Counter(
            EngineeringReportTextMapper.ftti_gap(x) for x in timing_inputs
            if x.get("hazard_time_limit_s") is None and x not in lost and x not in unsafe)
        return label + ("；含初始无制动避碰余量成员" if lost else "") + (
            "；含响应不能避碰成员" if unsafe else "") + (
            "；" + "、".join(f"{count}个成员{reason}" for reason, count in sorted(gaps.items())) if gaps else "")

    @staticmethod
    def ftti_gap(item: Mapping[str, Any]) -> str:
        return {
            "NO_ANALYSIS": "尚未建立响应模型",
            "CONTROL_PATH_UNRESOLVED": "缺少适用控制路径",
            "POST_STOP_MOTION_UNRESOLVED": "停车后保持待建模",
            "FAULT_CLOCK_UNRESOLVED": "故障起点或危险判据待绑定",
            "NO_CONTACT_WITHIN_MODEL": "模型时域内未建立接触边界",
        }.get(item.get("timing_trace", {}).get("pending_reason"), "待确定")

    @classmethod
    def safety_goal_timing(cls, timing_inputs: list[Mapping[str, Any]], *, detailed: bool = False) -> str:
        """Summarize sampled boundaries within the same declared response path.

        This is a display projection: no new timing computation or approval.
        Different response families, required controls and driver scopes remain
        separate, and unavailable members never disappear behind a known minimum.
        """
        groups: dict[tuple, list[Mapping[str, Any]]] = {}
        for item in timing_inputs:
            trace = item.get("timing_trace", {})
            command = trace.get("required_command_path") or {}
            key = (trace.get("response_model_family") or trace.get("response_model")
                   or item.get("physical_model", ""),
                   tuple(sorted(trace.get("required_controls") or ())),
                   item.get("driver_position", ""), item.get("time_origin", ""),
                   item.get("hazard_criterion", ""), trace.get("response_path_status", ""),
                   command.get("initiator", ""), tuple(sorted(command.get("independent_of") or ())))
            groups.setdefault(key, []).append(item)
        lines = []
        compact_paths = []
        for key, members in groups.items():
            family, controls, driver, origin, criterion, path_status, initiator, independent_of = key
            scope = []
            if driver:
                scope.append(cls.engineering_terms(driver))
            families = {"MOVING_TARGET_BRAKING": "移动目标制动并保持",
                        "RECTILINEAR_STATIONARY_TARGET_BRAKING": "静止目标有限包络制动并保持",
                        "MOVING_TARGET_FAULT_RELEASE": "解除异常减速并保持当时速度"}
            if family:
                scope.append(families.get(family) or cls._PHYSICAL_MODEL_TEXT.get(family, "所声明处置模型"))
            if controls:
                scope.append("依赖" + "、".join(cls._CONTROL_TEXT.get(c, "未分类控制路径") for c in controls))
            if path_status == "REQUIRED_NOT_ESTABLISHED":
                scope.append("独立响应通道待建立")
            if independent_of:
                scope.append("独立于" + "、".join(cls._CONTROL_TEXT.get(c, "未分类控制路径") for c in independent_of))
            if initiator:
                scope.append({"SYSTEM": "由系统发起", "REMOTE_OPERATOR": "由远程操作者发起",
                              "DRIVER": "由驾驶员发起", "OTHER_ROAD_USER": "由其他交通参与者发起"}
                             .get(initiator, "由方案声明的主体发起"))
            scope.append("从故障发生时计时" if origin == "FAULT_ONSET" else
                         "从模型初始状态计时" if origin == "MODEL_INITIAL_STATE" else "计时起点待绑定")
            if criterion:
                scope.append({"LOSS_OF_RESPONSE_AVOIDABILITY": "失去所选响应避碰能力",
                              "LOSS_OF_BRAKING_AVOIDABILITY": "失去制动避碰能力"}
                             .get(criterion, "所声明危险判据"))
            label = "；".join(scope) or "当前声明的处置范围"
            value = cls.ftti_value(members)
            lines.append(f"{label}：{value}（{len(members)}个成员）")
            numeric = [m["hazard_time_limit_s"] for m in members
                       if m.get("hazard_time_limit_s") is not None
                       and m.get("timing_trace", {}).get("status") not in {
                           "AVOIDABILITY_ALREADY_LOST", "NO_IMMEDIATE_SAFE_BRAKING_WINDOW"}]
            short_scope = ([cls.engineering_terms(driver)] if driver else [])
            short_scope.append(families.get(family) or cls._PHYSICAL_MODEL_TEXT.get(family, "所声明处置"))
            if path_status == "REQUIRED_NOT_ESTABLISHED":
                short_scope.append("通道待建立")
            compact_paths.append({"scope": "，".join(short_scope), "minimum": min(numeric) if numeric else None,
                                  "count": len(members), "missing": len(members) - len(numeric), "value": value})
        if not lines:
            return "尚无关联成员的时间边界；具体响应和缺口见场景明细。"
        if detailed:
            return "\n".join(lines)
        # Review cells show a bounded preview of strict sampled paths. The full
        # path labels, controls, clocks and all members remain in timing audit.
        compact_paths.sort(key=lambda p: (p["minimum"] is None,
                                          p["minimum"] if p["minimum"] is not None else 0))
        shown = compact_paths[:2]
        preview = []
        for index, path in enumerate(shown, 1):
            value = ("已计算最小 " + quantity(path["minimum"], "s") + " s" if path["minimum"] is not None
                     else path["value"])
            if path["minimum"] is not None and path["missing"]:
                value += f"；另{path['missing']}个成员无有效边界"
            preview.append(f"路径{index}（{path['scope']}）：{value}（{path['count']}个成员）")
        remaining = compact_paths[2:]
        if remaining:
            preview.append(f"另有{len(remaining)}条响应路径（{sum(p['count'] for p in remaining)}个成员），"
                           f"其中{sum(p['missing'] for p in remaining)}个成员无有效边界。")
        preview.append("各路径分别取已计算样本最小值；完整控制依赖、适用范围及未决原因见场景明细和时间追溯。")
        return "\n".join(preview)

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
        response_checks = []
        for item in timing_inputs or []:
            reason = str(item.get("reason", ""))
            trace = item.get("timing_trace", {})
            original = trace.get("original", {})
            response = trace.get("response_contact", {})
            if original.get("status") == "NO_CONTACT_WITHIN_MODEL" and response:
                checked_until = original.get("evaluated_until_s")
                detail = "以下轨迹时刻均从模型初态计时。"
                if checked_until is not None:
                    detail += "原轨迹检查至 " + quantity(checked_until, "s") + " s 未接触"
                    detail += ("，在自车停车时结束，未覆盖停车后。" if original.get("limiting_condition") == "EGO_STOP"
                               else "，仅覆盖声明时域。")
                if response.get("status") == "CONTACT":
                    detail += "即刻所选响应于 " + quantity(response["time_s"], "s") + " s 与同一目标接触"
                    detail += ("，接触发生在自车于 " + quantity(response["ego_stop_time_s"], "s") + " s 停车并保持之后。"
                               if response.get("contact_phase") == "HELD_AFTER_STOP" else
                               "，该响应保持当时速度行驶，并非停车。" if trace.get("response_motion") == "CONTINUE_AT_ONSET_SPEED"
                               else "，接触时自车尚未停稳。")
                elif response.get("status") == "NO_CONTACT_WITHIN_MODEL":
                    detail += "即刻所选响应检查至 " + quantity(response["evaluated_until_s"], "s") + " s 未接触；未建立响应延迟边界。"
                response_checks.append(detail)
            if reason.startswith("未形成该危险判据的数值边界："):
                code = reason.split("：", 1)[1]
                reason = boundary_reasons.get(code, "当前轨迹尚未形成有效时间边界，具体原因见时间追溯。")
            if EngineeringReportTextMapper.has_foreign_prose(reason):
                code = item.get("timing_trace", {}).get("pending_reason")
                reason = boundary_reasons.get(code) or EngineeringReportTextMapper.ftti_gap(item)
                reason = (reason + "；具体处置依据及原始解释见审计表原始分析说明。"
                          if reason != "待确定" else "")
            if any("\u4e00" <= ch <= "\u9fff" for ch in reason):
                explanations.append(EngineeringReportTextMapper.engineering_terms(reason))
        reasons = "；".join(dict.fromkeys(explanations))
        response_note = "".join(dict.fromkeys(response_checks))
        required_count = sum(item.get("timing_trace", {}).get("response_path_status") == "REQUIRED_NOT_ESTABLISHED"
                             for item in timing_inputs or [])
        required_note = (f"{required_count}个成员按尚需建立的独立系统响应评估，是否形成边界见各成员结果；该通道未被认定为现有能力，不计入C的避险收益。"
                         if required_count else "")
        lost = [item for item in timing_inputs or [] if item.get("timing_trace", {}).get("status") == "AVOIDABILITY_ALREADY_LOST"]
        lost_note = (f"{len(lost)}个成员在模型初始状态已无制动避碰余量：即使立即按假设净减速度制动，也不能在目标前停下。"
                     "原始0秒表示没有可用的响应延迟窗口，不是可供设计采用的正常FTTI，也不证明所有其他处置均无效。" if lost else "")
        calculated = [item for item in timing_inputs or [] if item.get("hazard_time_limit_s") is not None and item not in lost]
        if lost and not calculated:
            return required_note + lost_note + response_note + ("其余成员的具体时间缺口见逐成员追溯。" if len(lost) != len(timing_inputs or []) else "")
        if calculated:
            missing = len(timing_inputs or []) - len(calculated) - len(lost)
            moving = any(item.get("timing_trace", {}).get("response_model_family") == "MOVING_TARGET_BRAKING" for item in calculated)
            release = any(item.get("timing_trace", {}).get("response_model_family") == "MOVING_TARGET_FAULT_RELEASE" for item in calculated)
            return (required_note + "条件性时间边界：从故障发生到失去所选响应的避碰能力；"
                    + ("主表取已计算有效成员的最小值，逐成员数值见时间追溯。" if len(calculated) > 1 else "")
                    + ("解除异常减速的成员保持响应开始时的速度，在声明时域内检查与移动目标的间隔，不表示停车或瞬间恢复故障前速度。" if release else "")
                    + ("移动目标按已声明的匀速或反应后制动轨迹运动，检查自车故障行驶、制动及停车保持三段，在模型有效时段内确定首次失去避碰余量的边界。" if moving else
                       "依据同一轨迹、静止目标及剩余净减速假设计算。" if not release else "") +
                    "检测、决策和处置建立必须在该边界前完成；后续完成耗时不与此边界相加。"
                    "尚未经工程批准，不是完整ODD的FTTI；假设与逐成员计算见时间追溯。"
                    + lost_note + response_note + (f"仍有{missing}个成员尚无数值时间边界，具体控制、轨迹或输入缺口见逐成员追溯；已知最小值不代表全组结论。" if missing else ""))
        timings = [row for item in timing_inputs or [] for row in item.get("branch_timings") or [item]]
        available = [item for item in timings if item.get("contact_time_s") is not None]
        if not available:
            if response_note:
                return required_note + reasons + response_note + "所选响应的接触检查不能作为未干预轨迹的接触时限或 FTTI。"
            if timing_inputs and all(item.get("timing_trace", {}).get("pending_reason") == "NO_CONTACT_WITHIN_MODEL"
                                     for item in timing_inputs):
                return required_note + reasons
            return required_note + "缺少可用的接触时间；" + (reasons or "故障起点、危险判据及安全处置时间预算尚未确定，FTTI待定。")
        kinds = {item.get("contact_time_kind") for item in available}
        label = "首次接触" if kinds == {"first_contact_time_s"} else "轨迹接触"
        values = [item["contact_time_s"] for item in available]
        extent = (f"最小 {quantity(min(values), 's')} s、最大 {quantity(max(values), 's')} s"
                  "（仅为已计算样本统计，不代表连续可达范围）" if min(values) != max(values)
                  else quantity(values[0], "s") + " s")
        origin = ("从各模型初始状态计时" if all(
            item.get("time_origin") == "MODEL_INITIAL_STATE" for item in available)
            else "计时起点以各成员来源为准")
        return (required_note + f"已有{label}时间 {extent}（{origin}）；"
                + (reasons or "尚未绑定故障起点及危险判据") + "接触时间不能作为 FTTI。")


__all__ = ["EngineeringReportTextMapper"]
