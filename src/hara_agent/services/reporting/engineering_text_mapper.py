from __future__ import annotations

from collections.abc import Mapping
import re
from typing import Any


def _status(evidence: Any) -> str:
    value = getattr(evidence, "status", "")
    return str(getattr(value, "value", value))


class EngineeringReportTextMapper:
    """Map committed runtime status to compact Chinese review text.

    This layer owns presentation wording only.  It neither supplements project
    facts nor makes a risk-method decision.
    """

    _PENDING_VALUE = "Pending"
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

    @staticmethod
    def _has_reason(value: Any, token: str) -> bool:
        return token in str(value or "").upper()

    @classmethod
    def _display_mode(cls, mode: Any) -> str:
        return cls._MODE_TEXT.get(str(mode or "").strip().casefold(), "")

    @staticmethod
    def _number(value: Any) -> str:
        number = float(value)
        return f"{number:g}"

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
            facts.get("object_type", facts.get("road_user_type", "")) or ""
        ).casefold()
        if any(token in object_value for token in ("pedestrian", "person", "行人")):
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
            "OBJECT_PEDESTRIAN", "OBJECT_CYCLIST", "OBJECT_VEHICLE",
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
            ("自车速度", value("ego_speed_kph"), "km/h"),
            ("目标速度", value("object_speed_kph"), "km/h"),
            ("相对速度", value("relative_speed_kph"), "km/h"),
            ("接近速度", closing_speed, "km/h"),
            ("相对距离", value("relative_distance_m"), "m"),
            ("TTC", ttc, "s"),
        )
        parts = [
            f"{label}：{cls._number(number)} {unit}"
            for label, number, unit in quantities
            if isinstance(number, (int, float)) and not isinstance(number, bool)
        ]
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
            parts.append(f"未知覆盖策略：{policy}")
        action = str(control.get("unknown_policy_action", ""))
        if action and action not in {"NOT_INVOKED", ""}:
            parts.append(f"策略处理：{action}")
        return "；".join(parts)

    @staticmethod
    def analysis_basis(scenario: Any) -> str:
        provenance = getattr(scenario, "fact_provenance", {}) or {}
        if not isinstance(provenance, Mapping):
            return ""
        labels = {
            "ego_speed_kph": "自车速度", "object_speed_kph": "目标速度",
            "relative_distance_m": "相对距离", "ttc_s": "TTC",
            "driver_position": "驾驶员配置",
        }
        parts = []
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
            if source_locations:
                description.append("来源 " + ", ".join(dict.fromkeys(source_locations)))
            if description:
                parts.append(f"{label}：{'；'.join(description)}")
        return "；".join(parts)

    def scenario(self, scenario: Any, *, variant_count: int = 1) -> tuple[str, str]:
        """Render validated structured facts without exposing Method vocabulary."""
        facts = getattr(scenario, "facts", {}) or {}
        categories = self._categories(scenario)
        location = self._chinese_source(
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
            "OBJECT_PEDESTRIAN", "OBJECT_CYCLIST", "OBJECT_VEHICLE",
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
        details.append(f"分析变体：{self.variant_text(scenario)}")
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
        return "Not applicable" if _status(evidence) == "NOT_APPLICABLE" else self._PENDING_VALUE

    def potential_harm(self, value: Any, *, severity: Any) -> str:
        if str(value or "").strip():
            return str(value)
        if _status(severity) != "FINALIZED":
            return "待S评定完成后确定"
        return self._PENDING_VALUE

    def severity_rationale(self, evidence: Any, trace: Mapping[str, Any]) -> str:
        if _status(evidence) == "FINALIZED":
            return "已按当前方法完成 S 评定。"
        reason = str(trace.get("pending_reason", "") or getattr(evidence, "review_reason", ""))
        if self._has_reason(reason, "MISSING_RELATIVE_SPEED"):
            return "缺少该危险事件的实际相对速度，S 暂不评定。"
        if self._has_reason(reason, "MISSING_ROAD_USER_TYPE"):
            return "缺少交通参与者类型，S 暂不评定。"
        if self._has_reason(reason, "MISSING_COLLISION_TYPE"):
            return "缺少碰撞类型信息，S 暂不评定。"
        return "缺少 S 评定所需项目事实，S 暂不评定。"

    def exposure_rationale(self, evidence: Any, trace: Mapping[str, Any]) -> str:
        if _status(evidence) == "FINALIZED":
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
            atom_text = "、".join(atoms) if atoms else "无方法要素（S0 短路）"
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
        reason = trace.get("decision_status", "") or getattr(evidence, "review_reason", "")
        if self._has_reason(reason, "METHOD_BRANCH_UNRESOLVED") or self._has_reason(reason, "UNKNOWN_BRANCH_POLICY_UNSPECIFIED"):
            return "未知分支策略未定义，C 暂不评定。"
        return "缺少 C 评定所需控制上下文或方法分支，C 暂不评定。"

    def asil_rationale(self, evidence: Any, trace: Mapping[str, Any]) -> str:
        if _status(evidence) == "FINALIZED":
            return "已按当前 ASIL 矩阵完成评定。"
        return "S/E/C 未全部确定，ASIL 暂不评定。"

    @staticmethod
    def ftti_rationale() -> str:
        return "当前运行未启用 FTTI 计算。"


__all__ = ["EngineeringReportTextMapper"]
