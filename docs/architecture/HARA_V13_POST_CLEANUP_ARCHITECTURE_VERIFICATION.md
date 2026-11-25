# HARA_V13 Post-Cleanup Architecture Verification Report

> 历史设计与核验记录：本文中的“当前”、缺口和运行结论仅对应下述日期或提交。现行实现与使用方式见[开发交接手册](../handover/README.md)；历史记录索引见[文档目录](../README.md#历史开发与验证记录)。

日期：2026-08-21  
审阅基线：`68c0bfd`（Runtime Cleanup + Cutover），当前工作树 HEAD 为 `77fc921`。  
验证原则：先验证能力与调用链，不恢复旧 Runtime、不修改生产计算逻辑。

## 1. 结论

本次 Cutover 的主计算链已经切换到 Template-driven Runtime，并且 Domain/Profile 不再参与运行期评分。当前生产图可以从 Item 文档一路执行到草稿 Excel：

`Item → ProjectFacts/Functions → Guideword → Malfunction/Hazard → Method Scenario → Scenario causal evidence → MethodContract S/E/C → ASIL → Safety Goal → ReportContract Excel`

但是，**不能把 68c0bfd 判定为“没有能力损失”**。它同时删除了 Scenario Evidence V2、typed causal mechanism、Provider schema/conformance、real-provider/differential evaluation、FTTI service 和原有完整 HARA 回归测试。这些能力没有全部被等价替换。当前 Scenario 主链仍存在并可运行，但从 V2 的强类型机制契约退回到了 `scenario-evidence-v1 + dict causal_chain`；这是结构与验证能力的降级，而不是简单的文件搬迁。

因此本报告判定：

- Template-driven Cutover：**通过**。
- Domain 脱钩：**通过**。
- 当前 Scenario 四跳因果链存在且 fail-closed：**通过（V1 范围）**。
- 68c0bfd 无静默功能减少：**不通过**。
- 当前代码可支持 Draft HARA：**通过**。
- 当前代码具备与 Cleanup 前等价的 Provider/Scenario/FTTI 验证能力：**不通过**。

## 2. Item 到 Report 的当前完整调用链

生产入口由 `hara_agent.cli` 进入 `HARAApplication.run()`：

1. `TemplateRoleCompiler.compile_method()` 扫描模板、发现角色并编译 `MethodContract` 和 `ReportContract`。
2. `build_hara_agent_graph()` 注册唯一生产图。
3. `INITIALIZE`：`read_item_document()` 将 DOCX/PDF 转为带位置的 document blocks。
4. `EXTRACT`：`extract_item_artifacts()` 调用 `ItemArtifactExtractionAgent`，生成 typed `ItemDefinitionFacts` 与 `FunctionDefinition`；缺失项通过有界 evidence routing 和 targeted fact extraction 补抽取。
5. `FUNCTIONS`：`GuidewordApplicabilityAgent` 对模板编译出的全部 Guidewords 做完备性判断。
6. `HAZOP`：`MalfunctionHazardAgent` 生成 Malfunction、functional effect、vehicle-level hazard 和初始 causal chain。
7. `MALFUNCTIONS`：`MethodScenarioCandidateService` 用 `MethodContract.scenario_model + ProjectFacts` 生成候选；`ScenarioFeasibilityAgent` 执行四跳因果与证据契约验证；`ScenarioRiskFactAgent` 只解释模板要求的中性 scenario facts。
8. `SCORING`：`MethodRiskFactBindingService` 绑定 canonical facts；`MethodRuleScoringService` 只执行编译后的 S/E/C rules；`MethodContractASILService` 只查当前 `MethodContract.asil`。
9. `SAFETY_GOALS`：`MethodSafetyGoalService` 生成确定性的 SG/Safe-State proposal，并保持工程评审门禁。
10. `QUALITY_GATE`：任何 PENDING、REJECTED、error 均阻止正式发布。
11. `RENDER`：`HARAExcelRenderer` 只使用 `ReportContract` 坐标，复制模板包、原子写入并 reopen 验证。`--allow-draft` 在质量门阻断时生成带水印草稿。

关键装配位置：

- `src/hara_agent/application.py:53-183`
- `src/hara_agent/workflow/semantic_graph.py:125-227`
- `src/hara_agent/workflow/nodes/item_artifacts.py`
- `src/hara_agent/workflow/nodes/scenarios.py`
- `src/hara_agent/workflow/nodes/scoring.py`
- `src/hara_agent/workflow/nodes/safety_goals.py`
- `src/hara_agent/services/reporting/excel_renderer.py`

## 3. Scenario causal chain 当前在哪里

当前能力分为三层：

1. `models/malfunction.py` 的 `MalfunctionCandidate.causal_chain: list[str]` 保存从失效到 vehicle-level hazard 的初始链，并要求至少两段。
2. `services/semantic/scenario_agent.py` 的 `scenario-feasibility-v9` 要求每个 retained Scenario 提供：
   - `m_to_b`
   - `b_to_i`
   - `i_to_h`
   - `h_to_harm`
   - `breakpoint`
   - 每跳 `claim + basis_type + evidence_refs`
3. `services/semantic/scenario_evidence.py` 的 `validate_evidence_contract()` 校验四跳完整性、breakpoint 一致性、EvidenceRef 可解析性、证据 kind、ASSUMPTION 禁止进入 positive chain、risk dimension 与 causal 结论的一致性。

人工 5 km/h 制动失败 case 的确定性验证结果：

- 完整四跳链：`ACCEPTED`。
- 仅保留 “brake failure → effect” 的压缩链：`REJECTED / REQUIRED_HOP_MISSING / b_to_i`。

所以 Agent 的核心因果链没有完全消失；但当前 `ScenarioFeasibilityAssessment.causal_chain` 是 `dict[str, Any]`，不是 68c0bfd 前 `causal_mechanism.py` 提供的强类型 mechanism/edge/support 模型。

## 4. scenario-evidence-v2 对应关系

**没有等价的新模块。**

68c0bfd 删除了：

- `contracts/scenario_evidence_v2.py`
- `contracts/causal_mechanism.py`
- `semantic/scenario_provider_contract.py`
- `semantic/scenario_provider_schema.py`
- `semantic/provider_contract_conformance.py`
- V1/V2 differential、V10 schema/parser、provider snapshot 和 real-provider metrics 测试

当前运行时使用：

- `SCENARIO_ASSESSMENT_CONTRACT_VERSION = "scenario-evidence-v1"`
- `ScenarioFeasibilityAssessment.causal_chain: dict[str, Any]`
- `services/semantic/scenario_evidence.py`
- `services/semantic/scenario_agent.py`（prompt v9）

当前 V1 能阻止缺跳、假设、非法引用和部分 evidence kind 错误，但它自己的注释明确标记 prompt registry 为 `provenance-unaware`。被删 V2 的 typed support、mechanism catalog/fingerprint、Provider schema/conformance 和 differential evaluation 没有在别处重建。因此，不能把 `scenario_evidence.py` 称为 V2 的“新位置”；准确说法是：**生产 Runtime 回退并收敛到了较轻的 V1 契约。**

## 5. 68c0bfd 删除项替代审阅

### 5.1 已等价或基本等价接管

| 删除的核心能力 | 当前接管者 | 结论 |
|---|---|---|
| `domains/*`、`config/domains/avp/profile.json` | `TemplateRoleContract → MethodContract`、`Method*Service` | 正确切换；Runtime 无 Domain import |
| `scenario_candidate_service.py` | `method_scenario_service.py` | 已接管，候选由 Method scenario dimensions + ProjectFacts 驱动 |
| `scoring_service.py` | `method_rule_service.py` | 已接管，不做 prose/keyword 评分 |
| `risk_aggregation_service.py` | `workflow/nodes/scoring.py` + Method services | 已接管 |
| `safety_goal_service.py` | `method_safety_goal_service.py` | 已接管，保持 PENDING/approval gate |
| `template_inputs.py`、`template_scoring.py` | `template/` compiler + `MethodContract.required_fact_specs` | 已接管 |
| `template_contract.py`、`report_generator.py` | `ReportContract` + `HARAExcelRenderer` | 已接管 |
| old item/function workflow nodes | `workflow/nodes/item_artifacts.py` | 合并接管 |
| old controllers/engines/main executor | `cli.py` + `application.py` + typed workflow graph | 正确收敛为单 Runtime |
| Word/PDF parser scripts | `services/extraction/document_reader.py` | 当前生产输入范围内已接管 |
| function/scoring checkers | `FunctionValidator`、Method rule fail-closed、quality gate | 基本接管 |
| `asil_matrix.py` | `MethodContractASILService` | 正确接管；ASIL 只来自 active MethodContract |

### 5.2 有意删除、无需恢复为 Runtime

| 删除项 | 判断 |
|---|---|
| `compatibility/v13_json.py`、compatibility node | 不应恢复；会形成第二入口/兼容 Runtime |
| `ScenarioRef.json`、`HazardRef.json`、`SafetyStateRef.json`、`ScorabilityRef.json` | 不应恢复为可执行规则源；它们会与 Item/Template 形成额外权威 |
| `experience_library.py` | 不应恢复为隐式决策源；若未来需要，只能作为非执行参考或显式受治理 EvidenceProvider |
| AVP fixture/profile | 不应恢复到生产链；领域数据只能来自甲方 Item/Template |

### 5.3 未被等价替代，构成能力/验证缺口

| 删除项 | 当前状态 | 影响 |
|---|---|---|
| `causal_mechanism.py` | 仅剩 `dict causal_chain` + V1 validator | typed edge/support/mechanism identity 丢失 |
| `scenario_evidence_v2.py` | 无等价替代 | provenance-aware / typed V2 契约丢失 |
| Provider schema/contract/conformance | 无等价替代 | 无法证明不同 Provider 都遵守同一机器契约 |
| V1/V2 differential、order invariance、real-provider metrics | 当前 Scenario harness 只覆盖重复样本和错误计数 | 语义漂移、顺序敏感、Provider 特异性缺少自动检测 |
| `ftti_service.py`、`scripts/ftti.py` | 无替代；prompt/risk dimensions 仍出现 `ftti` | 声明与可执行能力不一致，属于悬空能力 |
| `eval_hara.py`、`scenario_smoke.py`、原 `test_agent_core.py` 全链路覆盖 | 无永久 full-graph synthetic test | 120 个测试不能直接证明全链能力未减少 |
| 大型 Scenario/Hazard reference library | 不再执行，也无新 coverage source | 架构上正确去权威化，但 Scenario 数量/覆盖率可能实际下降，必须通过甲方 gold set/eval 测量 |

## 6. 分层测试结果

| 层 | 命令/检查 | 结果 |
|---|---|---|
| Template Compiler | `pytest tests/unit/test_full_template_compiler.py -v` | 15 passed；包含 Guideword 14→15、Scenario dimension mutation、规则 mutation、SourceRef、冲突 fail-closed |
| ProjectFacts | `pytest tests/unit -k fact -v` | 40 passed；覆盖原子事实、provenance、speed context、hash-bound binding、targeted extraction |
| Scenario | `pytest -k scenario -v` | 17 passed；覆盖 evidence errors、risk facts、contract metrics；不包含 V2/provider conformance |
| Method scoring | Method rule/binding/runtime injection tests | 14 passed；缺事实/边界歧义保持 PENDING，ASIL 只来自 MethodContract |
| Safety Goal / ASIL | `pytest -k "safety_goal or asil" -v` | 3 passed |
| Static Domain check | `DomainProfile/AVPDomainPolicy/DomainRegistry` 搜索 | Runtime 无命中 |
| Static fallback check | `fallback` 搜索 | 只有显式 aggregate-speed opt-in、LLM format repair、数据字段和普通默认参数；未发现 missing fact → AVP default |
| Manual causal case | 四跳/压缩链对照 | 四跳 accepted；压缩链 fail-closed |
| Doctor | active template | `ready_for_draft=true`，`ready_for_release=false`，MethodContract OK |

当前模板编译为 `READY_WITH_WARNINGS`，14 Guidewords、5 Scenario dimensions、8 Required facts。非阻断 warning 包括：

- `EXPOSURE_METHOD_SELECTION_UNRESOLVED`
- `AMBIGUOUS_RESULT`
- `RULE_RANGE_GAP`
- `SEMANTIC_VARIABLE_UNRESOLVED`
- `UNRESOLVED_RULE_VARIABLE`
- `ASIL_NA_SEMANTICS_UNRESOLVED`
- `WORKFLOW_COORDINATE_MISMATCH`
- `MALFORMED_TEMPLATE_TEXT`

这与 fail-closed 目标一致：模板可用于 Draft，但不能声称 release-ready；Exposure T/F 不会自动取 E4。

## 7. 完整 Synthetic HARA 结果

本次使用受控 Synthetic Provider 跑的是当前生产 `build_hara_agent_graph()`，不是单独调用 service。执行包含：文档读取、核心抽取、14 Guideword 完整评估、Malfunction/Hazard、Method Scenario、四跳 Scenario Evidence、Method S/E/C、ASIL、Safety Goal 和 Excel draft render。

结果：

```json
{
  "provider_calls": [
    "extract_core_item_artifacts",
    "assess_guideword_applicability",
    "derive_malfunctions_and_hazards",
    "assess_scenario_feasibility"
  ],
  "functions": 1,
  "malfunctions": 1,
  "retained_scenarios": 1,
  "causal_hops": ["m_to_b", "b_to_i", "i_to_h", "h_to_harm"],
  "risk_results": 1,
  "sec": ["S1", "E4", "C3"],
  "asil": "B",
  "safety_goals": 1,
  "quality_gate": "pending_engineering_review",
  "draft_excel_generated": true
}
```

S/E/C、ASIL、SG 均为 `PENDING`，这是正确结果：Synthetic Provider 只能产生待评审语义结论，不能伪造 FINALIZED 工程权威。草稿文件位于：

`tmp/post-cleanup-synthetic/synthetic-hara-report.xlsx`

当前 `HARAState` 没有独立顶层 `project_facts`、`hazards`、`evidence` 字段：

- ProjectFacts 位于 `item_definition["typed"]`；
- Hazard 位于 Malfunction 和 Scenario assessment 中；
- Evidence 位于 SourceRef、fact provenance、FactRegistry 与 scenario assessment 中；
- 风险字段名为 `risk_results`，不是 `risk_assessment`。

这是状态结构收敛，不等于数据完全丢失；但它降低了架构验收时的可观测性。后续应增加只读 typed view/accessor，而不是复制出第二份状态真源。

## 8. 额外耦合审阅

Runtime 评分逻辑中没有发现 5 km/h、30 km/h、driver-inside 或 AVP 默认值。`pedestrian/cyclist/rear/front/side` 仍出现在：

- Template role discovery 的结构语义 signature；
- Severity rule parser 将模板文本映射到 canonical FactType；
- Scenario prompt 的非执行 evidence illustrations。

它们不是 S/E/C 分值硬编码，但会限制“任意不同术语模板”的泛化能力。尤其 `method_compiler.py` 的 collision subject 词表属于编译器 ontology adapter，应后续迁移为可扩展、版本化的 parser vocabulary 或模板内显式 canonical annotation；不能让它演变为新的 DomainProfile。

## 9. 建议的下一步（不恢复旧 Runtime）

1. 保留当前唯一 Runtime 和 MethodContract 评分链，不回滚 68c0bfd。
2. 在当前 Scenario v9 路径上恢复一个**领域中立、强类型、provenance-aware** 的 causal/evidence contract；可借鉴被删 V2，但不得恢复 mechanism catalog 作为业务规则源。
3. 恢复 Provider schema/conformance、order-invariance 和 differential evaluation，绑定当前 contract version；它们属于测试治理，不是第二 Runtime。
4. 将本次 Synthetic full-graph case 固化为永久 integration test，并断言四跳链、evidence refs、MethodContract 评分来源、SG 与 draft renderer。
5. 对 FTTI 做明确产品决策：若 Template/Report 要求则编译成 MethodContract executor；否则从 active prompt/schema/risk dimension 中移除，避免“看起来支持、实际不可计算”。
6. 用甲方 gold set 比较 Cleanup 前后 Hazard 数量、Scenario coverage、evidence completeness、PENDING 分布和 SG coverage；只有这一层能判断 reference library 删除后输出质量是否下降。

在完成第 2-6 项之前，不建议把当前状态描述为“68c0bfd 完整无损”；更准确的工程表述是：**Template-driven 主链已成功 Cutover，但 Scenario contract/evaluation 能力需要补回。**
