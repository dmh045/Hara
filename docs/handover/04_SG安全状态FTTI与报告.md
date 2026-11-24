# 04 SG、安全状态、FTTI与报告

[返回交接总览](README.md)

本文以提交 `d6a400e7010d62ff5dda56efd8215f23729bb3ec` 为代码交接基线。它说明接手者应沿哪些生产职责排查和修改，不新增一套分析方法。最新交付的数量、验证时点和未决分类见[第07章](07_验证结果与已知限制.md)。这里的“已计算”均须结合来源、适用范围和审批状态理解。

## 1. 先分清四种对象

SG描述需要避免的危险行为；Safe State候选描述特定成员可尝试达到的状态及控制依赖；FTTI链路评估所选响应的条件性时间边界；报告把上述对象投影为可审核的工程表达。四者互有关联，但不能相互代替。

例如，“避免非预期加速”可以是多个驾驶员配置共享的SG意图，但车内踏板、车外独立系统命令、解除异常制动后的继续行驶，属于不同响应路径。一个路径算出了边界，不能给另一个路径补上能力或审批。C3也不能直接推导一个FTTI，碰撞接触时间更不能直接填写到FTTI栏。

实际执行链路如下。时间评估早于SG汇总就会发生，后续节点继续调用同一个执行器，不能把图中的多次调用误解为多套时间算法：

```text
结构化物理/控制方案
  → 方案响应验证（必要时调用ftti，产生负面/纠正诊断）
  → 当前scope的因果与风险准入判断
  → score_structured_scenarios
      分支S/E/C/ASIL；调用ftti写入风险assessment的ftti_input
  → aggregate_safety_goals / MethodSafetyGoalService
      再用ftti核验当前响应；抑制不安全候选，建立SG及成员/缺口
  → 正式质量门 / 受授权的草稿输出路径
  → HARAReportProjectionService
      按成员重取时间证据，生成主表/明细/SG路径显示
  → HARAReportWorkbookRenderer + 重开验证 + 报告审计
```

阶段顺序见[生产图SCORING→SAFETY_GOALS节点注册](../../src/hara_agent/workflow/semantic_graph.py#L339)，评分期的时间记录见[score_structured_scenarios()中的ftti_input](../../src/hara_agent/workflow/nodes/scoring.py#L667)，方案期的调用见[响应验证诊断](../../src/hara_agent/workflow/physical_analysis.py#L637)。其他主要入口是[SG工作流节点](../../src/hara_agent/workflow/nodes/safety_goals.py#L8)、[时间输入服务](../../src/hara_agent/services/analysis/risk_calculation_input_service.py#L223)、[时间执行器](../../src/hara_agent/services/analysis/deterministic_risk_executor.py#L18)和[报告投影](../../src/hara_agent/services/reporting/projection.py#L790)。不要在renderer里补计算，也不要在mapper里发起语义生成。

## 2. SG如何产生、保持身份并汇总

### 2.1 身份与正文是两件事

当前生产实现是[MethodSafetyGoalService](../../src/hara_agent/services/analysis/method_safety_goal_service.py#L9)。`_intent_id()`把功能名称、引导词、失效描述分别去空白、casefold，组合后取SHA-256前10位，生成 `SG-METHOD-…`。场景速度、距离、驾驶员位置不参与这个共享意图的ID；这些限定保存在成员关联和响应候选中。

`register_intent()`保存简短意图、首次原始SG表达、功能及场景关联、已知最高ASIL、Safe State候选，以及方法来源。重复汇总时，[aggregate_safety_goals()](../../src/hara_agent/workflow/nodes/safety_goals.py#L8)优先保留既有 `original_safety_goal`。报告改善语句时，应保持SGID、原始表达、成员和状态，不要以“更好读”为由重新生成目标。

这个ID算法也意味着：功能名称或失效描述发生真实修改，可能改变SGID。迁移时必须核对关联，不应把它当随机ID漂移后强行替换。另一方面，报告里的 `HARA_…` 审核组编号是显示定位，不是稳定的场景业务身份；跨版本对照优先使用 `scenario_id` 和失效ID。

### 2.2 QM、未知ASIL和Max.ASIL

工作流跳过已确定QM的风险成员；对非QM以及ASIL尚未确定的成员保留安全意图。不能因为某个成员未算完，就让其从目标和缺口中消失；也不能为了让QM行有内容而补造SG。

实际S/E/C到ASIL的判定由[MethodContractASILService.determine()](../../src/hara_agent/services/analysis/asil_service.py#L33)查询当前编译矩阵。SG目录当前用 `MethodSafetyGoalService.ASIL_RANK` 对标准QM/A/B/C/D作已知结果摘要排序，这个常量不负责计算ASIL，也不是另一张矩阵。接手者若扩展结果标签或改变排序含义，须同时检查目录汇总与 `result_order()`；现有五总体的SG最高值已按各自活动矩阵顺序独立核验。未知成员仍需明确保留，已知最高值不代表全体已批准。

### 2.3 状态不会因“有文字”自动批准

`is_approved`读取SG/Safe State方法是否还要求语义推导。工作流还检查关联风险状态、候选和方法审批，通常将语义候选保留为 `PENDING`，并加入 `pending_reviews`。`derivation_status`、计算完成、因果契约通过和工程审批不是同一个状态轴。

正式放行由[ReleaseGateValidator.evaluate()](../../src/hara_agent/services/validation/release_gate.py#L32)只读检查方法、项目事实、因果契约、风险、目标和未决审核；它不修复数据，也不替人批准。`--allow-draft`允许形成草稿，不等于消除这些阻断项。

## 3. Safe State候选从哪里来，怎样判可达

### 3.1 候选的最小工程含义

候选通常来自结构化物理/控制方案的 `hypothesis.safe_state_proposal`。其契约在[PhysicalScenarioPlan及控制校验](../../src/hara_agent/contracts/physical_scenario_plan.py#L281)，实例化职责在[PhysicalHypothesisInstantiationService](../../src/hara_agent/services/analysis/physical_hypothesis_instantiation_service.py#L482)。需要一起阅读：

| 字段 | 审核含义 |
|---|---|
| `description` | 要达到的状态，不是能力证明 |
| `required_controls` | 此提案实际依赖的控制，不能混入失效控制 |
| `transition` | 已支持的响应模型、参数和停止后运动 |
| `conditions`、`basis_refs` | 成立条件、来源或分析依据 |
| `driver_positions` | 提案适用的人员配置，不能借另一分支 |
| `required_command_path` | 尚需建立的独立系统通道，不是现成接口 |

`remaining_controls`表示本次有范围限定的保留能力，不代表动作已经触发。对人工制动要核对驾驶位和故障后路径；对远程干预，App存在不等于有独立制动接口；对他方避让，要使用相应轨迹和适用性证据。人员专属方案的选择由[controls_for_driver()](../../src/hara_agent/contracts/physical_scenario_plan.py#L119)承担。其受验证的singleton request scope回退只服务原适用配置，不可默认驾驶员在位。

### 3.2 “停下”不等于“避免接触”

有限包络交互需检查同一目标在响应前、制动中和停车保持后的运动。目标可以继续接近或横穿，停止位置也可能仍在其路径上。只有停止耗时，或者只检查到本车停车瞬间，都不足以证明Safe State。

特别是原轨迹得到 `NO_CONTACT_WITHIN_MODEL` 时，不能提前返回“安全”。原轨迹可能只算到 `EGO_STOP`；响应后停车保持阶段的检查范围可能更长。应检查所选响应是否也无接触，并保留两段各自时域。若即刻响应仍接触，记录所选响应不可达；若双方均无接触，也只是说明该即时响应在声明时域内已检查，不能创造一个任意延迟均安全的正边界。相关反例见[响应证据复用测试](../../tests/unit/test_response_evidence_reuse.py)。

两类常见响应需要分清：`CONSTANT_NET_DECELERATION`描述净减速，有限交互通常还需明确 `HOLD_POSITION`；`RELEASE_DECELERATION_TO_CONSTANT_SPEED`描述解除异常减速后维持当时速度，结果仍在运动，不能显示成停车保持。二者都必须经过当前控制依赖和方法适用性校验。

### 3.3 负面证据、撤回与当前重验

[工作流的 `_retain_rejected_response()`](../../src/hara_agent/workflow/physical_analysis.py#L263)把被撤回方案、其原始输入快照和负面时间追溯留在 `rejected_response_evaluation`。不能删除这段历史来把“已经算出不安全”改成“缺输入”，也不能把历史提案重新放回当前候选。

时间执行器可在计算副本中恢复旧提案，并使用当前方法、来源、控制、人员和分支约束重验。只有当前仍得到负面，才继续显示负面。若当前缺命令路径、结果未定或变为正值，应报告重验差异，保持撤回，不能因此重新批准方案。

一个实际修复点是当前DERIVED轴向/TTC缓存持久化后，旧快照的 `longitudinal_contact=False` 与新值失配。[_rejected_response_current_physics_proof()](../../src/hara_agent/services/analysis/deterministic_risk_executor.py#L33)只接受已证明的特定缓存变化：其余响应输入精确相同，现有物理materializer重算出相同完整当前快照和TTC，所用输入的来源权威一致，然后同一执行器在冻结/当前视图下的结果字典也必须一致。它不是忽略字段的通用兼容层。对象、数值、来源或当前控制不相容仍阻断；无单独历史provenance快照的旧记录，也不能被描述为拥有这种不存在的证明。四个实际世界的正反例在[撤回重验测试](../../tests/unit/test_withdrawn_axis_cache_revalidation.py)。

## 4. FTTI链路到底算什么

### 4.1 数据对象与方法边界

[FTTIAssessmentInput](../../src/hara_agent/contracts/risk_calculation.py#L742)明确区分 `contact_time_s`、`hazard_time_limit_s`、`safe_state_transition_time_s`、`time_origin`、`hazard_criterion`、分支时间和 `timing_trace`。调用方需要保留这些字段的语义，不应只消费一个浮点数。

`RiskCalculationInputService.ftti()`组织输入；`FTTIMethodExecutor.evaluate()`执行编译后的 `timing_analysis`。模型范围、响应类型、危险判据和来源来自MethodContract；运行时不直接打开YAML。治理来源可从[项目分析策略中的时间配置](../../method_assets/fusa_baseline_v1/normalized/project_analysis_policy.yaml#L261)追到编译结果，但该文件不是让report mapper临时读取的规则表。

### 4.2 起始窗口和完成耗时不能相加填写

对适用的静止目标，现有[braking_avoidability_boundary()](../../src/hara_agent/services/analysis/scenario_physics.py#L584)求满足以下临界关系的响应启动延迟：

```text
v0·τ + a·τ²/2 + (v0+a·τ)²/(2·b_net) = 可用路径距离
```

这里 `b_net` 是有来源/条件声明的净减速度，不是程序默认能力；可用距离也不是随手把中心距离当保险杠净距。假设仅作解释的输入为v0=1 m/s、a=0、可用路径4 m、b_net=2 m/s²，则启动边界为3.75 s，之后停止耗时0.5 s。不能将两者相加成4.25 s再填作允许检测/决策延迟。真实系统的检测、决策和有效响应建立延迟须落在响应启动窗口以内。

若故障发生到模型初态有已建立的偏移，`hazard_time_limit_s`按同一故障时钟组合；缺偏移不能默认为零。模型时钟接触时间和故障时钟危险边界不能直接大小比较。该程序也不凭一个数值补全实际诊断周期、通信或执行器延迟。

### 4.3 有限目标和无窗口状态

[moving_braking_avoidability_boundary()](../../src/hara_agent/services/analysis/scenario_physics.py#L666)复用分段目标/自车轨迹及接触包络，寻找与即时安全响应连通的首次失去安全间隔边界。它保留故障行驶、响应和保持阶段；一般横穿结果不能未经单调性条件证明就盲目二分。模型有效时域以外不外推，曲率或其他模型也须先满足已接入响应的适用条件。

审核时应按证据状态区分：

| 状态 | 正确解释 |
|---|---|
| `CONDITIONAL_BOUNDARY`且时钟已绑定 | 所选条件性路径的数值窗口，仍非整车批准 |
| 同状态但 `FAULT_CLOCK_UNRESOLVED` | 响应已检查，但还不是FTTI数值 |
| `AVOIDABILITY_ALREADY_LOST` | 模型初态已无停车避碰余量，不能显示成可用的“0秒安全预算” |
| `NO_IMMEDIATE_SAFE_BRAKING_WINDOW` | 当前即刻响应仍不安全，不能推断所有其他策略都失败 |
| `NO_UNMITIGATED_CONTACT_BOUNDARY` | 缺可建立的延迟边界；可另外存在即时响应无接触证据 |
| 控制/响应模型未决 | 保留具体缺口，不能与已算负面合并成一个Pending |

旧内部结果有时保留零值或故障时钟偏移；报告按负面状态处理，不能仅因字段非空就拿它参与正时间最小值。

### 4.4 按一个实际成员逐层定位

收到“某行FTTI不对”的反馈，先取明细里的场景ID、所属失效和分支ID，再到 `ftti_timing_inputs` 找同一成员。先看它有无正窗口、是否已算负面、是否只有接触证据；不要从主表显示的四舍五入结果反推原始物理参数。继续沿 `timing_trace.hypothesis` 找被实际检查的提案，再核对原计划中的成员控制评估。只看到一个相似description，不足以认定是同一个方案。

随后核对时钟：模型初态是什么阶段、故障起点在哪里、偏移是否有依据。如果例子的模型起点比故障晚1.7 s，故障时钟边界应结合这个偏移解释；不能把故障时钟的值与未加偏移的模型接触时间直接比较后判断“边界超过碰撞”。反过来，偏移已知也不能补出尚未建模的检测和通信耗时。

最后核对来源权威，而非只看数值：这是项目直接事实、方法条件、分析假设还是DERIVED缓存？适用scope是否仍指当前失效和场景？有条件的响应净减速度不是车辆标定能力。来源或阶段不一致时，不能复制另一个成员的时间，也不能靠改字段名使已有值通过检查。若当前模型确实不支持返回轨迹、转向逃逸或位姿退出，应留下具体缺口，回到现有方案/方法职责解决。

同样需要区分“同值的直接输入”和“可以刷新的计算缓存”。当前轴向修复会把物理上纯纵向的接触标为纵向；SIDE只是接触表面，零速度目标的方向单位向量也不能凭空产生横向速度。但这不授权替换原项目事实。当前视图使用[scenario_with_current_physics()](../../src/hara_agent/services/analysis/scenario_equivalence.py#L87)，它复用来源/冲突感知的物理materializer；显示使用当前派生值，响应仍围绕原冻结轨迹和已验证重验生命周期。不能为追求各层字典字面一致而把历史快照一起改写。

## 5. 报告如何保留独立成员与关联

[HARAReportProjectionService.project()](../../src/hara_agent/services/reporting/projection.py#L790)把状态、方法、风险追溯和报告schema组织成view model。它区分显示分组、实际风险成员和碰撞分支。最新固定总体有2389个风险成员、2071个主表审核组；主表行数变少不是删除风险，关联必须在明细和 `grouping_map` 中闭合。

现行主表分组键明确包含成员的 `display_tuple`；生成审核组还包含审核状态、机制、对象、驾驶员及交互条件等，见[projection分组构造](../../src/hara_agent/services/reporting/projection.py#L936)。因此每一主表行是一组单值S/E/C/ASIL或明确待评定，不把混合tuple作为正常多值主表行。最高已知结果选择也必须取一条真实完整分支；未完整时保留实际partial tuple，不能跨分支把最大的S、E、C拼成一个从未发生过的tuple。

[最高已知tuple及组内摘要](../../src/hara_agent/services/reporting/projection.py#L552)还保留分布、共同维度和未决说明，用于不完整/历史状态的诊断与明细解释。这些共同S或E不构成跨tuple合组授权。主表等级栏只显示合法等级或待评定；理由、成员分布和缺口进入对应说明栏。明细中的 `scenario_id`、主表HARA-ID、分支ID、SGID必须可互相追溯，跨表连接不能依赖数组排序。

Safe State摘要由[_safe_state_display()](../../src/hara_agent/services/reporting/projection.py#L656)及mapper结合候选、缺口和同成员时间证据形成。[safe_state_outcome()](../../src/hara_agent/services/reporting/engineering_text_mapper.py#L1476)会比对同一提案的描述、控制、条件、transition和身份；只有匹配的已验证证据才能显示“在已检查范围内避免接触”。未验证只显示目标状态及待验证范围；需建立通道的方案还须保留设计要求限定。

[safety_goal_timing()](../../src/hara_agent/services/reporting/engineering_text_mapper.py#L1586)按响应族、依赖控制、驾驶员、时钟、判据及独立路径要求分组。各组分别取已计算样本最小值，有限预览显示主要路径，同时保留其他路径数和无有效边界成员；完整路径在 `safety_goal_timing_evidence`。不能把某个SG的一个最小值当作所有成员、所有ODD的连续最小值。

## 6. 中文、模板和重导分别改在哪里

中文工程表达在[EngineeringReportTextMapper](../../src/hara_agent/services/reporting/engineering_text_mapper.py)和[报告语言配置](../../report_assets/hara_report_v1.yaml)。SG显示取实际关联成员的对象与机制；原始表达继续留在目录及SG单元格批注中。038实例通过保留逗号分句中的明确“驾驶员”主体，修正“防止无法感知……”残句；未知主体、否定和不确定表达不得被补写成新的危险判断，见[共享主体回归](../../tests/unit/test_report_shared_subject.py)。

外文控制条件使用现有 `analysis_note()`给出中文审核提示，并保留原始控制方案及typed控制依赖；不伪译、也不删除原始条件。slash检测仅在明确的控制条件显示场景开启，不能把camera/USSC/lidar/radar等传感器列表当成外文叙事后吞掉可见性限定。原始审计引用、字段ID、产品和接口缩写与面向读者的自由英语要分开判断。

[ReportSchema](../../src/hara_agent/services/reporting/report_schema.py#L40)定义canonical字段和展示配置，[renderer.render()](../../src/hara_agent/services/reporting/canonical_renderer.py#L210)加载模板、应用样式、写入view、原子保存后重开验证。静态字段只按canonical身份分层合并；动态S/E/C、ASIL、SG和说明字段不能合并。输出约定的 `04_HARA` 等sheet名不用于识别输入方法角色。方法来源hash、报告schema hash、样式模板hash各有职责，不能互相替代。

仅改文字时使用[OfflineReportRebuilder.rebuild()](../../src/hara_agent/services/reporting/offline_rebuild.py#L25)，从已提交状态及review artifacts重建报告，不触发Provider或重新批准评分。CLI入口是[rebuild-report](../../src/hara_agent/cli.py#L324)。报告语言YAML改变会更新schema hash；它是应记录的报告身份变化，不是评分变化。

本地已经取得最新交付证据时，可用下面的最小例子重建到新路径。方法路径取自该项目proof，不能为了方便改成另一个默认baseline；此例不调用评分或语义生成。

```python
import json
from pathlib import Path
from hara_agent.services.reporting import OfflineReportRebuilder

release = Path("output/closure_continuation_20261123/release_reports")
proof = json.loads((release / "AVP_release_proof.json").read_text())
OfflineReportRebuilder().rebuild(
    checkpoint_path=Path(proof["checkpoint"]),
    method_baseline_path=Path(proof["method_baseline"]),
    report_style_template_path=Path("references/HARA_Template_AI_20260327.xlsx"),
    review_root=release / "review",
    output_path=Path("output/handover_check/AVP.xlsx"),
    audit_output_dir=Path("output/handover_check/AVP_audit"),
)
```

## 7. 常见修改的最小定位与验证

| 现象 | 先检查 | 不应采用的捷径 |
|---|---|---|
| SG错对象、缺主体 | 关联成员、`_grounded_safety_intent()`及语言配置 | 改原SGID、按单个样例全文匹配 |
| 负面响应变“缺输入” | 撤回记录、输入快照、当前控制guard、DERIVED重验 | 忽略全部snapshot字段或复活候选 |
| FTTI突然变0/接触时间 | 时钟、判据、响应启动窗口、完整时间trace | 把接触后停止耗时相加、默认能力 |
| 组内评级看似矛盾 | 实际branch tuple、主表分组和明细ID | 各维度分别取最大再拼接 |
| 中文或格式问题 | mapper、ReportSchema、renderer及报告审计 | 在Excel里手工改评分或用另一生成器 |

最低回归应包含一个合法正例、一个不相容控制/来源反例、一个负面或无边界例，以及原始对象/审批不变检查。重导后核 `report_projection_audit.json`、SG完整时间证据、内容和样式审计，再看实际工作簿。语义、方法、prompt或cache身份变化属于行为修复；不能以“只是整理”跳过正常评分、因果与恢复边界验证。

### 建议的单成员修复记录

接手者应在提交说明中记录：原始场景/分支/SG身份，用户看到的错误单元格，真正失效的契约，输入和来源前后差异，沿用哪个现有职责，以及哪些字段被允许改变。例如，只修SG共享主体，允许变化是显示正文和相应显示依据；评级、FTTI数值、原始SG、成员身份和审批应逐值不变。修了轴向计算缓存，则必须明确布尔分类和TTC的预期变化，同时证明对象、时域、接触时刻、速度与所选响应没有被顺带替换。

报告回归不应只比较总行数。应按稳定身份比较实际成员及分支tuple、SG关联和状态；比较时间结果及完整路径范围；检查原始来源仍可追溯；再检查主表和明细的显示变化是否仅落在已声明字段。浮点数显示精度、报告schema hash等非业务差异可以单独记录，但白名单必须有具体原因，不能泛化成忽略所有hash、文字或provenance。

对正式运行或已有检查点的修复，先保留输入和原始响应，使用新输出路径生成可审核结果。若涉及重新调用Provider，要先核对该任务当前有效的授权和账本；仅重导则使用原生离线入口。不要提交一份手改Excel而留下旧状态，也不要只修一个ID的输出，让同一职责下的其他成员继续丢证据。

## 8. 输出工作簿逐页阅读

当前工程报告 schema 声明以下九张业务/审计页。名称是**输出约定**，不允许反过来作为固定方法角色识别条件；模板保留的其他参考页也不能自动变成数值权威。来源为[报告 schema](../../report_assets/hara_report_v1.yaml)及现有 projection/renderer。

| 页 | 读什么 | 重点检查 |
|---|---|---|
| `00_Summary` | 运行范围、数量、计算和审核概况 | 已有条件性数值与无数值分开；样本范围不冒充全量 |
| `01_Item Definition` | 项目边界、模式、ODD 与输入事实 | 训练/执行、不同功能允许范围不能串用 |
| `02_Functions` | 来源功能、输出和前提 | 覆盖全部已提取功能，而非只列最终有风险的功能 |
| `03_Malfunctions` | 功能偏差、失效与危险行为 | 一个输出的幅值/速率偏差不在展示时擅自拆成未分析的新失效 |
| `04_HARA` | 工程主表：工况、危险事件、伤害、完整评级、依据、SG/状态/FTTI | 每行一组 S/E/C/ASIL；说明来源和假设；动态列不合并 |
| `04A_Scenario Detail` | 组内成员、具体物理条件、参数、分支和结果 | 主表归组后原身份仍完整可查；不能只保留代表成员 |
| `05_Method Basis` | 活动方法及计算依据 | 方法来源、规则引用和适用范围可核查，不混入非执行参考 |
| `06_Safety Goal` | 安全意图、关联成员、最高已知等级、安全状态与时间信息 | SG必须关联主表；未决成员不被已知最高值覆盖 |
| `99_Audit` | 来源、哈希、状态、追踪与映射 | 与运行证据一致；审计保留内部标识，主表避免原始JSON |

主表 canonical 字段包括 `hara_id`、`malfunction_id`、`scenario_id`、`hazardous_event_id`、功能、引导词、失效、危险、运行场景、细化工况、危险事件、潜在伤害、四项评级及各自依据、FTTI及依据、SG关联、安全状态、审核状态、澄清项与备注。字段映射由编译契约/报告schema负责；改一列应先改相应投影或配置，不在renderer硬编码一个新的固定列坐标。

阅读顺序建议：先看Summary确认范围，再从主表选一行到Scenario Detail看完整成员，随后查Method Basis和Audit验证评级依据，最后回到SG页确认意图和成员范围。不能仅看SG的Max.ASIL就判断全部危险事件已经闭环。
