# 故障后物理场景自主构建：统一关闭清单

> 本文件保留首次实现的历史记录；其中输入继承、坡向与报告性质的缺陷已由后续独立复核指出。最新状态以 [F01—F06 定向关闭清单](Autonomous_Physics_Followup_20261018.md) 为准。

日期：2026-10-18。依据：用户当轮要求和《HARA_故障后物理场景自主构建_详细开发提示词_20261018.md》。

## 结论和验证身份

普通 `analyze` 已接入显式物理分析调度；确定性评分器没有网络调用。已有物理生产者不足时，新任务提出条件性假设，进入原物理、因果验证、S/E/C、分支 ASIL 和报告链。

**代码与离线生产链已验证；真实 Provider 未验证。** 本轮没有新的付费预算，真实请求为 0，没有追加或重置历史账本。本包模拟响应验证接线，不证明真实 LLM 工程推理质量。不能把样本数算成真实已验证危险事件数。

- 开始时本地 HEAD 和 fetch 后 origin/master 均为 `9ea5016ac7d20601ceaed62fe2a4602e677a93d2`，无受跟踪文件修改。本轮未提交或推送。
- 保留未跟踪的个人配置及工作簿；交付不包含 `.env`、凭据、个人配置。
- Item：`input/ItemDef.docx`，SHA256 `c8448397033587957e27130dff7414980d4581b10c4dced76a0f1b6e73d69981`。
- 方法：`method_assets/fusa_baseline_v1/manifest.yaml`，hash `1eb4b56a2529b25484f28ebf5c7dbbb8f328905e58fe2177da5318ab84e3b2ac`，compiler `yaml-baseline-compiler-v5`。未修改 YAML 业务规则。
- 报告模板：`references/HARA_Template_AI_20260327.xlsx`，由原 ReportContract 报告器生成；表格工具只读检查。
- 来源 checkpoint：`runtime/agent/full-p6-20261012-002.checkpoint.json`，hash `f4bb59e6ca83a86b06b7a181f62151462ac27548796325a08642635ba7d5f0ba`，脚本前后逐字节一致。
- 样本从真实来源已提取的项目、功能、失效开始，在 `MALFUNCTIONS` 阶段进入 `HARAApplication.run()`。重新生成场景，没有评分前手填模型、碰撞分支、速度、TTC 或批准。

## 生产路径和边界

```text
当前项目事实 / FM / MethodContract
→ 原候选与物理生产者
→ workflow.construct_physical_scenarios（已有有效分支则跳过）
→ PhysicalScenarioAgent → 原始响应保存 → PhysicalScenarioPlan 严格解析
→ PhysicalHypothesisInstantiationService → derive_scenario_physics
→ ScenarioFeasibilityAgent（具体新条件重新判断）
→ 原 StructuredRiskScoringService / 活动 ASIL 矩阵
→ checkpoint / RiskExecutionTrace / 原 ReportContract 报告器
```

契约允许模型、对象几何、点/范围/符号参数、依据、适用性、未覆盖条件及失效/剩余控制；禁止评分、批准、碰撞速度、TTC、collision_branches、任意附加字段。最多 3 个假设，一次本地 Schema 修复；共用既有预算与响应缓存。

实现模型：纵向恒加速度首次接触、坡道重力减残余阻力的溜车、有纵向运动的侧向首次接触。当前目标须静止；侧向目标须沿纵向路径延伸。加速度可正、零、负；可显式使用独立未失效限速器的分段加速/匀速模型。正常 7 km/h 只约束初始工况，不作为通用故障速度上限。

假设仍为 `LLM_INFERENCE / LLM_ANALYSIS_HYPOTHESIS / PENDING`。`VALIDATED + CONDITIONAL` 只表示程序允许计算，不是批准。原文事实和方法来源保持原权威，程序派生保留输入依赖；假设无需在原文逐字出现。

范围采样采用代表值、端点、活动方法边界和区间内点，单参数变化，不做多维笛卡尔积。每个输入候选最多 9 个实例；未取点记入 `deferred_points`。汇总所有已算点的最大 S、最高 ASIL 及其同一组 SEC。**采样一致性不等于连续区间或车辆真实上界已证明。**

新对象经现有 synthesis OBJECT shortlist 绑定，其他故障前 Atom 保持不变。假设对象导致的 E 同样标为试算；无绑定或维度适用性不足只影响对应 E。

## W1—W6

“离线生产链”指真实来源上下文和生产请求/解析/消费者，模型响应为显式模拟。

| 包 | 代码状态 / 入口 | 实际证据与收益 | 真实验证 / 缺口 |
|---|---|---|---|
| W1 | 已接入 application、workflow/physical_analysis、nodes/scenarios | 42 条制动跳过新请求；64 条新实例在因果阶段前生成 | 未做真实 Provider |
| W2 | physical_scenario_plan、physical_scenario_agent | 32 个模拟方案保存原始请求/响应并严格解析；无需原文现成加速度 | 真实方案合理性、失败分布未测 |
| W3 | physical_hypothesis_instantiation_service、scenario_physics | 驱动/溜车/侧向均计算首次接触；坡度单位、初始静止、限速、有效期回归通过 | 移动目标、任意曲率/轮胎动力学、多阶段反转未覆盖 |
| W4 | 原评分器及矩阵，新增物理假设汇总 | 点值 64 条条件 S、48 条条件 ASIL；范围自然触发 14 个多点组汇总 | 侧向 C 无活动方法；连续域不变性未证明 |
| W5 | 原响应/场景缓存、risk trace、projection/mapper | 方案复用 32；报告区分试算、来源、具体 Pending；无平行报告器 | 跨 MF 近似方案复用未启用；真实传输未测 |
| W6 | 新离线脚本、22 个定向测试、原总体审计脚本 | 874 passed、1 skipped；106 点值/218 范围样本；988 当前盘点/1270 固定回归 | 后续须有预算的四类真实小样本 |

## T01—T20

以下每项真实 Provider 状态均为“未测：无新预算”。

| ID | 代码 / 离线验证 | 实际证据及尚未覆盖范围 |
|---|---|---|
| T01 | 已实现、生产链通过 | MF-F02-002 原始模拟假设生成 14 个物理实例，无原文加速度也可条件计算；真实模型质量未测 |
| T02 | 已实现、通过 | 42 制动对照跳过；`test_existing_direct_path_never_requests_a_physical_plan` |
| T03 | 已实现、通过 | 模型/参数保持 PENDING；来源、假设、推导分开；旧原文否定/上下文反例保留 |
| T04 | 已实现、通过 | POINT/RANGE/SYMBOLIC；范围经生产链，符号条件单测给边界而不编造数值 |
| T05 | 已实现、通过 | 上界/全范围证明均为 false；报告限定所列假设 |
| T06 | 已实现、通过 | v0=0、a=2、d=2 → 10.182338 km/h；独立剩余限速器测试通过；示例非车辆实测 |
| T07 | 已实现、通过 | a=0、a=-0.5 保持/降低速度；没有强制超过 7 km/h |
| T08 | 已实现、通过 | 36 条溜车；% 坡度转换、残余阻力、静止初态后运动；残余阻力仍属假设 |
| T09 | 已实现、通过 | 方向显式给出并校验前/后接近几何；侧向不允许转向从静止自行生出运动 |
| T10 | 已实现、通过 | S/C 同轨迹；侧向时间不当纵向 TTC；有效期改变清除旧接触时间和速度 |
| T11 | 已实现、通过 | JSON 恢复、响应重解析、预算耗尽前保存、相关来源变化后缓存失效；真实断网未测 |
| T12 | 已实现、通过 | 逐分支活动矩阵核查；最高 S 不覆盖其他高 ASIL，决定 SEC 不混拼 |
| T13 | 部分满足 | 原始范围响应自动形成 14 个多点组并触发最大值/采样一致性；完整连续域/联合参数不变性未证明 |
| T14 | 已实现、通过 | 14 个侧向条件 S；C 局部 LATERAL_C_METHOD_NOT_AVAILABLE；E 独立计算 |
| T15 | 已实现、通过 | 模型不创造 Atom ID；OBJECT 用既有 shortlist；故障速度不重选故障前 Atom |
| T16 | 已实现、通过 | 驾驶员条件族共享 32 个方案；来源/方法/目录/提示词/模型身份改变 key；跨 MF 近似共享未启用 |
| T17 | 已实现、通过 | 精确合并 0、方案复用 32、共享因果请求节省 21、完整因果复用 0 分列；模拟 token 0 |
| T18 | 已实现、通过 | Excel 标离线样本、试算、事实/假设/派生及具体原因；不将样本称全项目 |
| T19 | 已实现、通过 | 完整回归；1270 固定总体；原中文来源、分支字段、有效期、伤害机制反例保留 |
| T20 | 已满足 | 全轮真实请求 0；独立 Run ID/输出；历史 checkpoint、缓存和报告保留 |

## 实测结果

根目录：`output/autonomous_physics_20261018/`。最终交付为 `sample/` 和 `sensitivity/`；较早 sample01 等是调试记录，保留但不作为最终证据。

### 点值样本：4 / 61 个失效

| 失效 | 候选 / 可算轨迹 | S | E | C | ASIL |
|---|---:|---|---|---|---|
| 制动 MF-F01-001 | 42 / 42 | 42 既有规则 | 42 计算 | 42 计算 | 42 计算 |
| 驱动 MF-F02-002 | 14 / 14 | 14 试算 | 12 试算、2 未决 | 14 试算 | 12 试算、2 未决 |
| 转向 MF-F03-002 | 14 / 14 | 14 试算 | 12 试算、2 未决 | 14 方法未决 | 14 未决 |
| 驻车保持 MF-F08-001 | 36 / 36 | 36 试算 | 34 计算、2 未决 | 36 试算 | 36 试算 |

制动虽按既有规则算出数值，因果仍是模拟，不能算真实新增事件。溜车两条 E 未决在 S0 条件下由原矩阵零评级规则证明条件 QM，没有填造 E。

- 模拟物理请求 32；复用 32；新实例 64；物理冲突/拒绝 0。
- 模拟因果请求 85，独立判断 106：42 制动合并为 21 个双条请求，其余 64 单条；完整因果结论复用 0。
- 模拟事实请求 22，明确返回未找到；总模拟请求 139。
- 模拟因果保留/排除/待定为 106/0/0，仅是接口测试，不能当准确率。
- Schema 修复/响应命中/方案命中为 0/0/0；恢复与修复由单测验证。
- 模拟输入/输出 token 均 0；真实调用、真实新验证事件均 0。
- 新物理层精确去重 0，不用数目改变冒称覆盖优化。

### 范围样本

相同来源四类，仅驱动响应变为加速度 `[0.2,4] m/s²`、代表值 2。共 218 条：42 制动、126 驱动、14 转向、36 溜车。32 个物理方案、197 个因果请求、44 个事实请求，共 273 个模拟请求。176 条条件 S、152 条条件 ASIL；14 个多点组。驱动每输入最多 9 点；未取点和方法边界保留在 `physical_analysis.sensitivity`。不能把采样点数当机制覆盖或真实最大能力证明。

### 两份总体

1. **当前生成盘点**：61 个已提取失效，3 个基础来源上下文，988 个当前候选；98 个已有物理路径、890 个需方案，对应 445 个非驾驶员条件组、潜在方案共享 445。所有组为 NOT_REQUESTED，新假设/调用为 0。见 `current_population/`。
2. **固定历史回归**：1270 身份，886 风险记录；E 880 计算、6 Pending；S/C/ASIL 各 886 Pending，与最近 P7R 修订口径一致。36 物理重验标志、42 新因果 Pending；评分前事实变更 0、绑定冲突 0。它没有调用新 Agent，不能用来证明新生产者成功或失败。见 `fixed_population/fixed_population_summary.json`。

## 验证及交付

- `.venv/bin/python -m pytest -q`：**874 passed、1 skipped**；跳过的是未显式启用的历史 R3 工件守卫（`HARA_RUN_HISTORICAL_GUARDS`），不是 live 测试通过。真实 Provider 验证另记“未执行：无新预算”。
- `python -m compileall -q src scripts`、`git diff --check`：通过。
- 活动方法/报告模板 doctor：`ready_for_draft=true`，`ready_for_release=false`；编译正常，未绕过既有发布状态。
- `qa/verify.py`：同轨迹接触/速度、假设 PENDING、侧向无 TTC、逐分支矩阵、决定 ASIL 的 SEC、报告范围、动态列不合并；结果见 `validation/output_checks.json`。
- 原模板 `ASIL_Table!U5=#N/A`、`U6=#VALUE!` 两处已有错误原样保留；新增报告页无公式错误。未修改方法模板掩盖这些单元格。
- `qa/inspect.mjs` 只读渲染摘要/HARA/详情 PNG 检查，无二次导出。

交付包括：

- `sample/HARA_Autonomous_Physics_Offline.xlsx`、`sensitivity/HARA_Autonomous_Physics_Offline.xlsx`。
- 两目录的原始模拟请求/响应、checkpoints、response caches、review 风险追踪。
- `source_context.json` 为原 checkpoint 的项目/功能/失效上下文摘录，保留原 hash，去除旧场景事实和历史评分；包内复现时增加 `--source-checkpoint output/autonomous_physics_20261018/source_context.json`。不必复制 363 MB 历史总体 checkpoint。
- `Autonomous_Physics_Code_Changes.patch`（含新增文件）及 `Autonomous_Physics_Implementation_20261018.zip`（完整变更源码、样本、来源/方法/模板、验证和清单）。
- `delivery_manifest.json`：文件 hash、基准 HEAD 和验证身份。

没有重建 P6 Atom 选择器、平行评分器/报告器，没有扩展 SG / Safe State / FTTI 方法；保留历史草稿与关联。

## 新全量准备结论与后续命令

**代码已具备有预算真实样本的条件，但尚缺真实非制动样本通过证据，本轮不放行新全量。** 不需要逐行补动态参数，也不要求侧向 C 或全部 Pending 先归零。

先以制动、驱动、溜车、转向各一类真实样本检查工程适用性、因果、Schema/传输和恢复。建议另行给 40 次请求作为第一段预算上限，包括重试和上下游调用；不保证 40 次能完成四类。当前完整离线点值样本需要 139 个模拟请求，不能承诺 20 次覆盖全部。

下列命令**本轮未执行**；获得有效预算后将 N 替换为批准上限，使用全新路径：

```bash
.venv/bin/python -m hara_agent analyze \
  --item input/ItemDef.docx \
  --method-baseline method_assets/fusa_baseline_v1/manifest.yaml \
  --report-template references/HARA_Template_AI_20260327.xlsx \
  --output output/physical_live_new/HARA_Physical_Sample.xlsx \
  --run-dir runtime/agent/physical_live_new --run-id physical-live-new \
  --operating-mode parking_vehicle_control --allow-draft --max-workers 1 \
  --sample-function-limit 4 \
  --sample-function-id F01 --sample-function-id F02 --sample-function-id F03 --sample-function-id F08 \
  --sample-malfunction-limit 4 \
  --sample-malfunction-id MF-F01-001 --sample-malfunction-id MF-F02-002 \
  --sample-malfunction-id MF-F03-002 --sample-malfunction-id MF-F08-001 \
  --sample-parent-scenario-limit 1 --sample-scenario-pair-limit 128 \
  --provider-attempt-limit N
```

若提取版本改变身份，先按当前身份预览选样；不手填碰撞字段。恢复沿用同一 Run ID、预算和 `--resume`，不重置已用次数。

四类真实样本通过后可另建诊断性总体：去掉 ID 选择，Function/Malfunction/parent 上限按当前 `9/61/3`、pair 上限覆盖盘点规模，另给请求预算。仅第一轮物理方案就对应 445 次请求；一次 Schema 修复可达 890 次响应，传输重试、因果和风险事实另计。445 不是全链总成本；预算应在真实样本之后估计。

离线复现（输出目录须不存在，客户端无网络实现）：

```bash
.venv/bin/python scripts/validate_autonomous_physics_offline.py --output output/physics_replay_new
.venv/bin/python scripts/validate_autonomous_physics_offline.py --range-proposals --output output/physics_range_new
.venv/bin/python scripts/validate_autonomous_physics_offline.py --inventory-only --output output/physics_inventory_new
```
