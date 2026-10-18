# 自主物理分析定向收口：F01—F06 统一关闭清单

日期：2026-10-18。依据：用户《HARA_自主物理分析_定向收口与真实验证_详细操作提示词_20261018.md》及补充《Autonomous_Physics_实现包_独立复核_20261018.md》。两份文档均完整读取。

## 结论、起点与授权

**离线收口完成；真实 Provider 验证未执行。** 普通 `HARAApplication` 已验证点值、范围及符号提议到报告的生产链。本轮没有有效的新付费预算，真实请求为 0；没有使用历史 19/20 的剩余记录作为新额度。

- 开始 HEAD 与 SSH 核对远端 master 均为 `9ea5016ac7d20601ceaed62fe2a4602e677a93d2`。自主物理实现当时是本地未提交增量；先保存 `start_tree/` 和 SHA256 清单，再做局部修复，无 reset/clean。
- HTTPS fetch 一度连接失败，SSH `ls-remote` 成功核实远端。用户随后明确要求每轮更新完成后推送；本轮按该授权提交、推送项目代码，并在 AGENTS.md 记录长期约定。
- 历史 checkpoint、响应缓存、报告及个人未跟踪文件保留。`.env`、密钥、个人配置和运行大文件不提交。
- 沿用活动 YAML MethodContract、既有物理服务、S/E/C 执行器、ASIL 矩阵及 ReportContract 渲染器。没有修改数值表、E 聚合政策或引入平行评分器。
- 交付根目录：`output/autonomous_physics_followup_20261018/`。以 `final/point`、`final/range`、`final/symbolic` 为最终生产样本；其余目录是保留的调试记录。

### 来源和缺失附件

Item 为 `input/ItemDef.docx`，SHA256 `c8448397033587957e27130dff7414980d4581b10c4dced76a0f1b6e73d69981`。活动方法 hash 为 `1eb4b56a2529b25484f28ebf5c7dbbb8f328905e58fe2177da5318ab84e3b2ac`；报告模板 hash 为 `f3d2e4f7a16cbfda2d9e30c5f437ed94bb1b51affd635dc73215b8d307fb87ef`。

来源上下文摘录来自保留的 `full-p6-20261012-002` checkpoint，只使用项目、功能和失效定义；不导入历史碰撞输入、因果批准或风险结论。原始历史 checkpoint hash 为 `f4bb59e6ca83a86b06b7a181f62151462ac27548796325a08642635ba7d5f0ba`。

下载目录已找到补充独立复核，但未找到其中引用的 `independent_edge_cases.py`、结果 JSON、`uphill_sample_rows.json`、`grade_direction_index.csv`。因此：

- 保存并运行了按复核完整条件重建的 5 个反例，标记为“提示词反例重建”，没有冒称执行了作者原脚本。
- 修复后用 `replay_saved_prompt_cases.py` 重放**修复前已冻结的同一输入和原始提议**，不依赖后来修改的夹具。
- 原 12 条上坡样本从本地原 `sample` checkpoint 逐身份提取、复算，保存完整前后来源及风险记录。

## F01—F06

| ID | 根因与实际修改 | 代码/离线生产证据与收益 | 真实验证与具体剩余项 |
|---|---|---|---|
| F01 | 整个父 facts 混入新模型。现有实例化服务新增当前模型输入视图：显式 POINT/RANGE/SYMBOLIC 优先；省略参数按对象、阶段、角色、单位、坐标、模型选择继承。清除旧派生值与 `travel_to_impact_m` 别名，保留父原件和排除理由。 | 符号有/无旧加速度都不再产生唯一速度/TTC；同阶段同对象的加速度仍可继承。符号样本经普通应用到 Excel，显示参数关系和活动方法边界。 | 离线通过；真实模型提出符号假设的质量未测。未扩展任意坐标变换或车辆动力学。 |
| F02 | 仅保护项目来源，未保护方法已选状态/目标。现保护全部适用来源；明确 `condition_changes` 才建立新初态/新目标，并通过现有 synthesis 重绑相关 Atom、可见条件、分析身份和描述。无关 Atom 保留，既有约束校验继续执行。 | 静止 Atom + 初速 7 和同一方法移动目标 10→0 均报告具体冲突。点值生产样本有 18 条合法新初态，绑定/文字均一致；64 条新假设全部重新因果验证。 | 移动目标仍是模型范围缺口；允许另建明确的静止对象，但不声称同一移动对象已解决。 |
| F03 | 正坡度幅值被默认解释为前进。新增显式坡向参考、车头相对下坡朝向、车体正轴、重力及运动方向；同一物理服务计算 signed gravity 与沿运动的净加速度。 | 上/下坡、道路参考同向/反向、目标在反方向不接触、保持力足够不启动均通过。原 12 条：6 UPHILL + 6 DOWNHILL 假设，6 方向变化、2 SEC/ASIL 组合变化；不翻倍。 | 这两种朝向来自明确标记的离线提议，均非来源确认。真实 Agent 对未知朝向的提议合理性未测。 |
| F04 | 复用口径与范围成本需透明。保持原有限取点和所有分支比较；分别统计精确去重、共同证据、方案复用、完整因果复用。补一轮 workflow 显式语义纠正，并用同一预算与响应缓存落盘。 | 点值 32 方案供 64 新实例；方案复用 32，完整因果复用 0。范围本次 162 条、206 次模拟请求；并未把范围采样一致当连续域证明。五组入口仅 10 个身份，避免把 106/162 或 445 组搬成真实调用。 | 未证明范围点可共享完整因果结论，所以如实计费估计；本轮没有宣称新的付费成本优化率。 |
| F05 | 摘要 finalized/pending 混淆无值与试算；SG 缺少页内限定。改为三类数值状态，SG 页与每条结果标草稿、最高 ASIL 仅覆盖已计算成员。参数、坡向及合法继承投影到详情。 | 点值 S=42/64/0，E=42/62/2，C=42/50/14，ASIL=42/50/14（既有/条件/无值）。本次点值 3 条 SG 是离线/条件性草稿，不与历史 5 条相加。 | 侧向 C 的活动方法缺口保留；未扩展 SG、Safe State 或 FTTI 业务方法。 |
| F06 | 缺少可靠的定向验证、生产小样本入口和完整交付依赖。按反例→正例→点值→范围→符号生产回放执行，补薄 CLI，复用普通应用/预算器。 | 5 个冻结反例前后输出；20 个新增定向单测；42 个自主物理相关测试；完整回归、doctor、编译及差异检查。包内含未改动的 docs/architecture、stage skills、报告/方法资产与来源。 | 原作者反例脚本附件未收到；真实 Provider 未执行，不能宣称真实小样本验收或新全量已获准。 |

## 生产调用与输入一致性

```text
来源上下文 → 原场景生成 → 已有物理路径判断
→ workflow 显式物理方案任务 → 原始 JSON 响应保存
→ 当前模型输入视图 / 条件重绑 → 原物理服务推导
→ 具体子场景因果验证 → 原 S/E/C 与分支 ASIL → 原报告器
```

生产集成没有在评分前插入碰撞分支、碰撞速度、TTC 或批准。模拟客户端只返回提议、因果判断及“未找到”事实响应。明确假设仍是 `LLM_INFERENCE / PENDING / CONDITIONAL`。源数据与方法值不因此被重新标为已批准假设。

`CORRECTION_REQUESTED` 在请求前 checkpoint；首次原始响应、具体冲突、有限纠正、最终实例均保存。每个计划最多一次语义纠正；每次 schema 有一次既有修复机会，传输重试及所有阶段共享 `ProviderAttemptBudget`。预算耗尽原样抛出并保存，恢复不重复已保存的提议请求。新契约、提示词、相关来源/对象/阶段身份变化使旧缓存失效。

## T01—T18 验收定位

| 项 | 离线证据 | 状态 |
|---|---|---|
| T01、T02 | `test_explicit_symbol_cannot_use_parent_number_or_alias`、`test_omitted_parameter_inherits_only_applicable_role_and_object` | 通过 |
| T03、T05 | 静止/移动目标反例；明确另建目标正例 | 通过 |
| T04、T10 | 普通应用的 18 条初态变更；现有 materializer 重绑；父事实保留、子状态 PENDING、新 ID/指纹 | 通过 |
| T06 | 旧派生清理、metadata 对象身份与 grade 分支依赖失效测试；原 P7R 分支变更回归 | 通过 |
| T07—T09 | body/road 两类参考、未知朝向假设、反向目标不接触、保持力足够正例 | 通过 |
| T11、T12 | 独立轨迹公式核对、逐分支活动矩阵复算、原最高 ASIL 同 SEC 回归 | 通过 |
| T13 | 42 条制动跳过物理 Agent；既有规则输出继续计算 | 通过 |
| T14 | `final/point`、`final/range`、`final/symbolic` 普通应用及 Excel；符号无碰撞速度/TTC，保留关系 | 通过 |
| T15 | 原始请求计数、范围 deferred points、范围不变性/车辆上界均 false | 通过 |
| T16 | 摘要三分类逐项对账，SG 页本地 PNG 只读查看 | 通过 |
| T17 | 当前授权预算 0，历史账本未追加 | 未执行真实验证 |
| T18 | 首次 schema 响应保存、语义纠正预算中断、同 context 恢复与来源改变失效 | 离线通过；真实传输未测 |

## 样本、规模与真实调用

所有下列响应均标 `OFFLINE_SCRIPTED`。样本数不是新增真实已验证危险事件数。

| 运行 | 候选/风险记录 | 物理方案 | 因果请求 | 事实请求 | 模拟总数 | 真实请求 |
|---|---:|---:|---:|---:|---:|---:|
| 点值生产回放 | 106 | 32 | 85 | 22 | 139 | 0 |
| 范围生产回放 | 162 | 32 | 141 | 33 | 206 | 0 |
| 五组/双驾驶员入口 | 10 | 4 | 9 | 2 | 15 | 0 |
| 同入口，驱动改符号提议 | 10 | 4 | 9 | 2 | 15 | 0 |

点值：42 制动、14 驱动、14 侧向、36 溜车；64 条条件 S、50 条条件 ASIL。范围：70 驱动，其余相同；120 条条件 S、106 条条件 ASIL。范围相对点值新增 56 个候选和 56 次因果请求、11 次事实请求；没有强行复用不同点的结论。

之前 218 条范围样本把驱动初速统一改成 0；本次保留当前初态 7，活动方法边界在同一加速度区间中的分布发生变化，产生 162 条。**这一数量变化不是新去重收益。** 新物理层精确合并仍为 0、方案复用 32、共享证据的双驾驶员制动合批节省 21 次请求、完整因果结论复用 0，四者分开记录。

原 12 条上坡前后逐条见 `evidence/uphill_12_comparison.csv`，完整输入、假设、物理与风险见同名 JSON。保持原行数不是目标；部分速度幅值未变，是相同坡度/距离/阻力下反向目标条件的计算结果，不代表旧方向解释正确。

## 可执行的有限真实验证入口

新增薄脚本 `scripts/validate_physical_followup_sample.py`：先用真实来源和生产生成器选定五个条件族，两个驾驶员各一身份；通过新增的有界生成身份选择项，普通 `HARAApplication` 在物理任务前筛选这些身份。脚本不提供任何故障物理输入或评分捷径。

仅预览，不构造 Provider、不消耗请求：

```bash
.venv/bin/python scripts/validate_physical_followup_sample.py \
  --source-context output/autonomous_physics_20261018/source_context.json \
  --output output/physical_live_preview_NEW --run-id physical-live-NEW
```

未来收到新的明确预算后，用**新的独立输出**运行以下命令；`N` 须替换为该授权的正整数，本轮未执行：

```bash
.venv/bin/python scripts/validate_physical_followup_sample.py \
  --source-context output/autonomous_physics_20261018/source_context.json \
  --output output/physical_live_NEW --run-id physical-live-NEW \
  --live --max-provider-attempts N
```

恢复时追加 `--resume`，保留原输出、Run ID、样本及同一预算账本；不得换目录重置额度。范围/符号是有意义的真实返回时继续记录，但首轮提示优先每族一个代表条件。15 次模拟请求仅作调度参考，不承诺真实模型同样次数；语义/schema/传输重试也占授权预算。

离线重放命令见包内脚本：`validate_autonomous_physics_offline.py --source-checkpoint ... --output NEW`；范围追加 `--range-proposals`。五组入口加 `--offline`，符号检查再加 `--symbolic-drive`。所有输出目录必须新建，恢复仅走明确 `--resume`。

## 验证与交付

- 最终完整 pytest：**894 passed、1 skipped（76.25 s）**，见 `validation/pytest_final.log`；历史 R3 工件守卫的 opt-in 跳过单列，不作为真实 Provider 测试通过。
- `python -m compileall -q src scripts`、`git diff --check` 通过。
- 活动方法与报告模板 doctor：`ready_for_draft=true`、`ready_for_release=false`；原编译警告和正式发布条件保留。
- `qa/verify.py` 独立核对同轨迹速度/时间、分支 ASIL、条件状态、初态绑定、摘要分类、SG 限定及动态列无合并。结果在 `validation/output_checks.json`。
- 原模板 `ASIL_Table!U5=#N/A`、`U6=#VALUE!` 保留；新报告页没有新增公式错误。表格工具只读渲染，没有二次导出改写生产工作簿。
- 完整本轮差异与针对实施起点的增量差异分别交付；包内含完整必要源码、测试、docs、stage skills、方法和报告资产、来源摘录及最终样本的原始响应/checkpoint/risk trace。个人配置和凭据不打包。

### 新全量准备结论

1. **技术接线与离线条件已具备**：非制动方案能沿同一运行时生成有效条件性结果，已知输入混用与坡向问题已修复。
2. **进入真实诊断全量的验证条件尚未齐全**：仍须先用上述小样本证明真实提议、有限纠正及传输恢复有效，再决定真实诊断全量；不要求先清零侧向 C 或符号 Pending。
3. **预算不具备**：本轮 0 新授权，不能启动真实样本或全量。
4. **不能称为正式 HARA 交付**：当前 Excel 是离线/条件性样本；移动目标模型、侧向 C 方法、连续范围证明和真实车辆上界仍未覆盖。
