# HARA V13 FuSa YAML 与模板对账记录

> 历史设计与核验记录：本文中的“当前”、缺口和运行结论仅对应下述日期或提交。现行实现与使用方式见[开发交接手册](../handover/README.md)；历史记录索引见[文档目录](../README.md#历史开发与验证记录)。

日期：2026-09-01  
活动模板：`references/HARA_Template_AI_20260327.xlsx`  
模板哈希：`f3d2e4f7a16cbfda2d9e30c5f437ed94bb1b51affd635dc73215b8d307fb87ef`

本文是来源审计和修订清单，不是运行时规则。只有写入活动模板、重新编译为
`MethodContract` 并通过工程确认的规则才可执行。

> 2026-09-03 更新：用户已明确批准新增受治理的 YAML Baseline 方法源。
> 下文“不得让运行时读取 YAML 覆盖 MethodContract”的要求仍然成立；实现方式是
> `manifest + YamlBaselineCompiler -> MethodContract`，运行时 evaluator 不直接读取 YAML。
> 显式 `--template` 的既有路线保持不变。

## 结论

`fusa_agent.zip` 不是单一版本、单一权威等级的法规库。其 YAML 同时包含：

- 声称来自 ISO 26262 或 VDA 702 的标准内容；
- IAV 项目表反推值和内部方法；
- 工程经验、行业默认值及项目翻译层；
- `DRAFT`、few-shot、fallback 和测试期望值。

因此不得把 ZIP 中的 YAML 整体导入活动模板，也不得让运行时代码直接读取它覆盖
`MethodContract`。可采纳内容必须逐项保留来源、版本、适用域和审批状态。

## 可确定修正的模板错误

以下修订由活动模板自身的 `AI-process` 与 `05_HARA` 表头对账得出，不依赖外部规则，
可以直接修改：

1. `AI-process` 第 10 行：细化场景输出列由 `I` 改为 `H`；该行所有输出引用同步修改。
2. `AI-process` 第 11 行：细化场景输入列由 `I` 改为 `H`，Hazardous Event 输出列由
   `H` 改为 `I`；该行所有输入、输出引用同步修改。
3. `AI-process` 第 17 行：`C0,11,C2或C3` 改为 `C0,C1,C2或C3`。
4. `AI-process` 第 19 行：Safety Goal 输出列由 `T` 改为 `S`；该行所有输出引用同步修改。
5. `AI-process` 第 20 行：Safety Goal 输入列由 `T` 改为 `S`，Safe State 输出列由
   `U` 改为 `T`；该行所有输入、输出引用同步修改。

## 必须先做工程选择的规则

### Severity

活动模板 `Severity!I3:I26` 使用变量 `v`，但未说明它是自车速度、相对速度、碰撞速度
还是 delta-V；同时存在 `S0 (S1)`、`S2 (S3)` 等无选择条件的结果，并缺少侧碰
64–80 区间。

附件也不能自动消除该歧义：

- `iso26262_spec.yaml` 声称采用 delta-V，并给出 4/20/40 km/h 等阈值；
- `severity_table.yaml` 使用按碰撞对象区分的相对速度区间；
- `domain_rules/avp_low_speed.yaml` 又规定行人最低 S2，并明确说明来自 IAV 项目反推与
  工程共识。

工程确认时必须选择一个方法版本，并在模板中明确：速度变量语义、单位、边界开闭、
碰撞构型、道路使用者类型、适用域和冲突优先级。在此之前保留 fail-closed。

### Exposure

活动模板同时提供 T（持续时间）、F（频率）和 VDA 场景映射，但没有声明选择优先级或
多维组合关系。附件 `domain_rules/avp_low_speed.yaml` 的 `combine_method: min` 是项目规则，
而 `e_dimension_rules.yaml` 混有 VDA、ISO 和工程经验，不能作为自动覆盖依据。

工程确认时应在模板新增规范性规则区，至少声明：

- T、F、场景映射分别在什么条件下使用；
- 多维场景是独立、依赖、条件依赖还是不可组合；
- 各关系的合并算法、优先级及边界；
- VDA 条目是规范性规则还是仅供参考。

### Controllability

附件内部存在不同 TTC 表：

- `controllability_rules.yaml` 的 IAV 表为 0–1.5、1.5–3、3–5 秒；
- `iso26262_spec.yaml` 为 C3 ≤3、C2 3–4、C1 4–5、C0 >5 秒；
- `c_profiles/iav_avp.yaml` 标记为后续 IAV 内部截图版本。

活动模板当前采用驾驶员可避免比例而不是 TTC。不得把三套表叠加。工程确认时需选择
主方法，并明确 TTC、驾驶员位置、直接控制能力、远程干预、其他道路使用者可避免性等
修正项如何进入最终 C。

### ASIL

活动模板同时使用 `NA` 与 `QM`。附件矩阵使用 `QM`，但不能证明活动模板中的 `NA`
必然等价于 `QM`。在确认语义前必须保留区分；确认后应在模板中写出唯一映射规则。

## 禁止自动导入的附件内容

- `fallback_scenario_dimensions`：Item Definition 缺少事实时兜底，违反项目事实不得由
  模板示例或默认值补齐的契约。
- 默认反应时间、加速度、减速度和 FTTI 参数：混有项目反推、行业默认和 `DRAFT`。
- `expected_asil_coverage.yaml`：属于测试期望，不是业务规则。
- few-shot、coupling/infeasible examples：只可作为非执行参考，除非模板明确标为规范性。
- AVP 项目特化规则：必须标明项目、ODD 和审批人，不能冒充 ISO 通用规则。

## 已同步修复的运行时问题

- 场景风险事实的证据 ID 现在按 malfunction/scenario 对隔离，禁止引用同批次其他场景
  独有的证据。
- 缺失、非法或越界的 Provider 结果进行一次有界修复；仍失败时记录为未解析，不再使
  数小时流程整体崩溃。
- 速度含义未解析时不再请求 LLM 猜测速度，避免把 20 km/h 上限加工成 22.5/30 km/h
  等伪事实。
- ROAD_USER_TYPE 只在已接地的车辆—道路使用者碰撞下请求，其他碰撞标记不适用。
- Unicode 与历史乱码形式的不等号均可标准化，避免文档符号编码差异导致项目速度事实
  提取失败。

## 2026-09-01 工程流程补充后的目标链路

以下内容按用户提供的工程流程作为模板修订目标，但仍须在活动模板中形成规范性表格后
才可执行：

1. Item Definition 抽取以全文为输入，输出相关项定义、功能、状态、ODD、性能限制及
   其精确来源；不得固定为某一章节。
2. 场景库采用“方法 atom 全集与项目 ODD 子集求交 → 多维笛卡尔积 → 规范性约束规则”
   的路径。稀有不等于不可能，例如高速道路出现行人不得作为物理不可能直接删除，
   应保留并在 Exposure 中反映低频。
3. HAZOP 保留完整 Function × Guideword Y/N 矩阵；N 和无可信危害组合在报告中保留
   N/A 及理由，但不进入场景与评分。
4. Severity 输入必须是可溯源的碰撞对象、构型和经批准的速度语义；具体表可替换，
   Python 不固定阈值。
5. Exposure 先对各独立维度形成 Z/F 候选，再按显式的依赖关系和有序组合规则计算；
   “无 E 值”必须区分 `NOT_APPLICABLE` 与 `MISSING/UNRESOLVED`，后者不得静默忽略。
6. Controllability 由项目事实选择 profile，Python 计算 TTC 并执行 profile 查表；输出
   必须含 profile_id、rule_id、matched band 和 inputs_used，LLM 只能润色理由。
7. ASIL 只查活动 MethodContract 矩阵；FTTI 位于 ASIL 之后，但在模板尚未提供预算、
   输入和公式前保持 `PENDING_METHOD_SEMANTICS`。
8. 仅 ASIL A/B/C/D 形成 Safety Goal；具体反向语义和 Safe State 仍通过模板方法及项目
   能力约束生成。

仍需工程澄清：原始描述中“驾驶员在/不在车内”的 C3 条件前后矛盾。代码已提供 profile
选择与 TTC 查表接口，但不会在澄清前硬编码“驾驶员不在车内必为 C3”。
