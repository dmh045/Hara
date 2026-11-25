# HARA 项目文档索引

当前阅读入口为[开发与技术交接手册](handover/README.md)及
[可视化工作流详解](handover/HARA_Workflow_Engineering_Detail_20261124.html)。
手册按源码职责和五份输入的最新修订结果组织；完整阶段日志在下方单独索引。

## 当前开发、运行与审核说明

| 内容 | 文档 |
|---|---|
| 端到端架构与职责 | [01 架构与端到端流程](handover/01_架构与端到端流程.md) |
| 五份固定输入与复现 | [输入路径、校验和与测试资产说明](../input/README.md) |
| 输入、来源及方法治理 | [02 输入抽取、数据契约与方法治理](handover/02_输入抽取数据契约与方法治理.md) |
| 场景、物理、复用与评分 | [03 场景、物理与风险计算](handover/03_场景物理与风险计算.md) |
| 安全目标、响应和报告 | [04 SG、安全状态、FTTI与报告](handover/04_SG安全状态FTTI与报告.md) |
| 安装、配置、运行与恢复 | [05 部署、配置、运行与恢复](handover/05_部署配置运行与恢复.md) |
| 开发、测试与维护 | [06 开发、测试与维护](handover/06_开发测试与维护.md) |
| 五份输入结果与实际缺口 | [07 验证结果与已知限制](handover/07_验证结果与已知限制.md) |
| 输入、报告和运行证据移交 | [08 资产移交与接手清单](handover/08_资产移交与接手清单.md) |
| 文档本身的核查证据 | [09 文档验证记录](handover/09_本次文档验证记录.md) |
| 离线重评分专题 | [offline-risk-rescoring](offline-risk-rescoring.md) |
| 模板编译专题 | [Full Template Compiler](architecture/FULL_TEMPLATE_COMPILER.md) |
| FUSA Exposure规则对照 | [fusa_v1_exposure_parity](fusa_v1_exposure_parity.md) |

13步正文、代码摘录和图示的维护入口为
[workflow_detail](handover/workflow_detail/README.md)。规范以根目录
[AGENTS.md](../AGENTS.md)、[SKILL.md](../SKILL.md)、活动输入和编译契约为准。

## 历史开发与验证记录

这些记录保留当时的问题、决定、反例和运行证据。文中的“当前”、请求预算、数量及完成状态，
均对应其记录时点；查现行状态使用上面的交接手册。原文件位置保留，已有源码、测试和证据链接继续有效。

| 阶段或专题 | 原始记录 |
|---|---|
| P6场景与覆盖 | [实现](P6_Implementation_Report.md)、[复核修正](P6_Review_Correction_Report.md)、[定向验证](P6_Targeted_Validation_20261012.md) |
| P7 / P7R | [P7](P7_Closure_20261014.md)、[P7R](P7R_Closure_20261016.md) |
| 自主物理及输入相容性 | [实现关闭清单](Autonomous_Physics_Closure_20261018.md)、[定向收口](Autonomous_Physics_Followup_20261018.md) |
| 复用、报告及后续跨输入修复 | [统一开发与验证日志](Reuse_Reporting_Closure_20261018.md)；包含截至2026-11-23的后续记录 |
| 阶段评审 | [2026-10-18—19成果汇报](HARA_阶段成果汇报_20261018-19.md)、[2026-11-11输入与复用评审](HARA_输入文档完整性与跨项目复用性评审_20261111.md) |
| 模板治理迁移 | [切换记录](architecture/TEMPLATE_DRIVEN_GOVERNANCE_MIGRATION.md)、[切换后审查](architecture/HARA_V13_POST_CLEANUP_ARCHITECTURE_VERIFICATION.md) |
| YAML基线建立与对账 | [运行契约建立记录](architecture/HARA_V13_YAML_BASELINE_RUNTIME.md)、[FUSA与模板对账](architecture/HARA_V13_FUSA_YAML_TEMPLATE_RECONCILIATION.md) |

## 2026-11-25 冗余清理核查

基线为`4c02677`。本轮是维护性清理，活动方法、提示契约和计算行为保持既有版本。
同一清单记录实际调用、保留实现、删除依据与验证。

| 项目 | 清理或保留决定 | 调用关系与验证依据 |
|---|---|---|
| C01：根README中的多轮历史摘要与旧Run ID命令 | 当前入口收敛到README与交接手册，历史细节从上方阶段记录查阅 | 历史段落重复已有阶段报告；未发现指向被移除README小节的仓库内锚点引用 |
| C02：旧参考指南与API占位页 | 移除`references/USAGE_GUIDE.md`与`references/api_reference.md`；重写目录说明 | 旧指南引用的引擎、解析器、报告器已经不存在；占位页没有项目API内容。检查CLI、配置、技能、导入、测试和文档链接后，未发现当前调用者 |
| C03：重复的原子审计文件写入 | CLI、场景合成、因果重验共用[`atomic_files`](../src/hara_agent/infrastructure/atomic_files.py)的两个函数，删除五份重复函数体 | 原内部函数名保留为导入别名，现有调用点全部使用共用实现；UTF-8、JSON缩进、精度、结尾换行、临时文件位置、原子替换与异常保持一致 |
| C04：其他相似持久化代码 | 保留检查点、ReviewArtifactWriter、契约写入和父HE恢复各自实现 | 检查点与审核记录有锁、fsync、容错或不同序列化要求；契约临时文件格式和父HE清理语义也不同。它们与C03不是同一行为，未机械合并 |
| C05：历史架构及阶段文档 | 保留独立证据；对容易被误读为现行指导的四份架构记录增加历史范围说明 | 独有设计决定、反例和提交依据继续可查；现行导航统一到交接手册 |
| C06：旧入口与兼容模块 | 仍有CLI、离线工具或旧状态恢复用途的模块保留；P001测试已迁入现有生产代码路径，旧入口删除 | 2026-11-25后续迁移完成；逐项断言映射、具体行为修正及验证范围见下方C06记录 |

共用文件写入的职责只覆盖原来相同的JSON/文本审计落盘。已有审核记录器绑定审核目录，
还有不同的序列化和持久化要求；让CLI或离线工具依赖它会混合职责。因此保留一个窄的基础设施函数模块，
不新增Agent、业务服务、调度或评分路径。

验证采用清理前后同一输入、方法和冻结响应：既有209项相关测试前后复跑；
代表风险状态、完整trace、方法契约及报告投影逐字段相等。另验证原子文件的并发读取与写入、
替换失败时旧文件保留和临时文件清理，现有预算、缓存、恢复及报告测试继续执行。
本轮完整pytest为**2181 passed、1 skipped，128.63秒**；doctor退出0，方法与草稿检查通过，
正式发布仍需具体运行的工程审核。compileall与diff检查通过。实际结果保存在
[`output/safe_cleanup_20261125/`](../output/safe_cleanup_20261125/)；该目录为本机验证证据，不随Git提交。
本轮Provider请求为0。

源码清理提交为`68370d4`；文档整理另行提交，两者可分别回滚。输入、方法资产、冻结夹具、响应、账本、历史报告、
个人配置及原有未提交文档均不作为清理对象。


### C06 后续：P001旧测试迁移与入口关闭

基线为`f8b05b2`。原根目录脚本依赖的`WordProcessor`、`FunctionChecker`和
`MalfunctionEngine`已被现行生产链替代；本轮将有效断言接入默认pytest范围，
再删除原入口。对应测试位于：

- [核心抽取契约](../tests/unit/test_core_item_artifact_contract.py)：固定AVP来源、九项功能及语义字段传递。
- [功能校验](../tests/unit/test_function_validator.py)：原Output复制描述反例及合法相近表达。
- [AVP生产阶段衔接](../tests/unit/test_avp_function_pipeline.py)：现行抽取Agent、引导词节点和失效节点，串行与并发离线验证。

| 旧断言意图 | 现行覆盖与取舍 |
|---|---|
| 固定AVP包含九项整车功能，名称简短、Output非空且有独立含义 | 沿真实DOCX的明确功能表进行抽取及来源校验，逐项核对功能成员和顺序；保留简短标签与输出断言 |
| 原11项ODD、至少10条状态跳转和至少1条回退依据未丢失 | 用现行DocumentReader核对11类环境原文、至少10条可定位跳转及故障回退原文；这些来源不再被当成额外功能 |
| 前提、触发、ODD、预期效果与回退字段正确传递 | 按现有类型使用preconditions、triggers、odd_constraints、consequences及单值fallback_behavior；明确给出的条件逐值保留，连同来源经过状态序列化恢复 |
| 所有功能都必须填写前提和触发 | 改为检查有来源的条件保留，以及未给出的条件仍为空；旧引擎统一填充的要求不再作为生产输入规则 |
| Output复制完整描述必须被发现 | 保留原中文反例。迁移时先复现现行缺口，随后在FunctionValidator补充去首尾空白后完全相同的检查；不加入模糊相似度或新模型请求。直接返回和候选缓存解析都测试拒绝；合法术语重叠及空描述仍可用 |
| source_table/source_row是可定位的表格来源 | 迁为SourceRef的精确location与excerpt，核对位置确实存在于固定DOCX且摘录属于对应块 |
| 126个引导词候选和70个失效，控制展开 | 这两个数属于退役引擎的固定规则。当前测试从活动MethodContract生成完整功能×引导词身份集合，只让DOWNSTREAM_CANDIDATE进入失效生成；检查数量与该集合相符、全局ID唯一、重复模型局部ID被规范化，以及原始功能语义未变 |

这次唯一生产行为修正是拒绝整段描述复制成Output。五份已保存项目的51条功能均通过新校验，
检查过程没有修改原记录。缓存读取继续经过现有校验入口；不改变Prompt版本、评分方法、审批、
Provider预算或其他缓存身份。新衔接测试使用明确标记的离线脚本响应，证明程序传递与边界，
不表示重新验证了真实模型抽取质量。

原三项测试的有效意图已在当前默认测试中有对应覆盖，旧入口已删除；没有增加skip、恢复旧引擎，
或为了沿用固定数量而改动当前方法。缺口修复提交为`de10f29`，测试入口迁移清理另行提交，可独立回滚。
本轮验证日志及五项目功能核对保存在[`output/p001_migration_20261125/`](../output/p001_migration_20261125/)。

C06本轮实际验证：相关37项测试通过；完整pytest为**2192 passed、1 skipped，127.44秒**；
doctor退出0，compileall和diff检查通过。唯一skip仍是显式启用的历史运行守卫。
