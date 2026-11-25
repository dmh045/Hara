# HARA V13

HARA V13 从 Item Definition 提取功能与运行条件，构建故障场景，进行物理和因果分析，
依据活动方法计算 S/E/C、ASIL，并生成关联 Safety Goal、Safe State、FTTI 的 Excel 报告。

## 文档与当前结果

- [开发与技术交接手册](docs/handover/README.md)：架构、输入与方法、完整分析链、部署恢复、测试维护及五份输入的最新修订结果。
- [可视化工作流详解](docs/handover/HARA_Workflow_Engineering_Detail_20261124.html)：13步生产流程、代码实例、来源与结果追溯；下载后用浏览器离线阅读。
- [文档索引](docs/README.md)：当前说明与历史记录的用途、清理核查及对应证据。

当前业务结果采用2026-11-23五份完整修订报告；日期、输入身份、方法快照和未决项见
[验证结果与已知限制](docs/handover/07_验证结果与已知限制.md)。原始输入和运行证据的移交范围见
[资产清单](docs/handover/08_资产移交与接手清单.md)。`output/`和`runtime/`不随Git提交，
五份固定输入现已统一纳入`input/`，路径、校验值及复现步骤见[输入与测试资产说明](input/README.md)。
完整历史运行状态与响应仍通过独立证据包移交。

## 安装与运行

已验证环境为 Python 3.12。以下为macOS/Linux示例；环境、配置与平台约束详见
[部署、配置、运行与恢复](docs/handover/05_部署配置运行与恢复.md)。

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -e ".[test]"

.venv/bin/python -m hara_agent doctor \
  --method-baseline method_assets/fusa_baseline_v1/manifest.yaml \
  --report-template references/HARA_Template_AI_20260327.xlsx

.venv/bin/python -m hara_agent analyze \
  --item input/ItemDef.docx \
  --method-baseline method_assets/fusa_baseline_v1/manifest.yaml \
  --report-template references/HARA_Template_AI_20260327.xlsx \
  --output output/HARA_Report.xlsx \
  --run-id project-run-001 \
  --operating-mode "<输入文档中的实际运行模式>" \
  --allow-draft
```

CLI在存在`.env`时读取本地配置，已有环境变量优先。模型地址、名称和API key由运行者配置。
`doctor`检查方法编译、报告和LLM配置，不发送模型请求；缺少API配置时会单独报告该项未就绪。
`--allow-draft`允许保留待审核状态输出草稿。已安装的等价命令为`hara-agent`。

显式使用模板作为方法来源时，将`--method-baseline`与`--report-template`替换为
`--template TEMPLATE.xlsx`。方法角色有歧义时，按`doctor`给出的候选运行
`confirm-template-role`；确认记录按模板哈希保存。具体命令见部署手册。

恢复使用相同输入与方法、原Run ID并添加`--resume`。新分析使用独立Run ID和输出路径。
范围选择、请求预算、并发、日志及缓存的配置统一见部署手册，历史运行命令留在各自阶段记录。

## 方法与数据如何流转

```text
显式方法模板 → TemplateRoleContract → MethodContract
受治理YAML基线 ───────────────────→ MethodContract
Item Definition → ProjectFacts / RiskFacts
MethodContract + 当前事实与假设 → 物理 / 因果 / S/E/C / ASIL
HARAState + ReportContract → 保留模板格式的Excel报告
```

方法来源负责工程规则，Item负责项目事实，报告模板负责布局。LLM提出有依据的语义判断与
分析假设，Python执行契约校验和确定性计算；评级来自活动方法，批准状态独立记录。

场景实例保留对象、阶段、坐标、来源及原始身份。精确去重、物理方案复用、共同证据共享和
完整因果复用分别统计。主表组织审核组，成员和分支明细保留完整评级及追溯关系。
实现与适用边界见[场景、物理与风险计算](docs/handover/03_场景物理与风险计算.md)及
[SG、安全状态、FTTI与报告](docs/handover/04_SG安全状态FTTI与报告.md)。

## 仓库布局

```text
src/hara_agent/template/            模板发现、角色与方法编译
src/hara_agent/method_sources/      受治理YAML基线编译
src/hara_agent/contracts/           类型契约与规则中间表示
src/hara_agent/services/extraction/ 来源提取与规范化
src/hara_agent/services/semantic/   有界语义任务
src/hara_agent/services/analysis/   物理与方法执行、事实绑定
src/hara_agent/services/reporting/  报告投影与模板渲染
src/hara_agent/infrastructure/      Provider、预算、响应缓存和文件适配
src/hara_agent/workflow/            调度、检查点与质量检查
method_assets/                     受版本管理的方法资产
references/                        报告模板及角色基线
report_assets/                     报告展示schema
skills/                            项目阶段操作规范
tests/                             离线测试与冻结夹具
docs/                              当前手册、专题说明与历史证据记录
```

## 验证与维护

```bash
.venv/bin/python -m pytest -q
.venv/bin/python -m compileall -q src scripts
git diff --check
```

交付前还需运行上述`doctor`。测试类型、固定响应、历史守卫和真实Provider验证分别说明于
[开发、测试与维护](docs/handover/06_开发测试与维护.md)。新付费请求使用当轮明确授权；
离线测试和清理本身不启动真实全量。

维护约束以[AGENTS.md](AGENTS.md)、[项目技能入口](SKILL.md)、活动输入及编译契约为准。
历史阶段数字和预算只属于记录中的对应运行，统一从[文档索引](docs/README.md#历史开发与验证记录)查阅。
