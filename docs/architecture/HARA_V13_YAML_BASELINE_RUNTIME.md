# HARA V13 YAML Baseline 运行契约

> 历史设计与核验记录：本文中的“当前”、缺口和运行结论仅对应下述日期或提交。现行实现与使用方式见[开发交接手册](../handover/README.md)；历史记录索引见[文档目录](../README.md#历史开发与验证记录)。

日期：2026-09-03

## 已接通

- 显式 `--template`：继续由 `TemplateRoleCompiler` 编译方法和报告契约。
- 未传 `--template`：由 `method_assets/fusa_baseline_v1/manifest.yaml` 选择唯一
  canonical YAML 资产，再由 `YamlBaselineCompiler` 编译为相同的 `MethodContract`。
- 报告模板只提供 `ReportContract` 和版式；`method_source_hash` 与
  `report_template_hash` 独立记录并分别校验。
- Baseline 已编译 8 个 Guideword、197 个 VDA atom、7 个执行维度、结构化
  Severity 相对速度表、Z/F 与 E 聚合策略、IAV AVP TTC profile/override，以及展开后的
  80 个 ASIL cell。
- S/E/C runtime 只消费 typed `StructuredRiskMethod`；不会直接打开 YAML。

## 当前明确保留的 warning

- `BASELINE_SCENARIO_BINDING_INCOMPLETE`：跨语言 ODD 到 atom 的槽位匹配以及完整的
  bounded candidate selection 尚未完成。已有精确匹配会保留 `scenario_atom_ids`，未匹配
  不会使用默认 atom。
- `SEVERITY_FALLBACK_UNCOMPILED`：相对速度/Delta-V 主表可执行，traffic-domain fallback
  尚未成为 typed contract。
- `FTTI_METHOD_UNCOMPILED`：FTTI YAML 已纳入 bundle hash，但公式路由与审计过的函数
  registry 尚未进入 runtime。

这些 warning 不影响 compiler/doctor 生成草稿，但意味着当前 YAML 路线不应被描述为
正式发布就绪；剩余缺口会保留可分类的 PENDING，不允许猜默认值。

## 命令

```powershell
$env:PYTHONPATH="src"

python -m hara_agent doctor `
  --method-baseline method_assets\fusa_baseline_v1\manifest.yaml `
  --report-template references\HARA_Template_AI_20260327.xlsx

python -m hara_agent analyze `
  --item input\ItemDef.docx `
  --method-baseline method_assets\fusa_baseline_v1\manifest.yaml `
  --report-template references\HARA_Template_AI_20260327.xlsx `
  --output output\HARA_Report_yaml_baseline_draft.xlsx `
  --run-id hara-yaml-baseline-v1 `
  --operating-mode Active `
  --max-workers 2 `
  --allow-draft
```

首次 Baseline 运行不要复用 Template 模式的 checkpoint；方法源哈希不同，恢复会被拒绝。
