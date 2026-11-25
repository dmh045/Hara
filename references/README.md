# 模板与参考资产

本目录保存当前报告模板和模板角色基线。运行方式统一见
[部署、配置、运行与恢复](../docs/handover/05_部署配置运行与恢复.md)，
实现入口见[开发与技术交接手册](../docs/handover/README.md)。

| 资产 | 用途 |
|---|---|
| [HARA_Template_AI_20260327.xlsx](HARA_Template_AI_20260327.xlsx) | YAML方法模式下提供报告布局；显式模板模式下也作为方法编译的输入 |
| [template-role-baselines](template-role-baselines/README.md) | 受版本管理的模板角色确认基线；按模板哈希核对 |

当前受治理的方法在[`method_assets/fusa_baseline_v1/manifest.yaml`](../method_assets/fusa_baseline_v1/manifest.yaml)，
报告展示schema在[`report_assets/hara_report_v1.yaml`](../report_assets/hara_report_v1.yaml)。
方法和报告模板分别记录哈希。

原`USAGE_GUIDE.md`介绍的旧脚本入口已被现有应用替代，内容已收敛到上述手册；
原`api_reference.md`为未填写的文档模板，已移除。需要调用程序时，使用
`python -m hara_agent --help`、[`HARAApplication`](../src/hara_agent/application.py)和
[代码实现索引](../docs/handover/workflow_detail/04_代码实现与问题定位索引.md)。
