# 五份输入与复现资产

本目录保存五份已用于跨项目分析的原始输入，作为交接与复现的固定来源。
四份FSB PDF于2026-11-25纳入版本管理，文件内容与历史分析使用的输入哈希一致。
AVP DOCX此前已经提交，文件内容与`tests/fixtures/avp/ItemDef.docx`一致。

## 固定路径

| 项目标签 | 仓库路径 | 文件内容 | PDF页数 | 字节数 |
|---|---|---|---:|---:|
| AVP | [ItemDef.docx](ItemDef.docx) | 原AVP Item Definition | — | 241615 |
| IPA-High | [FSB_IPA-High_2.2.pdf](FSB_IPA-High_2.2.pdf) | IPA-High 2.2功能规格 | 7 | 949857 |
| IPA-Main | [FSB_IPA-Main_2.2.pdf](FSB_IPA-Main_2.2.pdf) | IPA-Main 2.2功能规格 | 7 | 947855 |
| RPA | [FSB_RPA-High_2.2.pdf](FSB_RPA-High_2.2.pdf) | RPA-High 2.2功能规格 | 8 | 950322 |
| rTPA | [FSB_rTPA-High_2.2.pdf](FSB_rTPA-High_2.2.pdf) | rTPA-High 2.2功能规格 | 11 | 973632 |

代码、命令和新证据索引统一使用上表路径。`new reference/`中的本地同内容副本不重复提交，
也不是程序隐式读取的目录。生产`analyze`只读取显式`--item`路径，不会自动把五份资料混为一个项目。
新增项目应使用新文件名和Run ID，保留这五份固定来源。

四份PDF为功能规格，明确范围和所缺引用资料应按各自原文处理；它们不必具有与原AVP DOCX
相同的字段完整度。训练、搜索、执行阶段的速度与条件分别使用，不能沿用其他项目的约定。
具体处理原则见[输入抽取、数据契约与方法治理](../docs/handover/02_输入抽取数据契约与方法治理.md)。

## 零Provider核查

从仓库根目录执行。完整SHA256由同目录[SHA256SUMS](SHA256SUMS)维护，文件名采用仓库相对路径。

```bash
# macOS
shasum -a 256 -c input/SHA256SUMS

# Linux：使用 sha256sum -c input/SHA256SUMS
```

在已经安装项目依赖的Python环境中，用现有生产读取器检查文档可读性：

```bash
.venv/bin/python - <<'PY'
from pathlib import Path
from hara_agent.services.extraction import DocumentReader

for line in Path("input/SHA256SUMS").read_text().splitlines():
    _, name = line.split(maxsplit=1)
    document = DocumentReader().read(name.strip())
    assert document.blocks and document.text.strip(), name
    print(name.strip(), "blocks=", len(document.blocks), "characters=", len(document.text))
PY
```

上述操作只校验文件内容与来源读取，不发送模型请求。它不替代对功能、条件或风险结论的语义评审。

## 测试资产怎样使用

| 资产 | 路径 | 用途 |
|---|---|---|
| 五份原始输入 | 本目录五个文件 | 从头提取、跨项目分析及来源可读性检查 |
| 原AVP固定夹具 | [tests/fixtures/avp](../tests/fixtures/avp/README.md) | 固定功能表、来源和上下文回归，避免替换当前输入时改变历史断言 |
| 离线响应与代表状态 | `tests/fixtures/` | 复现已知反例，检查对象、阶段、来源、缓存、评级和报告行为 |
| 正常测试入口 | [pytest.ini](../pytest.ini)、`tests/` | `.venv/bin/python -m pytest -q`；默认离线，不需要API key |
| 活动方法 | [method_assets/fusa_baseline_v1/manifest.yaml](../method_assets/fusa_baseline_v1/manifest.yaml) | 编译为统一MethodContract后执行 |
| 报告布局 | [references/HARA_Template_AI_20260327.xlsx](../references/HARA_Template_AI_20260327.xlsx) | 报告模板，独立于方法源哈希 |

现有冻结测试覆盖具体程序行为，不等同于每次测试都对这五份输入重新执行真实全量。
测试分层和已有覆盖入口见[开发、测试与维护](../docs/handover/06_开发测试与维护.md)。

## 新分析与历史结果复现

新的完整分析使用当前源码、活动方法、所选输入和可用Provider配置。例如：

```bash
.venv/bin/python -m hara_agent analyze \
  --item input/FSB_IPA-High_2.2.pdf \
  --method-baseline method_assets/fusa_baseline_v1/manifest.yaml \
  --report-template references/HARA_Template_AI_20260327.xlsx \
  --output output/IPA-High_new.xlsx \
  --run-id ipa-high-new-run \
  --operating-mode "<该输入中的实际分析运行模式>" \
  --allow-draft
```

该命令会使用Provider，需使用有效的当轮授权；本次提交输入资料没有执行此命令。
其他项目只替换为对应输入、新Run ID及独立输出，按该项目选择实际运行模式。
并发、预算、日志和恢复操作见[部署与运行手册](../docs/handover/05_部署配置运行与恢复.md)。

要原样恢复或重导历史结果，还需要对应checkpoint、审核trace和方法快照。最新报告映射、
Run ID和保存位置见[资产移交清单](../docs/handover/08_资产移交与接手清单.md)。这些大文件及原始响应、账本
仍由`output/`、`runtime/`的独立证据包移交；仅取得原始输入不能代替保存的模型结果。
`.env`和API key由运行者独立配置。

AVP历史原始全量状态未保存`item_sha256`，本次校验值只确认当前固定DOCX及已有重评分来源，
不补造历史身份绑定。四份PDF则已与各自保存的输入哈希逐一核对。
