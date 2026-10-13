# P6 场景来源、语义冲突与候选规模修正

日期：2026-10-12。当前证据目录：`output/p6_review_bounded/`。

## 1. 为什么曾出现 7,650 条

这是本轮修正中的过度展开，不能解释成必要的危险事件数量。

原代码把 `odd_locations` 的 3 个位置范围与 `odd_road_types` 的 10 个道路/车位类型合并为 13 个 WHERE 父上下文。恢复来源条件后，所有上下文分别生成三组主机制变体；另加上下坡两个边界上下文，最后展开两个驾驶员配置。

本项目 58 个失效模式合计产生 85 组物理选项，因此中间结果为：

`85 × (13 + 2) × 3 × 2 = 7,650`。

这同时混用了位置与道路类型，并把来源条件与主机制变体重复相乘。该中间结果保存在 `output/p6_review/`，已由当前回放取代，未启动全量 Provider 分析。

## 2. 代码修正

1. **来源维度分离。** 有项目位置时，WHERE 只取 `odd_locations`；道路/车位类型保留在 `source_road_type_scope` 和 `source_operating_conditions.permitted_road_types`，作为明确的允许范围传给因果分析。它们不是同时成立的几何条件，也未被证明风险等价。
2. **来源条件保留。** Atom 标签不会删除父场景的室内/室外、道路和坡向条件。来源条件参与场景身份与严格去重，并进入紧凑审核记录和语义输入。
3. **语义约束应用到所有 Atom 维度。** 使用现有语义分类与方法候选链路检查位置一致性，修复停车 ODD 选入高速公路/快速路行为 Atom，以及室外父上下文选入室内停车场 Atom 的问题。
4. **一次覆盖主机制，再补来源代表。** 每组不同物理输入保留原覆盖计划的主机制变体；其他来源上下文保留一个代表及所有新增主机制变体。上下坡来源边界放在同一物理选项的一个位置锚点。没有按目标总数截断候选，也没有增加 Atom 选择器。
5. **采样与等价分开记录。** 未选择的“上下文 × 变体”组合记录为 `COMBINATION_NOT_EVALUATED`，保留候选身份、Atom、主机制代表与来源代表。它们没有被当作风险等价合并。严格去重仍要求完整结构化条件一致。
6. **失效缓存失效化。** 普通场景批次版本更新为 v8，语义提示版本更新为 v16，避免使用旧输入形状的因果结果。

未修改 FUSA E 计算公式、S/C 规则、ASIL 矩阵或方法资产。生产入口仍是普通 `analyze`；离线脚本直接调用相同生产服务。

## 3. 当前离线回放

| 指标 | 结果 |
| --- | ---: |
| 函数 / 失效模式 | 9 / 58 |
| 来源位置锚点 | 3 |
| 初始有界组合池（驾驶员展开前） | 1,275 |
| 未选择的上下文与变体组合 | 662 |
| 代表候选（驾驶员展开前） | 613 |
| 驾驶员配置展开后 | 1,226 |
| 当前严格等价合并 | 0 |
| Atom 非空 / 方法约束验证 | 1,226 / 1,226 |
| E 试算就绪 | 1,200 |
| E 输入未就绪 | 26 |
| 本次规模修正的新增 Provider 调用 | 0 |

1,226 相比中间的 7,650 减少约 84%。**613 是代表输入数量，1,226 是驾驶员分支候选数量；都不是已通过因果验证的危险事件数，也不等于实际 Provider 请求数。**

就绪的 E 试算：E1 50、E2 232、E3 778、E4 140。另 26 条为 `EXPOSURE_RELEVANT_DIMENSION_UNRESOLVED`，不能将其试算值作为正式 E。

### 来源与语义检查

- 室外 886 条、室内 170 条、AVP 自构图区域 170 条。
- 基础道路代表 850 条，上坡 15% 206 条，下坡 15% 170 条。
- WHERE：SO010 886，CN_parking_multistory 340。
- ROAD：FB001 850，FB003 376。显式 15% 坡度未被 FB001 覆盖替代。
- 当前候选中，审查所列高速公路/快速路词、FA037、PU008、室外与室内停车场冲突，以及 15% 来源坡度与道路 Atom 冲突的检查结果均为 0。
- 10 个原始道路/车位类型完整保留在每个候选的来源允许范围中；**这不证明不同车位几何已经被分别评价或可安全合并。**

### 对旧 3,060 次归并的结论

旧 255 个分组中的 3,060 个被移除候选全部涉及不同父记录。旧签名缺失来源条件，因此撤回其“已证明等价”的结论。这里没有认定 3,060 条都必须独立分析；需要区分重复来源标签与真实风险条件变化。

当前 662 个未选择组合属于有界代表采样，不能拿来替代等价证明。多轴组合覆盖、几何对碰撞构型/可控性的影响，仍需后续代表性因果验证和工程覆盖检查。

## 4. 此前小范围真实样本的实际状态

现场：`output/p6_review/live_sample/`；运行号 `p6-review-sample-20261012-001`。它发生于本次候选规模修正之前，不能作为当前版本的真实验收。

- 四个来源上下文、两种驾驶员配置，共 8 个样本。
- 8/8 次 Provider 预算已用完；包含一次现有 schema 修复调用，无网络重试。
- 日志记录 7 个场景批次返回；第 8 个场景请求因预算耗尽未发出。
- 首轮 4 个代表返回后，调度器继续独立评价其他驾驶员分支，未采用复用。
- 实际返回模型为 `doubao-seed-2-1-turbo-260628`，未宣称其等同于配置的请求模型。
- 原测试脚本在整体调用结束后才保存结构化结果，预算异常使已返回结果未落盘；日志和请求账本仍在。**不能从这些日志重建完整因果结论，未得到新的正式 S/E/C/ASIL。**

已在样本脚本中补上返回 payload 立即保存及完成波次保存，并用本地 Mock 验证后续异常不会删除已记录 payload。尚未再次调用 Provider 验证此记录修正。原 8/8 账本保留，未重置或追加请求。

## 5. 验证与可复现材料

- `python -m pytest -q`：755 passed、1 skipped。
- 跳过项为需显式启用的历史运行 guard；不表示新的在线因果验收通过。
- `python -m compileall -q src scripts`：通过。
- `git diff --check`：通过。
- 当前 YAML baseline + 报告模板的 `doctor`：可生成草稿，非发布就绪；既有方法绑定、FTTI、severity fallback 警告仍保留。
- 历史 checkpoint SHA256：`0427c6967413995274dbe7112be00deb6286c4194d4bd64f759bea77851d0317`，回放前后相同。
- 当前 Method source hash：`907ed1cfaf7557941d746b84c2d1d10f801716ba87a2200a9841c5efc702830d`。

完整交付包 `output/P6_Review_Correction_20261012.zip` 包含当前完整源码（含新增的 equivalence 与回放脚本）、测试、方法资产、输入、模板、复现所需历史运行文件和当前证据。`PACKAGE_MANIFEST.json` 列出各文件 SHA256。它是本地工作树快照，不代表已推送 GitHub 的提交。

解压后，在仓库目录准备 Python 3.12 环境并安装 `requirements.txt` 与本项目，再执行：

```bash
python -m pytest -q
python -m compileall -q src scripts
python -m hara_agent doctor \
  --method-baseline method_assets/fusa_baseline_v1/manifest.yaml \
  --report-template references/HARA_Template_AI_20260327.xlsx
python scripts/benchmark_p6_offline.py \
  --checkpoint runtime/agent/full-p5o-20261010-001.checkpoint.json \
  --corrected-report output/p5o_review_20261012/HARA_full_p5o_20261010_review_fixed.xlsx \
  --output output/reproduced_p6_review
python scripts/audit_p6_source_coverage.py \
  --before output/p6 --after output/reproduced_p6_review
```

这些复现命令不需要 Provider。当前没有新版正式 HARA Excel：离线候选和 E 试算不能冒充已完成因果、S/C/ASIL 的报告。下一步仍是对当前代表样本做因果及覆盖验证，再决定是否运行全量。
