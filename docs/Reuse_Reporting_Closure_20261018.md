# 场景去重、复用与报告表达：R1—R7 关闭清单

更新：2026-10-18。核对基线及开始时本地/远端 HEAD：`1e150b298d252adea73fa54b99a027e4ddceb2b8`。本轮保留工作树、历史运行、响应缓存、预算及个人配置。输出根目录：`output/reuse_reporting_20261018/`；最终样本：`delivery/samples/`。

## 结论

R1—R7 已落实到现有生产链，离线回归和来源样本通过。报告展示减少已证明；精确重复和明确非干预对照确实减少请求。冻结来源总体没有证明请求/token总量下降，不能将展示归组或共有证据计数当作这一收益。

本轮真实 Provider 请求 **0**，没有新的有效预算。T31未执行；原生Excel已重开，打印入口因Office未激活而禁用，T29部分完成。具备运行有预算代表样本的技术条件；尚未满足“真实代表样本有效后再启动全量”的条件，也没有全量预算。

## 统一关闭清单

| 项目 | 根因与实际修改 | 离线生产证据和效果 | 真实验证 / 剩余范围 |
|---|---|---|---|
| R1 保留自主输入 | 沿用HARAApplication、现有物理Agent/实例化器/MethodContract；只增实际求解计数，未改模型和评级规则 | 五族10身份：既有制动2条跳过新物理请求；其他4份方案生成8实例，进入因果、评分、SG、报告链 | 新代码真实模型返回未验证；侧向C局部未决、符号关系保留 |
| R2 精确任务与范围 | scenario_equivalence使用原始事实、对象、阶段、坐标、有效来源及方法/提示词/Provider版本；response_checkpoint合并同进程并发同请求；现有malfunction checkpoint保留任务映射 | ID不同但同输入、同Pending均2→1；4线程并发仅1次调用；恢复改ID零调用；矛盾缓存重新判断；范围162身份完整 | 不声称跨进程分布式锁；固定来源总体精确重复0；token未测量 |
| R3 合批与复用 | scenarios调度移除提前返回；精确任务→非干预证明→共有证据与独立判断；原parser重编译，physical_interaction/对象/子身份保留，complete=false不改写 | 非干预正例2→1，使能反例2→2；已能合批为1次的未证明对继续分别判断并记录证明；变化交互只共享MF→行为 | 固定来源脚本没有非干预证明，完整因果复用0；真实证明质量待T31 |
| R4 审核组 | projection分开审核展示与风险等价；包含Pending；最高ASIL取同一分支完整元组，并列不同元组留明细 | 主表106→6、162→6、10→4；明细106/162/10；绑定组合不自由相乘；主表↔明细链接，trace随包可打开 | 展示减少不等于候选减少、风险等价或正式上界 |
| R5 语言/SG | mapper、现有SG服务/节点使用全部关联失效的已有意图整理短句；原文/ID/关联保留catalog、审计和注释；具体Pending用中文和格式化已知输入 | 3条SG保持离线/条件性草稿；最高ASIL只覆盖已计算成员；Safe State/FTTI待确定 | 未扩展SG/Safe State/FTTI业务方法 |
| R6 数值显示 | quantity_format仅用于报告：速度/坡度/角度1位，其余物理量2位；保留小非零、负值、未知、区间包含性；分档边界来自编译方法 | 10.182338→10.2，0.154286→0.15，1.454723→1.45；边界显示原始不等关系；三份checkpoint重导前后整文件SHA相同 | 原始精度、方法阈值、风险键和缓存签名不变 |
| R7 纯等级 | 主表/明细/分支/SG仅等级或待评定；条件性与审核状态独立保留 | 全部评级格扫描无后缀/非法等级；条件性QM仍是条件性；没有批量FINALIZED | 没有正式批准或发布状态升级 |

## 实测规模与收益

基准使用同一固定checkpoint人口、原始输入和脚本响应。基线调度固化在测试fixture，旧Agent提示词由`git show 1e150b2`获取并随包保留。新旧原始请求/返回、身份、判断均落盘；输入哈希一致，输出身份和因果判断逐项相等。

| 总体 | 身份前/后 | 请求前→后 | 合批共有上下文前→后 | 校验共有MF→行为步骤前→后 | 完整因果复用 | prompt字符前→后 |
|---|---:|---:|---:|---:|---:|---:|
| 点值 | 106/106 | 85→85 | 21→21 | 0→73 | 0 | 987931→999535 |
| 范围 | 162/162 | 141→141 | 21→21 | 0→129 | 0 | 1633171→1654183 |
| 符号 | 10/10 | 9→9 | 1→1 | 0→5 | 0 | 102027→102727 |

这些是脚本客户端调用，非真实HTTP。共有步骤仍由当前返回和证据编译器校验，73/129/5不代表少发请求或完整结论复用。输入略有增加，没有token节省证据；脚本usage=0不是测得的真实token。

| 定向对照 | 请求前→后 | 身份前/后 |
|---|---:|---:|
| 精确重复 | 2→1 | 2/2 |
| 精确Pending | 2→1 | 2/2 |
| 驾驶员明确非干预 | 2→1 | 2/2 |
| 驾驶员影响使能 | 2→2 | 2/2 |
| 后段交互改变，仅共享上游 | 2→2 | 2/2 |

五族普通应用阶段账本：**4方案+9因果+2事实=15次脚本调用**；物理方案复用4、生成实例8、实际物理求解8、既有制动跳过2、精确物理输入复用0。各类收益分别记录，不混计。

## T01—T32 验收

“通过”指所列离线/静态证据，不包含真实模型。U=`tests/unit/test_reuse_reporting_acceptance.py`；B=`delivery/benchmark`；C=`delivery/controls`；S=`delivery/samples`；F=`delivery/five_family`；V=`validation`。

| 编号 | 结果 | 证据 |
|---|---|---|
| T01 | 离线通过 | F普通应用五族10身份；来源→首次脚本方案→物理→因果→风险→报告，未手填评分输入 |
| T02 | 通过 | 原自主物理followup符号隔离回归；S/symbolic参数关系，不借旧点值 |
| T03 | 通过 | 原输入兼容、坡向、目标、保持力反例及正例；F上/下坡两组 |
| T04 | 通过 | U、C/exact：2→1，两身份、对象绑定完整 |
| T05 | 通过 | U：同进程4线程同请求，1次调用、3次缓存命中 |
| T06 | 通过 | U、C/pending_exact：诊断2→1，仍Pending |
| T07 | 通过 | U：3.9999与4.0001显示相同但任务键不同，原值分档 |
| T08 | 通过 | U：坡向、真实对象、阶段、来源改变各2次独立请求 |
| T09 | 通过 | U/C开启paired_driver_requests仍执行证明；B含合批和单项实际尺寸 |
| T10 | 通过 | C非干预2→1；原P6流水线测试保留C1/C3独立分支 |
| T11 | 通过 | C使能反例2→2；依赖变化不复制受影响判断 |
| T12 | 通过 | U：complete=false不阻止精确任务复用，标记仍false |
| T13 | 通过 | U/C：仅MF→行为共用；后段当前子场景独立，physical_interaction完整 |
| T14 | 通过 | B/range162身份全保留，上游共享129次；接触关系改变仍独立判断，不宣称连续不变性 |
| T15 | 通过 | U恢复零新增、输入变化失效、改ID重绑、缓存冲突重发；原响应缓存契约/模型身份回归 |
| T16 | 通过 | U显示文字变化仍命中；S重导0 Provider |
| T17 | 通过 | U Pending归组；S条件性/未决均能进入6/6/4审核组 |
| T18 | 通过 | U同分支完整元组、最大S≠最高ASIL、并列不同元组；所有分支保留 |
| T19 | 通过 | U只保留前进＋前方、倒车＋后方实际组合 |
| T20 | 通过 | S/comparison.json及artifact检查：106/162/10身份完整 |
| T21 | 通过 | U全部既有意图及原文；SG页没有新增参数或批准 |
| T22 | 通过 | U多成员意图不采用第一条窄场景；catalog保留原文 |
| T23 | 通过 | U量纲例；S物理参数文本检查及checkpoint不变证明 |
| T24 | 通过 | U真零、小非零、负小值、未知、无有限值；原不接近反例保留 |
| T25 | 通过 | U活动编译方法边界的生产projection、不等关系、区间包含性 |
| T26 | 通过 | 样本所有评级格扫描；分支完整元组纯等级测试 |
| T27 | 通过 | 三份checkpoint重导前后整文件SHA相同，来源/状态/ID不变 |
| T28 | 通过 | U条件性QM与未决SG独立状态；SG页自身最高已知范围说明 |
| T29 | 部分通过 | 原生Excel重开；打印禁用。8张前后渲染、openpyxl重开、0断链/JSON/长物理小数；见下方限制 |
| T30 | 通过 | V/pytest_handoff.log：923 passed、1 skipped；compileall、diff --check、doctor通过 |
| T31 | 未执行 | 无当前有效预算；所有响应明确OFFLINE_SCRIPTED |
| T32 | 已明确 | 可进行有预算代表样本；真实有效后再诊断全量，目前未授权或启动全量 |

唯一跳过项是opt-in历史runtime guard（HARA_RUN_HISTORICAL_GUARDS未启用），与真实Provider未执行分开记录。doctor为草稿就绪、正式发布未就绪，保持原运行证据/SG/Safe State审核要求；本轮未新增Gate。

## 交付及复现

- 点值报告：`delivery/samples/point/HARA_Review.xlsx`；范围、符号在相邻目录。
- 新五族普通应用报告：`delivery/five_family/HARA_Review.xlsx`，同目录保存首次方案、因果/事实响应、checkpoint和物理/风险trace。
- 每份报告`audit/p2-report-rebuild/report_projection_audit.json`保存审核组、完整场景/风险身份、风险等价键、实际组合、逐分支元组。
- 前后图：`qa/final_Before_*.png`、`qa/final_HARA_Review_*.png`。Artifact只读渲染，没有替代生产报告生成器。
- 同条件原始请求/返回：`delivery/benchmark`；定向控制：`delivery/controls/comparison.json`。
- 完整源码、差异、输入和上述证据随`HARA_Reuse_Reporting_20261018.zip`交付；源码验证后提交推送GitHub。包内README给出复现路径。

以下均零Provider；新输出目录必须不存在。包解压根目录执行：

```bash
python scripts/validate_reuse_reporting.py --source evidence/frozen_population --output output/reproduce_reports
python scripts/benchmark_scenario_reuse.py --source evidence/frozen_population --output output/reproduce_benchmark --baseline-agent evidence/benchmark/baseline_scenario_agent.py
python scripts/validate_reuse_controls.py --output output/reproduce_controls
python scripts/validate_physical_followup_sample.py --source-context evidence/source_context.json --output output/reproduce_five_family --run-id reuse-reporting-reproduce --offline
python -m pytest -q
```

后续真实验证沿用现有validate_physical_followup_sample.py五族选择和持久账本。当前没有可填的授权额度，因此未执行--live。获得有效预算后只启动代表样本，方案/因果/事实/修复/重试共用限额，停止后原run续跑；通过后才为全量建立独立Run ID、输出和checkpoint。局部C、符号参数及其他明确Pending无需全部归零。

## 已知限制

1. 没有新真实响应或token测量，不承诺全量调用降幅；固定来源脚本总体调用未下降。
2. 原生打印预览未完成：Office未激活。Excel已重开，打印范围/重复标题正确。主表可横向滚动，明细保留长证据。
3. Artifact只读渲染在Hazard合并标题下出现单个残留字形；F4确认为Hazard、F5为空、合并/旋转正常，未为渲染器表现破坏模板。模板ASIL_Table!U5/U6原有#N/A/#VALUE!前后相同，不来自本轮评级。

不扩大已关闭反例审查。下一项实质验证是有预算的五族真实样本，通过前不宣称新全量已获放行。
