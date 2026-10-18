# 场景去重、复用与报告表达：R1—R7 关闭清单

更新：2026-10-18（独立复核后修订）。原实现基线：`1e150b298d252adea73fa54b99a027e4ddceb2b8`；本次定向修订开始时本地/远端 HEAD：`dd80b3e51230e4bd01da0778fecabdaf7ef11f2a`。保留历史运行、响应缓存、预算及个人配置。原交付根目录：`output/reuse_reporting_20261018/`；本次修订根目录：`output/reuse_reporting_review_20261018/`，新报告在 `reports_final/`。

## 结论

R1—R7 的既有生产链保留。独立复核确认了 R4 的未决组代表值、摘要计数缺陷，以及 R5 安全意图整理仅部分完成；这三项已在本次修订中修正并重导验证。报告展示减少已证明；原定向对照中精确重复和明确非干预确实减少请求。冻结来源总体没有证明请求/token总量下降，不能将展示归组或共有证据计数当作这一收益。

报告修订时真实 Provider 为 0；独立验收后用户另行授权最多 20 次，现已用完，T31 已执行但未通过端到端验证。真实调用与后续零调用重放见下节。原生打印仍未完成，T29部分完成；新全量尚不就绪，也没有全量预算。

## 三项修订的独立验收（2026-10-18）

已完整读取用户补充的《报告三项定向修订_独立复核_20261018.md》，针对提交 `22a15b24aa4f2eb185276dacf7d56951bd90531e`。该复核确认上述三项报告问题在所核源码、回归和样本范围内已修正，本轮定向报告收口关闭，不再作为真实代表样本的前置阻碍。

- 独立复核报告：定向 34 passed，默认 927 passed、2 skipped；额外跳过为独立环境未包含的旧 SEC 工件，与本地 928/1 的差异不是回归失败。
- 独立复核报告：800 个清单文件校验通过，三份生产重导五个核查工作表的已读取单元格与交付差异为 0，278 条明细/分支及原 checkpoint 保留。
- 验收范围仍是展示修订；278 条含重叠样本，不代表新增独立危险事件。组内 E 的“待评定”代表没有唯一共同值，成员 E2/E3 已计算值仍保留。
- SG 的三条来源匹配改写通过；真实模型更换危险事件措辞时可能回退到原意图。下一步在真实样本中观察，不扩展本轮架构或宣称通用语言能力通过。
- 使用既有 `validate_physical_followup_sample.py` 生成无网络预览：制动、驱动、上坡、下坡、侧向五族共 10 个来源身份；保留现有来源提取，新的物理/因果/风险结果由生产链产生。预览：`output/reuse_reporting_acceptance_20261018/preview/sample_plan.json`，实时请求 0，预算 0。
- 随后用户明确授权最多 20 次，使用独立 Run ID `reuse-reporting-live-20261018-a`；实际输出与预览目录分开。所有阶段和纠正/重试共用同一持久预算，历史余额未转作新额度，见下节执行结果。

三项报告修订保持关闭。本次真实验证发现的局部接口/枚举/持久化问题直接修复，不改方法数值、ASIL矩阵或已保存的模型返回。最新验证目录：`output/reuse_reporting_acceptance_20261018/validation/`。

## 有预算真实代表样本：20 次执行结果

输出根目录 `output/reuse_reporting_acceptance_20261018/`。`authorization.json`、`live/sample_plan.json`、`live_validation_summary.json` 和 `live/checkpoints/*.provider-attempts.jsonl` 分别记录授权、五族10身份、结果和唯一账本。当前**剩余额度0，未启动全量，没有新的风险评级或Excel**。

| 项目 | 实际结果 |
|---|---|
| 调用 | 20次：物理方案/纠正15次，因果5次；事实/风险评分尚未执行 |
| 返回与中断 | 19次成功返回，实际模型均为 `doubao-seed-2-1-turbo-260628`；第14次在主动停止时为 `KeyboardInterrupt`，没有取得返回，仍计预算。19份成功返回均已保存，不能把20条结束账本事件说成20份成功返回 |
| Token | 可核对19份返回：输入1,125,192，输出21,227，合计1,146,419；被中断请求的usage未知，未估算货币费用 |
| 修复前 | 9次物理请求未形成可用非制动实例；另有4次成功因果返回和1次中断，均针对修复前条件，不能复用于新物理子场景 |
| 同账本续跑 | 剩余6次中，驱动和侧向各1次首次返回通过实例化，共4个子场景；随后坡道纠正消耗余量，到20立即保存停止，没有换ID或清空账本 |
| 最终零调用重放 | `saved_response_replay_final/` 使用普通HARAApplication、不可联网的客户端和原响应缓存；4份真实方案缓存命中，生成6个子场景；在下坡新纠正请求缺少缓存处停止，新增请求0，原账本哈希不变 |
| 生产物理trace | `replayed_physical_instances.json`及重放checkpoint保存6个完整子场景、模型输入、来源、假设和派生trace；它们是已保存真实返回的确定性重放，不是新增真实测试 |
| 风险与报告 | 新物理子场景因果验证没有完成，风险记录0；没有新S/E/C/ASIL或样本Excel，原报告保持不变 |

### 本次确认并修正的代码问题

1. **物理输出契约反馈不具体。** SYMBOLIC 被写成 `value: "a_fault"`，校验只返回 `'symbol'`；提示中对象类型、控制名和来源ID容易被当作自然语言。现在显式列出活动方法的可用键、三种参数形态和来源ID；修复请求附带未修改的原错误JSON。错误包含参数名、缺失/多余字段。提示版本更新，使旧失败请求不会伪装成新提示的缓存命中。未知对象没有映射成车辆，未增加评分规则。
2. **同一输入的枚举大小写误报冲突。** 已选 `front` 与提议 `FRONT` 被误判成不同对象位置。现在位置/方向按既有枚举含义比较，风险类别通过活动MethodContract精确词汇映射；子场景规范拼写但保留原来源权威。不同前后位置、不同对象ID和不一致初速仍拦截，父场景不改写。
3. **阶段中断前的物理实例追踪不足。** 原checkpoint只记录已生成ID；后续方案耗尽预算时，完整实例仍需从方案重算。现在在既有instantiations记录内同时保存完整生成场景及派生trace。预算中断回归证明前一成功实例仍持久化，审批状态保持PENDING。没有新运行时或旁路评分。

新增9个用例；本地最终 **937 passed、1 skipped**，相关定向51 passed。`validation/pytest_handoff.log`、`persistence_targeted.log`、`compileall_handoff.log`、`doctor_handoff.log`及`diff_check.log`为最新门禁。唯一跳过仍是opt-in历史guard。

### 实际剩余缺口与全量判断

- 驱动距离、侧向净距、坡道残余阻力均由真实模型保留为符号；仅形成参数关系/条件，未得到数值碰撞速度、TTC或完整评级。没有替模型填值。
- 下坡方案把来源初速7改成0，却声明 `condition_changes.initial_state=false`；这是实质冲突，保持待纠正，未为得到结果而放宽校验。
- 上坡返回的文字仍声称向后溜车撞到前方目标；结构化输入为REVERSE+FRONT，没有数值接触结论。模型场景语义质量未通过验收，后续必须由当前轨迹的因果验证解决，不能继承旧父结论。
- 物理提示每次约7万—7.8万输入token，真实开销已经测得较高；本轮不声称整体降本。后续应针对重复上下文处理，不能以删去风险条件来缩小输入。
- 真实样本端到端未通过；当前技术结论为**不启动新全量**。下一步是剩余有界方案纠正和新子场景因果/风险链，须另有有效预算；不是重开三项已关闭报告问题，也不要求所有Pending归零。

本次证据包 `HARA_Representative_Live_Validation_20261018.zip` 保存源码差异、原始响应、唯一账本、停止前快照、零调用重放、完整实例trace及测试。它不含冒称新结果的历史Excel。有关实时口头状态以本节核对后的账本结果为准。

## 独立复核后定向收口

本节路径相对本次修订根目录。只修改现有报告投影、语言配置和重导脚本，没有重新评分、生成物理方案或因果请求。

| 复核问题 | 修复与实际结果 | 证据 |
|---|---|---|
| R4 未决组误用首成员 E | 没有最高已知 ASIL 时逐维检查共同值，并检查碰撞分支；点值/范围 HARA_002 的 14 成员仍为 S1×14、E3×12/E2×2，主表改为 S1/待评定/待评定/待评定，独立理由列保留 E 分布。已有最高 ASIL 的同分支完整元组逻辑不变 | `before.json`、`after_integrity.json`、生产投影审计 `partial_group_grade_distributions`；新回归同时验证共同 E3 正例和分支 E2 反例 |
| R4 摘要混用总体计数 | 分别显示主表审核组数、风险记录数、场景明细成员数；点值 6/106/106，范围 6/162/162，符号 4/10/10 | 三份 `reports_final/*/HARA_Review.xlsx` 摘要及 `after_integrity.json` |
| R5 SG 只是前缀拼接 | 报告语言配置按已有失效、车辆危险及必要的关联事件/工况原文匹配，三条显示为“防止驻车保持失效导致车辆非预期移动。”“防止泊车过程中非预期加速引发碰撞。”“防止泊车制动能力丧失导致碰撞。”；两条已有伤害描述忠实译为中文 | 生产投影审计 `safety_goal_display_evidence` 保留逐成员原文、条件、ID及匹配依据；正例、多成员与否定接触反例通过 |

SG 改写是已有意图的展示配置，不是新安全目标生成器或工程规则。未知、变化或不支持碰撞的来源保留原意图，不套用这三句话；未增加控制措施、响应时限、Safe State 或批准。所有 SG 原文、ID、关联、最高已知 ASIL 和审核状态保持不变。

三份源 checkpoint 整文件 SHA 保持不变，278 条明细的身份及完整成员/分支评级与前版逐项相同。五族样本本次也仅从旧 checkpoint 重导；其源文件与前版 ZIP 中对应文件哈希相同，未重复运行五族分析。

本地完整测试 **928 passed、1 skipped**（新增 5 个定向回归）；`validation/pytest_final.log`、`compileall.log`、`diff_check.log`、`doctor.log` 保存最终门禁。独立复核的 922/2 与前版 923/1 差异来自复核环境缺少旧 SEC 工件；本地本次唯一跳过仍为显式 opt-in 历史 guard，与真实 Provider 未运行分开记录。

链接证据更正：前版外部文件超链接扫描命中 0，不能据此宣称跨机器链接验证通过。本次检查的是 **278 个相对 trace 路径单元格及对应文件存在性**；仍没有外部文件超链接关系，未进行跨机器超链接测试。新包清单逐项覆盖除清单自身外的全部文件，明确记录实际数量；前版 SHA 清单只覆盖 144 项，不能称覆盖全部 590 个 ZIP 成员。

本次图像位于 `qa/final_{Before,HARA_Review}_{main,summary,goals}.png`，已查看三个修改区域。仍是 Artifact 只读渲染；本次没有新增原生打印验证。

## 统一关闭清单

| 项目 | 根因与实际修改 | 离线生产证据和效果 | 真实验证 / 剩余范围 |
|---|---|---|---|
| R1 保留自主输入 | 沿用HARAApplication、现有物理Agent/实例化器/MethodContract；补充输出契约反馈、枚举等价比较及完整实例持久化，未改模型和评级规则 | 既有离线五族10身份：制动2条跳过新物理请求；4份脚本方案生成8实例完成后链。新增真实/重放结果另见上节 | 真实修复后4实例，零调用重放6；新子场景因果/风险链未完成，不能将脚本8实例当真实结果 |
| R2 精确任务与范围 | scenario_equivalence使用原始事实、对象、阶段、坐标、有效来源及方法/提示词/Provider版本；response_checkpoint合并同进程并发同请求；现有malfunction checkpoint保留任务映射 | ID不同但同输入、同Pending均2→1；4线程并发仅1次调用；恢复改ID零调用；矛盾缓存重新判断；范围162身份完整 | 不声称跨进程分布式锁；固定来源总体精确重复0；token未测量 |
| R3 合批与复用 | scenarios调度移除提前返回；精确任务→非干预证明→共有证据与独立判断；原parser重编译，physical_interaction/对象/子身份保留，complete=false不改写 | 非干预正例2→1，使能反例2→2；已能合批为1次的未证明对继续分别判断并记录证明；变化交互只共享MF→行为 | 固定来源脚本没有非干预证明，完整因果复用0；真实证明质量待T31 |
| R4 审核组 | projection分开审核展示与风险等价；包含Pending；最高ASIL取同一分支完整元组；无完整结果时仅显示共同维度，另列分布；摘要使用实际组数 | 主表6/6/4，明细106/162/10；E3×12/E2×2未误作共同E；实际组合和全部身份保留；278个相对trace路径存在 | 展示减少不等于候选减少、风险等价或正式上界；跨机器超链接未测 |
| R5 语言/SG | 前版仅加“防止”未充分完成，本次用报告语言配置整理来源支持的意图；原文/ID/关联和逐成员依据保留审计；具体Pending用中文和格式化已知输入 | 3条SG为自然短句，仍离线/条件性草稿；最高ASIL只覆盖已计算成员；Safe State/FTTI待确定；伤害翻译保留限定 | 未知来源保留原意图，未扩展SG/Safe State/FTTI业务方法 |
| R6 数值显示 | quantity_format仅用于报告：速度/坡度/角度1位，其余物理量2位；保留小非零、负值、未知、区间包含性；分档边界来自编译方法 | 10.182338→10.2，0.154286→0.15，1.454723→1.45；边界显示原始不等关系；三份checkpoint重导前后整文件SHA相同 | 原始精度、方法阈值、风险键和缓存签名不变 |
| R7 纯等级 | 主表/明细/分支/SG仅等级或待评定；条件性与审核状态独立保留 | 全部评级格扫描无后缀/非法等级；条件性QM仍是条件性；没有批量FINALIZED | 没有正式批准或发布状态升级 |

## 前版实测规模与收益（本次未重新测量）

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

“通过”指所列离线/静态证据，不包含真实模型。U=`tests/unit/test_reuse_reporting_acceptance.py`；B=`delivery/benchmark`；C=`delivery/controls`；S=`delivery/samples`；F=`delivery/five_family`；V=`validation`。以上相对原交付；本次补充 Q=`tests/unit/test_report_review_corrections.py`，N=本次修订根目录。

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
| T17 | 通过 | U Pending归组；N保留6/6/4审核组，Q验证无完整结论时不冒用首成员E |
| T18 | 通过 | U同分支完整元组、最大S≠最高ASIL、并列不同元组；Q补充共同值及碰撞分支冲突；所有分支保留 |
| T19 | 通过 | U只保留前进＋前方、倒车＋后方实际组合 |
| T20 | 通过 | N/after_integrity.json：106/162/10身份与评级完整；摘要组数和总体分开 |
| T21 | 修订后通过 | Q三种来源支持的SG自然短句、否定接触反例；生产审计保留原文和状态，无新增参数或批准 |
| T22 | 通过 | U/Q多成员意图不采用第一条窄场景；不匹配的不同来源仍保留，审计有逐成员依据 |
| T23 | 通过 | U量纲例；S物理参数文本检查及checkpoint不变证明 |
| T24 | 通过 | U真零、小非零、负小值、未知、无有限值；原不接近反例保留 |
| T25 | 通过 | U活动编译方法边界的生产projection、不等关系、区间包含性 |
| T26 | 通过 | 样本所有评级格扫描；分支完整元组纯等级测试 |
| T27 | 通过 | 三份checkpoint重导前后整文件SHA相同，来源/状态/ID不变 |
| T28 | 通过 | U条件性QM与未决SG独立状态；SG页自身最高已知范围说明 |
| T29 | 部分通过 | 前版原生Excel重开、打印禁用；本次6张前后渲染及openpyxl重开，278个相对trace路径存在；外部关系0不等于跨机器链接测试通过 |
| T30 | 通过 | 最新 acceptance/validation/pytest_handoff.log：937 passed、1 skipped；compileall、diff --check、doctor通过；报告修订时为928/1 |
| T31 | 已执行，未通过端到端 | 20次预算已用完；19份真实返回保存，1次中断；修复后真实生成4子场景，零调用重放6，尚无新子场景完整因果/风险结果 |
| T32 | 已明确 | 新全量不就绪、未授权或启动；先完成上述有界缺口，报告三项无需重开 |

唯一跳过项是opt-in历史runtime guard（HARA_RUN_HISTORICAL_GUARDS未启用），与真实样本未完成分开记录。doctor为草稿就绪、正式发布未就绪，保持原运行证据/SG/Safe State审核要求；本轮未新增Gate。

## 交付及复现

### 报告独立复核修订包（22a15b2，历史证据保留）

新包：`output/reuse_reporting_review_20261018/HARA_Reuse_Reporting_Review_Fix_20261018.zip`。包含完整提交源码、本次相对 dd80b3e 的差异、冻结 checkpoint、前后 Excel/截图、原始身份映射、最终测试和完整性证据；没有新 Provider 响应。新点值 Excel：`output/reuse_reporting_review_20261018/reports_final/point/HARA_Review.xlsx`，范围/符号在相邻目录，五族重导在 `five_family/`。

新包解压根目录，以新输出目录重导三份报告：

```bash
PYTHONPATH=src python scripts/validate_reuse_reporting.py --source evidence/frozen_population --previous-reports evidence/previous_reports --output output/reproduce_reports
```

该报告修订包没有重新打入前版所有调用基准；下节的调度/五族原始响应及收益证据仍在原包。该包制作时尚无新预算，真实验证尚未执行；其后20次授权及本次结果见上节，不回写覆盖历史包。

### 原版实施包（历史证据保留）

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

真实验证沿用现有validate_physical_followup_sample.py五族选择和持久账本，现有授权20次已耗尽。之后须有新的明确授权才能发请求，不能换Run ID重置额度。方案/因果/事实/修复/重试共用限额；代表样本通过后才考虑独立全量Run ID、输出和checkpoint。局部C、符号参数及其他明确Pending无需全部归零。

## 已知限制

1. 已有19份新真实返回及其token，但完整真实样本尚未完成，不承诺全量调用降幅；固定来源脚本总体调用未下降。第14次主动中断仍计入20次预算，其usage未知。
2. 原生打印预览未完成：Office未激活。Excel已重开，打印范围/重复标题正确。主表可横向滚动，明细保留长证据。
3. Artifact只读渲染在Hazard合并标题下出现单个残留字形；F4确认为Hazard、F5为空、合并/旋转正常，未为渲染器表现破坏模板。模板ASIL_Table!U5/U6原有#N/A/#VALUE!前后相同，不来自本轮评级。

不扩大已关闭报告反例审查。五族真实样本已在20次上限内执行但未完成；下一项为上述具体缺口的有界续验，通过前不宣称新全量已获放行。
