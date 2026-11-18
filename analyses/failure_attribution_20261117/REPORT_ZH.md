# 既有结果失败归因与通用改进清单

日期：2026-11-17。范围：E-DARA 三题导入产物及其原生数据库；Stage A v4；Stage B v2 十五题五组记录。没有改动生成内容、生产代码、历史评分或数据库，没有调用模型/API。

## 1. 最重要的发现

1. **E-DARA 的三份参评产物均为 Proposal，已精确找到原生版本。**删除 `rq_coverage`、将 `ev-*` 引用替换为 `[REF]` 后，所有保留字段与参评 JSON 完全一致。此前“无法追溯”的状态已被本次核对解决。
2. **原生证据没有随文本完整传入外评。**三份 Proposal 的 claim 引用了 24 次证据记录，对应各库合计 23 个唯一证据 ID；全部存在，且引文可在保存的来源 section 中找到。外评把多项“未检索到”升级成“造假”，不支持把旧的 12 条过度断言直接当作 12 个原生失实事实。
3. **Workshop 后续确实完成了原生 Plan。**旧 `ws-round6.log` 是预算暂停时的中间记录；后续恢复日志、当前数据库和保存的交付快照一致显示 `plan_reviewed` / `qualified_plan_available`。之前仅据旧日志的判断需要更新。参评文件仍是较早阶段的 Proposal。
4. **外挂路由存在可重放的阻塞。**十五题有十四题初始优先搜证，十三题最终未改正文；这十三题均已有设计澄清项。四个实际“未新增证据”的案例已离线重放，均在搜证后退出，生成器调用为零。
5. **评价误差足以影响排序解释。**69/225 个维度因锚点不合格降为未决，其中模型原判 68 个 pass、1 个 fail；46 个正文片段不匹配，26 个证据引文不匹配，两类有重叠。22 个正文片段去掉首尾引号即可匹配，其中 19 个没有其他锚点障碍。这是格式诊断，不是重新评分或证明模型原判正确。

因此，当前优先级应是：**修复参评证据传递和评价可靠性，再改外挂通用路由；E-DARA 的大幅架构改造缺乏当前证据支持。**原生配置与完成状态需要准确记录，仍然保留所有失败。

## 2. E-DARA：产物来源与运行状态

| 题目 | 导入产物的精确原生版本 | 原生记录核对 | 可以作出的判断 |
| --- | --- | --- | --- |
| AIIB-82 | Proposal `rv-5b1fe45c1ba14b9af1338753a3d8f91f8482ec6534dcc7ec74b0bae852d56c2b`，来自 `aiib82_arxiv_run` | 对应库没有该 Proposal 的 Review bundle，也没有 Plan；对应 Policy 最后为 `paused_budget`，错误为 `Insufficient reservation in topic` | 这是预算暂停前的 Proposal，不是经最终审查的 Plan。另一个 `arxiv_run2` 的结果不能替代该产物的来源 |
| SmallARI KV | Proposal `rv-1344dc72f8a2aa7c033060301264008a2ce3ce2cf0a67682bcaee03695382060` | Proposal 有三份 complete Review，各项 pass；另有绑定该 Proposal 的 Plan `rv-288657…`，其一份轻量 Plan Review 存在一项 partial；Policy 为 `inconclusive` | 已有 Plan，但存在来源复核/登记义务；参评文本取的是 Proposal |
| Workshop Agentic AI | Proposal `rv-bbed824c80aa58d690517bab4528489424e74490e588a15b60e3ba75ad1c0dab` | Proposal 有三份 complete Review，各项 pass；Plan `rv-951a7d…` 有一份 complete、各项 pass 的 Review；恢复后为 `plan_reviewed`，交付快照列入 qualified Plan | 该题原生设计闭环已完成；这不是实验效果已证实，也不等于已有十五题成功运行 |

SmallARI 与 Workshop 的 Plan 采用 `research-stage/light-plan-v1`，保存配置要求一份 Reviewer，因此不能把“一份 Review”误判成少跑了两份。两题参评 Proposal 的保存评审协议是 `stage-rubrics/v2`，不能用当前源码中 v9 的存在推断它们使用了 v9。

扫描的八个原生研究库均没有 Candidate Comparison / Idea Portfolio 的保存记录。这表明本批运行不能用来实证支撑当前多候选比较机制的收益；它不表示其他历史运行或当前系统没有这些能力。

### 原生剩余问题的归属

- AIIB：可确认的完成障碍是预算预留失败，不能据此归因为科研假设不合格。修复/续跑之前需要完整累计费用及冻结预算，而非重置运行。
- SmallARI：Plan 的 partial 指向 LAVa robustness 证据复核/登记及 PyramidKV 12% parity 的遗留义务。日志显示作者明确不能以文字修订代替独立复核，随后结为 inconclusive。要进一步判断这是合理义务还是调度闭环缺口，需要检查义务的可执行路由；本次不直接宣布应删除义务或降低门槛。
- Workshop：没有继续按“仍未产出 Plan”处理的理由。应把已保存的 Plan 和 Proposal 分别登记，按统一比较对象选择参评阶段。

机器核对记录：[evidence.json](evidence.json) 的 `native`；完成与恢复证据：[completion_and_replays.json](completion_and_replays.json)。

## 3. E-DARA：为什么旧外评不能直接解释为本体不诚实

### 3.1 可定位的失真链

原生 Proposal 的 claim 引用具体 `ev-*` → 导出后全部变为 `[REF]` → 模型再次归一化、抽取 prior_facts → 用评测侧的小文献池重新检索 → 缺少原始来源的判断被记为 fabricated_or_missing / smoothing。

去身份盲化是合理需求，但删除引用解析能力使科学来源一起丢失。应匿名化来源编号并保留映射，而不是让裁判猜测 `[REF]` 指的是谁。

### 3.2 具体例子与归因

| 原有负面判断 | 原生证据核对 | 本次归因 |
| --- | --- | --- |
| AIIB：E2VID 的 zero-shot UDA / 不针对目标域重训描述“无来源且误述” | 原生引用 ESS 的对照设置，原文明确描述 E2VID baseline、zero-shot UDA 和无法针对该目标域重训 | 评测缺原文上下文；不能用 E2VID 的一般介绍否定某篇论文中具体 baseline 的设置 |
| AIIB：去掉 embedding consistency 导致 5.56% mIoU 下降“可能编造” | ESS 保存引文直接给出 5.56% drop | 缺少原始文献证据造成的过强负面判定；仍须核对具体实验条件，不能把未检出当作编造 |
| AIIB：parrot-caption 与 CLIP text-spotting bias“无来源” | 原生 `Parrot Captions Teach CLIP to Spot Text` 的多条来源引文可定位 | 导出/检索证据缺失；不支持据此认定本体造假 |
| SmallARI：LAVa Figure 4 与 generation/extraction 的预算差异“可能捏造” | 原生 LAVa 引文含 Figure 4、dynamic layer/head budgets 及 generation/extraction 条件 | 原始来源丢失；外评归一化还给出了与原生论文标题不同的 LAVa 名称 |
| Workshop：KG-CoI、LLM-SRBench 的 31%、每问题 1k calls“未核实或编造” | 对应原生来源都存在；31% 与 1k calls 有直接来源引文 | 文献报告与本项目观测需区分；保存文本是在转述来源，不能直接算成伪造本项目结果 |
| Workshop：“Prior literature … requires …”把工作坊观点升格为文献共识 | 原始 claim 是 “The packet motivates … and it states …”；措辞升级出现在归一化的 prior_fact | 评价前文本变换污染。需要判断原文，不应让归一化新增断言成为惩罚对象 |

关于 E2VID 的 motion-invariant embedding：外评认为这属于 ESS 而不是 E2VID。原生 ESS 段落实际明确说它使用 E2VID encoder/decoder 构造该 embedding。原 Proposal 的“E2VID evidence”写法可能需要更清楚地区分来源论文与复用组件，但本次证据不足以认定这是独立的事实错误，更不能据此直接改写答案。

上述是基于保存材料的分析判断，不是独立人工金标准，也没有确认每条引文的全部推论范围、日期和实际结论。它足以推翻“查不到即造假”的归因，不能证明 E-DARA 已经完全诚实或优于基线。

### 3.3 设计缺失也需要先核对原文和阶段

旧 Spec Locator 的缺失列表包含真实可能重要的问题，如 SmallARI 的 task→budget 映射、Workshop 的 gate 决策规则；也包含易受阶段和归一化影响的要求。当前证据不支持把所有标为 blocking 的项目直接当作必须修改的原生缺陷。

AIIB 导入文本序列化后为 24,201 字符，归一化入口只取前 24,000 字符，末尾 201 字符及闭合结构被截断。这是具体的输入边界问题，但本次没有证据证明它造成了主要负面评分，不能将全部失分归因于截断。

## 4. 外挂：十五题逐题动作与评分变化

H/C/N 分别是诚实、可信、新颖；P/F/U 为 pass/fail/unresolved。以下只复述原评分，不把它们当作真值。SmallARI 编号对应原 task_id 中的 expansion40 编号。

| 题目 | 初稿澄清阻塞项数 | 完整外挂实际动作 | 正文变化 | Direct H/C/N → 完整外挂 H/C/N |
| --- | ---: | --- | --- | --- |
| AIIB-10541 | 3 | 搜证 | 无 | PPP → UPP |
| AIIB-33 | 3 | 搜证 | 无 | PPP → PPU |
| AIIB-4717 | 3 | 搜证 | 无 | PPU → PPP |
| AIIB-7144 | 2 | 搜证 | 无 | FPP → FPU |
| AIIB-82 | 3 | 搜证 | 无 | FUP → FPP |
| SmallARI-001 KV | 3 | 澄清、搜证 | 有 | UUU → PPP |
| SmallARI-002 Unlearning | 1 | 搜证、局部修改、搜证 | 有 | FPP → PUP |
| SmallARI-029 CBM | 3 | 搜证 | 无 | UPP → UPU |
| SmallARI-040 Actor-Critic | 2 | 搜证 | 无 | PPP → PPP |
| SmallARI-011 TTA | 3 | 搜证 | 无 | PUP → UPP |
| Workshop Agentic AI | 3 | 搜证 | 无 | UPP → PUU |
| Survey-06 | 2 | 搜证 | 无 | UUU → PUP |
| Survey-08 KV | 4 | 搜证 | 无 | PPP → PPP |
| Workshop Foundation Models | 3 | 搜证 | 无 | UPU → PPP |
| Workshop World Models | 3 | 搜证 | 无 | UUU → UUU |

总计 41 个初始 blocking 标签，全部十五题初始 overclaim_count 为 0。这些是审计器输出；标签本身还需要原文验证。

### 4.1 路由为何没有执行修订

`choose_action` 先返回贡献审计的 `seek_evidence`，随后才检查 overclaim、gap 和 spec。`apply` 补搜后如果还是 seek_evidence 就立即 break。因此，新颖性证据不确定会阻止独立的设计澄清。

四个真实零新增搜证案例（AIIB-33、Workshop Agentic AI、Survey-06、Survey-08）已用保存候选和审计状态重放：禁用所有模型/生成调用，返回无新增证据，得到与历史完全相同的退出轨迹。这确认控制流原因；不证明放开澄清后科学质量必然提高。

另有两题最终 reason 为 `invalid_core_span`，说明审计模型给出的核心片段不能匹配候选原文。该状态属于审计输出定位失败，当前同样被解释为继续搜证。应该优先做有界审计输出纠正或标记审计不可用，而不是假设缺少新论文。

### 4.2 两个实际改动案例说明什么

- SmallARI-001：澄清补充了 probe 计算和预算重分配规则、样本划分及指标。它证明 callback 修订能执行。但 Direct 的三个 U 都来自锚点降级，不能把 UUU→PPP 全部算成新增科学价值。
- SmallARI-002：补搜发现 `Evaluating Deep Unlearning in Large Language Models` 已涉及 deductive recovery，局部修改收缩原 gap、增加归因，同时保留联合评估方向。该轨迹支持“有据纠偏可以保留研究问题”的机制可行性；最终仍未联合达标，且出现 invalid_core_span。
- 没有任何 regenerate 动作。本轮既不能证明重新构思有效，也不能证明它无效。

## 5. Self-Refine 为什么更好，以及评价中哪些差异不可靠

### 5.1 已经能看到的实质内容变化

以下来自保存的自我批评与改写文本，是具体改进候选，不等于专家已经确认方案正确：

| 例子 | 原稿问题 | Self-Refine 的处理 | 完整外挂为何未做 |
| --- | --- | --- | --- |
| AIIB-4717 | 用已知诊断标签构造 complexity score，可能把真值泄漏到推理；S2/S3 的输出和调用预算不清楚 | 改为从推理时可见变量计算复杂度；明确 S2/S3 共享五组输出，只改变聚合 | 一轮搜证后仍未决，退出 |
| AIIB-82 | no-adapter 对照直接把 raw event features 与 CLIP text embedding 比较，空间/容量匹配不清楚；RGB 使用边界模糊 | 补充对照条件、event-only 范围、多 split/seed 与成功判据 | 搜证优先阻断了澄清 |
| World Models | object-centric 组件处在主张中心，却是 “if time permits”；少量模型上的相关性不足以支持强泛化表述 | 将必要组设为必做，收窄主张为探索性受控比较，分解预算 | 搜证优先阻断了澄清 |

所以 Self-Refine 的优势不宜简单解释成“更会润色”。已有保存文本显示它能进行机制、对照、测量和资源方面的修订。外挂需要争取的是更可靠的证据约束与动作选择，而不是人为削弱这个基线。

### 5.2 当前达标差异不能全算内容收益

- 完整外挂 13 份同文候选独立复评后，出现 15 个维度标签变化、4 个联合判定变化（2 升2降）。相同正文和共同证据下，这些不能归功于内容干预。
- Self-Refine 相对 Direct 有 7 个联合达标新增、2 个丢失。7 个新增中，5 题 Direct 所有非 pass 都是锚点校验导致的 U。因此 9/15 对 4/15 的差距仍需要在更可靠评价下确认。
- 69 个锚点降级中，68 个原始模型判断是 pass。这不意味着应该直接恢复 pass：模型可能仍然判错；它说明目前未决标签混合了科学未知和审计格式失败。
- AIIB-82、AIIB-7144 的诚实 fail 都部分来自 `supporting_result` 中的预设数值被当作已发生结果。原字段名和生成任务有前瞻含义，但句子用了现在时。应分别诊断“预注册支持条件表述不清”和“明确声称已经测得”，不能无条件把二者都判成造假，亦不能只因处在该字段就豁免真实造假。

### 5.3 检索相关性仍有问题

最终公共证据池中，Survey-08 KV 包含页面排版论文和房地产驱逐文章；World Models 包含版权出版和胸腔手术论文。代码仅依赖少量关键词重合判断相关性，容易受 page / eviction / video / model 等多义词影响。

这些来源可以有真实日期和 DOI，却不构成对应研究机制的近邻工作。应把日期/身份准入与科学相关性分开校验，并记录为何某来源能支持某个具体判断。不能因为“提供的几篇论文都没做过”就自动给新颖性通过。

## 6. 改进清单与验证方式

| 优先级 | 负责组件 | 证据与原因 | 通用改进 | 如何验证，包括失败条件 |
| --- | --- | --- | --- | --- |
| P0 | **参评适配与证据传递** | 三题引用全部被压成 `[REF]`，对应原生来源可追回；参评阶段为 Proposal | 记录精确版本/阶段/配置；用匿名可解析 ID 替代内码，保留 claim→来源映射。将各组合法来源纳入同一任务的冻结公共评价池，并统一检查时间/防泄漏边界；只改变新评测输入，不改历史文本 | 本次三题应能恢复全部已有引用；原句不变；各组裁判看到相同池；来源不合格仍标未决。不能只给 E-DARA 特殊证据待遇 |
| P0 | **评价与归一化** | 同文 15 次分项变化；69 个锚点降级；原文认知状态被改写 | 外评读取原始候选；字段级稳定定位；保留原响应，对无歧义格式差异做确定性校验，其他定位错误有界重试。科学未决与评审错误分列；相同任务/正文/证据/评价配置共用主评价，另做重复一致性分析 | 以保存响应离线重放；19 个纯首尾引号案例应可定位；伪造引文/实质改写不得误放行。旧标签不覆盖，不把缓存一致性伪称为裁判准确 |
| P1 | **外挂动作调度** | 十三份原稿有澄清项但未修改；四例可精确重放 | 新颖性继续未决时，仍允许处理独立、有原文依据的归因/设计问题。区分证据不足、原文定位错误、真正贡献覆盖；每类有界处理并保留未知 | 四个现成轨迹上应出现可解释的澄清机会；没有合格问题则保留原稿；不能靠把 novelty 状态改成 pass 触发流程。先离线检查，再做小规模模型复测 |
| P1 | **外挂检索和近邻比较** | 科学相关性弱的文章进入公共证据池 | 围绕机制/主张生成查询；对候选来源做相关性筛选和证据用途标记；无近邻证据时保持未决 | 保存的跨领域误匹配作为诊断样本；用新题检验召回代价；不能以大量删除文献人为降低重复检出 |
| P2 | **E-DARA 实验配置与运行** | 本批冻结 rubric v2、无新比较器回执；一题预算暂停，一题义务未闭合，一题完成 | 保持架构，明确用于下一次实测的冻结协议、预算和终点；针对 SmallARI 追踪既有复核义务是否有合法执行路径；复用已完成 Workshop 记录 | 记录实际启用机制、最终产物与累计成本；不得把未运行的新机制写入旧结果，不能重复生成直到只留下成功样本 |

本轮先交付归因和清单，没有执行这些生产修改。建议下一次实施范围定为前两项 P0，再做 P1 路由；完成后才适合依据新结果决定 E-DARA 是否需要实质机制调整和更大规模比较。

## 7. 证据、复现与限制

- [主分析脚本](analyze.py)：只用标准库，遍历八个原生研究库和 Stage B 保存文件；数据库以 `mode=ro`、`PRAGMA query_only=ON` 打开。
- [完成状态与路由重放脚本](verify_completion_and_routing.py)：读取已保存恢复回执，重放四个零新增搜证轨迹；模型入口被设为若调用即报错。
- [完整逐题结构化证据](evidence.json)：包括原生版本、claims 和引文、外评归一化文本、十五题动作、原始/校验后评分与锚点诊断。
- [完成状态与重放记录](completion_and_replays.json)：Workshop 的准确后续完成状态及四个路由重放结果。
- 主分析核对 245 个输入文件的前后 SHA-256 一致；补充分析也检查其输入未变。源码仓库保持原状态。
- 本次没有联网补论文、没有生成新 idea、没有重新判分，也没有声称上述内容核对等于独立专家评价。新颖性与效果仍需后续对照实验。

复现时为主分析指定新的输出目录，以保留已有分析：

```bash
cd /workspace/edara/staging/04_external_eval_refs/edara_mvp_eval
python analyses/failure_attribution_20261117/analyze.py --out /path/to/new-analysis-directory
../.venv/bin/python analyses/failure_attribution_20261117/verify_completion_and_routing.py \
  --evidence /path/to/new-analysis-directory/evidence.json \
  --out /path/to/new-analysis-directory
```

两份脚本均以独占写入保存结果。全部分析均不要求模型凭据。
