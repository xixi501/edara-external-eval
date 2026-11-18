# E-DARA—外挂—论文贡献—现有证据对应表

核对日期：2026-11-17。E-DARA 源码基线：`848cab2a5f`，分支 `codex/trusted-idea-frontier-20261110`。本次仅梳理代码、协议和已有记录；没有修改系统或外挂实现，没有启动新实验。以下代码能力不表示每个历史 run 都启用了相应协议。

后续逐题核对更新：三份 E-DARA 参评文本均已精确映射到原生 Proposal；引用匿名化为 `[REF]` 导致外评证据缺失，旧过度断言减少数不能直接解释为修复了同等数量的真实失实。Workshop 后续恢复已完成 `plan_reviewed`，参评仍使用其 Proposal。具体证据、十五题路由重放及改进优先级见[失败归因报告](../analyses/failure_attribution_20261117/REPORT_ZH.md)。下文历史评分保留原值，其解释以该核对结果为补充。

## 1. 研究主线与目标

研究目标是产出诚实、可信、同时具有一定实质新颖性的科研 idea。E-DARA 是完整系统实现；独立外挂是对其中可迁移治理思想的抽象和独立实现，同时吸收了调研中的方法。两者允许实现不同，但不能把外挂新增能力倒推成 E-DARA 已有且已验证的能力。

论文需要分别检验：

- 系统有效性：在共同任务、可比较资源和独立外部评价下，E-DARA 是否优于基础模型、强增强基线和相关科研 Agent？
- 迁移有效性：保持宿主不变，外挂是否带来增益；固定外挂核心机制，增益是否跨模型、跨 Agent 保持？
- 机制解释：若存在增益，哪些治理机制起作用，付出了多少成本，是否牺牲研究价值或新颖性？

现有证据尚不足以把前两项写成已成立的实验结论。

### 三个目标的现有操作口径

| 目标 | 当前可以评价的内容 | 不能直接推导的结论 |
| --- | --- | --- |
| 诚实 | 事实、推断、假设与预期结果是否区分；已有工作是否归因；关键未知是否披露；是否出现有据的夸大 | 模型具有诚实意图；所有未验证陈述都为假 |
| 可信 | 关键前提有证据或被明确标为假设；机制自洽；检验能区分核心主张；资源要求合理 | 尚未执行的实验已经成功；研究结论已经被实证证明 |
| 一定程度的新颖性 | 相对所提供的截止日前相关文献，是否存在实质且可检验的区分 | 全球首次；一定有高科学价值；文献中未检出就证明新颖 |

`pass / fail / unresolved` 沿用 Stage B 口径。三项联合达标与三个分项一起报告。内部 Gate、Strict Gate、Spec Completeness 是诊断指标；Method Retention 是文本词项保留指标，不是科学新颖性或机制语义保留的证明。

## 2. 共同思想与实现对应

| 共同治理思想 | E-DARA 当前实现 | 外挂当前实现 | 关系与重要差异 | 能提出的贡献主张及证据状态 |
| --- | --- | --- | --- | --- |
| 区分事实与待检验假设，保留未知 | 原生 Review 明确不要求假设已经被证实；修订路由标注 fact / inference / hypothesis / design_requirement；分阶段继承 carried_limits | 三态结构 prior_facts / prospective_hypotheses / unknown_dependencies；未知依赖包含解决路径、最小校准检验和 fallback | 思想对应，字段和执行方式不同；外挂三态数据结构不能说是直接搬出了原生同名组件 | 候选贡献：在纠正事实性夸大的同时保留可检验假设。三题有审计改善信号，尚无对应机制的独立因果消融 |
| 论断和批评都要绑定证据 | 文献边界过滤、正文/证据原子准入；Review 和修订责任绑定版本、字段及 evidence ID / packet pointer | 截止日前文献准入；贡献审计核对 proposal_span 与 evidence_quote；无有效锚点时保留未决 | E-DARA 有完整内部证据与版本结构；外挂从外部文本重新抽取。精确引文核对能防止虚构锚点，但不能确定性证明语义蕴含 | 候选贡献：减少无根据肯定和无根据否定。代码路径已存在；Stage B 也暴露锚定失败与证据不足，尚不能声称审计准确率已得到真实标注验证 |
| 区分旧成分与仍值得检验的实质差异 | Candidate Comparator 的 duplicate / unconfirmed / contrast_supported 与 ContrastBasis；可读取候选未直接引用的 corpus_evidence；另有可选 Idea Portfolio | ClaimNov 适配、组合余项分析；core / component 分开审计；covered / distinct_hypothesis / unresolved 决策 | E-DARA 比较多个候选并安排后续研究；外挂主要评估单个宿主输出。外挂对多文献组合的显式审计不是原生比较器的逐行抽取 | 候选贡献：治理既有贡献归因，同时保留有据的待检验差异。当前缺乏充分证据证明提升了实质新颖性；组合关系判断仍依赖模型 |
| 按问题类型选择干预，而非只给一个总分 | 独立 Review 后按 plan_local / evidence_search / upstream_change / external_limit / user_scope 路由；允许有依据的上游重开 | retain_and_test / seek_evidence / local_patch / clarify / regenerate；生成反馈通过 callback 返回宿主适配器 | 共同思想是分型反馈；原生处理跨阶段责任，外挂处理单候选外部反馈。当前实际 Stage B callback 只接入了大模型生成器 | 候选贡献：把诊断转为可执行的治理动作。代码路径存在；十五题无 regenerate，且全部最终停在 seek_evidence，核心重构收益未获实测验证 |
| 干预时保留仍有效的研究内容 | 不可变版本与父版本依赖，字段定位修订，跨阶段 fidelity 检查 | 每轮至多四处局部 span 修改、独立澄清追加通道、文本保留护栏；反馈重构另走 callback | 原生没有同样的四处补丁限制。高文本重合只说明少改字；必要的核心重构不能被一律视为失败 | 候选贡献：控制不必要改动，同时允许有依据的实质修订。三题文本保留较高；未证明科学机制、新颖性或研究价值被同等保留 |
| 生成、评判与记录分离 | Author / Reviewer / Comparator / Policy 分工；独立输入包；阶段评审与仲裁；数据库版本与操作账本 | 生成与审计接口分离；冻结候选后用共同证据池外评；保存候选历史、请求和费用，配对换序 | 外挂轻量，不具备完整原生权限与状态机。Stage B 内部审计与最终评价仍用同一 GLM 模型，输入隔离不等于裁判误差独立 | 工程基础已建立，可支持复查和失败分析；本身不是科研质量提高的证据 |

上述对应是当前实现的概念和代码对照，不是已经完成逐模块来源追踪的证明。若论文要叙述具体模块的历史提炼过程，还应绑定当时版本和开发记录。

### 对应代码入口

- 原生总体结构：[ARCHITECTURE.md](/workspace/edara/repos/framework/docs/ARCHITECTURE.md)。
- 原生认知边界、分阶段要求与独立评审：[reviews.py](/workspace/edara/repos/framework/src/edara/framework/research/reviews.py)、[stage_rubrics_v8.py](/workspace/edara/repos/framework/src/edara/framework/design/stage_rubrics_v8.py)、[stage_rubrics_v9.py](/workspace/edara/repos/framework/src/edara/framework/design/stage_rubrics_v9.py)。
- 原生新颖性比较：[candidate_comparison.py](/workspace/edara/repos/framework/src/edara/framework/research/candidate_comparison.py)、[idea_portfolio.py](/workspace/edara/repos/framework/src/edara/framework/research/idea_portfolio.py)。两种协议要依据具体 run 的冻结配置区分，不能把所有能力累加成每次都启用。
- 原生问题路由：[repair_routing.py](/workspace/edara/repos/framework/src/edara/framework/research/repair_routing.py)。
- 外挂三态审计与结构：[epistemic_audit.py](/workspace/edara/staging/04_external_eval_refs/edara_mvp_eval/src/edara_mvp_eval/epistemic_audit.py)、[schemas.py](/workspace/edara/staging/04_external_eval_refs/edara_mvp_eval/src/edara_mvp_eval/schemas.py)。
- 外挂贡献审计、干预与反馈：[governance.py](/workspace/edara/staging/04_external_eval_refs/edara_mvp_eval/src/edara_mvp_eval/governance.py)、[middleware.py](/workspace/edara/staging/04_external_eval_refs/edara_mvp_eval/src/edara_mvp_eval/middleware.py)、[feedback.py](/workspace/edara/staging/04_external_eval_refs/edara_mvp_eval/src/edara_mvp_eval/feedback.py)。
- 外挂现有评测入口：[stage_b.py](/workspace/edara/staging/04_external_eval_refs/edara_mvp_eval/src/edara_mvp_eval/stage_b.py)、[stage_b_review.py](/workspace/edara/staging/04_external_eval_refs/edara_mvp_eval/src/edara_mvp_eval/stage_b_review.py)。

## 3. 论文贡献与现有证据

| 拟主张的贡献 | 已有直接证据 | 当前可以说 | 尚缺什么 |
| --- | --- | --- | --- |
| C1：一个面向诚实、可信及一定新颖性的完整科研设计 Agent | 原生多阶段实现；三题导入产物；原生运行与失败记录 | 已实现相应系统和治理流程，已有探索性比较 | 可追溯的原生产物、实际启用机制、闭环状态与统一外评；相对强基线和外部 Agent 的可靠多题优势 |
| C2：将共同治理思想实现为可外接组件 | 外挂包独立于原生状态机；可处理 Direct、AI-Scientist-v2、E-DARA 的导入文本；已有生成 callback | 已实现可适配的外部审计与反馈原型，已验证多来源文本接入 | 固定机制在多个基础模型上的增益；外部 Agent 实际接收反馈并修订的闭环证据；接入成本与依赖边界 |
| C3：兼顾诚实可信与新颖性的干预机制有效 | 三题局部修改结果；十五题五组实验及完整负面结果 | 有初步改善信号和明确失败边界，已识别路由问题 | 可靠外评下的稳定净增益；等预算强基线；关键机制消融；独立确认题集 |

这些是待验证的贡献表述，不能把“具有实现”替换为“取得优势”，也不能把多个已有方法的组合本身直接称为学术首创。

### Stage A 三题：多来源输出的初步干预证据

以 `pilot3_full_compare_v4` 为准，不能与更早 v3 数值混用：

- 模型审计判出的过度断言数量：E-DARA 导入文本 12→0；AI-Scientist-v2 导入文本 7→1；Direct 2→2。
- 局部外挂对各自原稿的三题盲评均为 3–0；但 Direct 外挂对 Self-Refine 为 0 胜、2 负、1 平。
- E-DARA 导入原稿与 Direct 的盲评为 0–3，当前这些结果不支持 E-DARA 优势。
- 协议记录指出该轮 24 个最终产物的贡献审计均无合格准入证据；Strict Gate 全为零、ClaimNov veto 全为零。不能据此确认新颖性或进行稳定门禁排序。
- 这验证了外挂对不同来源文本的处理，不等于外部 Agent 已在自身工作流中响应外挂反馈。

来源：[v4 summary](/workspace/edara/staging/04_external_eval_refs/edara_mvp_eval/outputs/pilot3_full_compare_v4/summary.md)、[v4 results](/workspace/edara/staging/04_external_eval_refs/edara_mvp_eval/outputs/pilot3_full_compare_v4/results.json)、[Stage B 协议中的历史指标解释](/workspace/edara/staging/04_external_eval_refs/edara_mvp_eval/STAGE_B_PROTOCOL.md)。本次直接从 results 核对了六组过度断言总数。

### Stage B 十五题：同一基础生成器上的五组比较

| 组别 | 全部十五题联合达标 | 新增十二题联合达标 |
| --- | ---: | ---: |
| Direct | 4/15 | 4/12 |
| Self-Refine | 9/15 | 7/12 |
| Retrieve-and-Revise | 5/15 | 4/12 |
| 局部外挂 | 4/15 | 2/12 |
| 完整反馈外挂 | 5/15 | 4/12 |

本次从 `results.json` 重新汇总了上表。完整反馈外挂与 Direct 的净差不足以支持稳定增益；报告中的配对 bootstrap 95% 区间为 −20.0 至 +33.3 个百分点。Self-Refine 是必须保留的强基线。

结项分析还显示：完整反馈外挂只改动了 2/15 份正文；13 份同文产物的独立复评中，4 份联合判定翻转，2 升2降。不能把同文评分变化算成干预的内容收益。十五题全部最终处于继续搜证状态，核心重新构思未触发；搜证优先及提前退出可能阻断本可进行的诚实纠偏和设计澄清。

这轮已完成 75 份候选评价，但没有 E-DARA 或外部科研 Agent 的十五题原生比较。它也不是跨基础模型实验。不同题目来源不等于不同宿主。

来源：[结项报告](/workspace/edara/staging/04_external_eval_refs/edara_mvp_eval/outputs/mvp15_ablation_v2/REPORT_ZH.md)、[逐题结果](/workspace/edara/staging/04_external_eval_refs/edara_mvp_eval/outputs/mvp15_ablation_v2/results.json)。

### 原生 E-DARA 产物与资格对应（后续核对已补齐）

已核对 `edara_outputs.jsonl` 有三个唯一 task_id。后续只读逐库比对确认，三行都来自原生 Proposal：移除 `rq_coverage`、遮蔽 `ev-*` 引用后，其余字段完全匹配。不能仅凭行数认定三题都完成了原生 Plan 闭环。

- 现有 `collect.py` 要求精确终态 Plan、完整版本依赖、独立评审与验收回执；目前该目录没有其 `acceptance.json`，现有导入行的格式也不同于它写出的 `proposal_text` 格式。逐库精确匹配已提供另一条产物追溯证据。
- SmallARI 对应库最终为 `inconclusive`，存在带未决义务的 Plan；AIIB 导入文本来自 `aiib82_arxiv_run`，其 Policy 为 `paused_budget`，没有该 Proposal 的 Review。`arxiv_run2` 是另一分支，不是该导入文本的来源。
- Workshop 的旧 round6 日志为预算暂停；后续 `ws-budget-resume.log`、对应库与交付快照一致记录 `plan_reviewed` / `qualified_plan_available`，已完成原生 Plan 闭环。当前三题外评仍使用 Proposal，不能解释为三个终态 Plan 的比较。
- 论文应分别报告生成产物、审查完成、原生资格与外部质量；内部严格通过不是全部样本能否接受外评的前提。

来源：[导入产物](/workspace/edara/staging/04_external_eval_refs/pilot3_runs/edara/edara_outputs.jsonl)、[原生导出检查器](/workspace/edara/staging/04_external_eval_refs/pilot3_runs/edara/collect.py)、[SmallARI 日志](/workspace/edara/staging/04_external_eval_refs/pilot3_runs/edara/smallari-round3.log)、[Workshop 日志](/workspace/edara/staging/04_external_eval_refs/pilot3_runs/edara/ws-round6.log)、[AIIB outcome](/workspace/edara/staging/04_external_eval_refs/pilot3_runs/edara/aiib82_arxiv_run2/outcomes/aiib-82.json)。

## 4. 应如何解释“从系统提炼外挂”

现阶段适合表述为：以 E-DARA 的证据约束、未知保留、独立评审和分型反馈为共同设计基础，结合调研中的贡献归因与局部修订方法，形成外部治理实现，并分别检验系统效果与跨宿主效果。

需要明确三类差异：

1. E-DARA 特有：多阶段 Idea→Method→Proposal→Plan、内部证据原子与版本树、多候选编排、阶段评审/仲裁、跨阶段责任与运行控制。
2. 外挂独立实现或新增：从任意文本抽取三态声明、显式 core/component 贡献审计、多文献组合余项分析、四处局部补丁、外部 callback 接口。
3. 公共评价基础：共同证据池、冻结候选、双向盲评与费用记录。这些是检验机制效果的工具，不能同时作为机制自证有效的理由。

当前“不绑定某个模型/Agent”主要是接口层面的设计目标和初步事实，不是已完成全面迁移验证，也不是不使用模型。`claimnov_adapter.py` 仍依赖本地 ClaimNov 源码路径和适配；实际部署还需要统一候选字段、证据格式及反馈入口。

调研文件中已有 RINI、ClaimNov、Lit2Test、IdeaAMBIG 等方法启发，外挂代码也显式复用 ClaimNov 并在 prompt 中注明启发来源。因此，可检验的候选创新点应聚焦于：如何将证据支持、认知状态和贡献差异联合用于干预选择，使宿主纠正失实表述并继续探索。其相对已有工作的独特性还需专门文献核验；本次未把调研中的外部效果数字作为已核实事实引用。

## 5. 由对应表得到的下一步实验任务

| 任务 | 直接要回答的问题 | 应保留的证据 |
| --- | --- | --- |
| 对齐 E-DARA 导入文本与原生版本，再补齐共同题目的缺失运行 | C1 的参评对象究竟是什么；系统是否有优势 | 源码/配置版本、输出版本、实际启用机制、交付状态、失败、累计费用、共同外评 |
| 改进外挂通用路由与评价一致性 | 有哪些问题可在新颖性未决时仍被有效纠正；收益是否只是评审波动 | 旧稿、新稿、诊断、动作、反馈、证据池；相同任务/正文/证据/评价配置的同文共评或一致性分析 |
| 固定外挂，接入第二基础模型和一个真实外部 Agent 反馈入口 | C2 是否超出“能读不同来源的文本” | 接入适配、宿主实际响应、改写与重生成次数、成本、原版及等预算 Self-Refine 对照 |
| 在实现冻结后用新题确认，做小规模关键机制消融 | C3 的效果能否复现，是否由声称的机制产生 | 全任务分母、三轴与联合结果、不确定性、失败类别、成本及消融差值 |

这些任务不以人工标注到位为启动条件。当前可用自动证据锚定、冻结独立外评、换序/重复一致性分析与可确定真值的受控案例推进；受控案例只能支持对应机制行为，不能替代开放科研题目的真实质量证据。论文结论应明确限定于所用自动评价和文献范围。

当前最重要的两处缺口是：E-DARA 本体缺少可靠的统一比较证据；外挂已有的较大一轮实验尚未显示稳定优于强基线。后续工作应分别补这两条证据链，保留负面结果，不通过修改单题答案或放宽结果定义来制造成功。
