# 交接说明

日期：2026-11-18　｜　上一阶段由 Codex（额度耗尽）转 DeepSeek Harness 接手完成

> **最新状态（2026-11-18 傍晚）**：本仓库已包含 claim 级事实审计 pass 与两处修复。诚实轴仪器修复（校准 12/12 + 6/6）；Step 1 召回 7/9、落实 7/7；Step 2（27 份，修复前）可信 +0.222 [0.014, 0.417]、诚实 +0.104、R−E +0.077、新颖 0，2 条新增缺陷；之后修复了强制落实字段映射与槽规则，**held-out 尚未验证**。完整接管文档见 [`docs/HANDOFF_20261118.md`](docs/HANDOFF_20261118.md)，Step 2 结果见 [`docs/STEP2_RESULT_20261118.md`](docs/STEP2_RESULT_20261118.md)，数据见 [`results/20261118_claim_audit/`](results/20261118_claim_audit/)。


## 0. 一句话

治理外挂已在 38 份冻结原稿上完成同稿配对试验：C 对 B 可信 **+0.222 [0.089, 0.344]**，E 对 D **−0.167 [−0.322, −0.044]**，新颖度无变化。诚实轴仪器随后修复并以 12/12 + 6/6 通过校准，但自然稿 58–69% 未决。2026-11-18 下午定位到审计的断言覆盖缺口，加入 claim 级事实审计 pass；Step 2（27 份）显示可信保持、诚实方向转正（R−E +0.077），但预注册判据未全过，两处副作用已修复，**held-out 待验证**。协议 v1.2 保持。

## 1. 仓库与数据位置

| 内容 | 位置 |
|---|---|
| 本仓库（外评与治理外挂） | `github.com/xixi501/edara-external-eval`（私有） |
| 实验服务器工作副本 | `/workspace/edara/staging/04_external_eval_refs/edara_mvp_eval` |
| E-DARA 主体源码 | `github.com/xixi501/edara`，分支 `codex/trusted-idea-frontier-20261110`，HEAD `848cab2a5f` |
| 正式试验运行目录 | `/workspace/edara/staging/04_external_eval_refs/edara_mvp_eval/outputs/paired_v1_20261118_v12` |
| 原始输出与请求记录 | 同上，体积约 3.7G，不随仓库分发 |

工作区规则：`/workspace/edara` 本身不是 Git 仓库；`staging/` 不属于 Git。本仓库是从 staging 整理出的独立副本，不改变原工作目录。

## 2. 这一轮做了什么

### 2.1 仪器修复（协议 v1.2）

旧仪器的问题：评委输出格式稍有偏差就整份作废（诚实轴 98% 未决）；两轮校准与正式试验用的不是同一套投票程序；校准门禁在两次抽样间会翻转；编辑补丁重叠会整臂回退；追加上限 240 字符卡掉了 22 次合法修改。

v1.2 的改动（详见 `docs/EVAL_PROTOCOL_V1_2_ZH.md`）：

- 三次独立判断取多数（正序、倒序、正序重采样），校准与试验同一程序；
- 比较配对差而不是各臂绝对等级；
- 未知带宽只计不对称未知；
- 评委输出逐条机械归一化（字段漂移、四种引用写法、缺失 ID、标签缺失、删除但仍在、重复新增），每次修复记入 `repairs`；
- 追加上限 240 → 400 字符；
- 审计预算只统计实际执行的调用，缓存复用单独计量。

交接版另外做了四处**路径端口化**，判读逻辑未改：`WORKSPACE` 支持 `EDARA_WORKSPACE` 环境变量；协议文件可从 `docs/` 读取；输出目录约束在 `outputs/` 不存在时放宽；`freeze` 在 `data/inputs_<cohort>.json` 存在时直接复制冻结输入。因此交接版不能用于续跑旧 manifest 目录，只用于新建目录的复现。

### 2.2 新增混淆项审计（治疗变更）

C/E 臂的审计新增 `confound_audit`：针对核心假设列出能产生同一观测的替代解释，并给出能分开二者的具体实验臂。诊断排第一位；编辑未落实时，程序发起一次独立调用强制补齐（≤400 字符追加，独立预算，不占四处修改名额），并把 `addressed` 记入 `confound_enforcement`。

两轮 pilot 验证：aiib-7144 与 aiib-10541 的可信轴从 −2 负翻为平，评委的 control 检查从 no 变 yes。

### 2.3 正式试验

38 份冻结原稿（Direct 15、AI-Scientist 14、E-DARA 9，缺 7 份），五臂、两组配对、三轴判读。结果见第 3 节。

## 3. 结果

完整表格与解读见 [`results/RESULTS_20261118_ZH.md`](results/RESULTS_20261118_ZH.md)，原始汇总见 `results/core_summary.md`，逐稿判定见 `results/per_manuscript.csv` 与 `results/core_results.json`。

| 比较 | 轴 | 胜/负/平/未决 | 净胜率 | 95% 区间 |
|---|---|---|---:|---|
| C vs B | 可信 | 10/2/24/2 | +0.222 | [+0.089, +0.344] |
| C vs B | 新颖 | 3/1/30/4 | +0.044 | [−0.044, +0.156] |
| E vs D | 可信 | 1/7/28/2 | −0.167 | [−0.322, −0.044] |
| E vs D | 新颖 | 1/4/23/10 | −0.060 | [−0.167, +0.060] |

剔除 3 个开发题后：C vs B 可信 +0.278 [+0.153, +0.389]；E vs D 可信 −0.208 [−0.389, −0.056]。

独立核验：`results/verification.json`，原稿未改、证据一致、缺失 7 份、预算合规。执行：B/C/E 各 38/38 改稿，D 37/38；异常 2 次。费用约 $3.5（记录值含复用重复计量，见核验备注）。

## 4. 已经能下的结论

- 仪器可用：机械失败清零，未决主要来自评委分歧与证据不足。
- 结构化审计（C vs B）对可信度有小幅正向，合并区间不含零；三个宿主方向一致，单宿主区间多跨零。
- 完整工作流（E vs D）对可信度为负向，剔除开发题后 0 胜 6 负；整稿重构比局部修补更容易丢掉显式对照或判败规则。
- 新颖度没有可报告的变化。诚实轴缺数据。

## 5. 未决与风险

| 编号 | 问题 | 影响 | 建议 |
|---|---|---|---|
| R1 | 诚实轴校准未过（9/12 与 4/6），本轮无诚实数据 | 论文的诚实性主张缺直接数字 | 先修三处机械问题（证据格式、`@1` 编号映射、same 允许改写锚点），重跑校准；过了再补判 38 份诚实轴，约 $0.6 |
| R2 | 「相对原稿的绝对等级」的基线由独立调用测得 | 绝对水平的升降含测量噪声 | 报告里标明为描述性指标；若要用，把原稿放进配对判读重测 |
| R3 | 可信量表是二值的，丢一个要素掉 1–2 级 | 等级差会放大措辞变化 | 已有逐检查项记录，可做要素级分析 |
| R4 | extension（25 份 Direct 单宿主复核）未跑 | Direct 的证据只有 15 份 | 主样本结论稳定后再跑，约 $2 |
| R5 | 隐藏变量：评委模型单一（GLM-5.3-flash） | 跨模型泛化未验证 | 下一轮换评委复判 |
| R6 | E 臂整稿重构与 D 的差异未消融 | 无法归因到具体动作 | 加「只保留/只修补/只重构」的动作消融 |

## 6. 接手后的建议顺序

1. 修 R1 的三处机械问题，重跑校准，补判诚实轴。
2. 用 `results/per_manuscript.csv` 做要素级分析（哪个检查项在哪些宿主上掉分）。
3. 视结果决定是否跑 extension 与评委换模型复判。
4. 若要写论文，先锁协议与结果，再动图表。

## 7. 运行手册

### 环境

- Python 3.10+，`pip install -e .`；外部依赖 `alphaideabench`（本地 checkout）与 ClaimNov（路径注入，见 `src/edara_mvp_eval/claimnov_adapter.py`）。
- 凭据通过环境变量注入，不入库：`DEEPSEEK_API_KEY`（编辑器）、`AUTODL_API_KEY`（评委/审计）。
- 单元测试：`pytest -q`（108 项）。

### 运行

```bash
# 新目录冻结 → 校准 → 正式试验
python pilots/run_paired_isolated.py --phase freeze --out outputs/paired_<date> --calibration-set reserve
python pilots/run_paired_isolated.py --resume --phase calibrate --out outputs/paired_<date> --calibration-set reserve
python pilots/run_paired_isolated.py --resume --phase core --out outputs/paired_<date> --calibration-set reserve
# 独立核验
python pilots/verify_paired_trial.py --run outputs/paired_<date> --cohort core
# 导出逐稿表
python analyses/export_results.py outputs/paired_<date> results/per_manuscript.csv
```

- 长任务用命名 tmux 会话，日志重定向到文件；本轮用 `edara-v12`。
- 成本：38 份正式试验约 $4–5；校准约 $0.2；38 份诚实轴补判约 $0.6。

### 协议版本

- `docs/EVAL_PROTOCOL_V1_ZH.md`：v1，已冻结，历史结果绑定它。
- `docs/EVAL_PROTOCOL_V1_2_ZH.md`：当前版本。manifest 记录 `protocol_version` 与文件哈希，改动必须开新目录。

## 8. 红线

- 不覆盖或重写已有运行目录、原始响应、失败记录和费用。
- 不修改 `repos/framework` 的源码（外评只读使用其产物）；E-DARA 主体变更走它自己的仓库与规则。
- 不提交凭据、模型缓存、PDF、真实 run、日志。
- 校准未过的轴不发布正式效果数字。
- 同一份稿件不被重复计入分母；缺失样本进 `missing`，不进分母。
