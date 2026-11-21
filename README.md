# E-DARA 外部治理与统一评测外挂（edara-external-eval）

本仓库是 E-DARA 研究生成治理思想的独立外挂实现与统一对比评测框架（Trustworthy-Novelty Middleware + Unified Evaluation Harness），可独立作用于 `direct`、`AI-Scientist-v2` 与 `E-DARA` 冻结原稿（含 `proposal` 与 `experiment_plan`），并以 **Lit2Test（五维位置去偏盲评）+ AlphaIdeaBench（结构维度）+ ClaimNov（组合新颖度）** 评测修订收益。

E-DARA 主体框架代码位于 [`xixi501/edara`](https://github.com/xixi501/edara)（`codex/trusted-idea-frontier-20261110`）。

---

## 核心组件

- `src/edara_mvp_eval/host_adapter.py`：多宿主字段适配（`filter_proposal_fields`、`_ROLE_BASES`、`to_lit2test_card`、`to_alphaideabench_idea`、`confound_addressed` 与 `collect_design_text` 同源对齐，以及 `_title_anchor_tokens` / `preserves_related_work_anchors` 防误删已入库文献护栏）。
- `src/edara_mvp_eval/confound_audit.py`：假设特异性替代解释 / 混淆因子与鉴别对照臂审计。
- `src/edara_mvp_eval/factual_audit.py`：Claim 级事实审计、未入库外部命名实体扫描（`audit_unverified_entities`）、跨字段定量阈值与证伪一致检查（`audit_draft_consistency`）、残余未执行 `pre-registered` 断言拦截，以及高信息优先证据切片（`_prioritize_passages`）。
- `src/edara_mvp_eval/epistemic_audit.py` & `spec_locator.py` & `governance.py`：前瞻边界区分、证据边界准入及由 `factual_audit` 派生零冗余 `_tri_state_from_factual`。
- `src/edara_mvp_eval/feedback.py` & `middleware.py` & `paired_trial.py`：五动作状态路由（含 `guided_revise` 多字段协同修订、预算守恒与结构/文献锚点保留护栏、单次恢复重试、`clarify` 就地澄清放行及搜证后强制落实）。
- `src/edara_mvp_eval/unified_eval.py` & `claimnov_adapter.py` & `llm_client.py`：`Lit2Test` 正反互换去偏五维评判、`AlphaIdeaBench` 批量五维打分与 `ContextVar` + facet 缓存并发 `ClaimNov`。

---

## 最新实证结果（2026-11-20：`N=68` 三裁判汇总 & `ext15` 扩样本 & `Unified v4`）

- **全量 30 题合并总表（`N=68` = `v4` 38 + `ext15` 30，0 题重叠，0 未决）**（[`results/extension_15_20261120_qwen3/crossjudge_3way_ext15_and_combined68.md`](results/extension_15_20261120_qwen3/crossjudge_3way_ext15_and_combined68.md)）：
  - **三裁判多数决（`E_vs_D`）**：`38/17/13`，题目加权净胜率 **`+0.283`**（`95% CI [+0.067, +0.483]`），五维总分差 **`+0.938`**（`95% CI [+0.157, +1.636]`）；分裁判净胜率分别为 GLM-5.3-flash `+0.294`（`39/18/11`）、DeepSeek-v4.1-flash `+0.244`（`37/19/12`）、Qwen3.8-flash `+0.322`（`38/15/15`）。
  - **客观机制台账（`N=68`）**（[`results/extension_15_20261120_qwen3/mechanism_audit_stats_combined68.json`](results/extension_15_20261120_qwen3/mechanism_audit_stats_combined68.json)）：虚假流程包装词残留 `D: 17 稿 32 处` → `E: 0/0`；未核验外部实体 `D: 367` → `E: 169`（`-54.0%`）；核心三要素齐备率 `E: 68/68 (100%)`；混杂控制臂落入设计字段 `E: 36/38 (94.7%)`；原宿主字段结构保留 `68/68 (100%)`。
- **独立 15 题扩样本 `ext15`（`N=30`）**（[`results/extension_15_20261120_qwen3/extension_summary.md`](results/extension_15_20261120_qwen3/extension_summary.md) | [`mechanism_audit_stats_ext15.json`](results/extension_15_20261120_qwen3/mechanism_audit_stats_ext15.json)）：Qwen3.8-flash（`16/16` 满分校准，[`calibration.json`](results/extension_15_20261120_qwen3/calibration.json)）主评 `E_vs_D` `16/9/5/0`（净胜 `+0.233`，总分差 `+0.867`）；三裁判多数决 `15/10/5`（净胜 `+0.167`）。
- **Unified `v4` 主队列（`N=38`）及三裁判复核**：
  - **统一校准**：`10/12`（substantive）+ `4/4`（sham ties），`qualified = True`（[`results/unified_pilot_20261119_v4/calibration.json`](results/unified_pilot_20261119_v4/calibration.json)）；三裁判多数决 `23/7/8`，净胜率 **`+0.400`**（[`results/unified_pilot_20261119_v4_qwen_judge/crossjudge_3way_full38.md`](results/unified_pilot_20261119_v4_qwen_judge/crossjudge_3way_full38.md)）；机制审计台账见 [`results/unified_pilot_20261119_v4/mechanism_audit_stats_v4.md`](results/unified_pilot_20261119_v4/mechanism_audit_stats_v4.md)。
  - **全量 `core` 队列（15 个 topic / 38 份冻结原稿，0 未决）**（[`results/unified_pilot_20261119_v4/core_summary.md`](results/unified_pilot_20261119_v4/core_summary.md)）：
    - **`E_vs_D`**：Lit2Test 胜/负/平/未决 **`25/7/6/0`**，净胜率 **`+0.489`**（`95% CI [+0.200, +0.744]`），五维总分差 **`+1.383`**（`95% CI [+0.350, +2.228]`），`ClaimNov +0.005`。五个分维度全部为正（`falsifiability +0.594`、`decisive_metric +0.444`、`minimality_feasibility +0.289`、`grounding +0.050`、`hypothesis_specificity +0.006`）。
    - **纯 held-out 29 份（剔除 3 题 pilot）**：`E_vs_D` **`20/7/2/0`**，净胜率 **`+0.472`**（`95% CI [+0.125, +0.778]`），总分差 **`+1.299`**。
    - **分宿主（`E_vs_D`）**：`ai_scientist` **`13/1/0/0`**（净胜 **`+0.857`**，总分差 **`+2.786`**）；`edara` **`7/1/1/0`**（净胜 **`+0.667`**，总分差 **`+1.389`**，其中 `experiment_plan` **`4/0/1/0`** `+1.800`）；`direct` `5/5/5/0`。
    - **`C_vs_B`（≤4 处局部补丁消融对照）**：`6/14/18/0`（净胜 `-0.244`，总分差 `-0.617`），与 `E_vs_D` 拉开 `+0.733` 净胜率差距。
  - **`pilot` 子集（9 份 / 3 题）**（[`results/unified_pilot_20261119_v4/pilot_summary.md`](results/unified_pilot_20261119_v4/pilot_summary.md)）：`E_vs_D` **`5/0/4/0`**（净胜 **`+0.556`**，总分差 **`+1.722`**）。

---

## 目录结构

```text
src/edara_mvp_eval/      # 外挂治理引擎、多宿主适配与统一评测器
tests/                   # 离线单元测试套件（129 项）
pilots/                  # 运行入口（run_unified_eval.py, run_paired_isolated.py, verify_paired_trial.py 等）
configs/                 # 模型端点 profile（不含任何 API key，凭据通过环境变量注入）
data/                    # 冻结任务、输入快照与校准用例
docs/                    # 评审协议与证据映射文档
results/                 # unified v4、ext15 (N=30)、N=68 三裁判汇总、校准与客观机制台账产物
```

---

## 测试与复现

```bash
# 运行离线单测
PYTHONPATH=src pytest -q

# 运行统一评测校准与对照（需设置模型凭据环境变量及依赖包 alphaideabench / claimnov）
python pilots/run_unified_eval.py --phase all --out outputs/unified_run
```
