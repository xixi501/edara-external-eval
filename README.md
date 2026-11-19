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

## 最新实证结果（Unified v4，2026-11-19）

- **统一校准**：`10/12`（substantive）+ `4/4`（sham ties），`qualified = True`（[`results/unified_pilot_20261119_v4/calibration.json`](results/unified_pilot_20261119_v4/calibration.json)）。
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
tests/                   # 离线单元测试套件（127 项）
pilots/                  # 运行入口（run_unified_eval.py, run_paired_isolated.py, verify_paired_trial.py 等）
configs/                 # 模型端点 profile（不含任何 API key，凭据通过环境变量注入）
data/                    # 冻结任务、输入快照与校准用例
docs/                    # 评审协议与证据映射文档
results/                 # 最新 unified_pilot_20261119_v4 汇总与校准产物
```

---

## 测试与复现

```bash
# 运行离线单测
PYTHONPATH=src pytest -q

# 运行统一评测校准与对照（需设置模型凭据环境变量及依赖包 alphaideabench / claimnov）
python pilots/run_unified_eval.py --phase all --out outputs/unified_run
```
