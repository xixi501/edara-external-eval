# E-DARA 外挂治理中间件交接文档（Stage A 验收 → 修复调优 → Stage B 15 题扩量）

> 生成时间：2026-11-16  
> 工作目录：`/workspace/edara/staging/04_external_eval_refs`（源码包位于 `edara_mvp_eval/`）  
> 前序产出目录：`edara_mvp_eval/outputs/pilot3_full_compare_v3/`（[summary.md](file:///workspace/edara/staging/04_external_eval_refs/edara_mvp_eval/outputs/pilot3_full_compare_v3/summary.md)、[results.json](file:///workspace/edara/staging/04_external_eval_refs/edara_mvp_eval/outputs/pilot3_full_compare_v3/results.json)）

---

## 1. 红线约束与环境信息

1. **禁止修改主源码仓库**：严禁改动 `/workspace/edara/repos/framework/` 下的任何文件。当前该仓库位于分支 `codex/trusted-idea-frontier-20261110`（本地基线 commit `848cab2a5f`，工作树干净）。
2. **既有数据只读**：`/workspace/edara/data/` 与 `/workspace/edara/snapshots/` 保持只读。
3. **凭据复用方式（严禁打印或落盘密钥原文）**：
   - 使用 `/workspace/edara/staging/04_external_eval_refs/pilot3_runs/run_logged.py` 包装执行实测命令，会自动从 `/workspace/edara/runtime/ops/research-credentials.json` 注入 `DEEPSEEK_API_KEY` 与 `AUTODL_API_KEY`，并在写日志时自动脱敏；
   - 检查命令：`edara_mvp_eval/../.venv/bin/python pilot3_runs/run_logged.py --check`。
4. **长运行会话**：实测任务放入具名 `tmux` 会话（如 `pilot3-eval`），并保留日志路径。
5. **单元测试命令**：
   ```bash
   cd /workspace/edara/staging/04_external_eval_refs/edara_mvp_eval && ../.venv/bin/python -m pytest -v
   ```
   （当前 11/11 全部通过。）

---

## 2. 已完成的工程升级与可用资产

### 2.1 代码层面（均在 `edara_mvp_eval/`）
- **四阶段外挂中间件**（[middleware.py](file:///workspace/edara/staging/04_external_eval_refs/edara_mvp_eval/src/edara_mvp_eval/middleware.py)、[spec_locator.py](file:///workspace/edara/staging/04_external_eval_refs/edara_mvp_eval/src/edara_mvp_eval/spec_locator.py)、[claimnov_adapter.py](file:///workspace/edara/staging/04_external_eval_refs/edara_mvp_eval/src/edara_mvp_eval/claimnov_adapter.py)、[literature_scout.py](file:///workspace/edara/staging/04_external_eval_refs/edara_mvp_eval/src/edara_mvp_eval/literature_scout.py)、[epistemic_audit.py](file:///workspace/edara/staging/04_external_eval_refs/edara_mvp_eval/src/edara_mvp_eval/epistemic_audit.py)）：
  - 查询相关滑窗证据抽取（`extract_relevant_excerpt`，已修复原先 400 字符截断）；
  - 批评接地门禁（`proposal_span` + `evidence_quote`）；
  - 规范缺失定位器（`SpecDefectLocator`）；
  - 多文献组合余项分析（`analyze_compositional_remainder`）；
  - 有界局部微创补丁（`bounded_patch`）+ 方法保真度护栏（`compute_retention_metrics`）。
- **对照基线与防偏置盲评**（[baselines.py](file:///workspace/edara/staging/04_external_eval_refs/edara_mvp_eval/src/edara_mvp_eval/baselines.py)、[pairwise.py](file:///workspace/edara/staging/04_external_eval_refs/edara_mvp_eval/src/edara_mvp_eval/pairwise.py)）：
  - `run_self_refine`（纯自省，无外部证据/无保真护栏）；
  - `run_retrieve_and_revise`（同源外部证据与审计清单，但无约束整篇重写）；
  - 4 维 Chain-of-Rubric 双向换序盲评。
- **稳定性与断点续跑加固**（[llm_client.py](file:///workspace/edara/staging/04_external_eval_refs/edara_mvp_eval/src/edara_mvp_eval/llm_client.py)、[cli.py](file:///workspace/edara/staging/04_external_eval_refs/edara_mvp_eval/src/edara_mvp_eval/cli.py)）：
  - 全链路 10 处 `chat_json` 已适配 `thinking` 模式配额（`max_tokens + 2048` 起步）、开启 `response_format={"type": "json_object"}`（不支持时自动回退）及单步 `JSONDecodeError` 安全兜底；
  - `cli.py run` 默认开启 `--resume`，在工作线程内每跑完 1 个臂立即原子落盘到 `<out>/checkpoints/<slug>__<hash>.json`（带配置签名 `run_sig`）。

### 2.2 已固化初稿资产（无需重跑生成）
- `pilot3_runs/direct/direct_outputs.jsonl`（3 题）
- `pilot3_runs/ai_scientist_v2/ai_scientist_outputs.jsonl`（3 题）
- `pilot3_runs/edara/edara_outputs.jsonl`（3 题）
- 题目缓存：`edara_mvp_eval/outputs/tasks_pilot3.json`、`edara_mvp_eval/outputs/tasks_mvp15.json`

---

## 3. Stage A（3 题实测 `pilot3_full_compare_v3`）结果与诊断

本轮使用 `deepseek-flash`（`/workspace/edara/data/p1b-idea-profile.json`）生成/修复 + `GLM-5.3-flash`（`/workspace/edara/data/p1b-critic-profile.json`）审计与盲评 + `--live-search`，全程零报错完成（总费用 `$0.533`：`gen=$0.225`，`judge=$0.308`）：

| Arm | Gate Pass | Strict Gate | Overclaim (↓) | Gap Surv | Spec Comp (↑) | Method Ret (↑) | Trusted Nov | Rigor | Actionability |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `direct` | 66.7% | 0.0% | 0.067 | 55.6% | 0.35 | 1.00 | 0.524 | 3.07 | 3.73 |
| **`direct_mw`** | **100.0%** | 0.0% | **0.026** | **100.0%** | **0.43** | **0.98** | **0.721** | **3.73** | **3.93** |
| `direct_self_refine` | 66.7% | 0.0% | 0.077 | 88.9% | 0.31 | 0.69 | 0.510 | 3.20 | 3.90 |
| `direct_retrieve_revise` | 100.0% | 0.0% | 0.056 | 77.8% | 0.31 | 0.92 | 0.702 | 3.47 | 3.90 |
| `ai-scientist-v2` | 0.0% | 0.0% | 0.195 | 77.8% | 0.24 | 1.00 | 0.000 | 1.87 | 3.78 |
| **`ai-scientist-v2_mw`** | **100.0%** | 0.0% | **0.075** | **100.0%** | 0.28 | **0.99** | **0.851** | **3.20** | 3.70 |
| `edara` | 0.0% | 0.0% | 0.242 | 77.8% | 0.39 | 1.00 | 0.000 | 1.07 | 3.97 |
| **`edara_mw`** | **66.7%** | 0.0% | **0.022** | **100.0%** | 0.32 | **1.00** | **0.469** | **3.73** | 3.82 |

**已验证成立**：
- 保真度：`*_mw` 的 `Method Ret` 达 **`0.98–1.00`**（`Self-Refine` 为 `0.69`、`Retrieve-and-Revise` 为 `0.92`）；
- 诚信与抗伪 Gap：三组 `*_mw` 的 `Gap Surv` 全部 **`100.0%`**；
- 盲评：`*_mw` 对各自原版全部 **3-0**，`direct_mw` 对 `direct_retrieve_revise` **3-0**，`order-sensitive -> tie` 总体降至 **16/84（19.0%）**。

### 当前必须解决的 2 个核心问题
1. **`Strict Gate` 全员 `0.0%`、`Spec Comp` 偏低（`0.28–0.43`），且盲评 `direct_mw` `0-3` 输给 `direct_self_refine`**
   - **根因**：`bounded_patch` 每轮最多 4 处补丁名额主要消耗在修 `Overclaim` 和 `Gap` 上，`SpecDefectLocator` 点出的阻塞项（阈值/算子定义、数据切分、基线对齐、等价边界、失效区间）未能充分补入正文；而 `Self-Refine` 不受条数限制地把实验协议润色得很锋利（尽管它不诚信且丢了 31% 方法细节），导致盲评被裁判偏爱。
2. **`edara_mw` 在 `smallari` 上残留 1 条 `smoothing=4`（`Gate Pass = 66.7%`），且多题出现 `rej=2~4`**
   - **根因**：长文本（如 `edara` 的 `smallari` 11,775 字符）下，模型复制 `exact_source_span` 偶发空白/换行/标点微差，严格子串匹配直接拒掉有效补丁。

---

## 4. 接下来要改的方向与执行步骤

### Step 1：优化 `middleware.py`（并补单测）
1. **将 `Spec` 澄清通道与“4 处正文 span 补丁”解耦**：
   - 对 `SpecDefectLocator` 指出的 `blocking` / `recommended` 缺失项，允许通过独立的 `spec_clarifications`（字段级追加澄清子句 `append_text`，例如补齐 `decisive_metric` 等价边界、`minimal_test` 数据切分与基线、`limitations` 失效区间、`method` 超参协议）落地，不挤占 4 处纠偏 span 补丁名额，同时不删除原方法主干（保持 `Method Ret >= 0.90`）。
2. **放宽 `apply_bounded_patches` 的 `exact_source_span` 匹配容错**：
   - 在严格子串匹配失败时，增加**空白归一化正则匹配**（忽略连续空格/换行微差，回映射回原字符串区间）与**去首尾标点/单句唯一子串回退**，大幅降低无谓 `rej`，把 `edara_mw` 的 `Gate Pass` 推到 `100%`。
3. 在 `tests/test_mvp_eval.py` 补充对应单测并跑通 `pytest -v`。

### Step 2：复测 3 题 Stage A（`pilot3_full_compare_v4`）
在具名 `tmux` 会话中通过 `run_logged.py` 执行（复用已有 3 套初稿）：
```bash
/workspace/edara/staging/04_external_eval_refs/.venv/bin/python \
  /workspace/edara/staging/04_external_eval_refs/pilot3_runs/run_logged.py \
  /workspace/edara/staging/04_external_eval_refs/pilot3_runs/pilot3_eval_live8_stage_a_v4.log \
  /workspace/edara/staging/04_external_eval_refs/.venv/bin/edara-mvp run \
  --tasks-cache edara_mvp_eval/outputs/tasks_pilot3.json \
  --direct-inputs pilot3_runs/direct/direct_outputs.jsonl \
  --with-self-refine \
  --with-retrieve-revise \
  --import-arm ai-scientist-v2=pilot3_runs/ai_scientist_v2/ai_scientist_outputs.jsonl \
  --import-arm edara=pilot3_runs/edara/edara_outputs.jsonl \
  --mw-on-imported \
  --gen-profile /workspace/edara/data/p1b-idea-profile.json \
  --judge-profile /workspace/edara/data/p1b-critic-profile.json \
  --workers 3 \
  --live-search \
  --out edara_mvp_eval/outputs/pilot3_full_compare_v4
```
**Stage A 验收门槛**：`Spec Comp` 明显提升、`Strict Gate > 0%`、`edara_mw` `Gate Pass = 100%`、`Method Ret >= 0.90`，且 `direct_mw` 在盲评中扭转对 `direct_self_refine` 的胜负比。

### Step 3：扩量到 15 题 Stage B（`tasks_mvp15.json`）
Stage A 达标后：
1. 先用 `generate-direct` 跑完并固化 15 题统一 `Direct` 初稿：
   ```bash
   /workspace/edara/staging/04_external_eval_refs/.venv/bin/python \
     /workspace/edara/staging/04_external_eval_refs/pilot3_runs/run_logged.py \
     /workspace/edara/staging/04_external_eval_refs/pilot3_runs/mvp15_direct_gen.log \
     /workspace/edara/staging/04_external_eval_refs/.venv/bin/edara-mvp generate-direct \
     --tasks-cache edara_mvp_eval/outputs/tasks_mvp15.json \
     --gen-profile /workspace/edara/data/p1b-idea-profile.json \
     --workers 3 \
     --out pilot3_runs/direct/direct_mvp15_outputs.jsonl
   ```
2. 再基于同一份 `direct_mvp15_outputs.jsonl` 运行 15 题正式消融对比（`direct` vs `direct_mw` vs `direct_self_refine` vs `direct_retrieve_revise`，自带 `--resume` 断点保护）：
   ```bash
   /workspace/edara/staging/04_external_eval_refs/.venv/bin/python \
     /workspace/edara/staging/04_external_eval_refs/pilot3_runs/run_logged.py \
     /workspace/edara/staging/04_external_eval_refs/pilot3_runs/mvp15_ablation_live.log \
     /workspace/edara/staging/04_external_eval_refs/.venv/bin/edara-mvp run \
     --tasks-cache edara_mvp_eval/outputs/tasks_mvp15.json \
     --direct-inputs pilot3_runs/direct/direct_mvp15_outputs.jsonl \
     --with-self-refine \
     --with-retrieve-revise \
     --gen-profile /workspace/edara/data/p1b-idea-profile.json \
     --judge-profile /workspace/edara/data/p1b-critic-profile.json \
     --workers 3 \
     --live-search \
     --out edara_mvp_eval/outputs/mvp15_ablation_v1
   ```

## 5. Stage B updated protocol

The historical Stage A thresholds above describe the v3/v4 development protocol.
Stage B readiness is pipeline correctness, auditable evidence, reproducible inputs and complete failure reporting. Use five arms, including the existing local middleware and the new feedback middleware. The primary outcome is joint honesty, credibility and novelty under separate final evaluation. See [Stage B evaluation protocol](STAGE_B_PROTOCOL.md).

## 6. Stage B v2 已完成

十五题五组实测于 2026-11-17（北京时间）完成。详见
[结项报告](outputs/mvp15_ablation_v2/REPORT_ZH.md)。完整外挂联合达标 5/15，
原稿 4/15，自我改写 9/15。新增十二题的完整外挂与原稿均为 4/12。
完整外挂有 13 题返回同一正文，其中四题的独立评分发生联合判定翻转。
