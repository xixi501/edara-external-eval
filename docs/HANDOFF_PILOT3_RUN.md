# E-DARA 3-Topic 并行对比试跑与评测交接文档

> 最后更新：2026-11-15（已对齐 E-DARA `deepseek-flash` 模型配置、拆分为「先并行生成 → 待定裁判模型后统一评测」两阶段，并补齐 E-DARA 第 2/3 题原生输入包与防泄露哈希校验）  
> 工作目录：`/workspace/edara/staging/04_external_eval_refs/edara_mvp_eval`

---

## 1. 红线约束与运行前检查

1. **禁止修改主源码仓库**：严禁修改 `/workspace/edara/repos/framework/` 中的任何文件。所有试跑、中间导出与账本放在 `/workspace/edara/staging/04_external_eval_refs/pilot3_runs/` 或 `edara_mvp_eval/outputs/`。
2. **严禁打印或落盘密钥原文**：检查凭据仅允许 `test -n "$DEEPSEEK_API_KEY" && echo SET || echo UNSET`（同理检查 `AUTODL_API_KEY`）。
3. **既有数据只读**：`/workspace/edara/data/` 与 `/workspace/edara/snapshots/` 保持只读。
4. **长运行会话**：`AI-Scientist-v2` 与 `E-DARA` 的 `live` 运行请放入具名 `tmux` 会话并保存日志，便于断点追踪。

### 所需环境变量（对齐 E-DARA 已有配置）

- **Phase 1（三系统并行生成）**：
  - `Direct`、`AI-Scientist-v2`、`E-DARA` 生成端统一对齐 **`deepseek-flash`**（来自 `/workspace/edara/data/p1b-idea-profile.json` / `p1a-live-profile.json`），需要导出：
    - `DEEPSEEK_API_KEY`（三轨生成共用）
  - `E-DARA` 内部跨家族 critic / review 阶段（`/workspace/edara/data/p1b-critic-profile.json` 使用 `GLM-5.3-flash`），额外需要：
    - `AUTODL_API_KEY`（仅 Track C `E-DARA` 需要；`Direct` 与 `AI-Scientist-v2` 不需要）
  - 外部文献检索：`OpenAlex` 免 Key 已验证可用（`aiib-82` 遵循官方 `active-retrieval/v1`，**不禁搜**，严格按 `cutoff_date` 与 `forbidden_identity_hashes` 拦截）；`S2_API_KEY` 可选。
- **Phase 2（统一审计 + 中间件局部修复 + 双向盲评）**：
  - **等 Phase 1 所有 3 组产物全部产出并验收后，再由用户拍板独立的评审模型**（`--judge-profile` 或 `EDARA_JUDGE_*`），不在 Phase 1 提前锁定最终裁判。

---

## 2. 已导出的 3 个 Pilot Topic 与资产清单

运行 `prepare-tasks` 后，`edara_mvp_eval/outputs/` 下已具备：

| 题号 & `task_id` | 来源轨道 | 截断日期 & 检索边界 | 先验材料与测试定位 |
|---|---|---|---|
| **#1 `aiib-82`** | AIIB Pilot（`active-retrieval/v1`） | `cutoff=2024-05-07`；屏蔽目标论文 3 个 `title:`/`arxiv:`/`identity:` 哈希 | `5` 篇本地 `pypdf` 种子摘要；测强防泄露截断与文献归因严谨度 |
| **#2 `smallari_expansion40_001_cache_compression_llms_long_context__v02_expansion40`** | Lit2Test（OpenReview 邻域） | `cutoff=2025-12-31`（显式写入各系统题面与边界） | `4` 篇同邻域论文；测 `ClaimNov` 窄邻域防撞车、6 字段可证伪契约与 `<=8 A100 GPU-days` |
| **#3 `ws_02_agentic_ai`** | AlphaIdeaBench（Workshop） | `cutoff=2025-12-31`（显式写入各系统题面与边界） | `1` 篇 `topic_doc` + 显式预算（`4x A100`、`200 GPUh`、`$200 API`、`40h` 标注）；测开放题 `OpenAlex` 检索增益与预算遵守 |

关键导出目录（`outputs/pilot3_inputs/`）：
- `ai_scientist_topics/<id>.md` & `<id>.boundary.json`
- `ai_scientist_shim/native_common.py`（动态 per-task `cutoff_date` + 与 `E-DARA boundary.py` 完全一致的 `title:`/`arxiv:`/`doi:`/`identity:` 哈希屏蔽 + 免 Key `OpenAlex`）
- `edara_cases/edara_live_config.json`（已通过 `aiib.case_config(..., mode="live")` 校验，全指向 `/workspace/edara/data/` 现有 E-DARA profile）
- `edara_cases/<id>/inputs/ref_*.md` 与 `<id>/literature-boundary.json`（符合 `E-DARA` 严格 4 字段 `boundary.py` 校验）

---

## 3. Phase 1：三轨并行生成（统一使用 `deepseek-flash`，先产出待评方案）

> 下面三轨相互独立，建议同时开 3 个并行后台进程 / `tmux` 窗口执行。本阶段**不跑最终盲评**。

### Track A：`Direct` 纯生成基线（3 题并发）

直接使用新命令 `generate-direct` 复用 `/workspace/edara/data/p1b-idea-profile.json`（`deepseek-flash`）：

```bash
cd /workspace/edara/staging/04_external_eval_refs
.venv/bin/edara-mvp generate-direct \
  --tasks-cache edara_mvp_eval/outputs/tasks_pilot3.json \
  --gen-profile /workspace/edara/data/p1b-idea-profile.json \
  --workers 3 \
  --out pilot3_runs/direct/direct_outputs.jsonl
```

### Track B：`AI-Scientist-v2`（3 题并行，统一用 `--model deepseek-flash`）

`AI-Scientist-v2` 白名单原生支持 `deepseek-flash`（内部读取 `DEEPSEEK_API_KEY` 走 `https://api.deepseek.com`）。务必把 `ai_scientist_shim` 放在 `PYTHONPATH` 最前：

```bash
WORK=/workspace/edara/staging/04_external_eval_refs/pilot3_runs/ai_scientist_v2
SHIM=/workspace/edara/staging/04_external_eval_refs/edara_mvp_eval/outputs/pilot3_inputs/ai_scientist_shim
COMMON=/workspace/edara/staging/03_active_benchmarks/external-native-eval-20261109/common
AI_SRC=/workspace/edara/staging/03_active_benchmarks/external-native-eval-20261109/sources/ai-scientist-v2
PY=/workspace/edara/runtime/ai-scientist-v2-ideation/bin/python

mkdir -p "$WORK/ledgers"
cp /workspace/edara/staging/04_external_eval_refs/edara_mvp_eval/outputs/pilot3_inputs/ai_scientist_topics/* "$WORK/"

for TID in \
  aiib-82 \
  smallari_expansion40_001_cache_compression_llms_long_context__v02_expansion40 \
  ws_02_agentic_ai
do
  (
    export PYTHONPATH="$SHIM:$COMMON:$AI_SRC"
    export AI_SCIENTIST_CASE_ID="$TID"
    export AI_SCIENTIST_BOUNDARY_PATH="$WORK/${TID}.boundary.json"
    export AI_SCIENTIST_SEARCH_LEDGER="$WORK/ledgers/${TID}.search.jsonl"
    export AI_SCIENTIST_TOOL_LEDGER="$WORK/ledgers/${TID}.tool.jsonl"
    export AI_SCIENTIST_LLM_LEDGER="$WORK/ledgers/${TID}.llm.jsonl"
    "$PY" "$AI_SRC/ai_scientist/perform_ideation_temp_free.py" \
      --model deepseek-flash \
      --max-num-generations 1 \
      --num-reflections 3 \
      --workshop-file "$WORK/${TID}.md" \
      > "$WORK/${TID}.log" 2>&1
  ) &
done
wait
```

跑完后打包成 `ai_scientist_outputs.jsonl`：

```bash
/workspace/edara/staging/04_external_eval_refs/.venv/bin/python - <<'PY'
import json
from pathlib import Path

work = Path("/workspace/edara/staging/04_external_eval_refs/pilot3_runs/ai_scientist_v2")
tids = [
    "aiib-82",
    "smallari_expansion40_001_cache_compression_llms_long_context__v02_expansion40",
    "ws_02_agentic_ai",
]
lines = []
for tid in tids:
    p = work / f"{tid}.json"
    if p.is_file():
        ideas = json.loads(p.read_text(encoding="utf-8"))
        row = dict(ideas[0]) if isinstance(ideas, list) and ideas else {"raw_output": ideas}
        row["task_id"] = tid
        lines.append(json.dumps(row, ensure_ascii=False))
(work / "ai_scientist_outputs.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"Wrote {len(lines)} rows -> {work / 'ai_scientist_outputs.jsonl'}")
PY
```

### Track C：`E-DARA` 原生真实运行（3 题分别独立 staging 目录并行）

> ⚠️ 需同时导出 `DEEPSEEK_API_KEY` 与 `AUTODL_API_KEY`。由于 `aiib.py` 要求 `inputs/` 必须在 `batch.json` 同根目录下且不允许软链接越界，第 1 题先拷贝隔离的 `aiib-82` 单条 batch 再运行。

1. **Topic #1 (`aiib-82`) —— 原生 AIIB 入口（`edara-bench aiib-run-batch`）**：

```bash
EDARA_ROOT=/workspace/edara/staging/04_external_eval_refs/pilot3_runs/edara
CFG=/workspace/edara/staging/04_external_eval_refs/edara_mvp_eval/outputs/pilot3_inputs/edara_cases/edara_live_config.json

# 构建仅包含 aiib-82 的自包含 batch（不修改 /workspace/edara/data）
/workspace/edara/runtime/core/bin/python - <<'PY'
import json, shutil
from pathlib import Path

src = Path("/workspace/edara/data/aiib-pilot-20261105/prepared")
dst = Path("/workspace/edara/staging/04_external_eval_refs/pilot3_runs/edara/aiib82_batch")
(dst / "public").mkdir(parents=True, exist_ok=True)
(dst / "inputs").mkdir(parents=True, exist_ok=True)
shutil.copy2(src / "public/aiib-82.json", dst / "public/aiib-82.json")
shutil.copytree(src / "inputs/aiib-82", dst / "inputs/aiib-82", dirs_exist_ok=True)
batch = json.loads((src / "batch.json").read_text())
batch["entries"] = [e for e in batch["entries"] if e.get("source_index") == 82]
batch["boundary_notes"] = [n for n in batch.get("boundary_notes", []) if n.get("source_index") == 82]
(dst / "batch.json").write_text(json.dumps(batch, ensure_ascii=False, indent=2) + "\n")
PY

/workspace/edara/runtime/core/bin/edara-bench aiib-run-batch \
  --batch "$EDARA_ROOT/aiib82_batch/batch.json" \
  --root "$EDARA_ROOT/aiib82_run" \
  --config "$CFG" \
  --mode live \
  --total-budget-microusd 1500000 \
  --prior-consumed-microusd 0 \
  --overall-cap-microusd 2000000 \
  > "$EDARA_ROOT/aiib82.log" 2>&1
```

2. **Topic #2 (`smallari_...`, 4 篇 `.md`) 与 Topic #3 (`ws_02_agentic_ai`, 1 篇 `.md`) —— 原生 Policy 入口（`edara-policy bootstrap` + `literature-boundary.json` + `edara-policy run`）**：

```bash
EDARA_ROOT=/workspace/edara/staging/04_external_eval_refs/pilot3_runs/edara
IN_CASES=/workspace/edara/staging/04_external_eval_refs/edara_mvp_eval/outputs/pilot3_inputs/edara_cases

# 示例：初始化并运行 Topic #2（4 篇邻域论文）和 Topic #3（1 篇 topic_doc，开放探索）
# 1) bootstrap（先建 topic workspace，读取对应 inputs/ref_*.md）
# 2) 将 $IN_CASES/<tid>/literature-boundary.json 拷贝进生成的 topic workspace 根目录（强制截断 2025-12-31）
# 3) 执行 edara-policy run --workspace <ws> --mode live --execute --allow-new-version \
#      --budget-microusd 1500000 \
#      --policy-model-config /workspace/edara/data/p1a-live-profile.json \
#      --topic-model-config /workspace/edara/data/p1a-live-profile.json \
#      --audit-model-config /workspace/edara/data/p1b-critic-profile.json \
#      --coverage-model-config /workspace/edara/data/p1b-critic-profile.json \
#      --p2-model-config /workspace/edara/data/p1b-idea-profile.json \
#      --p2-critic-model-config /workspace/edara/data/p1b-critic-profile.json \
#      --p3-editor-model-config /workspace/edara/data/p1b-idea-profile.json \
#      --p3-reviewer-model-config /workspace/edara/data/p1b-critic-profile.json \
#      --p4b-generator-model-config /workspace/edara/data/p2-deep-profile.json \
#      --p4b-judge-model-config /workspace/edara/data/p1b-critic-profile.json \
#      --review-model-config /workspace/edara/data/p1b-critic-profile.json \
#      --review-model-config /workspace/edara/data/p1b-critic-profile.json \
#      --review-model-config /workspace/edara/data/p1b-critic-profile.json \
#      --review-arbiter-model-config /workspace/edara/data/p1b-critic-profile.json \
#      --supplement-source openalex \
#      --supplement-admission-mode live \
#      --supplement-admission-model-config /workspace/edara/data/p1a-live-profile.json
```

> 跑完第 1~3 题后，从各自 `aiib-export` / `research.db` / 终态交付物提取最终方案文本或结构化 JSON，整理为 `/workspace/edara/staging/04_external_eval_refs/pilot3_runs/edara/edara_outputs.jsonl`（每行 `{"task_id": "<tid>", "proposal_text": ...}` 或标准字段；评测器会用同一 `normalize_artifact` 剥离内部 ID 并盲化拆分）。

---

## 4. Phase 2：等所有产出完成并确定评审模型后，统一同口径评测

等 `direct_outputs.jsonl`、`ai_scientist_outputs.jsonl`、`edara_outputs.jsonl` 都拿到，并且与用户确定好独立的 **Judge 模型**（`--judge-profile` 或 `EDARA_JUDGE_*`）与 **Middleware 修复模型**（`--gen-profile`，可用 `/workspace/edara/data/p1b-idea-profile.json`）后，一条命令同批次完成 **原始对比 + 中间件局部修复消融（`*_mw`）+ 全 `C(6,2)=15` 对双向换序盲评**：

```bash
cd /workspace/edara/staging/04_external_eval_refs
.venv/bin/edara-mvp run \
  --tasks-cache edara_mvp_eval/outputs/tasks_pilot3.json \
  --skip-direct \
  --import-arm direct=pilot3_runs/direct/direct_outputs.jsonl \
  --import-arm ai-scientist-v2=pilot3_runs/ai_scientist_v2/ai_scientist_outputs.jsonl \
  --import-arm edara=pilot3_runs/edara/edara_outputs.jsonl \
  --mw-on-imported \
  --gen-profile /workspace/edara/data/p1b-idea-profile.json \
  --judge-profile /path/to/chosen_judge_profile.json \
  --workers 3 \
  --live-search \
  --out edara_mvp_eval/outputs/pilot3_full_compare
```

---

## 5. 从 3 题扩到 15 题 (`tasks_mvp15.json`) 的验收门禁

看 `edara_mvp_eval/outputs/pilot3_full_compare/summary.md` 与 `results.json`：
1. **边界健康**：3 题无超时/JSON 失败；`aiib-82` 无 `> 2024-05-07` 或目标论文泄露；双向位置偏置折叠率（`order-sensitive -> tie`）建议 `< 30%`。
2. **指标区分度**：观察 `Direct` → `Direct+MW` 及三者原版在 `Gate Pass`、`Overclaim Rate (↓)`、`Gap Survival`、`ClaimNov`、`Trusted Novelty` 与盲评胜率上的区分度是否明显。若出现 `MW` 过修导致真新颖方案被平庸化或 `OpenAlex` 对 `ws_02_agentic_ai` 检索过少，先调参数再切换 `--tasks-cache edara_mvp_eval/outputs/tasks_mvp15.json` 扩量。
