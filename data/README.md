# 数据说明

| 文件 | 内容 | 来源 |
|---|---|---|
| `inputs_core.json` | 15 个 topic、38 份冻结原稿、统一证据包、缺失清单 | 2026-11-17 冻结，正文 SHA-256 固化 |
| `inputs_pilot.json` | 3 个开发题、9 份原稿的试跑输入 | 同上 |
| `inputs_extension.json` | 25 个题、25 份 Direct 原稿（本轮未跑） | 同上 |
| `calibration_cases.json` | 54 个评委校准案例（每轴 12 实质 + 6 表达） | 由 `paired_calibration.py` 生成并冻结 |
| `unified_calibration_cases.json` | Lit2Test 细微破坏与同义替换校准案例快照 | 由 `unified_eval.py` 冻结 |
| `tasks_mvp15.json` | 15 题历史三轨 MVP 任务定义快照 | 历史任务集 |
| `benchmark_50_topics.json` | 锁定 50 题分层主基准清单（10 领域 × 5 题，25 Lit2Test + 25 AlphaIdeaBench，含 wave1/2/3） | 2026-11-22 固化 |
| `tasks_benchmark50.json` | 50 题完整 `TaskSpec` 离线复现快照（无外部原始库时自动回退读取） | 由 `load_benchmark50_tasks()` 导出 |

## 为什么只有这些

原始运行目录约 3.7G，包含逐请求 API 响应、审计缓存和全部候选稿；其中请求记录含提供方响应，不宜分发。本仓库保留：

1. 足以复现判读与汇总的冻结输入；
2. 足以复现校准的案例集；
3. `results/` 下的汇总和逐稿判定。

## 正文哈希

`inputs_*.json` 中每份原稿都有 `sha256`（归一化后正文）与 `raw_sha256`（原始文本）。复现时 `paired_trial` 会校验原稿未被改动；任何差异都会报错而不是静默重算。

## 缺失样本

`core` 的 45 个宿主位置中有 7 份缺失（E-DARA 6、AI-Scientist 1），记入 `missing`，不进分母。这是宿主生成阶段的结果，不是本轮筛选。
