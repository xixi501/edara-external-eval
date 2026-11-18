# 测量与机制修复：离线验收回放

日期：2026-11-17。全部检查只读保存记录，不使用网络和模型调用。

## 验收结论

| 项 | 判据 | 结果 |
|---|---|---|
| E1 | 三份文本逆变换等于原行；23 个唯一 evidence ID；引文全部可在原生 section 定位 | 通过 |
| E2 | 22/22 首尾引号 span 可定位；19/19 纯格式案例恢复原判；0 个负样本误放行 | 通过 |
| E3 | 证据换序与字段换序不改变评价输入 | 通过 |
| E4 | 带 blocking spec 的题目在搜证耗尽后仍可选到非搜证动作 | 通过 |
| E5 | 3/3 已知误匹配被拒；入池召回损失 ≤ 10% | 通过 |
| E6 | pytest ≥ 55 passed | 通过 |
| E7 | 历史产物 sha256 不变 | 通过 |

## E1 导出可逆与证据可定位

| 题目 | 阶段 | 引用数 | 来源数 | 逆变换等于原行 | 引文未定位 |
|---|---|---:|---:|---|---:|
| aiib-82 | proposal | 9 | 4 | 是 | 0 |
| smallari_expansion40_001_cache | proposal | 11 | 3 | 是 | 0 |
| ws_02_agentic_ai | proposal | 16 | 3 | 是 | 0 |

- 唯一原生 evidence ID：**23**；全部逆变换一致：True；全部引文可定位：True

## E2 锚点回归（69 个历史失败）

- 首尾引号可修复的 span：22，其中新定位成功：22
- 纯格式案例（无引文问题）：19，新定位成功：19
- 引文重新匹配：8/28（涉及 26 行）
- 负样本（篡改 span）误放行：0/20
- 判定模式分布：['edge_quotes', 'exact', 'not_found']

## E3 评价输入确定性

- 证据换序后 packet 不变：True
- 候选字段换序后文本不变：True
- packet sha256：`c3325c53a1ce8af3…`（12 条证据）

## E4 调度可执行性

- 题目数：15；带 blocking spec：15
- 新策略下 clarify 立即可选：15
- 搜证耗尽后 clarify 仍可选：15
- 历史实际发生非搜证动作的题目：2

| 题目 | blocking | 初始动作 | 新策略可选 | 历史动作 |
|---|---:|---|---|---|
| aiib-10541 | 3 | seek_evidence | clarify, seek_evidence | seek_evidence |
| aiib-33 | 3 | seek_evidence | clarify, seek_evidence | seek_evidence |
| aiib-4717 | 3 | seek_evidence | clarify, seek_evidence | seek_evidence |
| aiib-7144 | 2 | seek_evidence | clarify, seek_evidence | seek_evidence |
| aiib-82 | 3 | seek_evidence | clarify, seek_evidence | seek_evidence |
| smallari_expansion40_001_cac | 3 | retain_and_test | clarify | clarify, seek_evidence |
| smallari_expansion40_002_unl | 1 | seek_evidence | local_patch, clarify, seek_evidence | local_patch, seek_evidence |
| smallari_expansion40_011_tes | 3 | seek_evidence | clarify, seek_evidence | seek_evidence |
| smallari_expansion40_029_bot | 3 | seek_evidence | clarify, seek_evidence | seek_evidence |
| smallari_expansion40_040_cri | 2 | seek_evidence | clarify, seek_evidence | seek_evidence |
| survey_06_llm_safety_jailbre | 2 | seek_evidence | clarify, seek_evidence | seek_evidence |
| survey_08_kv_cache_compressi | 4 | seek_evidence | clarify, seek_evidence | seek_evidence |
| ws_02_agentic_ai | 3 | seek_evidence | clarify, seek_evidence | seek_evidence |
| ws_09_world_models_simulatio | 3 | seek_evidence | clarify, seek_evidence | seek_evidence |
| ws_18_scientific_foundation_ | 3 | seek_evidence | clarify, seek_evidence | seek_evidence |

## E5 检索相关性

- 三个已知误匹配被拒：3/3
- 已入池来源重放：187，新门驳回：16（8.6%）

## E7 历史产物不变

- `../pilot3_runs/edara/edara_outputs.jsonl`：未变
- `outputs/mvp15_ablation_v2/REPORT_ZH.md`：未变
- `outputs/mvp15_ablation_v2/results.json`：未变
- `outputs/pilot3_full_compare_v4/summary.md`：未变
- 全部未变：True

## E6 测试

- pytest：**75 passed**

## 结论与边界

- 这七项只证明**测量与调度的失效模式已被消除**，不证明外挂的质量增益；后者需要 Phase 1 的付费重评。
- E2 中恢复的是模型的原判，不是证明原判正确。19 个纯格式案例只是重新进入科学判定，仍可能是错的。
- 28 条引文里仍有 20 条无法在给定段落中找到，它们继续作为未决，属于引用本身不可核对，不是格式问题。
- E5 的召回代价为 8.6%，已接近 10% 上限；Phase 1 若发现关键来源被误删，应改为按机制词加权而不是继续收紧。
- 本轮没有发起任何付费调用；smoke 目录里的金额是 dry-run 的模拟值，不是真实计费。

