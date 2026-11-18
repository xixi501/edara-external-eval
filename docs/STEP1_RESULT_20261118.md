# Step 1 结果：claim 级事实审计 + 编辑槽规则（9 个案例）

## 结论

- **审计召回 7/9**（gate ≥6/9 通过；C 子集 3/4 通过）。
- **编辑落实 7/7**（所有被点出的缺陷都改了；gate ≥6/9 通过）。
- 两个漏点不是随机噪声：都是「数字/覆盖」类断言，审计判了 supported，但评委和原文证据都表明是缺陷。

## 逐案

| 案例 | 宿主 | 比较 | claim | 审计点出 | 编辑落实 | 说明 |
|---|---|---|---|---|---|---|
| aiib-7144 | direct | E vs D | c5 | 是 | 是 | supporting_result 改为 "Target result"；指标/对照保留 |
| smallari_001 | ai_scientist | E vs D | c13 | **否** | — | DynamicKV 5% 不是其报告预算（1.7%），审计漏判 |
| smallari_002 | direct | C vs B | c7 | 是 | 是 | 结果段改为 "Predicted outcome (pre-registered)" |
| smallari_002 | ai_scientist | C vs B | c1 | 是 | 是 | related_work 补上两篇来源的归属 |
| smallari_002 | ai_scientist | E vs D | c1 | 是 | 是 | 同上 |
| ws_02 | ai_scientist | E vs D | c8 | 是 | 是 | approach 改为 "We predict" |
| ws_02 | edara | C vs B | c3 | **否** | — | 把已提供的 LLM-SRBench 标成 unresolved，审计漏判 |
| survey_08 | ai_scientist | C vs B | c10 | 是 | 是 | "improves" 改为 "we will evaluate whether" |
| survey_08 | ai_scientist | E vs D | c10 | 是 | 是 | 同上 |

## 编辑行为

- 9 个案例都用了 4 处编辑；被点出的事实缺陷都占到了至少一个槽。
- 9 个案例都没有触发混淆项强制重试（confound enforcement 0/9），说明这一次事实修复没有挤掉强制落实；但也没有测到「需要强制落实 + 事实修复」同时发生的最坏情况。
- 编辑不是简单删句：c10 改成计划句、c1 补归属、c7 改成预测、c8 改成 "We predict"，都是实质修正。

## 两个漏点的含义

1. **数字/预算类断言**：审计能识别「预测写成已观察」，但对「数字与来源不符」不敏感（DynamicKV 5% vs 1.7%）。
2. **证据覆盖类断言**：把已提供的论文标成 unresolved，审计没有核对 packet 覆盖。

如果 Step 2 只靠这一版审计，诚实改善的上限是 7/9 左右的案例，不是全部。要覆盖这两类，审计 prompt 还需补「数字核对」和「packet 覆盖核对」两条规则。

## 成本

Step 1 三次运行（初版、槽规则、修 prompt）合计约 $0.15，36 次请求左右。
