# E-DARA 外评与治理外挂

本仓库是 E-DARA 科研 idea 生成系统的**外部评测框架**与**可信新颖性治理外挂（Trustworthy-Novelty Middleware）**的独立实现。它把治理思想从 E-DARA 中抽象出来，作用在其他 Agent（Direct、AI-Scientist）或本系统的提案上，并用同稿配对试验测量它是否改善提案的诚实性、可信度与新颖性。

> Evaluation harness and governance middleware for research-proposal agents. It audits a frozen proposal, applies bounded repairs, and measures the effect against an equally-budgeted conventional revision on the same source draft.

## 从哪里开始

1. [`docs/HANDOFF_20261118.md`](docs/HANDOFF_20261118.md)：最新交接（目标、设计、结果、下一步）。
2. [`ARCHITECTURE.md`](ARCHITECTURE.md)：数据流、模块职责、关键不变量。
3. [`HANDOFF.md`](HANDOFF.md)：上一阶段的交接、结论、运行手册、红线。
3. [`docs/EVAL_PROTOCOL_V1_2_ZH.md`](docs/EVAL_PROTOCOL_V1_2_ZH.md)：判读规则。
4. [`results/RESULTS_20261118_ZH.md`](results/RESULTS_20261118_ZH.md)：本轮结果与解读。

## 当前状态

- 仪器版本：协议 v1.2；诚实轴三处机械问题已修复，三轴校准 12/12 + 6/6 通过。
- 主样本：15 个 topic、38 份冻结原稿（Direct 15、AI-Scientist 14、E-DARA 9，缺 7 份）。
- 主试验：C 对 B 可信 **+0.222 [0.089, 0.344]**（要素级 +0.316）；E 对 D **−0.167 [−0.322, −0.044]**；新颖度无变化。
- 后续（2026-11-18 下午）：新增 claim 级事实审计 pass。Step 1（9 例）审计召回 7/9、编辑落实 7/7；Step 2（27 份，修复前）C 对 B 可信 +0.222 [0.014, 0.417]、诚实 +0.104、R−E +0.077、新颖 0；随后修复了强制落实字段映射与槽规则两处 bug，**held-out 尚未验证**。
- 接管入口：[`docs/HANDOFF_20261118.md`](docs/HANDOFF_20261118.md)；Step 2 结果：[`docs/STEP2_RESULT_20261118.md`](docs/STEP2_RESULT_20261118.md)；数据：[`results/20261118_claim_audit/`](results/20261118_claim_audit/)。
- E-DARA 主体代码在另一个仓库 [`xixi501/edara`](https://github.com/xixi501/edara)，本仓库不修改它。

## 试验设计（简述）

每份原稿派生五个版本，做两组配对比较：

| 臂 | 处理 | 比较 |
|---|---|---|
| A | 冻结原稿 | 参照 |
| B | 常规审阅 + 最多 4 处局部补丁 | C 的对照 |
| C | 外挂结构化审计 + 同一编辑器 + 4 处上限（含混淆项强制落实） | **C − B** |
| D | 自我检索后整稿改写，最多 3 轮 | E 的对照 |
| E | 完整外挂五动作路由（搜证/澄清/局部/重构/保留） | **E − D** |

评委在不知道臂别、诊断和编辑理由的情况下，对同一份稿的两个版本做盲评，每个配对三次（正序、倒序、正序重采样），按多数配对差判定胜负平。三个轴独立评分：

- **诚实**：`(R − E) / K`，净纠偏；
- **可信**：0–3 级，检查假设、实验、对照、指标、判败规则；
- **新颖**：−1 / 0 / +1，相对截止日前文献的实质贡献变化。

完整规则见 [`docs/EVAL_PROTOCOL_V1_2_ZH.md`](docs/EVAL_PROTOCOL_V1_2_ZH.md)（v1 已冻结，供历史结果对照）。

## 目录

```
ARCHITECTURE.md         架构说明（数据流、模块、不变量）
HANDOFF.md              交接说明（现状、未决、运行手册）
configs/                模型端点 profile（不含凭据）
src/edara_mvp_eval/     外挂与评测实现
  confound_audit.py    假设特异性混淆项审计（v1.2 新增）
  paired_review.py     三轴盲评、机械归一化、多数票判定
  paired_trial.py      五臂试验、预算、缓存与汇总
  middleware.py        审计→有界修补→复核
  epistemic_audit.py   证据、缺口、规格、ClaimNov 审计
pilots/                 运行入口与核验脚本
tests/                  单元测试（108 项）
analyses/               历史分析脚本与报告（repair_v3 等）
docs/                   协议、交接、贡献—证据对应表
data/                   冻结输入、校准案例、任务清单
results/                正式试验的汇总与逐稿判定
```

## 环境

- Python 3.10+
- 依赖：`pydantic`、`numpy`、`openai`、`rank-bm25`、`typer`、`rich`、`pyyaml`、`pypdf`、`alphaideabench`
- `alphaideabench` 与 `claimnov` 为外部项目，需单独获取：
  - AlphaIdeaBench：见上级目录的同名 checkout 或上游仓库，`pip install -e <path>`
  - ClaimNov：`claimnov_adapter.py` 通过路径注入使用，部署时设置 `CLAIMNOV_SRC` 或按该文件注释放置
- 凭据通过环境变量注入，不入库：`DEEPSEEK_API_KEY`、`AUTODL_API_KEY`

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
pytest -q
```

## 复现一次试验

```bash
# 1. 冻结样本（写入新目录的 manifest 与 inputs）
python pilots/run_paired_isolated.py --phase freeze \
  --out outputs/paired_<date> --calibration-set reserve

# 2. 校准（三轴各 12 实质 + 6 表达，门槛 ≥10/12 且 ≥5/6）
python pilots/run_paired_isolated.py --resume --phase calibrate \
  --out outputs/paired_<date> --calibration-set reserve

# 3. 正式试验（通过校准的轴才进入）
python pilots/run_paired_isolated.py --resume --phase core \
  --out outputs/paired_<date> --calibration-set reserve

# 4. 独立核验（原稿未改、证据一致、预算与计量）
python pilots/verify_paired_trial.py --run outputs/paired_<date> --cohort core
```

凭据注入与日志脱敏参考 `pilots/run_logged.py`（仓库内为通用版本，实际部署路径按环境调整）。

## 版本说明

本仓库是交接用的**源代码基准（source of truth）**。它和实验服务器上的执行副本只有三处路径端口化差异，判读逻辑与试验结果不受影响：

| 文件 | 差异 |
|---|---|
| `src/edara_mvp_eval/paired_trial.py` | `WORKSPACE` 支持 `EDARA_WORKSPACE`；协议可从 `docs/` 读取；无 `outputs/` 时放宽输出目录约束；`freeze` 自动复制 `data/` 中的冻结输入 |
| `pilots/run_governance_pilot.py` | 不再假设固定的父目录层级 |
| `tests/test_paired_trial.py` | 缺少原始宿主输出时改用 `data/` 中的冻结输入与校准案例 |

模型端点配置从 `data/p1b-*.json` 复制到 `configs/`，不含任何凭据；API key 只从环境变量读取。

## 数据说明

- `data/inputs_core.json`：38 份冻结原稿、任务、统一证据包与缺失清单。正文哈希固化，复现时不再重新生成。
- `data/calibration_cases.json`：54 个评委校准案例（每轴 12 实质 + 6 表达）。
- `data/tasks_mvp15.json`：15 题任务定义。
- `data/README.md`：数据来源、哈希与缺失样本说明。
- 若要重新生成校准案例（而不是使用冻结的 `calibration_cases.json`），需要把 Lit2Test 的 checkout 放在本仓库的上级目录，和原工作区布局一致；否则测试会用冻结案例。
- `results/`：正式试验的汇总与逐稿判定，不含原始 API 响应全文与完整运行目录（体积过大，且含提供方响应）。
- 原始运行目录（逐请求记录、审计缓存）保存在实验服务器，不随仓库分发。

## 已知限制

见 [`HANDOFF.md`](HANDOFF.md) 的「未决与风险」一节，主要包括诚实轴校准稳定性、评委绝对等级的可重复性、整稿重构的代价，以及未跑的单宿主复核。

## 许可

尚未指定。仓库当前为私有，移交前由所有者决定许可证。
