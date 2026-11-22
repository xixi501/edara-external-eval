# 运行与复现实验入口（`pilots/`）

## 1. 当前核心脚本（Benchmark-50 主线与论文图表生成）

| 脚本 | 用途与示例命令 |
| :--- | :--- |
| `run_benchmark50_wave.py` | 运行 `Benchmark-50` 指定波次（`wave1` / `wave2` / `wave3`）的 `Direct`、`AI-Scientist-v2`、`E-DARA` 五臂对照与三评委盲评（支持 `--dry-run` 零成本离线验证）。 |
| `run_evoscientist_wave.py` | 运行 `EvoScientist`（`research-ideation v2.1.2` 五阶段进化 + ELO 锦标赛协议）原稿生成、五臂对照与三评委盲评。 |
| `generate_paper_artifacts.py` | 一键汇总已完成的各波次与多宿主盲评结果，输出论文级 LaTeX `booktabs` 三线表（`.tex`）、Markdown 报告与矢量对比图（`.svg`）。 |
| `verify_paired_trial.py` | 独立核验配对试验目录的原稿 SHA-256、证据池一致性及调用预算上限。 |

### 快速生成论文全套表格与矢量图

```bash
python pilots/generate_paper_artifacts.py --results-dir results --out-dir results/paper_ready
```

## 2. 早期阶段复现与辅助脚本

其余脚本（如 `run_unified_eval.py`、`run_paired_isolated.py`、`run_governance_pilot.py`、`run_feedback_ablation.py`、`export_host_inputs.py` 等）对应早期 `Pilot-3`、`MVP-15` 与 `Extension-15` 阶段的探索及归档结果复现，予以保留以确保历史实验记录可追溯。
