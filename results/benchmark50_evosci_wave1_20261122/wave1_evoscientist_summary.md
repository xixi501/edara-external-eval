# EvoScientist (`evoscientist-research-ideation/v2.1.2`) `wave1` 5-Arm + 3-Judge Evaluation (`N=10`)

## 1. Aggregate `E_vs_D` Across 3 Calibrated Blind Judges

| Judge | Win / Loss / Tie | Topic Net Wins (95% CI) | Score Sum Delta (95% CI) | Order Sens. |
| --- | :---: | :---: | :---: | ---: |
| Qwen3.8-flash (Primary) | `5 / 2 / 3` | **`+0.300`** (`[-0.200, +0.800]`) | **`+0.700`** (`[-0.500, +1.850]`) | 3 / 10 |
| DeepSeek-v4.1-flash | `2 / 6 / 2` | **`-0.400`** (`[-0.900, +0.100]`) | **`-1.450`** (`[-3.150, +0.350]`) | 2 / 10 |
| GLM-5.3-flash | `2 / 2 / 6` | **`+0.000`** (`[-0.400, +0.400]`) | **`+0.250`** (`[-0.750, +1.250]`) | 5 / 10 |
| **3-Judge Majority Vote** | `2 / 2 / 6` | **`+0.000`** (`[-0.400, +0.400]`) | **`-0.167`** (`[-1.250, +0.983]`) | 0 / 10 |

- **Single-Pass Ablation (`C_vs_B`, Qwen)**: `1W / 4L / 5T` (`net_wins=-0.300`, `sum_delta=-0.350`)
- **ClaimNov Compositional Novelty Delta (`E_vs_D`)**: `+0.017`
- **AlphaIdeaBench Deltas (`E_vs_D`)**: `testability=+0.200`, `context_grounding=+0.000`, `verification_feasibility=+0.200`, `method_novelty=+0.200`, `insight_novelty=+0.100`
- **Original Keys Preserved (`E`)**: `10 / 10` (`100%`)
- **Process Hype (`A -> D -> E`)**: `8 -> 22 -> 0`
- **Unverified External Entities (`A -> D -> E`)**: `39 -> 47 -> 15`
- **Total Metered Cost (USD)**: `$2.9896`

## 2. Topic-by-Topic Breakdown

| Topic | EvoSci Winner ELO | `E` Trajectory | Qwen | DeepSeek | GLM | **Majority (`Δsum`)** |
| --- | :---: | --- | :---: | :---: | :---: | :---: |
| `survey_01_llm_reasoning_cot` | `1531.26` | `['guided_revise', 'seek_evidence', 'guided_revise']` | `win` | `loss` | `tie` | **`tie` (`-1.000`)** |
| `survey_02_vlm_hallucination` | `1531.26` | `['guided_revise', 'seek_evidence', 'guided_revise']` | `loss` | `loss` | `loss` | **`loss` (`-3.500`)** |
| `smallari_expansion40_010_problems_inve` | `1531.26` | `['guided_revise', 'clarify', 'local_patch', 'guided_revise']` | `loss` | `tie` | `tie` | **`tie` (`+0.500`)** |
| `ws_02_agentic_ai` | `1531.26` | `['guided_revise', 'clarify', 'local_patch', 'guided_revise']` | `tie` | `loss` | `tie` | **`tie` (`-1.667`)** |
| `smallari_expansion40_002_unlearning_ma` | `1531.26` | `['guided_revise', 'seek_evidence', 'guided_revise']` | `win` | `win` | `win` | **`win` (`+2.667`)** |
| `smallari_expansion40_011_test_time_ada` | `1531.26` | `['guided_revise', 'clarify', 'retain_and_test']` | `win` | `loss` | `tie` | **`tie` (`-0.167`)** |
| `survey_08_kv_cache_compression` | `1531.26` | `['guided_revise', 'seek_evidence', 'guided_revise']` | `tie` | `tie` | `tie` | **`tie` (`+0.167`)** |
| `smallari_expansion40_008_graph_network` | `1531.26` | `['guided_revise', 'seek_evidence', 'clarify']` | `win` | `loss` | `tie` | **`tie` (`-0.167`)** |
| `survey_05_mechanistic_interpretability` | `1531.26` | `['guided_revise', 'seek_evidence', 'guided_revise']` | `win` | `win` | `win` | **`win` (`+2.833`)** |
| `smallari_expansion40_003_quantum_circu` | `1531.26` | `['guided_revise', 'local_patch', 'clarify', 'guided_revise']` | `tie` | `loss` | `loss` | **`loss` (`-1.333`)** |
