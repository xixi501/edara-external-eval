# EvoScientist (`evoscientist-research-ideation/v2.1.2`) `wave2` 5-Arm + 3-Judge Evaluation (`N=20`)

## 1. Aggregate `E_vs_D` Across 3 Calibrated Blind Judges

| Judge | Win / Loss / Tie | Topic Net Wins (95% CI) | Score Sum Delta (95% CI) | Order Sens. |
| --- | :---: | :---: | :---: | ---: |
| Qwen3.8-flash (Primary) | `7 / 6 / 7` | **`+0.050`** (`[-0.300, +0.400]`) | **`+0.725`** (`[-0.125, +1.575]`) | 7 / 20 |
| DeepSeek-v4.1-flash | `7 / 9 / 4` | **`-0.100`** (`[-0.500, +0.300]`) | **`-0.800`** (`[-2.050, +0.450]`) | 4 / 20 |
| GLM-5.3-flash | `6 / 6 / 8` | **`+0.000`** (`[-0.350, +0.350]`) | **`+0.200`** (`[-0.475, +0.900]`) | 7 / 20 |
| **3-Judge Majority Vote** | `7 / 7 / 6` | **`+0.000`** (`[-0.350, +0.350]`) | **`+0.042`** (`[-0.783, +0.875]`) | 0 / 20 |

- **Single-Pass Ablation (`C_vs_B`, Qwen)**: `5W / 5L / 10T` (`net_wins=+0.000`, `sum_delta=+0.025`)
- **ClaimNov Compositional Novelty Delta (`E_vs_D`)**: `+0.005`
- **AlphaIdeaBench Deltas (`E_vs_D`)**: `testability=+0.050`, `context_grounding=-0.050`, `verification_feasibility=+0.200`, `method_novelty=-0.200`, `insight_novelty=+0.150`
- **Original Keys Preserved (`E`)**: `20 / 20` (`100%`)
- **Process Hype (`A -> D -> E`)**: `25 -> 71 -> 1`
- **Unverified External Entities (`A -> D -> E`)**: `76 -> 86 -> 22`
- **Total Metered Cost (USD)**: `$5.9926`

## 2. Topic-by-Topic Breakdown

| Topic | EvoSci Winner ELO | `E` Trajectory | Qwen | DeepSeek | GLM | **Majority (`Δsum`)** |
| --- | :---: | --- | :---: | :---: | :---: | :---: |
| `smallari_expansion40_001_cache_compres` | `1516.0` | `['guided_revise', 'seek_evidence', 'local_patch']` | `loss` | `loss` | `loss` | **`loss` (`-2.833`)** |
| `survey_03_rag_systems` | `1531.26` | `['guided_revise', 'local_patch', 'clarify', 'seek_evidence']` | `win` | `win` | `tie` | **`win` (`+1.833`)** |
| `ws_09_world_models_simulation` | `1531.26` | `['guided_revise', 'seek_evidence', 'guided_revise']` | `tie` | `loss` | `loss` | **`loss` (`-1.833`)** |
| `smallari_expansion40_025_audio_visual_` | `1531.26` | `['guided_revise', 'clarify', 'seek_evidence', 'guided_revise']` | `tie` | `loss` | `tie` | **`tie` (`-0.667`)** |
| `smallari_expansion40_013_motion_genera` | `1531.26` | `['guided_revise', 'seek_evidence', 'guided_revise']` | `win` | `win` | `win` | **`win` (`+3.667`)** |
| `survey_14_diffusion_3d_generation` | `1531.26` | `['guided_revise', 'seek_evidence', 'guided_revise']` | `tie` | `tie` | `tie` | **`tie` (`-0.167`)** |
| `smallari_expansion40_040_critic_reinfo` | `1531.26` | `['guided_revise', 'clarify', 'regenerate']` | `tie` | `win` | `tie` | **`tie` (`+1.000`)** |
| `survey_04_llm_agent_collaboration` | `1531.26` | `['guided_revise', 'seek_evidence', 'guided_revise']` | `win` | `tie` | `win` | **`win` (`+1.833`)** |
| `survey_06_llm_safety_jailbreak` | `1531.26` | `['guided_revise', 'seek_evidence', 'guided_revise']` | `loss` | `tie` | `tie` | **`tie` (`-0.167`)** |
| `smallari_expansion40_006_backdoor_atta` | `1531.26` | `['guided_revise', 'seek_evidence', 'guided_revise']` | `win` | `win` | `win` | **`win` (`+1.500`)** |
| `smallari_expansion40_009_detection_out` | `1531.26` | `['guided_revise', 'clarify', 'seek_evidence', 'guided_revise']` | `tie` | `win` | `tie` | **`tie` (`+0.000`)** |
| `smallari_expansion40_014_spurious_corr` | `1531.26` | `['guided_revise', 'seek_evidence', 'clarify']` | `loss` | `loss` | `loss` | **`loss` (`-1.833`)** |
| `smallari_expansion40_023_quantization_` | `1531.26` | `['guided_revise', 'seek_evidence', 'guided_revise']` | `tie` | `loss` | `loss` | **`loss` (`-1.000`)** |
| `smallari_expansion40_030_pruning_langu` | `1516.0` | `['guided_revise', 'clarify', 'retain_and_test']` | `loss` | `loss` | `loss` | **`loss` (`-3.167`)** |
| `smallari_expansion40_012_series_time_a` | `1531.23` | `['guided_revise', 'clarify', 'seek_evidence', 'guided_revise']` | `win` | `win` | `win` | **`win` (`+2.667`)** |
| `ws_38_temporal_graph_learning` | `1531.26` | `['guided_revise', 'regenerate', 'local_patch']` | `win` | `tie` | `tie` | **`tie` (`+1.000`)** |
| `smallari_expansion40_029_bottleneck_co` | `1531.26` | `['guided_revise', 'seek_evidence', 'local_patch']` | `loss` | `loss` | `tie` | **`loss` (`-1.833`)** |
| `smallari_expansion40_005_memorization_` | `1531.26` | `['guided_revise', 'clarify', 'seek_evidence']` | `loss` | `loss` | `loss` | **`loss` (`-2.667`)** |
| `smallari_expansion40_019_protein_langu` | `1531.26` | `['guided_revise', 'clarify', 'seek_evidence', 'guided_revise']` | `tie` | `win` | `win` | **`win` (`+1.500`)** |
| `ws_18_scientific_foundation_models` | `1531.26` | `['guided_revise', 'local_patch', 'clarify', 'guided_revise']` | `win` | `loss` | `win` | **`win` (`+2.000`)** |
