# Benchmark-50 `wave2` Multi-Host Evaluation Report (`N=42` Manuscripts across `20` Domains)

## 1. Aggregate `E_vs_D` Across 3 Calibrated Blind Judges

| Judge | Win / Loss / Tie / Unres | Topic Net Wins (95% CI) | Score Sum Delta (95% CI) | Order Sens. |
| --- | :---: | :---: | :---: | ---: |
| Qwen3.8-flash (Primary) | `20 / 10 / 12 / 0` | **`+0.225`** (`[-0.025, +0.475]`) | **`+0.971`** (`[+0.300, +1.658]`) | 11 / 42 |
| DeepSeek-v4.1-flash | `15 / 13 / 14 / 0` | **`+0.042`** (`[-0.200, +0.267]`) | **`-0.071`** (`[-0.958, +0.767]`) | 14 / 42 |
| GLM-5.3-flash | `21 / 10 / 11 / 0` | **`+0.250`** (`[+0.000, +0.475]`) | **`+0.583`** (`[+0.013, +1.150]`) | 11 / 42 |
| **3-Judge Majority Vote** | `19 / 11 / 12 / 0` | **`+0.183`** (`[-0.067, +0.425]`) | **`+0.494`** (`[-0.133, +1.119]`) | 0 / 42 |

## 2. Unified Secondary Axes (Qwen3.8-flash + ClaimNov) & Deterministic Mechanisms

- **ClaimNov Compositional Novelty Delta**: `-0.000`
- **AlphaIdeaBench Deltas**: `testability=+0.158`, `context_grounding=-0.092`, `verification_feasibility=+0.125`, `method_novelty=+0.050`, `insight_novelty=+0.092`
- **Process Hype Defects**: `A=20 -> D=74 -> E=5`
- **Unverified External Entities**: `A=229 -> D=215 -> E=71`
- **Blocking Confounds Addressed**: `D=0/0` -> `E=0/0`
- **Trajectory Integrity (`E`)**: `errors=0`, `clean_closed_loop=34/42`, `consolidated=30`, `orig_keys_kept=42/42`

## 3. By Host (`Qwen3.8-flash Primary` & `3-Judge Majority`)

| Host | n | Qwen W/L/T (`Net`, `Δsum`) | 3-Judge Majority W/L/T (`Net`, `Δsum`) |
| --- | :---: | :---: | :---: |
| `direct` | 20 | `7/5/8` (`+0.100`, `+0.550`) | `7/6/7` (`+0.050`, `-0.142`) |
| `ai_scientist` | 20 | `12/5/3` (`+0.350`, `+1.375`) | `11/5/4` (`+0.300`, `+1.117`) |
| `edara` | 2 | `1/0/1` (`+0.500`, `+1.500`) | `1/0/1` (`+0.500`, `+1.333`) |

## 4. Manuscript-by-Manuscript Breakdown

| Topic | Host | Cohort | `E` Trajectory | Qwen | DeepSeek | GLM | **Majority (`Δsum`)** |
| --- | :---: | :---: | --- | :---: | :---: | :---: | :---: |
| `smallari_expansion40_001_cache_com` | `direct` | `v4` | `changed` `['guided_revise', 'clarify', 'seek_evidence', 'guided_revise']` | `win` | `tie` | `tie` | **`tie` (`+0.667`)** |
| `smallari_expansion40_001_cache_com` | `ai_scientist` | `v4` | `changed` `['guided_revise', 'seek_evidence', 'local_patch']` | `loss` | `tie` | `loss` | **`loss` (`-0.833`)** |
| `smallari_expansion40_001_cache_com` | `edara` | `v4` | `changed` `['guided_revise', 'clarify', 'local_patch', 'guided_revise']` | `tie` | `loss` | `win` | **`tie` (`+0.333`)** |
| `survey_03_rag_systems` | `direct` | `ext15` | `changed` `['guided_revise', 'clarify', 'retain_and_test']` | `win` | `tie` | `win` | **`win` (`+1.000`)** |
| `survey_03_rag_systems` | `ai_scientist` | `ext15` | `changed` `['guided_revise', 'seek_evidence', 'guided_revise']` | `win` | `win` | `win` | **`win` (`+3.000`)** |
| `ws_09_world_models_simulation` | `direct` | `v4` | `changed` `['guided_revise', 'seek_evidence', 'guided_revise']` | `win` | `win` | `win` | **`win` (`+2.000`)** |
| `ws_09_world_models_simulation` | `ai_scientist` | `v4_supplement` | `changed` `['guided_revise', 'seek_evidence', 'guided_revise']` | `win` | `tie` | `tie` | **`tie` (`+2.500`)** |
| `smallari_expansion40_025_audio_vis` | `direct` | `new25` | `changed` `['guided_revise', 'seek_evidence', 'clarify']` | `loss` | `loss` | `loss` | **`loss` (`-3.667`)** |
| `smallari_expansion40_025_audio_vis` | `ai_scientist` | `new25` | `changed` `['guided_revise', 'seek_evidence', 'clarify']` | `win` | `win` | `win` | **`win` (`+3.000`)** |
| `smallari_expansion40_013_motion_ge` | `direct` | `new25` | `changed` `['guided_revise', 'seek_evidence', 'guided_revise']` | `tie` | `loss` | `loss` | **`loss` (`-2.500`)** |
| `smallari_expansion40_013_motion_ge` | `ai_scientist` | `new25` | `changed` `['guided_revise', 'seek_evidence', 'guided_revise']` | `tie` | `loss` | `win` | **`tie` (`+0.500`)** |
| `survey_14_diffusion_3d_generation` | `direct` | `new25` | `changed` `['guided_revise', 'clarify', 'seek_evidence', 'guided_revise']` | `tie` | `win` | `tie` | **`tie` (`+1.000`)** |
| `survey_14_diffusion_3d_generation` | `ai_scientist` | `new25` | `changed` `['guided_revise', 'seek_evidence', 'guided_revise']` | `win` | `win` | `win` | **`win` (`+5.000`)** |
| `smallari_expansion40_040_critic_re` | `direct` | `v4` | `changed` `['guided_revise', 'seek_evidence', 'guided_revise']` | `tie` | `tie` | `tie` | **`tie` (`-0.333`)** |
| `smallari_expansion40_040_critic_re` | `ai_scientist` | `v4` | `changed` `['guided_revise', 'seek_evidence', 'clarify']` | `tie` | `tie` | `win` | **`tie` (`+1.000`)** |
| `survey_04_llm_agent_collaboration` | `direct` | `new25` | `changed` `['guided_revise', 'seek_evidence', 'guided_revise']` | `win` | `tie` | `win` | **`win` (`+1.500`)** |
| `survey_04_llm_agent_collaboration` | `ai_scientist` | `new25` | `changed` `['guided_revise', 'seek_evidence', 'guided_revise']` | `win` | `win` | `win` | **`win` (`+4.333`)** |
| `survey_06_llm_safety_jailbreak` | `direct` | `v4` | `changed` `['guided_revise', 'seek_evidence', 'clarify']` | `loss` | `loss` | `loss` | **`loss` (`-3.000`)** |
| `survey_06_llm_safety_jailbreak` | `ai_scientist` | `v4` | `changed` `['guided_revise', 'seek_evidence', 'guided_revise']` | `win` | `win` | `win` | **`win` (`+4.500`)** |
| `smallari_expansion40_006_backdoor_` | `direct` | `ext15` | `changed` `['guided_revise', 'seek_evidence', 'guided_revise']` | `loss` | `loss` | `loss` | **`loss` (`-3.000`)** |
| `smallari_expansion40_006_backdoor_` | `ai_scientist` | `ext15` | `changed` `['guided_revise', 'seek_evidence', 'guided_revise']` | `win` | `tie` | `win` | **`win` (`+1.500`)** |
| `smallari_expansion40_009_detection` | `direct` | `ext15` | `changed` `['guided_revise', 'regenerate', 'seek_evidence']` | `tie` | `loss` | `loss` | **`loss` (`-1.500`)** |
| `smallari_expansion40_009_detection` | `ai_scientist` | `ext15` | `changed` `['guided_revise', 'regenerate', 'local_patch']` | `loss` | `loss` | `loss` | **`loss` (`-2.833`)** |
| `smallari_expansion40_014_spurious_` | `direct` | `ext15` | `changed` `['guided_revise', 'seek_evidence', 'guided_revise']` | `tie` | `win` | `win` | **`win` (`+1.167`)** |
| `smallari_expansion40_014_spurious_` | `ai_scientist` | `ext15` | `changed` `['guided_revise', 'seek_evidence', 'guided_revise']` | `loss` | `loss` | `tie` | **`loss` (`-2.167`)** |
| `smallari_expansion40_023_quantizat` | `direct` | `ext15` | `changed` `['guided_revise', 'seek_evidence', 'guided_revise']` | `win` | `win` | `win` | **`win` (`+2.167`)** |
| `smallari_expansion40_023_quantizat` | `ai_scientist` | `ext15` | `changed` `['guided_revise', 'clarify', 'seek_evidence', 'guided_revise']` | `tie` | `tie` | `win` | **`tie` (`+0.333`)** |
| `smallari_expansion40_030_pruning_l` | `direct` | `new25` | `changed` `['guided_revise', 'clarify', 'retain_and_test']` | `loss` | `tie` | `tie` | **`tie` (`-0.167`)** |
| `smallari_expansion40_030_pruning_l` | `ai_scientist` | `new25` | `changed` `['guided_revise', 'clarify', 'retain_and_test']` | `loss` | `loss` | `loss` | **`loss` (`-2.833`)** |
| `smallari_expansion40_012_series_ti` | `direct` | `ext15` | `changed` `['guided_revise', 'seek_evidence', 'guided_revise']` | `tie` | `loss` | `tie` | **`tie` (`-1.500`)** |
| `smallari_expansion40_012_series_ti` | `ai_scientist` | `ext15` | `changed` `['guided_revise', 'seek_evidence', 'guided_revise']` | `win` | `tie` | `win` | **`win` (`+1.333`)** |
| `ws_38_temporal_graph_learning` | `direct` | `new25` | `changed` `['guided_revise', 'seek_evidence', 'guided_revise']` | `tie` | `tie` | `tie` | **`tie` (`-0.667`)** |
| `ws_38_temporal_graph_learning` | `ai_scientist` | `new25` | `changed` `['guided_revise', 'seek_evidence', 'guided_revise']` | `win` | `win` | `tie` | **`win` (`+0.167`)** |
| `smallari_expansion40_029_bottlenec` | `direct` | `v4` | `changed` `['guided_revise', 'local_patch', 'clarify', 'guided_revise']` | `win` | `win` | `win` | **`win` (`+4.000`)** |
| `smallari_expansion40_029_bottlenec` | `ai_scientist` | `v4` | `changed` `['guided_revise', 'clarify', 'retain_and_test']` | `win` | `tie` | `win` | **`win` (`+0.667`)** |
| `smallari_expansion40_029_bottlenec` | `edara` | `v4` | `changed` `['guided_revise', 'seek_evidence', 'clarify']` | `win` | `win` | `win` | **`win` (`+2.333`)** |
| `smallari_expansion40_005_memorizat` | `direct` | `new25` | `changed` `['guided_revise', 'seek_evidence', 'guided_revise']` | `tie` | `win` | `tie` | **`tie` (`+1.000`)** |
| `smallari_expansion40_005_memorizat` | `ai_scientist` | `new25` | `changed` `['guided_revise', 'clarify', 'seek_evidence', 'guided_revise']` | `loss` | `loss` | `loss` | **`loss` (`-1.833`)** |
| `smallari_expansion40_019_protein_l` | `direct` | `ext15` | `changed` `['guided_revise', 'seek_evidence', 'guided_revise']` | `win` | `tie` | `win` | **`win` (`+2.000`)** |
| `smallari_expansion40_019_protein_l` | `ai_scientist` | `ext15` | `changed` `['guided_revise', 'seek_evidence', 'guided_revise']` | `win` | `win` | `tie` | **`win` (`+2.000`)** |
| `ws_18_scientific_foundation_models` | `direct` | `v4` | `changed` `['guided_revise', 'local_patch', 'clarify', 'guided_revise']` | `loss` | `loss` | `loss` | **`loss` (`-3.000`)** |
| `ws_18_scientific_foundation_models` | `ai_scientist` | `v4` | `changed` `['guided_revise', 'seek_evidence', 'guided_revise']` | `win` | `win` | `win` | **`win` (`+3.000`)** |

- **Metered Cost (USD)**: `New A/B/C/D = $0.4369`, `Arm E = $1.1761`, `3-Judge Blind Eval = $3.7382`, `Total = $5.3511`
