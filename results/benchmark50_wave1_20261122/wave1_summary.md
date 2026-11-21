# Benchmark-50 `wave1` Canary Evaluation Report (`N=22` Manuscripts across `10` Domains)

## 1. Aggregate `E_vs_D` Across 3 Calibrated Blind Judges

| Judge | Win / Loss / Tie / Unres | Topic Net Wins (95% CI) | Score Sum Delta (95% CI) | Order Sens. |
| --- | :---: | :---: | :---: | ---: |
| Qwen3.8-flash (Primary) | `11 / 8 / 3 / 0` | **`+0.150`** (`[-0.250, +0.500]`) | **`+1.167`** (`[-0.042, +2.325]`) | 3 / 22 |
| DeepSeek-v4.1-flash | `10 / 7 / 5 / 0` | **`+0.150`** (`[-0.117, +0.467]`) | **`+0.825`** (`[-0.292, +2.009]`) | 5 / 22 |
| GLM-5.3-flash | `8 / 7 / 7 / 0` | **`+0.050`** (`[-0.267, +0.350]`) | **`+0.642`** (`[-0.125, +1.367]`) | 7 / 22 |
| **3-Judge Majority Vote** | `9 / 6 / 7 / 0` | **`+0.133`** (`[-0.117, +0.417]`) | **`+0.878`** (`[-0.097, +1.867]`) | 0 / 22 |

## 2. Unified Secondary Axes (Qwen3.8-flash + ClaimNov) & Deterministic Mechanisms

- **ClaimNov Compositional Novelty Delta**: `-0.007`
- **AlphaIdeaBench Deltas**: `testability=+0.283`, `context_grounding=-0.167`, `verification_feasibility=+0.067`, `method_novelty=+0.150`, `insight_novelty=+0.033`
- **Process Hype Defects**: `D=9 ms (20)` -> `E=1 ms (3)`
- **Unverified External Entities**: `D=128` -> `E=46`
- **Blocking Confounds Addressed**: `D=0/0` -> `E=0/0`
- **Trajectory Integrity (`E`)**: `errors=1`, `clean_closed_loop=20/22`, `consolidated=21`, `orig_keys_kept=22/22`

## 3. By Host (`3-Judge Majority`)

| Host | n | Win / Loss / Tie | Topic Net Wins | Score Sum Delta |
| --- | :---: | :---: | ---: | ---: |
| `direct` | 10 | `3/3/4` | `+0.000` | `+0.150` |
| `ai_scientist` | 10 | `5/2/3` | `+0.300` | `+1.750` |
| `edara` | 2 | `1/1/0` | `+0.000` | `+0.000` |

## 4. Manuscript-by-Manuscript Old `E` vs Upgraded `E` Comparison

| Topic | Host | Old `E` Status / Verdict (`Δsum`) | New `E` Trajectory | Qwen | DeepSeek | GLM | **Majority (`Δsum`)** |
| --- | :---: | --- | --- | :---: | :---: | :---: | :---: |
| `survey_01_llm_reasoning_cot` | `direct` | `changed` / `loss` (`-3.500`) | `changed` `['guided_revise', 'seek_evidence', 'guided_revise']` | `loss` | `loss` | `loss` | **`loss` (`-3.333`)** |
| `survey_01_llm_reasoning_cot` | `ai_scientist` | `changed` / `win` (`+3.000`) | `changed` `['guided_revise', 'seek_evidence', 'guided_revise']` | `win` | `win` | `win` | **`win` (`+4.667`)** |
| `survey_02_vlm_hallucination` | `direct` | `changed` / `loss` (`-3.000`) | `changed` `['guided_revise', 'clarify', 'seek_evidence', 'guided_revise']` | `loss` | `loss` | `loss` | **`loss` (`-2.000`)** |
| `survey_02_vlm_hallucination` | `ai_scientist` | `changed` / `tie` (`+2.000`) | `changed` `['guided_revise', 'seek_evidence', 'guided_revise']` | `tie` | `tie` | `loss` | **`tie` (`-0.167`)** |
| `smallari_expansion40_010_problems_` | `direct` | `changed` / `win` (`+3.500`) | `changed` `['guided_revise', 'clarify', 'seek_evidence', 'guided_revise']` | `win` | `win` | `win` | **`win` (`+3.000`)** |
| `smallari_expansion40_010_problems_` | `ai_scientist` | `changed` / `win` (`+2.500`) | `changed` `['guided_revise', 'seek_evidence', 'guided_revise']` | `win` | `win` | `win` | **`win` (`+4.000`)** |
| `ws_02_agentic_ai` | `direct` | `changed` / `tie` (`+0.000`) | `changed` `['guided_revise', 'local_patch', 'guided_revise']` | `win` | `loss` | `tie` | **`tie` (`+0.167`)** |
| `ws_02_agentic_ai` | `ai_scientist` | `changed` / `win` (`+5.500`) | `changed` `['guided_revise', 'local_patch', 'seek_evidence', 'guided_revise']` | `win` | `win` | `win` | **`win` (`+5.333`)** |
| `ws_02_agentic_ai` | `edara` | `changed` / `win` (`+2.500`) | `changed` `['guided_revise', 'local_patch', 'guided_revise']` | `win` | `win` | `tie` | **`win` (`+3.000`)** |
| `smallari_expansion40_002_unlearnin` | `direct` | `changed` / `loss` (`-2.000`) | `changed` `['guided_revise', 'regenerate', 'seek_evidence', 'guided_revise']` | `loss` | `tie` | `tie` | **`tie` (`-0.833`)** |
| `smallari_expansion40_002_unlearnin` | `ai_scientist` | `changed` / `loss` (`-5.000`) | `changed` `['guided_revise', 'regenerate', 'seek_evidence']` | `loss` | `tie` | `tie` | **`tie` (`-0.333`)** |
| `smallari_expansion40_002_unlearnin` | `edara` | `revision_error:JSONDecodeError` / `loss` (`-5.000`) | `budget_exhausted` `['guided_revise', 'seek_evidence', 'guided_revise']` | `loss` | `loss` | `loss` | **`loss` (`-3.000`)** |
| `smallari_expansion40_011_test_time` | `direct` | `changed` / `tie` (`+0.500`) | `changed` `['guided_revise', 'regenerate', 'seek_evidence', 'guided_revise']` | `tie` | `win` | `tie` | **`tie` (`+0.833`)** |
| `smallari_expansion40_011_test_time` | `ai_scientist` | `changed` / `win` (`+4.000`) | `changed` `['guided_revise', 'seek_evidence', 'guided_revise']` | `win` | `win` | `win` | **`win` (`+4.000`)** |
| `survey_08_kv_cache_compression` | `direct` | `changed` / `win` (`+2.500`) | `changed` `['guided_revise', 'seek_evidence', 'guided_revise']` | `win` | `win` | `win` | **`win` (`+2.500`)** |
| `survey_08_kv_cache_compression` | `ai_scientist` | `changed` / `win` (`+4.500`) | `changed` `['guided_revise', 'seek_evidence', 'guided_revise']` | `loss` | `loss` | `loss` | **`loss` (`-2.833`)** |
| `smallari_expansion40_008_graph_net` | `direct` | `revision_error:invalid_rewrite` / `loss` (`-5.000`) | `changed` `['guided_revise', 'seek_evidence', 'guided_revise']` | `tie` | `tie` | `tie` | **`tie` (`+0.000`)** |
| `smallari_expansion40_008_graph_net` | `ai_scientist` | `changed` / `loss` (`-4.000`) | `changed` `['guided_revise', 'clarify', 'regenerate', 'guided_revise']` | `win` | `tie` | `tie` | **`tie` (`+2.000`)** |
| `survey_05_mechanistic_interpretabi` | `direct` | `changed` / `tie` (`+0.500`) | `changed` `['guided_revise', 'local_patch', 'seek_evidence', 'guided_revise']` | `loss` | `loss` | `loss` | **`loss` (`-2.167`)** |
| `survey_05_mechanistic_interpretabi` | `ai_scientist` | `changed` / `win` (`+6.000`) | `changed` `['guided_revise', 'seek_evidence', 'guided_revise']` | `win` | `win` | `win` | **`win` (`+5.000`)** |
| `smallari_expansion40_003_quantum_c` | `direct` | `changed` / `win` (`+3.500`) | `changed` `['guided_revise', 'seek_evidence', 'guided_revise']` | `win` | `win` | `win` | **`win` (`+3.333`)** |
| `smallari_expansion40_003_quantum_c` | `ai_scientist` | `changed` / `win` (`+1.500`) | `changed` `['guided_revise', 'seek_evidence', 'guided_revise']` | `loss` | `loss` | `loss` | **`loss` (`-4.167`)** |

- **Metered Cost (USD)**: `Arm E = $0.7994`, `3-Judge Blind Eval = $1.8504`, `Total = $2.6498`
