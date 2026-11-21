# Benchmark-50 `wave1` Canary Evaluation Report (`N=22` Manuscripts across `10` Domains)

## 1. Aggregate `E_vs_D` Across 3 Calibrated Blind Judges

| Judge | Win / Loss / Tie / Unres | Topic Net Wins (95% CI) | Score Sum Delta (95% CI) | Order Sens. |
| --- | :---: | :---: | :---: | ---: |
| Qwen3.8-flash (Primary) | `11 / 6 / 5 / 0` | **`+0.250`** (`[-0.100, +0.583]`) | **`+1.700`** (`[+0.541, +2.817]`) | 5 / 22 |
| DeepSeek-v4.1-flash | `11 / 7 / 4 / 0` | **`+0.183`** (`[-0.150, +0.500]`) | **`+1.100`** (`[-0.225, +2.401]`) | 4 / 22 |
| GLM-5.3-flash | `10 / 6 / 6 / 0` | **`+0.200`** (`[-0.083, +0.500]`) | **`+0.967`** (`[-0.083, +1.933]`) | 6 / 22 |
| **3-Judge Majority Vote** | `10 / 7 / 5 / 0` | **`+0.150`** (`[-0.150, +0.450]`) | **`+1.256`** (`[+0.178, +2.250]`) | 0 / 22 |

## 2. Unified Secondary Axes (Qwen3.8-flash + ClaimNov) & Deterministic Mechanisms

- **ClaimNov Compositional Novelty Delta**: `-0.012`
- **AlphaIdeaBench Deltas**: `testability=+0.350`, `context_grounding=-0.267`, `verification_feasibility=+0.033`, `method_novelty=+0.083`, `insight_novelty=+0.100`
- **Process Hype Defects**: `D=9 ms (20)` -> `E=1 ms (3)`
- **Unverified External Entities**: `D=128` -> `E=36`
- **Blocking Confounds Addressed**: `D=0/0` -> `E=0/0`
- **Trajectory Integrity (`E`)**: `errors=0`, `clean_closed_loop=17/22`, `consolidated=16`, `orig_keys_kept=22/22`

## 3. By Host (`3-Judge Majority`)

| Host | n | Win / Loss / Tie | Topic Net Wins | Score Sum Delta |
| --- | :---: | :---: | ---: | ---: |
| `direct` | 10 | `3/5/2` | `-0.200` | `+0.300` |
| `ai_scientist` | 10 | `6/1/3` | `+0.500` | `+2.550` |
| `edara` | 2 | `1/1/0` | `+0.000` | `-1.167` |

## 4. Manuscript-by-Manuscript Old `E` vs Upgraded `E` Comparison

| Topic | Host | Old `E` Status / Verdict (`Δsum`) | New `E` Trajectory | Qwen | DeepSeek | GLM | **Majority (`Δsum`)** |
| --- | :---: | --- | --- | :---: | :---: | :---: | :---: |
| `survey_01_llm_reasoning_cot` | `direct` | `changed` / `loss` (`-3.500`) | `changed` `['guided_revise', 'seek_evidence', 'guided_revise']` | `loss` | `loss` | `win` | **`loss` (`-1.000`)** |
| `survey_01_llm_reasoning_cot` | `ai_scientist` | `changed` / `win` (`+3.000`) | `changed` `['guided_revise', 'seek_evidence', 'guided_revise']` | `win` | `win` | `win` | **`win` (`+4.833`)** |
| `survey_02_vlm_hallucination` | `direct` | `changed` / `loss` (`-3.000`) | `changed` `['guided_revise', 'seek_evidence', 'guided_revise']` | `win` | `tie` | `loss` | **`tie` (`+0.000`)** |
| `survey_02_vlm_hallucination` | `ai_scientist` | `changed` / `tie` (`+2.000`) | `changed` `['guided_revise', 'seek_evidence', 'guided_revise']` | `win` | `win` | `win` | **`win` (`+3.667`)** |
| `smallari_expansion40_010_problems_` | `direct` | `changed` / `win` (`+3.500`) | `changed` `['guided_revise', 'clarify', 'seek_evidence', 'guided_revise']` | `win` | `win` | `win` | **`win` (`+4.500`)** |
| `smallari_expansion40_010_problems_` | `ai_scientist` | `changed` / `win` (`+2.500`) | `changed` `['guided_revise', 'seek_evidence', 'clarify']` | `win` | `win` | `win` | **`win` (`+3.333`)** |
| `ws_02_agentic_ai` | `direct` | `changed` / `tie` (`+0.000`) | `changed` `['guided_revise', 'local_patch', 'clarify', 'guided_revise']` | `tie` | `win` | `tie` | **`tie` (`+1.167`)** |
| `ws_02_agentic_ai` | `ai_scientist` | `changed` / `win` (`+5.500`) | `changed` `['guided_revise', 'seek_evidence', 'guided_revise']` | `win` | `win` | `win` | **`win` (`+5.167`)** |
| `ws_02_agentic_ai` | `edara` | `changed` / `win` (`+2.500`) | `changed` `['guided_revise', 'local_patch', 'retain_and_test']` | `win` | `win` | `tie` | **`win` (`+1.667`)** |
| `smallari_expansion40_002_unlearnin` | `direct` | `changed` / `loss` (`-2.000`) | `changed` `['guided_revise', 'seek_evidence', 'guided_revise']` | `loss` | `loss` | `loss` | **`loss` (`-1.500`)** |
| `smallari_expansion40_002_unlearnin` | `ai_scientist` | `changed` / `loss` (`-5.000`) | `changed` `['guided_revise', 'regenerate', 'seek_evidence']` | `tie` | `tie` | `win` | **`tie` (`+0.667`)** |
| `smallari_expansion40_002_unlearnin` | `edara` | `revision_error:JSONDecodeError` / `loss` (`-5.000`) | `changed` `['guided_revise', 'seek_evidence', 'local_patch']` | `loss` | `loss` | `loss` | **`loss` (`-4.000`)** |
| `smallari_expansion40_011_test_time` | `direct` | `changed` / `tie` (`+0.500`) | `changed` `['guided_revise', 'seek_evidence', 'guided_revise']` | `loss` | `tie` | `loss` | **`loss` (`-0.167`)** |
| `smallari_expansion40_011_test_time` | `ai_scientist` | `changed` / `win` (`+4.000`) | `changed` `['guided_revise', 'regenerate', 'seek_evidence', 'guided_revise']` | `win` | `win` | `win` | **`win` (`+4.833`)** |
| `survey_08_kv_cache_compression` | `direct` | `changed` / `win` (`+2.500`) | `changed` `['guided_revise', 'seek_evidence', 'guided_revise']` | `win` | `win` | `tie` | **`win` (`+2.333`)** |
| `survey_08_kv_cache_compression` | `ai_scientist` | `changed` / `win` (`+4.500`) | `changed` `['guided_revise', 'seek_evidence', 'guided_revise']` | `tie` | `loss` | `tie` | **`tie` (`-0.333`)** |
| `smallari_expansion40_008_graph_net` | `direct` | `revision_error:invalid_rewrite` / `loss` (`-5.000`) | `changed` `['guided_revise', 'seek_evidence', 'clarify']` | `loss` | `loss` | `loss` | **`loss` (`-3.167`)** |
| `smallari_expansion40_008_graph_net` | `ai_scientist` | `changed` / `loss` (`-4.000`) | `changed` `['guided_revise', 'seek_evidence', 'clarify']` | `tie` | `tie` | `tie` | **`tie` (`+0.000`)** |
| `survey_05_mechanistic_interpretabi` | `direct` | `changed` / `tie` (`+0.500`) | `changed` `['guided_revise', 'seek_evidence', 'guided_revise']` | `tie` | `loss` | `loss` | **`loss` (`-2.333`)** |
| `survey_05_mechanistic_interpretabi` | `ai_scientist` | `changed` / `win` (`+6.000`) | `changed` `['guided_revise', 'seek_evidence', 'guided_revise']` | `win` | `win` | `win` | **`win` (`+5.500`)** |
| `smallari_expansion40_003_quantum_c` | `direct` | `changed` / `win` (`+3.500`) | `changed` `['guided_revise', 'seek_evidence', 'guided_revise']` | `win` | `win` | `win` | **`win` (`+3.167`)** |
| `smallari_expansion40_003_quantum_c` | `ai_scientist` | `changed` / `win` (`+1.500`) | `changed` `['guided_revise', 'seek_evidence', 'guided_revise']` | `loss` | `loss` | `tie` | **`loss` (`-2.167`)** |

- **Metered Cost (USD)**: `Arm E = $0.6713`, `3-Judge Blind Eval = $1.8214`, `Total = $2.4927`
