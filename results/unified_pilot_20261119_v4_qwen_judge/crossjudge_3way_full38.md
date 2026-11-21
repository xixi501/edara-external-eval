# 3-Way Cross-Family Blind Judge Verification (`N=38`, 15 Topics)

- **Reference Run (Frozen `D` vs `E`)**: `/workspace/edara/staging/04_external_eval_refs/edara_mvp_eval/outputs/unified_pilot_20261119_v4`
- **Judges**:
  1. `GLM-5.3-flash` (`p1b-critic-profile.json`, calibration `10/12 + 4/4`)
  2. `DeepSeek-flash / v4.1-flash-1017` (`p1a-live-profile.json` + `dashscope-deepseek-profile.json`, calibration `10/12 + 4/4`)
  3. `Qwen3.8-flash` (`dashscope-qwen-profile.json`, **calibration `12/12 + 4/4 = 16/16`**)
- **Structured Raw Output**: `crossjudge_3way_full38.json`

## 1. Aggregate `E_vs_D` Results (`N=38`, 15 Topics, 0 Unresolved)

| Judge | Win / Loss / Tie | Topic Net Wins (95% CI) | Score Sum Delta (95% CI) | Order Sens. |
| --- | :---: | :---: | :---: | ---: |
| **GLM-5.3-flash** | `25 / 7 / 6` | **`+0.489`** (`[+0.200, +0.744]`) | **`+1.383`** (`[+0.350, +2.228]`) | 6 / 38 |
| **DeepSeek-flash** | `22 / 9 / 7` | **`+0.322`** (`[+0.022, +0.611]`) | **`+1.328`** (`[+0.228, +2.283]`) | 7 / 38 |
| **Qwen3.8-flash** | `22 / 6 / 10` | **`+0.411`** (`[+0.122, +0.667]`) | **`+1.450`** (`[+0.467, +2.217]`) | 10 / 38 |
| **3-Judge Majority Vote** | **`23 / 7 / 8`** | **`+0.400`** (`[+0.111, +0.656]`) | **`+1.387`** | — |

### Per-Dimension Topic-Weighted Deltas (`N=38`)
| Lit2Test Dimension | GLM-5.3-flash | DeepSeek-flash | Qwen3.8-flash |
| --- | ---: | ---: | ---: |
| `falsifiability` | `+0.594` | `+0.739` | **`+0.756`** |
| `decisive_metric` | **`+0.444`** | `+0.250` | `+0.411` |
| `minimality_feasibility` | **`+0.289`** | `+0.067` | `+0.078` |
| `grounding` | `+0.050` | **`+0.133`** | **`+0.133`** |
| `hypothesis_specificity` | `+0.006` | **`+0.139`** | `+0.072` |

## 2. By Host (`N=38`)

| Host | n | GLM (`W/L/T`, net, sum) | DeepSeek (`W/L/T`, net, sum) | Qwen3.8-flash (`W/L/T`, net, sum) |
| --- | :---: | --- | --- | --- |
| `ai_scientist` | 14 | `13/1/0`, `+0.857`, `+2.786` | `10/2/2`, `+0.571`, `+2.536` | **`12/1/1`, `+0.786`, `+2.500`** |
| `edara` | 9 | `7/1/1`, `+0.667`, `+1.389` | `7/1/1`, `+0.667`, `+1.778` | `4/1/4`, `+0.333`, `+1.056` |
| `direct` | 15 | `5/5/5`, `+0.000`, `-0.100` | `5/6/4`, `-0.067`, `-0.133` | `6/4/5`, `+0.133`, `+0.633` |

## 3. Trajectory Cleanliness & Length-Controlled Stratification (`N=38`)

| Stratum | n | GLM (`W/L/T`, net) | DeepSeek (`W/L/T`, net) | Qwen3.8-flash (`W/L/T`, net) |
| --- | :---: | --- | --- | --- |
| Clean closed-loop (`guided_revise` end) | 20 | `16/2/2`, `+0.654` | `13/2/5`, `+0.487` | **`15/0/5`, `+0.756` (0 losses!)** |
| Clean closed-loop (`E/D <= 1.15x`) | 9 | `6/2/1`, `+0.286` | `4/2/3`, `+0.190` | **`4/0/5`, `+0.476` (0 losses!)** |
| Incomplete / broken `E` trajectory | 18 | `9/5/4`, `+0.273` | `9/7/2`, `+0.227` | `7/6/5`, `+0.136` *(all 6 losses here)* |

## 4. Inter-Judge Agreement (`N=38`)
- **DeepSeek vs Qwen3.8-flash**: Exact 3-class agreement `27/38` (`71.1%`), **`0 / 38` (`0%`) direct win↔loss flips**.
- **GLM vs Qwen3.8-flash**: Exact 3-class agreement `27/38` (`71.1%`), **`1 / 38` (`2.6%`) direct win↔loss flip**.
- **GLM vs DeepSeek**: Exact 3-class agreement `29/38` (`76.3%`), **`2 / 38` (`5.3%`) direct win↔loss flips**.
