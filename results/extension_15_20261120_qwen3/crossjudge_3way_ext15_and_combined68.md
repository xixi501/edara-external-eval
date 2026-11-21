# 15-Topic Expansion (`N=30`) & Combined 30-Topic (`N=68`) 3-Judge + Mechanism Evaluation Report

- **Extension Run Directory**: `/workspace/edara/staging/04_external_eval_refs/edara_mvp_eval/outputs/extension_15_20261120_qwen3`
- **v4 Reference Directory**: `/workspace/edara/staging/04_external_eval_refs/edara_mvp_eval/outputs/unified_pilot_20261119_v4`
- **Cross-Judge Outputs**:
  - `crossjudge_3way_ext15_and_combined68.json`
  - `mechanism_audit_stats_ext15.json` (`N=30`)
  - `mechanism_audit_stats_combined68.json` (`N=68`)
- **Blind Judges**:
  1. `Qwen3.8-flash` (`dashscope-qwen-profile.json`, calibration `12/12 + 4/4 = 16/16`)
  2. `DeepSeek-v4.1-flash-1017` (`dashscope-deepseek-profile.json`, calibration `10/12 + 4/4`)
  3. `GLM-5.3-flash` (`p1b-critic-profile.json`, calibration `10/12 + 4/4`)

---

## 1. Combined 30-Topic Results (`N=68`: `v4` 38 + `ext15` 30, 0 Overlap, 0 Unresolved)

### 1.1 Aggregate `E_vs_D` Across All 3 Blind Judges (`N=68`, 30 Topics)

| Judge | Win / Loss / Tie | Topic Net Wins (95% CI) | Score Sum Delta (95% CI) | Order Sens. |
| --- | :---: | :---: | :---: | ---: |
| **GLM-5.3-flash** | **`39 / 18 / 11`** | **`+0.294`** (`[+0.067, +0.506]`) | **`+0.833`** (`[+0.111, +1.511]`) | 11 / 68 |
| **DeepSeek-v4.1-flash** | **`37 / 19 / 12`** | **`+0.244`** (`[+0.033, +0.444]`) | **`+0.822`** (`[-0.042, +1.625]`) | 12 / 68 |
| **Qwen3.8-flash** | **`38 / 15 / 15`** | **`+0.322`** (`[+0.100, +0.533]`) | **`+1.158`** (`[+0.344, +1.889]`) | 15 / 68 |
| **3-Judge Majority Vote** | **`38 / 17 / 13`** | **`+0.283`** (`[+0.067, +0.483]`) | **`+0.938`** (`[+0.157, +1.636]`) | — |

### 1.2 Per-Dimension Topic-Weighted Deltas (`N=68`, 30 Topics)

| Dimension | GLM-5.3-flash | DeepSeek-v4.1-flash | Qwen3.8-flash |
| --- | ---: | ---: | ---: |
| Lit2Test `falsifiability` | **`+0.447`** | **`+0.578`** | **`+0.661`** |
| Lit2Test `decisive_metric` | **`+0.272`** | `+0.125` | **`+0.322`** |
| Lit2Test `minimality_feasibility` | **`+0.178`** | `+0.042` | `+0.072` |
| Lit2Test `grounding` | `+0.000` | `+0.058` | **`+0.067`** |
| Lit2Test `hypothesis_specificity` | `-0.064` | `+0.019` | **`+0.036`** |

- **AlphaIdeaBench & ClaimNov (`N=68` Primary Judges Combined)**:
  - `testability`: **`+0.233`** (`AI-Scientist` `n=29`: **`+0.345`**; `Direct` `n=30`: **`+0.167`**)
  - `verification_feasibility`: **`+0.006`** (`AI-Scientist` `n=29`: **`+0.138`**)
  - `method_novelty`: **`+0.011`** (`Direct` `n=30`: **`+0.067`**)
  - `insight_novelty`: **`+0.006`** (`Direct` `n=30`: **`+0.067`**; `E-DARA` `n=9`: **`+0.111`**)
  - `context_grounding`: `-0.122`
  - `ClaimNov novelty_delta`: **`-0.002`** (`Direct` `n=30`: **`+0.019`**)

### 1.3 By Host (`N=68`)

| Host | n | GLM (`W/L/T`, net, sum) | DeepSeek (`W/L/T`, net, sum) | Qwen3.8-flash (`W/L/T`, net, sum) | 3-Judge Majority (`W/L/T`, net, sum) |
| --- | :---: | --- | --- | --- | --- |
| **`ai_scientist`** | **29** | `20/5/4`, **`+0.517`**, `+1.810` | `18/6/5`, **`+0.414`**, `+1.707` | `21/4/4`, **`+0.586`**, `+2.069` | **`20/5/4`, `+0.517`, `+1.862`** |
| **`edara`** | **9** | `7/1/1`, **`+0.667`**, `+1.389` | `7/1/1`, **`+0.667`**, `+1.778` | `4/1/4`, **`+0.333`**, `+1.056` | **`7/1/1`, `+0.667`, `+1.407`** |
| **`direct`** | **30** | `12/12/6`, `+0.000`, `-0.217` | `12/12/6`, `+0.000`, `-0.217` | `13/10/7`, **`+0.100`**, `+0.350` | `11/11/8`, `+0.000`, `-0.028` |

### 1.4 By Trajectory Cleanliness (`N=68`)

| Stratum | n | GLM (`W/L/T`, net) | DeepSeek (`W/L/T`, net) | Qwen3.8-flash (`W/L/T`, net) | 3-Judge Majority (`W/L/T`, net) |
| --- | :---: | --- | --- | --- | --- |
| **Clean closed-loop (`guided_revise` end)** | **42** | `27/8/7`, **`+0.460`** | `25/7/10`, **`+0.433`** | `28/4/10`, **`+0.593`** | **`27/6/9`, `+0.487`** |
| Clean closed-loop (`E/D <= 1.15x`) | 19 | `9/7/3`, `-0.033` | `7/6/6`, `+0.022` | `7/3/9`, **`+0.189`** | `7/5/7`, `+0.011` |
| Incomplete / broken `E` trajectory | 26 | `12/10/4`, `+0.125` | `12/12/2`, `+0.094` | `10/11/5`, `+0.031` | `11/11/4`, `+0.062` |

### 1.5 Inter-Judge Agreement (`N=68`)
- **DeepSeek vs Qwen3.8-flash**: Exact 3-class agreement `53/68` (`77.9%`), **`0 / 68` (`0%`) direct win↔loss flips**.
- **GLM vs Qwen3.8-flash**: Exact 3-class agreement `53/68` (`77.9%`), **`1 / 68` (`1.5%`) direct win↔loss flip**.
- **GLM vs DeepSeek**: Exact 3-class agreement `55/68` (`80.9%`), **`2 / 68` (`2.9%`) direct win↔loss flips**.

---

## 2. Standalone Extension Cohort (`N=30`, 15 New Stratified Topics)

### 2.1 Aggregate `E_vs_D` (`N=30`, 15 Topics)

| Judge | Win / Loss / Tie | Topic Net Wins (95% CI) | Score Sum Delta (95% CI) | Order Sens. |
| --- | :---: | :---: | :---: | ---: |
| **GLM-5.3-flash** | `14 / 11 / 5` | **`+0.100`** (`[-0.200, +0.433]`) | **`+0.283`** (`[-0.750, +1.300]`) | 5 / 30 |
| **DeepSeek-v4.1-flash** | `15 / 10 / 5` | **`+0.167`** (`[-0.133, +0.467]`) | **`+0.317`** (`[-1.017, +1.600]`) | 5 / 30 |
| **Qwen3.8-flash** | `16 / 9 / 5` | **`+0.233`** (`[-0.100, +0.533]`) | **`+0.867`** (`[-0.500, +2.100]`) | 5 / 30 |
| **3-Judge Majority Vote** | **`15 / 10 / 5`** | **`+0.167`** (`[-0.167, +0.467]`) | **`+0.489`** (`[-0.739, +1.639]`) | — |

### 2.2 Extension Breakdown by Host, Benchmark & Cleanliness (`N=30`)

| Subgroup | n | GLM (`W/L/T`, net, sum) | DeepSeek (`W/L/T`, net, sum) | Qwen3.8-flash (`W/L/T`, net, sum) | 3-Judge Majority (`W/L/T`, net, sum) |
| --- | :---: | --- | --- | --- | --- |
| **`ai_scientist`** | 15 | `7/4/4`, `+0.200`, `+0.900` | `8/4/3`, `+0.267`, `+0.933` | **`9/3/3`, `+0.400`, `+1.667`** | **`8/4/3`, `+0.267`, `+1.167`** |
| **`direct`** *(incl. 1 error)* | 15 | `7/7/1`, `+0.000`, `-0.333` | `7/6/2`, `+0.067`, `-0.300` | `7/6/2`, `+0.067`, `+0.067` | **`7/6/2`, `+0.067`, `-0.189`** |
| **`AlphaIdeaBench_survey`** | 8 | `4/3/1`, `+0.125`, `+0.750` | `5/2/1`, `+0.375`, `+1.062` | `4/2/2`, `+0.250`, `+1.625` | **`4/2/2`, `+0.250`, `+1.146`** |
| **`Lit2Test_expansion40`** | 22 | `10/8/4`, `+0.091`, `+0.114` | `10/8/4`, `+0.091`, `+0.045` | `12/7/3`, `+0.227`, `+0.591` | **`11/8/3`, `+0.136`, `+0.250`** |
| **Clean closed-loop (`E`)** | **22** | `11/6/5`, **`+0.250`** | `12/5/5`, **`+0.375`** | `13/4/5`, **`+0.417`** | **`12/5/5`, `+0.333`** |
| Incomplete `E` *(incl. `008/direct`)* | 8 | `3/5/0`, `-0.200` | `3/5/0`, `-0.200` | `3/5/0`, `-0.200` | `3/5/0`, `-0.200` |

- **Extension Inter-Judge Agreement (`N=30`)**:
  - `GLM vs DeepSeek`: **`26/30 (86.7%)` exact agreement, `0/30 (0%)` direct win↔loss flips**
  - `GLM vs Qwen3.8-flash`: **`26/30 (86.7%)` exact agreement, `0/30 (0%)` direct win↔loss flips**
  - `DeepSeek vs Qwen3.8-flash`: **`26/30 (86.7%)` exact agreement, `0/30 (0%)` direct win↔loss flips**

---

## 3. Deterministic Objective Mechanism Ledger (`N=30` & `N=68`)

| Mechanism Metric | Extension (`N=30`) | Combined (`N=68`, 30 Topics) |
| --- | --- | --- |
| **Process hype defects (`manuscripts / total`)** | `A: 2/2`, `B: 2/2`, `C: 0/0`, `D: 9/15`, **`E: 0/0`** | `A: 6/9`, `B: 6/9`, `C: 1/3`, `D: 17/32`, **`E: 0/0`** |
| **Unverified external entities (`manuscripts / total`)** | `A: 30/158`, `B: 30/153`, `C: 28/122`, `D: 30/171`, **`E: 24/67 (-60.8% vs D)`** | `A: 67/357`, `B: 67/363`, `C: 62/282`, `D: 67/367`, **`E: 53/169 (-54.0% vs D)`** |
| **Core 3 design elements (`control / metric / decision / all_3`)** | `A/B/C: 28/30`, `D: 29/30`, **`E: 30/30 (100%)`** | `A/B/C: 62/68`, `D: 64/68`, **`E: 68/68 (100%)`** |
| **Confound control placed in design field (`required n`)** | `n=15`: `A: 4/15`, `D: 7/15`, **`C/E: 13/15 (86.7%)`** | `n=38`: `A: 15/38`, `D: 27/38`, **`C/E: 36/38 (94.7%)`** |
| **Original host schema key retention (`orig_keys_kept_all`)** | **`30/30 (100%)`**, `0` anchor drops | **`68/68 (100%)`**, `2` regression steps blocked by `E` guard |
| **Factual audit defect reduction (`step-0 -> final E`)** | `34` (`18` drafts) $\rightarrow$ `8` (**`-76.5%`**) | `74` (`41` drafts) $\rightarrow$ `16` (**`-78.4%`**) |
