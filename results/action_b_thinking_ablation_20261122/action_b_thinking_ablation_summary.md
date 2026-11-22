# Action B: Slow-Thinking (`thinking: enabled`) + Symmetric `8192`-Token Budget Ablation (`N=30` Manuscripts)

## 1. Aggregate `E_think vs D_think` Across 3 Calibrated Blind Judges

| Judge | Win / Loss / Tie | Topic Net Wins (95% CI) | Score Sum Delta (95% CI) | Order Sens. |
| --- | :---: | :---: | :---: | ---: |
| Qwen3.8-flash (Primary Independent) | `13 / 7 / 10` | **`+0.200`** (`[+0.000, +0.400]`) | **`+1.067`** (`[+0.417, +1.817]`) | 10 / 30 |
| DeepSeek-v4.1-flash | `10 / 4 / 16` | **`+0.200`** (`[+0.067, +0.333]`) | **`+1.150`** (`[+0.550, +1.733]`) | 16 / 30 |
| GLM-5.3-flash | `19 / 4 / 7` | **`+0.500`** (`[+0.300, +0.700]`) | **`+1.483`** (`[+1.133, +1.850]`) | 7 / 30 |
| **3-Judge Majority Vote** | `13 / 3 / 14` | **`+0.333`** (`[+0.200, +0.467]`) | **`+1.233`** (`[+0.850, +1.639]`) | 0 / 30 |

## 2. By Upstream Host (`Qwen3.8-flash Primary` & `3-Judge Majority`)

| Host | $N$ | Qwen W/L/T (`Net`, `ΔSum`) | **3-Judge Majority W/L/T (`Net`, `ΔSum`)** |
| --- | :---: | :---: | :---: |
| `ai_scientist` | 10 | `4/2/4` (`+0.200`, `+1.400`) | **`6/1/3`** (`+0.500`, `+1.817`) |
| `evoscientist` | 10 | `6/3/1` (`+0.300`, `+1.450`) | **`5/2/3`** (`+0.300`, `+1.317`) |
| `direct` | 10 | `3/2/5` (`+0.100`, `+0.350`) | **`2/0/8`** (`+0.200`, `+0.567`) |

## 3. 2×2 Epistemic Hygiene Comparison (`Fast 4096` vs. `Slow-Thinking 8192`) & Reasoning Token Telemetry

- **Schema Keys Preserved (`E_think`)**: `30 / 30`
- **Slow-Thinking Process Hype (`A -> D_think -> E_think`)**: `14 -> 91 -> 0`
- **Slow-Thinking Unverified Entities (`A -> D_think -> E_think`)**: `158 -> 159 -> 33`
- **Slow-Thinking Blocking Confounds Addressed (`A -> D_think -> E_think`)**: `5/14 -> 4/14 -> 12/14`
- **10-Topic Direct Initial Generation Probe (`Fast A` vs `Slow-Thinking A_think`)**: Process Hype `3 -> 8`, Unverified Entities `45 -> 59`
- **Metered Reasoning Tokens (`D_think` vs `E_think`)**: `D_think = 295,388`, `E_think = 1,112,372`
- **Total Metered Cost (USD)**: `$4.5238`

## 4. Paired Reference: Identical 30 Wave-1 Manuscripts under `Fast 4096`
- **Wave-1 Fast-4096 Qwen Primary**: `15W/7L/8T` (`Net = +0.267`, `ΔSum = +1.500`)
- **Wave-1 Fast-4096 3-Judge Majority**: `11W/8L/11T` (`Net = +0.100`, `ΔSum = +0.894`)
- **Wave-1 Fast-4096 Hygiene**: Hype `D=42 -> E=3`, Unverified `D=169 -> E=47`, Confounds `D=6/14 -> E=13/14`
