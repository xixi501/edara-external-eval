# Action A: Cross-Backbone Generalization (`Qwen3.8-flash` Host + Editor, `N=30` across `10` Domains)

## 1. Role-Inverted Blind Evaluation (`Arm E` vs. `Arm D`)

| Judge (Role) | Win / Loss / Tie | Topic Net Wins (95% CI) | Score Sum Delta (95% CI) | Order Sens. |
| --- | :---: | :---: | :---: | ---: |
| **DeepSeek-v4.1-flash (Independent Primary)** | `10 / 8 / 12` | **`+0.067`** (`[-0.333, +0.467]`) | **`-0.433`** (`[-2.000, +1.083]`) | 12 / 30 |
| **GLM-5.3-flash (Independent Primary)** | `13 / 8 / 9` | **`+0.167`** (`[-0.133, +0.467]`) | **`-0.100`** (`[-1.083, +0.900]`) | 9 / 30 |
| **Independent 2-Judge Consensus (Disjoint)** | `16 / 10 / 4` | **`+0.200`** (`[-0.233, +0.600]`) | **`-0.267`** (`[-1.525, +0.958]`) | 0 / 30 |
| Qwen3.8-flash (Same-Family Reference) | `12 / 10 / 8` | **`+0.067`** (`[-0.233, +0.333]`) | **`+0.350`** (`[-1.233, +1.750]`) | 8 / 30 |
| **3-Judge Majority Vote** | `14 / 8 / 8` | **`+0.200`** (`[-0.167, +0.533]`) | **`-0.061`** (`[-1.395, +1.194]`) | 0 / 30 |

## 2. By Upstream Host (`DeepSeek Independent`, `GLM Independent`, & `3-Judge Majority`)

| Host | $N$ | DeepSeek W/L/T (`Net`, `ΔSum`) | GLM W/L/T (`Net`, `ΔSum`) | **3-Judge Majority W/L/T (`Net`, `ΔSum`)** |
| --- | :---: | :---: | :---: | :---: |
| `ai_scientist` | 10 | `3/5/2` (`-0.200`, `-1.650`) | `4/5/1` (`-0.100`, `-1.900`) | **`3/5/2`** (`-0.200`, `-1.967`) |
| `evoscientist` | 10 | `4/1/5` (`+0.300`, `+1.100`) | `5/0/5` (`+0.500`, `+1.050`) | **`7/1/2`** (`+0.600`, `+1.650`) |
| `direct` | 10 | `3/2/5` (`+0.100`, `-0.750`) | `4/3/3` (`+0.100`, `+0.550`) | **`4/2/4`** (`+0.200`, `+0.133`) |

## 3. Deterministic Epistemic Hygiene on `Qwen3.8-flash` Backbone

- **Schema Keys Preserved (`E`)**: `30 / 30`
- **Process Hype (`A -> D -> E`)**: `0 -> 4 -> 0`
- **Unverified External Entities (`A -> D -> E`)**: `174 -> 186 -> 173`
- **Blocking Confounds Addressed (`A -> D -> E`)**: `7/24 -> 9/24 -> 19/24`
- **Trajectory Integrity (`E`)**: `errors=2`, `clean_closed_loop=28/30`
- **Total Metered Cost (USD)**: `$3.3072`
