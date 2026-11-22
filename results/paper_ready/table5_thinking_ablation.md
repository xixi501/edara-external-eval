| Regime / Stratum (`N=30` Paired Wave-1 Manuscripts) | Qwen Primary W/L/T (`Net`, `$\Delta$Sum`) | DeepSeek W/L/T (`Net`, `$\Delta$Sum`) | GLM W/L/T (`Net`, `$\Delta$Sum`) | **3-Judge Majority W/L/T (`Net [95% CI]`, `$\Delta$Sum [95% CI]`)** |
| :--- | :---: | :---: | :---: | :---: |
| `Fast Mode` (`thinking=off`, `4096` cap) | `15/7/8` (`+0.267`, `+1.50`) | `12/12/6` (`+0.000`, `+0.35`) | `12/7/11` (`+0.167`, `+0.83`) | `11/8/11` (`+0.100`, `+0.89`) |
| **`Slow-Thinking Mode` (`thinking=on`, `8192` cap)** | **`13/7/10`** (`+0.200`, `+1.07`) | **`10/4/16`** (`+0.200`, `+1.15`) | **`19/4/7`** (`+0.500`, `+1.48`) | **`13/3/14`** (**`+0.333`** `[+0.20, +0.47]`, **`+1.23`** `[+0.85, +1.64]`) |
| ↳ AI-Scientist-v2 (`Slow-Thinking`, $n=10$) | `4/2/4` (`+0.200`, `+1.40`) | `5/2/3` (`+0.300`, `+1.45`) | `8/1/1` (`+0.700`, `+2.60`) | **`6/1/3`** (`+0.500`, `+1.82`) |
| ↳ EvoScientist (v2.1.2) (`Slow-Thinking`, $n=10$) | `6/3/1` (`+0.300`, `+1.45`) | `3/2/5` (`+0.100`, `+0.85`) | `7/1/2` (`+0.600`, `+1.65`) | **`5/2/3`** (`+0.300`, `+1.32`) |
| ↳ Direct LLM (`Slow-Thinking`, $n=10$) | `3/2/5` (`+0.100`, `+0.35`) | `2/0/8` (`+0.200`, `+1.15`) | `4/2/4` (`+0.200`, `+0.20`) | **`2/0/8`** (`+0.200`, `+0.57`) |

- **The Reasoning-Without-Governance Paradox**: Enabling test-time slow-thinking in unconstrained `Arm D_think` **inflated fabricated process-hype claims from `A=14` & `D_fast=42` up to `D_think=91` (`+116.7%` over fast baseline)** while leaving unverified entities at `159` and addressing only `4/14` blocking confounds. Governed **`Arm E_think` reduced process hype `91 -> 0` (`-100.0%`)**, cut unverified entities **`159 -> 33` (`-79.2%`)**, and resolved **`12/14` (`85.7%`)** blocking confounds.
- **Initial Slow-Thinking Generation Probe (`n=10` Direct topics)**: Native slow-thinking during `Arm A` generation likewise increased process hype (`3 -> 8`) and unverified external entities (`45 -> 59`).
