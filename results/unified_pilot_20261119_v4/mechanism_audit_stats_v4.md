# `v4` Core (`N=38`, 15 Topics) Mechanism Audit & Invariant Ledger

- **Source Run**: `/workspace/edara/staging/04_external_eval_refs/edara_mvp_eval/outputs/unified_pilot_20261119_v4`
- **Structured Raw Metrics**: `mechanism_audit_stats_v4.json`
- **Scope**: Zero-cost deterministic + trajectory audit across all 38 frozen manuscripts (`direct=15`, `ai_scientist=14`, `edara=9`) comparing Original `A`, Generic Local `B`, Plugin Local `C`, Generic Multi-Round `D`, and Full Plugin `E`.

## 1. Cross-Host Schema Write-Back & Structural Fidelity (`E1`)

| Host | Manuscripts (`n`) | Original Keys Kept (`100%`) | Non-Empty `design_field` in `E` | Total Fields Changed in `E` |
| --- | ---: | ---: | ---: | ---: |
| `direct` | 15 | **`15 / 15` (`100%`)** | `15 / 15` (`100%`) | 154 |
| `ai_scientist` | 14 | **`14 / 14` (`100%`)** | `14 / 14` (`100%`) | 103 |
| `edara` (`proposal` + `experiment_plan`) | 9 | **`9 / 9` (`100%`)** | `9 / 9` (`100%`) | 183 |
| **Total** | **38** | **`38 / 38` (`100%`)** | **`38 / 38` (`100%`)** | **440** |

## 2. Deterministic Unattested Process Hype & Unverified External Entity Control (`E2`)

| Arm | Manuscripts w/ Unattested Process Hype (`pre-registered`, `already validated`) | Total Hype Defects | Manuscripts w/ Unverified External Entities | Total Unverified Entities |
| --- | ---: | ---: | ---: | ---: |
| **A (Original)** | 4 / 38 | 7 | 34 / 38 | 179 |
| **B (Generic Local)** | 4 / 38 | 7 | 35 / 38 | 186 |
| **C (Plugin Local)** | 1 / 38 | 3 | 30 / 38 | 139 |
| **D (Generic Multi-Round)** | **8 / 38** *(2x worse than A)* | **17** *(2.4x worse than A)* | 35 / 38 | 167 |
| **E (Full Plugin)** | **`0 / 38` (`0%`)** | **`0`** (`-100%`) | **`24 / 38`** | **`65`** (`-63.7% vs A`) |

- **Empirical Confirmation of LLM Overclaim Drift**: Unconstrained generic multi-round critique+revision (`Arm D`) **more than doubled** unattested process hype (`7 -> 17` occurrences across `4 -> 8` manuscripts), whereas our deterministic + schema-guided governance (`Arm E`) eliminated it to **`0 / 38` (`0`)**.

## 3. Confound Audit, Field Realization & Structural Element Coverage (`E3` / `E4`)

- **Confound Detection Rate (`Arm E` step 0)**: **`35 / 38` (`92.1%`)** manuscripts flagged with alternative explanations (`23` blocking, `12` recommended).
- **Mandatory Discriminating-Arm Targets (`targets_required_n = 25`)**:
  - Original `A`: **`0 / 25` (`0.0%`)** addressed in `design_field`
  - Generic Multi-Round `D`: **`20 / 25` (`80.0%`)**
  - Plugin Local `C`: **`24 / 25` (`96.0%`)** — initial local pass missed `8 / 25` (`32.0%`), **`confound_enforcement` triggered 8 times and recovered `8 / 8` (`100%`)** into `design_field`!
  - Full Plugin `E`: **`23 / 25` (`92.0%` overall; `23 / 24 = 95.8%` excluding the single `smallari_002/edara` JSONDecodeError crash)** — `guided_revise` directly realized discriminating controls on the first pass cleanly (`0` needed retries; `1` non-crash miss was `smallari_001/ai_scientist`).
- **Structural Design Elements (`control / metric / decision` out of 38)**:
  - `A`: `38 / 37 / 35`
  - `B`: `38 / 37 / 35`
  - `C`: `38 / 37 / 35`
  - `D`: `38 / 38 / 35` (left 3 missing explicit `decision` rules)
  - **`E`**: **`38 / 38 / 38` (`100%` complete across control, metric, and decision rules)**

## 4. Independent Factual Defects, Active Pre-Cutoff Search & Preservation Guard (`E5` / `E7`)

- **Factual Defect Reduction in `Arm E`**:
  - Initial (`step 0`): **`40` defects** across **`23 / 38` manuscripts** (`prediction_as_observed=18`, `unattested_process=10`, `fabrication=4`, `coverage_contradiction=3`, `numeric_mismatch=2`, `attribution=2`, `internal_inconsistency=1`).
  - Final (`last step`): **`8` remaining (`-80.0%`)**, including **`1` `fabrication`** which was the single crashed unedited `smallari_002/edara` draft (**`0` `fabrication` remaining across all 37 completed runs**).
- **Pre-Cutoff Evidence Retrieval (`seek_evidence`)**: Triggered on **`30 / 38`** manuscripts; retrieved **`58` papers** across **`26`** runs; **`20 / 26`** ran subsequent `guided_revise` synthesis cleanly while **`6 / 26`** hit the old `rounds=3` cap fixed in commit `fbbb12e`.
- **Deterministic Semantic Preservation Guard (`preserves_design_elements` & `preserves_related_work_anchors`)**:
  - During `Arm E` runtime, the guard intercepted and blocked **`2`** regressions (both `dropped_supplied_reference`), preventing supplied packet citation loss.
  - End-state audit (`A -> final` across all 38 manuscripts):
    - **`Arm E`**: **`0 / 38`** design/anchor violations (`dropped_supplied_reference = 0`).
    - **`Arm C`**: **`0 / 38`** design/anchor violations.
    - **`Arm D`** (unguarded): **`1 / 38`** design/anchor violation (`dropped_supplied_reference = 1`).
