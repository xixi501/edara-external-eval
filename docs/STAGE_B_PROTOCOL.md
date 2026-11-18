# Stage B evaluation protocol

The original tasks, initial drafts, and v3/v4 runs are historical inputs. The run manifest records their SHA-256 hashes. The v4 dataset has three tasks, 24 artifacts and 84 bidirectional comparisons (21 order-sensitive). The three development tasks appear in the fifteen-task Stage B set. Report the remaining twelve separately as expansion tasks.

| Target | Historical evidence | Stage B primary evidence |
|---|---|---|
| Honesty | Prior-fact audits, attribution and overclaim count/rate | Accurate credit and clear separation of fact, inference and prediction |
| Credibility | Gap audit, feasibility, falsification and specification | Supported premises or declared assumptions, coherent mechanism, discriminating test and feasible resources |
| Novelty | ClaimNov and raw novelty scores | Substantive testable difference relative to the dated common evidence |
| Diagnostics | Gate Pass, Strict Gate, Trusted Nov, Spec Comp, Method Ret | Retained in audit records. Method Ret measures text overlap. Trusted Nov applies an integrity/uncertainty filter. |

Each primary axis is pass, fail or unresolved. All three must pass to count toward the primary joint rate. All scheduled tasks are in the denominator. Report failures and unresolved cases distinctly.

| Finding | Interpretation |
|---|---|
| v4 flagged overclaims, E-DARA 12→0 and AI Scientist v2 7→1 | Improvement detected by the existing model audit |
| Direct 2→2 despite a lower rate | The historical denominator includes hypotheses and unknowns |
| v4 middleware vs each original 3–0, Direct middleware vs self-refine 0–2–1 | Initial quality signal, including a strong negative comparison |
| Raw novelty means Direct .682→.714, AI Scientist v2 .815→.821, E-DARA .695→.718 | Small model-scored movement on three tasks |
| v4 ClaimNov vetoes zero, Strict Gate passes zero | Those binary summaries do not discriminate in this pilot |
| Fixed evidence pilot 8/8 expected routes, harder pilot 11/12 | Demonstrated routing on curated evidence |
| Harder pilot ordinary retry and structured feedback both 12/12 | No incremental benefit detected on those cases |

v4 returned no admissible contribution evidence on all 24 final artifacts. The ten local development references lack dates or have only the cutoff year. Fix evidence ingestion before scaling. Use provenance-checked claim spans, not citation presence alone. The existing blind judge sees internal audit summaries and saves only one reason. Stage B freezes candidates and evaluates each against a common evidence pool with one GLM model. Aggregate cost accounting must use request ownership. Local patches address attribution and narrow defects. Verified full overlap calls for rebuilding the core claim. Missing evidence calls for search or an unresolved result. Persistent mechanism and test ambiguity calls for focused generator feedback.

## Execution

Use the Stage B module entry point under the existing environment. It always runs the five configured arms. Use distinct output directories for development and expansion. The task checkpoint includes a ledger of model requests, response hashes/inputs by identity, provider responses, candidate history and separately frozen final evidence. Provider SDK retries are internal to a request, so reported requests refer to calls visible to the runner.

From the external evaluation directory, run the development three-task evaluation with saved Direct drafts:

```bash
.venv/bin/python pilot3_runs/run_logged.py pilot3_runs/stage_b_regression_v1.log \
  .venv/bin/python -m edara_mvp_eval.stage_b \
  --tasks-cache edara_mvp_eval/outputs/tasks_pilot3.json \
  --direct-inputs pilot3_runs/direct/direct_outputs.jsonl \
  --gen-profile /workspace/edara/data/p1b-idea-profile.json \
  --judge-profile /workspace/edara/data/p1b-critic-profile.json \
  --out edara_mvp_eval/outputs/stage_b_regression_v1
```

After confirming development evidence is admitted and the five-arm run records all outcomes,
use the same command with `tasks_mvp15.json`, remove `--direct-inputs`, and choose
`mvp15_ablation_v2` and a fresh log. The workflow generates each Direct draft once, freezes all
candidates before final evaluation, and saves results and cost by task and stage. Resume a
run with the identical configuration, existing output directory and a new log path.

The development-to-expansion sequence is also available as
`bash pilot3_runs/run_stage_b.sh v2 initial` from the external evaluation directory.
For a restart, use the same version argument and a fresh attempt label to preserve logs.
The first live regression was interrupted after identifying irrelevant search matches. Its
original inputs, requests and responses remain under `outputs/stage_b_regression_v1`.
