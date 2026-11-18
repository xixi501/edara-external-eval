# E-DARA 15-Task MVP Evaluation & Trustworthy-Novelty Middleware Prototype

Standalone staging prototype under `staging/04_external_eval_refs/edara_mvp_eval/` (does not modify `repos/framework`).

## What It Combines

- **Task Suite (`15` tasks, built by `prepare-tasks`)**:
  - `5` local `aiib-pilot-20261105` cases (with `pypdf`-extracted seed fulltext snippets & `cutoff_date` / `forbidden_identity_hashes`)
  - `5` `Lit2Test` 4-paper OpenReview literature neighborhoods across 5 research subfields
  - `5` `AlphaIdeaBench` open workshop/survey tasks with explicit GPU/API/human budget envelopes
- **Reused Open-Source Components**:
  - `AlphaIdeaBench` dimension rubric loader (`alphaideabench.domain.dimensions`) grounded on pre-verified evidence
  - `ClaimNov` typed claim extraction, entailment profiling & `profile_veto` (with automatic `bm25_fallback` so chat-only endpoints work without `/embeddings`)
  - `Lit2Test` 6-field falsifiable schema & bidirectional A/B order-swapped pairwise judge (order-sensitive flips fold to `tie`)
- **New E-DARA Core Modules**:
  - `confound_audit.py`: hypothesis-specific confound audit. Names the alternative explanations that would reproduce the predicted observation and the arm that rules each one out, so the editor adds the decisive control instead of any control (structure follows AlphaIdeaBench testability and Lit2Test falsifiability checklists).
  - `epistemic_audit.py`: Tri-State Epistemic Audit (`prior_facts`, `prospective_hypotheses`, `unknown_dependencies`) + `SmoothScore`-style Support/Skeptic dual-pass + Pre-Cutoff `Gap-Premise` Counterexample Search + Pre-Execution Integrity Hard-Fail Gate -> `Trusted Novelty`
  - `middleware.py`: `TrustworthyNoveltyMiddleware` (`Audit -> Bounded Local Repair -> Remainder Re-Check`) runnable over any generator draft

## Quick Usage

Run offline dry-run verification & unit tests:

```bash
cd /workspace/edara/staging/04_external_eval_refs
(cd edara_mvp_eval && ../.venv/bin/python -m pytest -q)
.venv/bin/edara-mvp prepare-tasks
.venv/bin/edara-mvp generate-direct --tasks-cache edara_mvp_eval/outputs/tasks_pilot3.json --workers 3 --dry-run
.venv/bin/edara-mvp run --tasks-cache edara_mvp_eval/outputs/tasks_pilot3.json --workers 3 --dry-run
```

For full 3-topic parallel real runs (Direct, Direct+MW, AI-Scientist-v2, and E-DARA), see `HANDOFF_PILOT3_RUN.md`.

> 交接说明：本文件是外挂原型的历史 README，其中的相对路径按原工作区布局书写。本仓库的顶层 README 与 HANDOFF 为准。

Run live against your existing OpenAI-compatible Chat API (reuses E-DARA `ModelProfile` JSON or env vars; supports Chat-only endpoints via BM25 ClaimNov fallback and key-free OpenAlex pre-cutoff search via `--live-search`):

```bash
export EDARA_GEN_BASE_URL="https://your-gateway/v1"
export EDARA_GEN_API_KEY="..."
export EDARA_GEN_MODEL="your-generator-model"

# Optional separate judge model (recommended to avoid self-judging bias):
export EDARA_JUDGE_BASE_URL="$EDARA_GEN_BASE_URL"
export EDARA_JUDGE_API_KEY="$EDARA_GEN_API_KEY"
export EDARA_JUDGE_MODEL="your-judge-model"

.venv/bin/edara-mvp run \
  --tasks-cache edara_mvp_eval/outputs/tasks_pilot3.json \
  --workers 3 \
  --live-search \
  --out edara_mvp_eval/outputs/pilot_3
```

Or pass existing E-DARA profile JSON files and external arms (`ai-scientist-v2`, `edara`, plus optional `--mw-on-imported`):

```bash
.venv/bin/edara-mvp run \
  --tasks-cache edara_mvp_eval/outputs/tasks_pilot3.json \
  --gen-profile /path/to/gen_profile.json \
  --judge-profile /path/to/judge_profile.json \
  --import-arm ai-scientist-v2=/path/to/ai_outputs.jsonl \
  --import-arm edara=/path/to/edara_outputs.jsonl \
  --mw-on-imported \
  --workers 3 \
  --live-search
```

## Fixed-evidence governance pilot

`edara_mvp_eval.governance.audit_idea(task, draft, judge)` accepts a text or structured idea and
a replaceable JSON judge. It returns evidence-linked contribution findings and an action
(`correct_attribution`, `retain_and_test`, `regenerate`, or `seek_evidence`). Core combinations
are evaluated separately from their components. Findings express coverage relative to the
dated evidence supplied to the call.

From the parent `04_external_eval_refs` directory, use the frozen fixture in
`edara_mvp_eval/outputs/governance_pilot_v1/cases.json`. The fixture carries ClaimNov's public
labels, original paper IDs, recovered abstracts, and the IDs of any missing abstracts.
`pilots/freeze_governance_cases.py` builds a new fixture from public endpoints when needed.
The fixed-evidence comparison uses the saved `baseline` from before the audit change:

```bash
.venv/bin/python pilot3_runs/run_logged.py /path/to/new-audit.log \
  .venv/bin/python edara_mvp_eval/pilots/run_governance_pilot.py audit \
  --cases edara_mvp_eval/outputs/governance_pilot_v1/cases.json \
  --prior-phase edara_mvp_eval/outputs/governance_pilot_v1/baseline \
  --out /path/to/new-audit

.venv/bin/python pilot3_runs/run_logged.py /path/to/new-regeneration.log \
  .venv/bin/python edara_mvp_eval/pilots/run_governance_pilot.py regenerate \
  --cases edara_mvp_eval/outputs/governance_pilot_v1/cases.json \
  --prior-phase /path/to/new-audit \
  --out /path/to/new-regeneration
```

Use a named `tmux` session for a live run. The runner saves per-case artifacts and binds each
phase to its case set, source hashes, model profiles and upstream results. The regeneration arms
share the same dated evidence and generation settings. Only the feedback arm receives the audit
diagnosis. One blind evaluation call per valid candidate scores mechanism replacement,
unsupported novelty, and testability. These are diagnostic results against the frozen corpus.

## Three-arm feedback comparison

The frozen case set in `outputs/governance_feedback_v2/cases_frozen.json` contains a full collision
and a partial collision for each of six research topics. The variants share dated primary-source
evidence. Two partial cases adapt contrasts from ClaimNov's public majority annotations. The
remaining cases are construction controls. Dates and labels bound the comparison to the supplied
papers, and the candidate generators never receive the reference key or the later papers.

Run from the parent `04_external_eval_refs` directory in a named `tmux` session.

```bash
tmux new-session -s governance-feedback-12 -c "$PWD"
.venv/bin/python pilot3_runs/run_logged.py edara_mvp_eval/outputs/governance_feedback_v2/live.log \
  .venv/bin/python edara_mvp_eval/pilots/run_feedback_ablation.py \
  --cases edara_mvp_eval/outputs/governance_feedback_v2/cases_frozen.json \
  --out edara_mvp_eval/outputs/governance_feedback_v2/live --workers 3
```

Use a fresh log path and the same output directory to resume a matching manifest. The runner
checkpoints each audit, candidate and blind verdict. The ordinary arm sees the full frozen packet.
The highlight and feedback arms also see identical audit-selected excerpts. The feedback arm
additionally receives the attribution, coverage findings and suggested action. All three use the
same generator and one generation at the same output-token limit. Report paired gains and losses
against each adjacent arm separately for full and partial collisions. The shared audit and separate
blind reviewers use the same judge model. The result is a fixed-evidence diagnostic across six topic
pairs, with no claim of corpus-wide novelty.
