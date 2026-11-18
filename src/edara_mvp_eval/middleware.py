"""Attachable Trustworthy-Novelty Middleware:
Tri-State Routing + Multi-Paper Compositional Remainder + Spec Defect Locator
+ Bounded Local Span Patching + Deterministic Retention Guard (max 2 rounds).
"""
from __future__ import annotations

import difflib
import json
import re
from typing import Any

from pydantic import ValidationError

from .epistemic_audit import run_full_audit
from .factual_audit import clean_replacement_text
from .host_adapter import preserves_design_elements
from .literature_scout import gather_evidence_for_query, make_counterexample_query
from .llm_client import LLMRunner
from .schemas import (
    BoundedPatch,
    FullAuditBundle,
    NormalizedProposal,
    PriorWorkItem,
    RetentionMetrics,
    SpecClarification,
    SpecDefectItem,
    TaskSpec,
    UnknownDependencyClaim,
)

MAX_PATCHES_PER_ROUND = 4
MIN_METHOD_RETENTION = 0.90
_SPEC_TEXT_LIMIT = 720
_SPEC_FIELDS = frozenset({
    "method", "minimal_test", "decisive_metric", "supporting_result", "falsifying_result",
    "limitations", "resource_plan",
})
_EDGE_PUNCTUATION = " .,!?:;。！？、，：；“”‘’\"'«»…"
_TERMINAL_PUNCTUATION = ".!?。！？…"
ALLOWED_PATCH_FIELDS = frozenset(
    {
        "title",
        "literature_gap",
        "hypothesis",
        "method",
        "minimal_test",
        "decisive_metric",
        "supporting_result",
        "falsifying_result",
        "related_work",
        "limitations",
    }
)

_PATCH_SYSTEM = """You are a Trustworthy-Novelty Bounded Surgical Middleware (inspired by RINI + ClaimNov + IdeaAMBIG + E-DARA Epistemic Calibration).
Your job is to output BOUNDED LOCAL PATCHES—NEVER rewrite the whole proposal from scratch.

Strict Rules:
0. Design confounds come first. If `design_confounds` lists an alternative whose `status` is `absent`, your FIRST spec clarification must add its `discriminating_arm` verbatim into `minimal_test` (or `decisive_metric`). A generic baseline, a reworded protocol, or a related but non-discriminating ablation does not close this defect. If the auditor marks it `present`, do not duplicate it.
1. Output at most 4 patches in `patches`. Each patch edits ONE field using an EXACT unique substring (`exact_source_span`) from the current proposal text.
   - To narrow/fix an overclaimed sentence, copy that exact sentence/clause into `exact_source_span` and replace it with a scoped, evidence-grounded version.
   - Prioritize the specific hard-failing claims in the diagnostic. Find all surviving occurrences in the current proposal; use the shortest unique verbatim clause that actually states each claim.
   - Use an empty `exact_source_span` only for an attribution append. Put specification additions in the separate list below.
2. Preserve the author's core `hypothesis` and `method` mechanisms. NEVER delete valid `unknown_dependencies`.
3. Unknown-Dependency Contract: whenever a prior fact, empirical number, or interaction assumption lacks pre-cutoff evidence, DO NOT pretend it is settled AND DO NOT leave a vacuum—convert it via `append_unknown_contracts` into a structured triplet:
   {dependency, why_unresolved, resolution_path, minimal_calibration_test, fallback_branch}.
4. Use `spec_clarifications` once per listed defect ID, addressing blocking defects first. Defects with category `confound_control` are design confounds and take precedence over other specification defects. This list has its own allowance and only appends text to the defect's target field. For each defect, supply the minimal concrete executable decision that closes it, including a fixed value or reproducible calibration rule and fallback when relevant. Align dataset/split, matched baseline, decisive metric and falsifier with the original proposal. Treat additions as prospective protocol choices. Do not present measurements or source facts as established.
5. Compositional Attribution: if `compositional.attribution_requirements` lists prior papers, credit them explicitly in `related_work` while keeping `compositional_remainder` clear.
6. Fix eroded/falsified `gap_premises` in `literature_gap` (and `updated_gap_premises` if provided) toward the surviving regime boundary.
7. NEVER fabricate citations; cite ONLY from the supplied Pre-Cutoff References Summary.

Return ONLY valid JSON:
{
  "patches": [
    {
      "target_field": "literature_gap" | "hypothesis" | "method" | "minimal_test" | "decisive_metric" | "supporting_result" | "falsifying_result" | "related_work" | "limitations" | "title",
      "exact_source_span": "<verbatim substring from current field, or '' for bounded append>",
      "replacement_text": "<concise surgical replacement or append>",
      "edit_type": "scope_narrowing" | "attribution_credit" | "gap_refinement" | "unknown_conversion",
      "rationale": "<1 sentence>"
    }
  ],
  "spec_clarifications": [
    {"defect_id": 1, "target_field": "<field from the corresponding defect>", "append_text": "<one concrete protocol clarification>"}
  ],
  "append_unknown_contracts": [
    {
      "dependency": "...",
      "why_unresolved": "...",
      "resolution_path": "...",
      "minimal_calibration_test": "...",
      "fallback_branch": "..."
    }
  ],
  "updated_gap_premises": ["<optional narrowed gap premise strings>"],
  "repair_notes": ["<concise change log>"]
}"""


def _content_tokens(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]{4,}", (text or "").lower())}


def _token_retention(orig: str, edited: str) -> float:
    orig_toks = _content_tokens(orig)
    if not orig_toks:
        return 1.0
    edited_toks = _content_tokens(edited)
    return round(len(orig_toks & edited_toks) / len(orig_toks), 3)


def _needs_intervention(bundle: FullAuditBundle) -> bool:
    return not (
        bundle.gate.gate_pass
        and bundle.gate.strict_gate_pass
        and bundle.tri_state.max_smoothing_score <= 2
        and bundle.gap_audit.survival_rate >= 1.0
        and not bundle.compositional.attribution_requirements
        and not any(f.label == "defect" for f in getattr(bundle, "factual_findings", ()))
    )


def _unique_regex_span(text: str, pattern: str) -> tuple[tuple[int, int] | None, bool]:
    # Lookahead exposes overlapping occurrences, which are ambiguous as well.
    matches = re.finditer(r"(?=(" + pattern + r"))", text)
    first = next(matches, None)
    ambiguous = next(matches, None) is not None
    return (first.span(1) if first and not ambiguous else None), ambiguous


def _source_span(text: str, quoted: str, replacement: str) -> tuple[int, int] | None:
    if not quoted.strip():
        return None
    start = text.find(quoted)
    if start >= 0:
        return (start, start + len(quoted)) if text.find(quoted, start + 1) < 0 else None

    pattern = r"\s+".join(re.escape(word) for word in quoted.split())
    match, ambiguous = _unique_regex_span(text, pattern)
    if match or ambiguous:
        return match

    # Drop sentence-edge punctuation only. A copied fragment still has to
    # identify a single bounded substring with the words in their original order.
    core = quoted.strip().strip(_EDGE_PUNCTUATION).strip()
    if not re.search(r"\w", core) or re.search(r"[!?。！？]|\.\s", core) or re.match(r"^\.\d", quoted.strip()):
        return None
    pattern = r"\s+".join(re.escape(word) for word in core.split())
    match, _ = _unique_regex_span(text, r"(?<![\w.+−-])" + pattern + r"(?!\w|\.\d)")
    if not match:
        return None
    start, end = match
    if end < len(text) and replacement[-1] in _TERMINAL_PUNCTUATION and text[end] in _TERMINAL_PUNCTUATION:
        end += 1
    return start, end


def apply_bounded_patches(
    proposal: NormalizedProposal,
    patch_payload: dict[str, Any],
    *,
    max_edits: int = MAX_PATCHES_PER_ROUND,
    original_proposal: NormalizedProposal | None = None,
) -> tuple[NormalizedProposal, int, int, bool]:
    """Deterministically validate and apply bounded patches with a core-mechanism retention guard."""
    updated = proposal.model_copy(deep=True)
    applied = 0
    rejected = 0
    guard_reverted = False

    raw_patches = patch_payload.get("patches") or []
    if not isinstance(raw_patches, list):
        raw_patches = []

    for raw in raw_patches[:max_edits]:
        if not isinstance(raw, dict):
            rejected += 1
            continue
        try:
            patch = BoundedPatch.model_validate(raw)
        except ValidationError:
            rejected += 1
            continue
        if patch.target_field not in ALLOWED_PATCH_FIELDS:
            rejected += 1
            continue
        cur_val = str(getattr(updated, patch.target_field) or "")
        span = patch.exact_source_span
        rep = clean_replacement_text((patch.replacement_text or "").strip())
        if not rep:
            rejected += 1
            continue

        if span:
            location = _source_span(cur_val, span, rep)
            if location is None:
                rejected += 1
                continue
            start, end = location
            # Bound expansion so a single patch cannot smuggle a full rewrite.
            if len(rep) > max(480, int((end - start) * 2.8)):
                rejected += 1
                continue
            cand_val = cur_val[:start] + rep + cur_val[end:]
        else:
            if patch.edit_type not in {"spec_clarification", "attribution_credit"} or len(rep) > 480:
                rejected += 1
                continue
            sep = " " if cur_val and not cur_val.endswith((" ", "\n")) else ""
            cand_val = (cur_val + sep + rep).strip()

        # Core mechanism retention guard for hypothesis & method edits.
        before = original_proposal or proposal
        minimum = {"method": MIN_METHOD_RETENTION, "hypothesis": 0.55}.get(patch.target_field)
        if minimum is not None and _token_retention(getattr(before, patch.target_field), cand_val) < minimum:
            rejected += 1
            guard_reverted = True
            continue

        setattr(updated, patch.target_field, cand_val)
        applied += 1

    # Always preserve existing unknown_dependencies and append validated new UnknownDependencyContracts.
    existing_keys = {u.dependency.strip().lower()[:48] for u in updated.unknown_dependencies if u.dependency.strip()}
    for raw_u in patch_payload.get("append_unknown_contracts") or []:
        if not isinstance(raw_u, dict) or not str(raw_u.get("dependency") or "").strip():
            continue
        try:
            item = UnknownDependencyClaim.model_validate(raw_u)
        except Exception:
            continue
        k = item.dependency.strip().lower()[:48]
        if k not in existing_keys:
            existing_keys.add(k)
            updated.unknown_dependencies.append(item)

    # Optionally update narrowed gap_premises when provided.
    new_gaps = [str(g).strip() for g in (patch_payload.get("updated_gap_premises") or []) if str(g).strip()]
    if new_gaps:
        updated.gap_premises = new_gaps[:3]
    elif updated.literature_gap and updated.literature_gap != proposal.literature_gap:
        updated.gap_premises = [updated.literature_gap]

    # Re-sync prior_facts if prior-fact claims were modified out of scope: keep prior_facts aligned with text edits.
    if applied > 0 and updated.prior_facts:
        full_blob = f"{updated.title} {updated.literature_gap} {updated.method} {updated.related_work}".lower()
        kept_facts = []
        for pf in updated.prior_facts:
            # Drop prior_fact only if none of its core tokens appear anywhere in the edited draft anymore.
            pf_toks = _content_tokens(pf.claim)
            if not pf_toks or len(pf_toks & _content_tokens(full_blob)) >= max(1, len(pf_toks) // 4):
                kept_facts.append(pf)
        updated.prior_facts = kept_facts

    return updated, applied, rejected, guard_reverted


def append_spec_clarifications(
    proposal: NormalizedProposal,
    patch_payload: dict[str, Any],
    defects: list[SpecDefectItem],
) -> tuple[int, int]:
    """Append one bounded clarification per verified defect to the working proposal."""
    raw_clarifications = patch_payload.get("spec_clarifications", [])
    if not isinstance(raw_clarifications, list):
        return 0, 1
    used: set[int] = set()
    rejected = max(0, len(raw_clarifications) - len(defects))
    for raw in raw_clarifications[:len(defects)]:
        try:
            clarification = SpecClarification.model_validate(raw)
        except ValidationError:
            rejected += 1
            continue
        index = clarification.defect_id
        field = clarification.target_field
        text = clarification.append_text.strip()
        if (
            index in used or index > len(defects) or field not in _SPEC_FIELDS
            or field != defects[index - 1].target_field or not text or len(text) > _SPEC_TEXT_LIMIT
        ):
            rejected += 1
            continue

        if field == "resource_plan":
            prior = proposal.resource_plan.get("implementation_clarifications", "")
        else:
            prior = getattr(proposal, field)
        if not isinstance(prior, str) or " ".join(text.split()) in " ".join(prior.split()):
            rejected += 1
            continue
        appended = f"{prior}\n{text}" if prior else text
        if field == "resource_plan":
            proposal.resource_plan["implementation_clarifications"] = appended
        else:
            setattr(proposal, field, appended)
        used.add(index)
    return len(used), rejected


def compute_retention_metrics(
    original: NormalizedProposal,
    final: NormalizedProposal,
    *,
    rounds_used: int,
    patches_applied: int,
    patches_rejected: int,
    guard_reverted: bool,
    spec_clarifications_applied: int = 0,
    spec_clarifications_rejected: int = 0,
) -> RetentionMetrics:
    orig_core = " ".join(
        [original.title, original.literature_gap, original.hypothesis, original.method, original.minimal_test]
    )
    fin_core = " ".join(
        [final.title, final.literature_gap, final.hypothesis, final.method, final.minimal_test]
    )
    locality = round(difflib.SequenceMatcher(None, orig_core, fin_core).ratio(), 3)
    orig_unk = sum(
        1
        for u in original.unknown_dependencies
        if u.why_unresolved.strip() and (u.resolution_path.strip() or u.minimal_calibration_test.strip())
    )
    fin_unk = sum(
        1
        for u in final.unknown_dependencies
        if u.why_unresolved.strip() and (u.resolution_path.strip() or u.minimal_calibration_test.strip())
    )
    unk_ratio = round(min(1.0, fin_unk / orig_unk), 3) if orig_unk > 0 else 1.0
    return RetentionMetrics(
        method_token_retention=_token_retention(original.method, final.method),
        hypothesis_retention=_token_retention(original.hypothesis, final.hypothesis),
        unknown_retention_ratio=unk_ratio,
        edit_locality_ratio=locality,
        guard_reverted=guard_reverted,
        rounds_used=rounds_used,
        patches_applied=patches_applied,
        patches_rejected=patches_rejected,
        spec_clarifications_applied=spec_clarifications_applied,
        spec_clarifications_rejected=spec_clarifications_rejected,
    )


def _bundle_rank_key(bundle: FullAuditBundle) -> tuple:
    factual_defects = sum(1 for f in getattr(bundle, "factual_findings", ()) if f.label == "defect")
    return (
        1 if bundle.gate.gate_pass else 0,
        1 if bundle.gate.strict_gate_pass else 0,
        -factual_defects,
        -len(bundle.gate.hard_fail_reasons),
        1 if bundle.gap_audit.survival_rate >= 1.0 else 0,
        -bundle.spec_audit.blocking_count,
        round(bundle.gate.actionability_score, 3),
        round(bundle.gate.trusted_novelty, 3),
    )


def build_pre_gen_brief(task: TaskSpec, *, live_search: bool = False) -> str:
    """Optional BVSR + Lit2Test pre-generation hook: low-anchoring tension & anti-collision brief."""
    refs = list(task.prior_work[:4])
    if live_search and task.title:
        for hit in gather_evidence_for_query(task.title, task, live_search=True, top_k=3):
            if all(hit.title.lower() != r.title.lower() for r in refs):
                refs.append(hit)
    lines = [
        f"Pre-Cutoff Boundary: {task.cutoff_date}",
        "Avoid claiming novelty on already-established baseline mechanisms below; isolate an unaddressed tension or regime boundary:",
    ]
    for r in refs[:5]:
        lim = (r.limitation or r.key_contribution or r.abstract)[:180]
        lines.append(f"- {r.title} ({r.year or '?'}): {lim}")
    return "\n".join(lines)


class TrustworthyNoveltyMiddleware:
    """Drop-in post-draft wrapper runnable over any LLM or agent generator."""

    def __init__(
        self,
        *,
        live_search: bool = False,
        enable_embeddings: bool = False,
        max_rounds: int = 2,
    ) -> None:
        self.live_search = live_search
        self.enable_embeddings = enable_embeddings
        self.max_rounds = max(1, max_rounds)

    def _gather_counter_refs(self, bundle: FullAuditBundle, task: TaskSpec) -> list[PriorWorkItem]:
        ref_items: list[PriorWorkItem] = list(task.prior_work[:5])
        seen_titles = {p.title.strip().lower() for p in ref_items if p.title}
        if not self.live_search:
            return ref_items

        queries = [make_counterexample_query(g.premise) for g in bundle.gap_audit.items if g.status != "survived"]
        if not queries and bundle.proposal.literature_gap:
            queries = [make_counterexample_query(bundle.proposal.literature_gap)]
        added = 0
        for q in queries[:2]:
            for hit in gather_evidence_for_query(q, task, live_search=True, top_k=4):
                key = hit.title.strip().lower()
                if key and key not in seen_titles:
                    seen_titles.add(key)
                    ref_items.append(hit)
                    added += 1
                    if added >= 3:
                        return ref_items
        return ref_items

    def _build_diag(self, bundle: FullAuditBundle) -> dict[str, Any]:
        defect_prio = {
            "fabrication": 0,
            "prediction_as_observed": 1,
            "internal_inconsistency": 2,
            "numeric_mismatch": 3,
            "coverage_contradiction": 4,
            "attribution": 5,
            "scope_overclaim": 6,
            "unattested_process": 7,
        }
        factual = [
            f.model_dump()
            for f in sorted(
                (item for item in getattr(bundle, "factual_findings", ()) if item.label == "defect"),
                key=lambda row: (
                    defect_prio.get(getattr(row, "defect_type", "none"), 8),
                    0 if (row.evidence or row.basis) else 1,
                ),
            )
        ]
        unverified = list(getattr(bundle, "unverified_entities", ()) or ())
        rest = {
            "gate_failures": bundle.gate.hard_fail_reasons,
            **({"unverified_external_entities": unverified} if unverified else {}),
            "smoothed_or_overclaimed_facts": [
                a.model_dump()
                for a in bundle.tri_state.prior_fact_audits
                if a.smoothing_score >= 3
                or a.epistemic_overclaim
                or a.citation_status in {"fabricated_or_missing", "unverified"}
            ],
            "gap_findings": [g.model_dump() for g in bundle.gap_audit.items if g.status != "survived"],
            "spec_defects": [
                {"defect_id": i, **d.model_dump()} for i, d in enumerate(bundle.spec_audit.defects, start=1)
            ],
            "compositional": bundle.compositional.model_dump(),
            "governance": bundle.governance.model_dump() if bundle.governance else None,
            "claimnov_veto": bundle.claimnov.veto,
            "claimnov_evidence": bundle.claimnov.veto_evidence,
        }
        confounds = bundle.confound_audit.model_dump()
        if bundle.confound_audit.priority == "blocking":
            return {"design_confounds": confounds, **({"factual_findings": factual} if factual else {}), **rest}
        return {**({"factual_findings": factual} if factual else {}), "design_confounds": confounds, **rest}

    def _enforce_confounds(self, diag: dict, payload: dict, user: str, runner: LLMRunner) -> dict:
        """One targeted retry when the editor ignores a blocking design confound."""
        required = [d["defect_id"] for d in diag.get("spec_defects", [])
                    if d.get("category") == "confound_control" and d.get("severity") == "blocking"]
        result = {"required": required, "retried": False, "unaddressed": [], "payload": payload}
        if not required:
            return result
        covered = {row.get("defect_id") for row in payload.get("spec_clarifications") or []}
        missing = [defect_id for defect_id in required if defect_id not in covered]
        if not missing:
            return result
        result["retried"] = True
        reminder = (
            user
            + "\n\nMECHANICAL REQUIREMENT: spec_clarifications must contain one entry for each of these "
            f"blocking confound_control defect IDs: {missing}. Each entry must append the defect's "
            "discriminating_arm verbatim into minimal_test or decisive_metric. Other defects may wait."
        )
        try:
            retry = runner.chat_json(_PATCH_SYSTEM, reminder, max_tokens=4096, tag="bounded_patch_confound")
        except json.JSONDecodeError:
            result["unaddressed"] = missing
            return result
        covered = {row.get("defect_id") for row in retry.get("spec_clarifications") or []}
        result["unaddressed"] = [defect_id for defect_id in missing if defect_id not in covered]
        if not result["unaddressed"]:
            result["payload"] = retry
        return result

    def apply(
        self,
        task: TaskSpec,
        raw_draft: str | dict[str, Any],
        *,
        repair_runner: LLMRunner,
        audit_runner: LLMRunner,
        target_arm_name: str = "with_middleware",
        initial_audit: FullAuditBundle | None = None,
        history: list | None = None,
    ) -> tuple[dict[str, Any], FullAuditBundle, FullAuditBundle, float]:
        """Return (repaired_draft, before_audit, after_audit, middleware_overhead_usd)."""
        gen_cost_before = repair_runner.usage.cost_usd
        before = initial_audit or run_full_audit(
            task,
            f"{target_arm_name}__pre",
            raw_draft,
            audit_runner,
            live_search=self.live_search,
            enable_embeddings=self.enable_embeddings,
        )
        if not _needs_intervention(before):
            retention = compute_retention_metrics(
                before.proposal,
                before.proposal,
                rounds_used=0,
                patches_applied=0,
                patches_rejected=0,
                guard_reverted=False,
            )
            repaired = raw_draft if isinstance(raw_draft, dict) else before.proposal.model_dump()
            after = before.model_copy(update={"arm": target_arm_name, "retention": retention})
            return repaired, before, after, 0.0

        current_proposal = before.proposal.model_copy(deep=True)
        current_bundle = before
        best_bundle = before
        best_proposal = current_proposal
        total_applied = 0
        total_rejected = 0
        total_spec_applied = 0
        total_spec_rejected = 0
        any_guard_reverted = False
        rounds_used = 0

        for round_idx in range(1, self.max_rounds + 1):
            rounds_used = round_idx
            ref_items = self._gather_counter_refs(current_bundle, task)
            diag = self._build_diag(current_bundle)
            user = (
                f"Task ID: {task.task_id} (Repair Round {round_idx}/{self.max_rounds})\n"
                f"Budget Envelope: {json.dumps(task.budget, ensure_ascii=False)}\n\n"
                f"Pre-Cutoff References Summary:\n"
                + "\n".join(
                    f"- [{p.id} | {p.source}] {p.title} ({p.year}): {(p.limitation or p.abstract)[:300]}"
                    for p in ref_items
                )
                + f"\n\nCurrent Proposal Fields (copy exact substrings from here for exact_source_span):\n"
                f"{current_proposal.model_dump_json(indent=2)}\n\n"
                f"Structured Audit Diagnostics Requiring Bounded Surgical Patch:\n"
                f"{json.dumps(diag, ensure_ascii=False, indent=2)}"
            )
            try:
                patch_payload = repair_runner.chat_json(
                    _PATCH_SYSTEM,
                    user,
                    max_tokens=4096,
                    tag="bounded_patch",
                )
            except json.JSONDecodeError as exc:
                print(
                    f"[warn] {task.task_id} {target_arm_name} round={round_idx} bounded_patch JSON parse failed ({exc}); keeping best valid proposal.",
                    flush=True,
                )
                break
            enforcement = self._enforce_confounds(diag, patch_payload, user, repair_runner)
            patch_payload = enforcement.pop("payload")
            patched_prop, n_app, n_rej, reverted = apply_bounded_patches(
                current_proposal, patch_payload, original_proposal=before.proposal,
            )
            spec_app, spec_rej = append_spec_clarifications(
                patched_prop, patch_payload, current_bundle.spec_audit.defects,
            )
            total_applied += n_app
            total_rejected += n_rej
            total_spec_applied += spec_app
            total_spec_rejected += spec_rej
            any_guard_reverted = any_guard_reverted or reverted

            # Re-audit the patched proposal (Remainder Re-Check).
            if patched_prop == current_proposal:
                if history is not None:
                    history.append({"round": round_idx, "action": "local_patch", "status": "unchanged",
                                    "diagnostics": diag, "patch": patch_payload, "confound_enforcement": enforcement})
                break

            candidate_bundle = run_full_audit(
                task,
                target_arm_name,
                patched_prop,
                audit_runner,
                live_search=self.live_search,
                enable_embeddings=self.enable_embeddings,
            )
            current_proposal = candidate_bundle.proposal
            current_bundle = candidate_bundle
            preserved, regressions = preserves_design_elements(
                best_proposal,
                patched_prop,
                before_bundle=best_bundle,
                after_bundle=candidate_bundle,
            )
            accepted = preserved and _bundle_rank_key(candidate_bundle) >= _bundle_rank_key(best_bundle)
            if history is not None:
                history.append({"round": round_idx, "action": "local_patch", "status": "accepted" if accepted else "returned",
                                "diagnostics": diag, "patch": patch_payload, "confound_enforcement": enforcement,
                                "design_regressions": regressions,
                                "candidate": patched_prop.model_dump(), "audit": candidate_bundle.model_dump()})
            if accepted:
                best_bundle = candidate_bundle
                best_proposal = candidate_bundle.proposal

            if not _needs_intervention(candidate_bundle):
                break

        retention = compute_retention_metrics(
            before.proposal,
            best_proposal,
            rounds_used=rounds_used,
            patches_applied=total_applied,
            patches_rejected=total_rejected,
            guard_reverted=any_guard_reverted,
            spec_clarifications_applied=total_spec_applied,
            spec_clarifications_rejected=total_spec_rejected,
        )
        mw_cost = round(repair_runner.usage.cost_usd - gen_cost_before, 6)
        final_after = best_bundle.model_copy(
            update={
                "arm": target_arm_name,
                "proposal": best_proposal,
                "retention": retention,
            }
        )
        return best_proposal.model_dump(), before, final_after, mw_cost
