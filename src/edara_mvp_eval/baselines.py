"""Ablation baselines: Self-Refine (no external audit/retrieval) & Retrieve-and-Revise (unconstrained full rewrite)."""
from __future__ import annotations

import json
from typing import Any

from .epistemic_audit import normalize_artifact, run_full_audit
from .literature_scout import gather_evidence_for_query, make_counterexample_query
from .llm_client import LLMRunner
from .middleware import compute_retention_metrics
from .schemas import FullAuditBundle, NormalizedProposal, PriorWorkItem, TaskSpec

_SELF_REFINE_CRITIQUE_SYSTEM = """You are reviewing a draft research proposal using ONLY internal reasoning (no external search or tools).
Identify up to 4 weaknesses across overclaiming, unclear experimental test, budget realism, or novelty positioning.
Return JSON only: {"needs_revision": true | false, "feedback": ["<concise critique>"]}"""

_SELF_REFINE_REVISE_SYSTEM = """Revise the research proposal to address the self-critique feedback.
Return a complete JSON object with keys:
title, literature_gap, hypothesis, method, minimal_test, decisive_metric, supporting_result, falsifying_result, related_work, limitations, resource_plan."""

_RETRIEVE_REVISE_SYSTEM = """You are revising a research proposal given retrieved pre-cutoff papers and audit findings.
Rewrite the full proposal JSON to fix any overclaims, eroded gaps, or missing details while respecting the budget envelope.
Return ONLY valid JSON with keys:
title, literature_gap, hypothesis, method, minimal_test, decisive_metric, supporting_result, falsifying_result, related_work, limitations, resource_plan."""


def run_self_refine(
    task: TaskSpec,
    raw_draft: str | dict[str, Any],
    *,
    before_bundle: FullAuditBundle,
    repair_runner: LLMRunner,
    audit_runner: LLMRunner,
    target_arm_name: str = "direct_self_refine",
    max_rounds: int = 2,
    live_search: bool = False,
    enable_embeddings: bool = False,
) -> tuple[dict[str, Any], FullAuditBundle, float]:
    """Iterative prompt-only Self-Refine baseline (no external evidence/ClaimNov/SpecLocator/BoundedPatch)."""
    cost_before = repair_runner.usage.cost_usd
    current: dict[str, Any] = (
        dict(raw_draft) if isinstance(raw_draft, dict) else before_bundle.proposal.model_dump()
    )
    rounds_used = 0
    for r in range(1, max(1, max_rounds) + 1):
        rounds_used = r
        crit_user = (
            f"Task: {task.title} (Cutoff: {task.cutoff_date})\n"
            f"Budget: {json.dumps(task.budget, ensure_ascii=False)}\n\n"
            f"Current Draft:\n{json.dumps(current, ensure_ascii=False, indent=2)[:16000]}"
        )
        try:
            crit = repair_runner.chat_json(_SELF_REFINE_CRITIQUE_SYSTEM, crit_user, max_tokens=1024, tag="self_refine_critique")
        except json.JSONDecodeError as exc:
            print(f"[warn] {task.task_id} {target_arm_name} round={r} critique JSON parse failed ({exc}); keeping prior draft.", flush=True)
            break
        feedback = crit.get("feedback") or []
        if not crit.get("needs_revision", True) and not feedback:
            break
        rev_user = (
            f"Task Context:\n{task.prompt_context[:6000]}\n\n"
            f"Current Draft:\n{json.dumps(current, ensure_ascii=False, indent=2)[:12000]}\n\n"
            f"Self-Critique Feedback:\n{json.dumps(feedback, ensure_ascii=False, indent=2)}"
        )
        try:
            current = repair_runner.chat_json(_SELF_REFINE_REVISE_SYSTEM, rev_user, max_tokens=4096, tag="self_refine_revise")
        except json.JSONDecodeError as exc:
            print(f"[warn] {task.task_id} {target_arm_name} round={r} revise JSON parse failed ({exc}); keeping prior draft.", flush=True)
            break

    overhead = round(repair_runner.usage.cost_usd - cost_before, 6)
    audited = run_full_audit(
        task,
        target_arm_name,
        current,
        audit_runner,
        live_search=live_search,
        enable_embeddings=enable_embeddings,
    )
    retention = compute_retention_metrics(
        before_bundle.proposal,
        audited.proposal,
        rounds_used=rounds_used,
        patches_applied=0,
        patches_rejected=0,
        guard_reverted=False,
    )
    return current, audited.model_copy(update={"retention": retention}), overhead


def run_retrieve_and_revise(
    task: TaskSpec,
    *,
    before_bundle: FullAuditBundle,
    repair_runner: LLMRunner,
    audit_runner: LLMRunner,
    target_arm_name: str = "direct_retrieve_revise",
    live_search: bool = False,
    enable_embeddings: bool = False,
) -> tuple[dict[str, Any], FullAuditBundle, float]:
    """Unconstrained Retrieve-and-Revise baseline (same evidence & findings, but full-draft rewrite without bounded guard)."""
    cost_before = repair_runner.usage.cost_usd
    ref_items: list[PriorWorkItem] = list(task.prior_work[:5])
    seen_titles = {p.title.strip().lower() for p in ref_items if p.title}
    if live_search:
        queries = [make_counterexample_query(g.premise) for g in before_bundle.gap_audit.items if g.status != "survived"]
        if not queries and before_bundle.proposal.literature_gap:
            queries = [make_counterexample_query(before_bundle.proposal.literature_gap)]
        added = 0
        for q in queries[:2]:
            for hit in gather_evidence_for_query(q, task, live_search=True, top_k=4):
                k = hit.title.strip().lower()
                if k and k not in seen_titles:
                    seen_titles.add(k)
                    ref_items.append(hit)
                    added += 1
                    if added >= 3:
                        break
            if added >= 3:
                break

    findings = {
        "gate_failures": before_bundle.gate.hard_fail_reasons,
        "smoothed_or_overclaimed_facts": [
            a.model_dump()
            for a in before_bundle.tri_state.prior_fact_audits
            if a.smoothing_score >= 3 or a.epistemic_overclaim or a.citation_status == "fabricated_or_missing"
        ],
        "gap_findings": [g.model_dump() for g in before_bundle.gap_audit.items if g.status != "survived"],
        "claimnov_veto": before_bundle.claimnov.veto,
        "claimnov_evidence": before_bundle.claimnov.veto_evidence,
    }
    user = (
        f"Task ID: {task.task_id}\n"
        f"Budget Envelope: {json.dumps(task.budget, ensure_ascii=False)}\n\n"
        f"Pre-Cutoff References Summary:\n"
        + "\n".join(
            f"- [{p.id} | {p.source}] {p.title} ({p.year}): {(p.limitation or p.abstract)[:300]}"
            for p in ref_items
        )
        + f"\n\nDraft:\n{before_bundle.proposal.model_dump_json(indent=2)}\n\n"
        f"Audit Findings:\n{json.dumps(findings, ensure_ascii=False, indent=2)}"
    )
    try:
        revised = repair_runner.chat_json(_RETRIEVE_REVISE_SYSTEM, user, max_tokens=4096, tag="retrieve_revise")
    except json.JSONDecodeError as exc:
        print(f"[warn] {task.task_id} {target_arm_name} revise JSON parse failed ({exc}); keeping prior draft.", flush=True)
        revised = before_bundle.proposal.model_dump()
    overhead = round(repair_runner.usage.cost_usd - cost_before, 6)
    audited = run_full_audit(
        task,
        target_arm_name,
        revised,
        audit_runner,
        live_search=live_search,
        enable_embeddings=enable_embeddings,
    )
    retention = compute_retention_metrics(
        before_bundle.proposal,
        audited.proposal,
        rounds_used=1,
        patches_applied=0,
        patches_rejected=0,
        guard_reverted=False,
    )
    return revised, audited.model_copy(update={"retention": retention}), overhead
