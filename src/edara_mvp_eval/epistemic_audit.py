"""Core Evidence-First Tri-State Epistemic Audit, Gap-Premise Falsification, AlphaIdeaBench Grounded Check, and Pre-Execution Integrity Gate."""
from __future__ import annotations

import json
import statistics
from typing import Any

from alphaideabench.domain.dimensions import load_catalog
from pydantic import ValidationError

from .claimnov_adapter import analyze_compositional_remainder, run_claimnov_audit
from .factual_audit import audit_independent_claims, audit_unverified_entities
from .governance import audit_idea
from .host_adapter import filter_proposal_fields, to_lit2test_card
from .literature_scout import extract_relevant_excerpt, gather_evidence_for_query, make_counterexample_query
from .llm_client import LLMRunner
from .schemas import (
    ClaimNovAuditResult,
    CompositionalRemainderResult,
    FactualFinding,
    FullAuditBundle,
    GapAuditResult,
    GapPremiseAuditItem,
    HypothesisClaim,
    IntegrityGateResult,
    NormalizedProposal,
    PriorFactAudit,
    PriorFactClaim,
    SpecAuditResult,
    TaskSpec,
    TriStateAuditResult,
    UnknownDependencyClaim,
)
from .spec_locator import locate_spec_defects
from .confound_audit import audit_confounds, merge_confounds

_NORMALIZE_SYSTEM = """You are a blind research-artifact normalizer.
Convert ANY raw research proposal/plan (regardless of source agent style) into a strict blind JSON contract.
Strip all agent names, internal trace/evidence IDs (like e01, tree nodes), confidence hype, and formatting quirks.

CRITICAL: Classify claims into three disjoint epistemic buckets:
1. prior_facts: statements about ALREADY PUBLISHED papers, baseline numbers, or established literature limits (list cited_titles / cited_ids if referenced).
2. prospective_hypotheses: untested predictions or proposed mechanisms (pair with decisive_metric and falsifying_outcome).
3. unknown_dependencies: unresolved data, compute, parameter, or empirical prerequisites that CANNOT be known before running the experiment (record why_unresolved and resolution_path).
4. gap_premises: 1-3 crisp falsifiable premises claiming what prior work fails to do or leaves open (used for pre-cutoff counterexample search).

Return JSON matching:
{
  "title": "...",
  "literature_gap": "...",
  "hypothesis": "...",
  "method": "...",
  "minimal_test": "...",
  "decisive_metric": "...",
  "supporting_result": "...",
  "falsifying_result": "...",
  "related_work": "...",
  "limitations": "...",
  "resource_plan": {},
  "prior_facts": [{"claim": "...", "cited_titles": [], "cited_ids": []}],
  "prospective_hypotheses": [{"claim": "...", "decisive_metric": "...", "falsifying_outcome": "..."}],
  "unknown_dependencies": [{"dependency": "...", "why_unresolved": "...", "resolution_path": "...", "minimal_calibration_test": "...", "fallback_branch": "..."}],
  "gap_premises": ["..."]
}"""


def normalize_artifact(
    task: TaskSpec,
    raw_output: str | dict[str, Any],
    runner: LLMRunner,
) -> NormalizedProposal:
    if isinstance(raw_output, dict):
        filtered = filter_proposal_fields(raw_output) or raw_output
        raw_str = json.dumps(filtered, ensure_ascii=False, indent=2)
        fallback_card = to_lit2test_card(filtered)
    else:
        raw_str = raw_output
        fallback_card = to_lit2test_card(raw_output) if raw_output.lstrip().startswith("{") else {}
    user = (
        f"Task ID: {task.task_id}\n"
        f"Task Title: {task.title}\n"
        f"Budget Envelope: {json.dumps(task.budget, ensure_ascii=False)}\n\n"
        f"Raw Proposal Artifact:\n{raw_str[:24000]}"
    )
    try:
        data = runner.chat_json(_NORMALIZE_SYSTEM, user, max_tokens=4096, tag="normalize")
    except json.JSONDecodeError as exc:
        print(f"[warn] {task.task_id} normalize JSON parse failed ({exc}); using raw fallback.", flush=True)
        data = dict(raw_output) if isinstance(raw_output, dict) else {"title": task.title, "method": raw_str[:4000]}
    prior_facts = [PriorFactClaim.model_validate(x) for x in (data.get("prior_facts") or []) if isinstance(x, dict) and x.get("claim")]
    hypotheses = [HypothesisClaim.model_validate(x) for x in (data.get("prospective_hypotheses") or []) if isinstance(x, dict) and x.get("claim")]
    unknowns = [UnknownDependencyClaim.model_validate(x) for x in (data.get("unknown_dependencies") or []) if isinstance(x, dict) and x.get("dependency")]
    gap_premises = [str(x).strip() for x in (data.get("gap_premises") or []) if str(x).strip()]
    if not gap_premises and data.get("literature_gap"):
        gap_premises = [str(data["literature_gap"]).strip()]
    return NormalizedProposal(
        task_id=task.task_id,
        title=str(data.get("title") or fallback_card.get("title") or task.title),
        literature_gap=str(data.get("literature_gap") or fallback_card.get("literature_gap") or ""),
        hypothesis=str(data.get("hypothesis") or fallback_card.get("hypothesis") or ""),
        method=str(data.get("method") or fallback_card.get("method") or ""),
        minimal_test=str(data.get("minimal_test") or fallback_card.get("minimal_test") or ""),
        decisive_metric=str(data.get("decisive_metric") or fallback_card.get("decisive_metric") or ""),
        supporting_result=str(data.get("supporting_result") or fallback_card.get("supporting_result") or ""),
        falsifying_result=str(data.get("falsifying_result") or fallback_card.get("falsifying_result") or ""),
        related_work=str(data.get("related_work") or fallback_card.get("related_work") or ""),
        limitations=str(data.get("limitations") or fallback_card.get("limitations") or ""),
        resource_plan=data.get("resource_plan") if isinstance(data.get("resource_plan"), dict) else {},
        prior_facts=prior_facts,
        prospective_hypotheses=hypotheses,
        unknown_dependencies=unknowns,
        gap_premises=gap_premises[:3],
    )


_FACT_AUDIT_SYSTEM = """You audit ONE prior-fact claim from a research proposal using Support + Skeptic passes (SmoothScore-inspired + ClaimCheck grounding).
DO NOT punish forward-looking research hypotheses here—this claim asserts something about EXISTING prior literature or benchmarks.
Run two checks against the retrieved pre-cutoff evidence:
- Support Pass: does any retrieved paper actually support the cited fact or entity? Quote or summarize the matching retrieved snippet in evidence_quote.
- Skeptic Pass: does the claim commit Claim Smoothing (unsupported composition of distinct caveats, universal scope expansion from a narrow setting, misattribution, or citing fabricated/unverified claims)?
  Score smoothing_score on 1..5 where 1 = completely faithful to retrieved evidence (no smoothing/no overclaim), 2 = minor simplification, 3 = moderate caveat omission/smoothing, 4 = severe smoothing/unsupported composition or scope expansion, 5 = severe fabrication/contradicted overclaim.
Check also whether the claim disguises an unresolved/empirical unknown as an already-settled certainty (epistemic_overclaim).
CRITICAL: Anchor any critique in BOTH an exact substring from the claim/proposal (proposal_span) AND the retrieved evidence quote or explicit absence note (evidence_quote).

Return valid JSON only:
{
  "citation_status": "verified_local" | "verified_pre_cutoff" | "unverified" | "fabricated_or_missing",
  "smoothing_score": 1,
  "smoothing_type": "none" | "unsupported_composition" | "scope_expansion" | "misattribution" | "fabrication",
  "proposal_span": "<exact substring from the claim/proposal being audited>",
  "evidence_quote": "<verbatim quote or explicit missing-evidence note from retrieved sources>",
  "support_evidence": "<brief quote or summary>",
  "skeptic_counter_evidence": "<caveats, scope limits, or mismatch>",
  "epistemic_overclaim": true | false
}"""


def audit_tri_state_and_smoothing(
    proposal: NormalizedProposal,
    task: TaskSpec,
    judge_runner: LLMRunner,
    *,
    live_search: bool = False,
) -> TriStateAuditResult:
    audits: list[PriorFactAudit] = []
    proposal_blob = " ".join(
        [
            proposal.title,
            proposal.literature_gap,
            proposal.hypothesis,
            proposal.method,
            proposal.related_work,
            proposal.limitations,
            *(pf.claim for pf in proposal.prior_facts),
        ]
    ).lower()
    for pf in proposal.prior_facts[:12]:
        q = f"{pf.claim} {' '.join(pf.cited_titles)}".strip()
        evidence_items = gather_evidence_for_query(q, task, live_search=live_search, top_k=4)
        ev_text = "\n".join(
            f"- [{e.source} | {e.id} | {e.year or '?'}] {e.title}\n"
            f"  Abstract: {e.abstract[:500]}\n"
            f"  RelevantExcerpt: {extract_relevant_excerpt(q, e, max_chars=800)}"
            for e in evidence_items
        ) or "(no pre-cutoff evidence matched)"

        user = (
            f"Cutoff Date: {task.cutoff_date}\n"
            f"Prior-Fact Claim: {pf.claim}\n"
            f"Cited Titles: {pf.cited_titles} | Cited IDs: {pf.cited_ids}\n\n"
            f"Retrieved Pre-Cutoff Evidence:\n{ev_text}"
        )
        try:
            res = judge_runner.chat_json(_FACT_AUDIT_SYSTEM, user, max_tokens=1200, tag="tri_state_fact_audit")
        except json.JSONDecodeError as exc:
            print(f"[warn] {task.task_id} tri_state_fact_audit JSON parse failed ({exc}); marking unverified.", flush=True)
            res = {"citation_status": "unverified", "smoothing_score": 2, "smoothing_type": "none", "epistemic_overclaim": False}
        sc = int(min(5, max(1, int(res.get("smoothing_score", 2)))))
        cit_status = str(res.get("citation_status") or "unverified")
        if cit_status not in {"verified_local", "verified_pre_cutoff", "unverified", "fabricated_or_missing"}:
            cit_status = "unverified"
        sm_type = str(res.get("smoothing_type") or "none")
        if sm_type not in {"none", "unsupported_composition", "scope_expansion", "misattribution", "fabrication"}:
            sm_type = "none"

        prop_span = str(res.get("proposal_span") or pf.claim)[:200]
        ev_quote = str(res.get("evidence_quote") or res.get("support_evidence") or "")[:300]
        skeptic = str(res.get("skeptic_counter_evidence") or "")[:300]
        # ClaimCheck-inspired Critique Grounding Gate: penalizing critiques must anchor in proposal text + rationale.
        span_anchored = bool(prop_span.strip()) and (
            prop_span.strip().lower()[:40] in proposal_blob or pf.claim.strip().lower()[:40] in proposal_blob
        )
        evidence_anchored = bool(ev_quote.strip() or skeptic.strip())
        if task.strict_evidence:
            excerpts = [
                passage for e in evidence_items for passage in (
                    e.abstract[:500], extract_relevant_excerpt(q, e, max_chars=800),
                )
            ]
            quote = " ".join(ev_quote.split())
            evidence_anchored = len(quote) >= 12 and any(quote in " ".join(p.split()) for p in excerpts)
            if not evidence_anchored:
                cit_status, sm_type = "unverified", "none"
        critique_grounded = span_anchored and evidence_anchored
        if not critique_grounded and sc >= 3:
            sc = 2

        audits.append(
            PriorFactAudit(
                claim=pf.claim,
                citation_status=cit_status,  # type: ignore[arg-type]
                smoothing_score=sc,
                smoothing_type=sm_type,  # type: ignore[arg-type]
                support_evidence=str(res.get("support_evidence") or "")[:300],
                skeptic_counter_evidence=skeptic,
                proposal_span=prop_span,
                evidence_quote=ev_quote,
                critique_grounded=critique_grounded,
                epistemic_overclaim=bool(res.get("epistemic_overclaim", False)) and critique_grounded,
            )
        )

    # Prospective hypotheses are only flagged if they lack falsifiability bindings (never as unsupported facts).
    untested_hyp_cnt = len(proposal.prospective_hypotheses)
    honest_unknown_cnt = sum(
        1
        for u in proposal.unknown_dependencies
        if u.why_unresolved.strip() and (u.resolution_path.strip() or u.minimal_calibration_test.strip())
    )
    unop_unknown_cnt = len(proposal.unknown_dependencies) - honest_unknown_cnt
    overclaims = sum(1 for a in audits if a.epistemic_overclaim or a.smoothing_score >= 3)
    denom = max(1, len(audits) + untested_hyp_cnt + len(proposal.unknown_dependencies))
    max_sm = max((a.smoothing_score for a in audits), default=1)
    mean_sm = round(statistics.mean(a.smoothing_score for a in audits), 3) if audits else 1.0
    fabrication = any(a.citation_status == "fabricated_or_missing" or a.smoothing_type == "fabrication" for a in audits)

    return TriStateAuditResult(
        prior_fact_audits=audits,
        untested_hypothesis_count=untested_hyp_cnt,
        honest_unknown_count=honest_unknown_cnt,
        unoperationalized_unknown_count=unop_unknown_cnt,
        epistemic_overclaim_count=overclaims,
        max_smoothing_score=max_sm,
        mean_smoothing_score=mean_sm,
        epistemic_overclaim_rate=round(overclaims / denom, 3),
        citation_fabrication_detected=fabrication,
    )


_GAP_FALSIFY_SYSTEM = """You perform Pre-Cutoff Gap-Premise Falsification.
Given a proposal's motivating Gap Premise (e.g., "No prior work tests X" or "Existing methods assume Y") and retrieved papers strictly prior to the cutoff date, decide whether the gap premise is:
- "falsified": a pre-cutoff paper ALREADY directly addresses/solves the claimed gap or proves the stated limitation false;
- "partially_eroded": prior work addresses a close variant, so the universal gap claim must be narrowed;
- "survived": sufficiently relevant pre-cutoff evidence was assessed and did not falsify the gap within the supplied set;
- "unresolved": the supplied evidence is missing, too weak, or too indirect to assess the premise.

Return valid JSON only:
{
  "status": "survived" | "partially_eroded" | "falsified" | "unresolved",
  "counterexample_title": "<exact title or null>",
  "counterexample_year": "<year or null>",
  "justification": "<1-2 sentences citing evidence>"
}"""


def audit_gap_premises(
    proposal: NormalizedProposal,
    task: TaskSpec,
    judge_runner: LLMRunner,
    *,
    live_search: bool = False,
) -> GapAuditResult:
    items: list[GapPremiseAuditItem] = []
    for premise in (proposal.gap_premises or [proposal.literature_gap])[:3]:
        if not premise.strip():
            continue
        q = make_counterexample_query(premise)
        cands = gather_evidence_for_query(q, task, live_search=live_search, top_k=5)
        cands = [candidate for candidate in cands if candidate.abstract.strip() or candidate.fulltext_snippet.strip()]
        if not cands:
            items.append(GapPremiseAuditItem(premise=premise, status="unresolved", justification="No usable evidence retrieved."))
            continue
        ev_text = "\n".join(
            f"- [{c.source} | {c.year or '?'}] {c.title}\n  Abstract: {c.abstract[:500]}\n  RelevantExcerpt: {extract_relevant_excerpt(q, c, max_chars=550)}"
            for c in cands
        )
        user = (
            f"Cutoff Date: {task.cutoff_date}\n"
            f"Gap Premise: {premise}\n\n"
            f"Pre-Cutoff Candidate Counterexamples:\n{ev_text}"
        )
        try:
            res = judge_runner.chat_json(_GAP_FALSIFY_SYSTEM, user, max_tokens=1024, tag="gap_falsify")
        except json.JSONDecodeError as exc:
            print(f"[warn] {task.task_id} gap_falsify JSON parse failed ({exc}); marking unresolved.", flush=True)
            res = {"status": "unresolved", "justification": "Invalid audit response."}
        st = str(res.get("status") or "unresolved")
        if st not in {"survived", "partially_eroded", "falsified", "unresolved"}:
            st = "unresolved"
            res = {"justification": "Unrecognized audit status."}
        if task.strict_evidence and st in {"falsified", "partially_eroded"}:
            title = str(res.get("counterexample_title") or "").strip().casefold()
            if not any(c.title.strip().casefold() == title for c in cands):
                st = "unresolved"
                res = {"justification": "No matched counterexample title in supplied evidence."}
        try:
            finding = GapPremiseAuditItem(
                premise=premise,
                status=st,  # type: ignore[arg-type]
                counterexample_title=res.get("counterexample_title"),
                counterexample_year=res.get("counterexample_year"),
                justification=str(res.get("justification") or "")[:350],
            )
        except ValidationError:
            finding = GapPremiseAuditItem(premise=premise, status="unresolved", justification="Invalid audit fields.")
        items.append(finding)
    if not items:
        return GapAuditResult(items=[], survival_rate=1.0, any_falsified=False)
    survived = sum(1 for i in items if i.status == "survived")
    any_fals = any(i.status == "falsified" for i in items)
    return GapAuditResult(
        items=items,
        survival_rate=round(survived / len(items), 3),
        any_falsified=any_fals,
        unresolved_count=sum(i.status == "unresolved" for i in items),
    )


def evaluate_alphaideabench_axes(
    proposal: NormalizedProposal,
    task: TaskSpec,
    evidence_brief: str,
    judge_runner: LLMRunner,
) -> dict[str, Any]:
    """Use AlphaIdeaBench rubric dimensions WITH pre-verified evidence injected (fixing AlphaIdeaBench's retrieval-blind flaw)."""
    catalog = load_catalog()
    selected = (
        "motivation_method_consistency",
        "technical_feasibility",
        "data_feasibility",
        "budget_feasibility",
        "verification_feasibility",
    )
    rubric_lines = []
    for dim_id in selected:
        if dim_id in catalog.dimensions:
            rubric_lines.append(f"- {dim_id}: {catalog.dimensions[dim_id].question}")

    system = (
        "You score a normalized research proposal on grounded AlphaIdeaBench dimensions (1-5 integer/float) "
        "USING the supplied pre-cutoff Evidence Brief and Budget Envelope. "
        "Do NOT punish an honestly declared unknown dependency that names a valid resolution path. "
        "Return JSON with keys: budget_feasible (bool), motivation_method_consistency, technical_feasibility, "
        "data_feasibility, budget_feasibility, verification_feasibility, novelty_score (1-5)."
    )
    user = (
        f"Budget Envelope:\n{json.dumps(task.budget, ensure_ascii=False)}\n\n"
        f"Rubric Questions:\n" + "\n".join(rubric_lines) + "\n\n"
        f"Evidence Brief (Pre-Cutoff & Smoothing Audit):\n{evidence_brief}\n\n"
        f"Proposal:\n{proposal.model_dump_json(indent=2)}"
    )
    try:
        res = judge_runner.chat_json(system, user, max_tokens=1200, tag="aib_eval")
    except json.JSONDecodeError as exc:
        print(f"[warn] {task.task_id} aib_eval JSON parse failed ({exc}); using neutral 3.0 scores.", flush=True)
        res = {}
    return {
        "budget_feasible": bool(res.get("budget_feasible", True)),
        "motivation_method_consistency": float(res.get("motivation_method_consistency", 3.0)),
        "technical_feasibility": float(res.get("technical_feasibility", 3.0)),
        "data_feasibility": float(res.get("data_feasibility", 3.0)),
        "budget_feasibility": float(res.get("budget_feasibility", 3.0)),
        "verification_feasibility": float(res.get("verification_feasibility", 3.0)),
        "novelty_score": float(res.get("novelty_score", 3.0)),
    }


def compute_integrity_gate(
    proposal: NormalizedProposal,
    tri: TriStateAuditResult,
    gap: GapAuditResult,
    claimnov: ClaimNovAuditResult,
    aib: dict[str, Any],
    spec: SpecAuditResult | None = None,
) -> IntegrityGateResult:
    """Pre-Execution Integrity Hard-Fail Gate -> Trusted Novelty."""
    reasons: list[str] = []
    if tri.citation_fabrication_detected:
        reasons.append("citation_fabrication")
    if tri.max_smoothing_score >= 4:
        reasons.append(f"severe_claim_smoothing(max={tri.max_smoothing_score})")
    if gap.any_falsified:
        reasons.append("gap_premise_falsified_pre_cutoff")
    if claimnov.veto:
        reasons.append("claimnov_method_collision_veto")
    if not proposal.decisive_metric.strip() or not proposal.falsifying_result.strip():
        reasons.append("missing_falsification_contract")
    if not bool(aib.get("budget_feasible", True)) or float(aib.get("budget_feasibility", 3.0)) < 2.0:
        reasons.append("budget_infeasible")

    gate_pass = len(reasons) == 0
    spec_res = spec or SpecAuditResult()
    strict_gate_pass = gate_pass and spec_res.blocking_count == 0 and gap.unresolved_count == 0
    # Combine ClaimNov continuous novelty [0,1] (rescaled 1..5) and grounded AIB novelty (1..5), normalized 0..1.
    cn_scaled = 1.0 + 4.0 * max(0.0, min(1.0, claimnov.novelty_score))
    aib_nov = max(1.0, min(5.0, float(aib.get("novelty_score", 3.0))))
    raw_nov_unit = round(((0.5 * cn_scaled + 0.5 * aib_nov) - 1.0) / 4.0, 3)

    # Rigor & Actionability diagnostics (calibrated 0..4 / 1..5 scale)
    rigor = max(0.0, 4.0 - 0.8 * tri.epistemic_overclaim_count - (1.2 if gap.any_falsified else 0.0))
    aib_act_mean = (
        float(aib.get("technical_feasibility", 3.0))
        + float(aib.get("data_feasibility", 3.0))
        + float(aib.get("verification_feasibility", 3.0))
    ) / 3.0
    spec_scaled = 1.0 + 4.0 * spec_res.completeness_score
    actionability = round(
        0.75 * aib_act_mean + 0.25 * spec_scaled - 0.4 * tri.unoperationalized_unknown_count,
        3,
    )
    trusted_novelty = raw_nov_unit if gate_pass and gap.unresolved_count == 0 else 0.0
    return IntegrityGateResult(
        gate_pass=gate_pass,
        strict_gate_pass=strict_gate_pass,
        hard_fail_reasons=reasons,
        novelty_score_raw=raw_nov_unit,
        trusted_novelty=trusted_novelty,
        actionability_score=max(0.0, round(actionability, 3)),
        rigor_score=round(rigor, 3),
        spec_completeness=spec_res.completeness_score,
        budget_feasible=bool(aib.get("budget_feasible", True)),
    )


def _expanded_prior_corpus(
    proposal: NormalizedProposal,
    task: TaskSpec,
    *,
    live_search: bool,
) -> list:
    if not live_search:
        return list(task.prior_work)
    from .tasks_loader import normalize_title_hash

    merged = list(task.prior_work)
    seen = {normalize_title_hash(p.title) for p in merged if p.title}
    method_q = f"{proposal.title} {proposal.method} {proposal.hypothesis}".strip()[:220]
    for item in gather_evidence_for_query(method_q, task, live_search=True, top_k=6):
        h = normalize_title_hash(item.title)
        if h not in seen:
            seen.add(h)
            merged.append(item)
    return merged


def _tri_state_from_factual(
    proposal: NormalizedProposal,
    findings: list[FactualFinding],
) -> TriStateAuditResult:
    """Derive TriStateAuditResult directly from batch factual findings to avoid duplicate serial LLM calls."""
    audits: list[PriorFactAudit] = []
    for f in findings:
        ev_quote = f.evidence[0].quote if f.evidence else ""
        if f.label == "supported":
            audits.append(
                PriorFactAudit(
                    claim=f.quote,
                    citation_status="verified_local",
                    smoothing_score=1,
                    smoothing_type="none",
                    proposal_span=f.quote[:200],
                    evidence_quote=ev_quote[:300],
                    critique_grounded=True,
                    epistemic_overclaim=False,
                )
            )
        elif f.label == "defect":
            is_fab = f.defect_type == "fabrication"
            sm_type = (
                "fabrication" if is_fab
                else "misattribution" if f.defect_type in {"attribution", "numeric_mismatch", "coverage_contradiction"}
                else "scope_expansion"
            )
            score = 5 if is_fab else (4 if f.defect_type in {"scope_overclaim", "prediction_as_observed", "attribution", "numeric_mismatch", "coverage_contradiction"} else 3)
            audits.append(
                PriorFactAudit(
                    claim=f.quote,
                    citation_status="fabricated_or_missing" if is_fab else ("verified_local" if ev_quote else "unverified"),
                    smoothing_score=score,
                    smoothing_type=sm_type,  # type: ignore[arg-type]
                    skeptic_counter_evidence=f.reason[:300],
                    proposal_span=f.quote[:200],
                    evidence_quote=ev_quote[:300],
                    critique_grounded=True,
                    epistemic_overclaim=True,
                )
            )
        else:
            audits.append(
                PriorFactAudit(
                    claim=f.quote,
                    citation_status="unverified",
                    smoothing_score=2,
                    smoothing_type="none",
                    proposal_span=f.quote[:200],
                    evidence_quote=ev_quote[:300],
                    critique_grounded=bool(ev_quote),
                    epistemic_overclaim=False,
                )
            )
    untested_hyp_cnt = len(proposal.prospective_hypotheses)
    honest_unknown_cnt = sum(
        1
        for u in proposal.unknown_dependencies
        if u.why_unresolved.strip() and (u.resolution_path.strip() or u.minimal_calibration_test.strip())
    )
    unop_unknown_cnt = len(proposal.unknown_dependencies) - honest_unknown_cnt
    overclaims = sum(1 for a in audits if a.epistemic_overclaim or a.smoothing_score >= 3)
    denom = max(1, len(audits) + untested_hyp_cnt + len(proposal.unknown_dependencies))
    max_sm = max((a.smoothing_score for a in audits), default=1)
    mean_sm = round(statistics.mean(a.smoothing_score for a in audits), 3) if audits else 1.0
    fabrication = any(a.citation_status == "fabricated_or_missing" or a.smoothing_type == "fabrication" for a in audits)
    return TriStateAuditResult(
        prior_fact_audits=audits,
        untested_hypothesis_count=untested_hyp_cnt,
        honest_unknown_count=honest_unknown_cnt,
        unoperationalized_unknown_count=unop_unknown_cnt,
        epistemic_overclaim_count=overclaims,
        max_smoothing_score=max_sm,
        mean_smoothing_score=mean_sm,
        epistemic_overclaim_rate=round(overclaims / denom, 3),
        citation_fabrication_detected=fabrication,
    )


def run_full_audit(
    task: TaskSpec,
    arm: str,
    raw_output: str | dict[str, Any] | NormalizedProposal,
    judge_runner: LLMRunner,
    *,
    live_search: bool = False,
    enable_embeddings: bool = False,
    include_factual_audit: bool = True,
    run_tri_state: bool = False,
) -> FullAuditBundle:
    cost_before = judge_runner.usage.cost_usd
    if isinstance(raw_output, NormalizedProposal):
        artifact = raw_output.model_dump()
        proposal = raw_output
    else:
        artifact = raw_output
        proposal = normalize_artifact(task, artifact, judge_runner)
    prior_corpus = _expanded_prior_corpus(proposal, task, live_search=live_search)
    factual_findings = (
        audit_independent_claims(artifact, task, judge_runner, prior_work=prior_corpus)
        if include_factual_audit
        else []
    )
    unverified_entities = (
        audit_unverified_entities(artifact, task, prior_work=prior_corpus)
        if include_factual_audit
        else []
    )
    if run_tri_state or not include_factual_audit:
        tri = audit_tri_state_and_smoothing(proposal, task, judge_runner, live_search=live_search)
        defect_count = sum(1 for f in factual_findings if f.label == "defect")
        if defect_count > tri.epistemic_overclaim_count:
            tri = tri.model_copy(update={"epistemic_overclaim_count": defect_count})
    else:
        tri = _tri_state_from_factual(proposal, factual_findings)
    gap = audit_gap_premises(proposal, task, judge_runner, live_search=live_search)
    cn = run_claimnov_audit(proposal, prior_corpus, judge_runner, enable_embeddings=enable_embeddings)
    comp = analyze_compositional_remainder(proposal, cn)
    governance = audit_idea(task, artifact, judge_runner, claimnov=cn, prior_work=prior_corpus)
    spec = locate_spec_defects(proposal, task, judge_runner)
    confound = audit_confounds(proposal, task, judge_runner)
    spec = merge_confounds(spec, confound)
    brief = (
        f"Overclaims={tri.epistemic_overclaim_count}, MaxSmooth={tri.max_smoothing_score}, "
        f"GapFalsified={gap.any_falsified}, ClaimNovVeto={cn.veto}, ClaimNovScore={cn.novelty_score:.3f}, "
        f"SpecCompleteness={spec.completeness_score:.2f} (blocking={spec.blocking_count}), "
        f"CompositionalNovelty={comp.is_compositional_novelty}"
    )
    aib = evaluate_alphaideabench_axes(proposal, task, brief, judge_runner)
    gate = compute_integrity_gate(proposal, tri, gap, cn, aib, spec=spec)
    return FullAuditBundle(
        task_id=task.task_id,
        arm=arm,
        proposal=proposal,
        tri_state=tri,
        gap_audit=gap,
        claimnov=cn,
        spec_audit=spec,
        confound_audit=confound,
        factual_findings=factual_findings,
        unverified_entities=unverified_entities,
        compositional=comp,
        governance=governance,
        aib_scores=aib,
        gate=gate,
        audit_cost_usd=round(judge_runner.usage.cost_usd - cost_before, 6),
    )
