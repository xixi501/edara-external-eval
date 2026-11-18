"""Evidence-bound contribution audit for black-box research idea generators."""

from __future__ import annotations

import json
from datetime import date
from typing import Any, Protocol

from pydantic import BaseModel, Field, ValidationError

from .host_adapter import as_draft_dict, filter_proposal_fields
from .schemas import ClaimNovAuditResult, ContributionFinding, IdeaGovernanceResult, PriorWorkItem, TaskSpec

_SYSTEM = """Audit the contribution claims of ONE research idea using only the supplied dated evidence.
Keep attribution separate from novelty. For each separately advertised contribution, select the
contiguous draft clause defining the contribution, and mark it core. Respect an explicit statement
that a mechanism is the sole contribution. Audit a proposed integration or interaction as one core
claim; separately list constituent methods as components. Evidence covering A and B separately does
not establish A+B or its novelty. A source may cover a mechanism even when terminology differs.
Ordinary comparisons, baselines, measurements, planned tests, and expected results that merely
evaluate the advertised mechanism are not additional contributions. Omit them from the findings
unless the draft independently presents a new measurement method or a distinct scientific result
or hypothesis as a contribution. Ignore generic novelty declarations without a mechanism.

Every finding must identify one verbatim, contiguous span of the draft. Use the shortest span that
still expresses the assessed claim. Mark a claim covered only if a supplied
source describes that contribution directly or the supplied passages together explicitly establish it.
For a reused component, use role component. For claims the draft presents as a new contribution, use core.
If the draft mentions a distinct, falsifiable contrast to the closest supplied work, use distinct_hypothesis
and cite the passage that establishes that contrast; this is only relative to that source.
Otherwise use unresolved. Never use remembered papers or background knowledge to supply absent evidence.
A source marked date_basis=participant_provenance may be used only to check whether a quoted
sentence exists in the supplied passage; do not use it to argue that a claim is novel or covered.
Flag whether the draft already gives the covered source credit. Do not invent an alternative contribution.

Return a JSON object with a claims array. Each item has:
role (core or component), status (covered, distinct_hypothesis or unresolved),
proposal_span (verbatim from draft), evidence (array of {paper_id, quote}, quote verbatim from source),
rationale (brief reason), and already_credited (boolean).
Always account for the draft's core contribution(s), including when none can be resolved."""

_MAX_PASSAGE = 3600


class AuditModel(Protocol):
    def chat_json(
        self, system: str, user: str, *, max_tokens: int, tag: str,
    ) -> dict[str, Any]: ...


class _ProposedFindings(BaseModel):
    claims: list[ContributionFinding] = Field(min_length=1, max_length=16)


def _collapse(value: str) -> str:
    return " ".join(value.split())


def _pre_cutoff(evidence: PriorWorkItem, cutoff: date) -> bool:
    value = evidence.publication_date or evidence.year
    if not value or evidence.source == "topic_doc":
        return False
    try:
        if len(value) == 4:
            return int(value) < cutoff.year
        return date.fromisoformat(value) < cutoff
    except ValueError:
        return False


def admissible(task: TaskSpec, paper: PriorWorkItem) -> bool:
    from .literature_scout import _candidate_identity_hashes

    identities = set().union(*(
        _candidate_identity_hashes(paper.title, value or "") for value in (paper.id, paper.url)
    ))
    if identities & set(task.forbidden_title_hashes):
        return False
    if task.strict_evidence and not (paper.version and paper.date_source):
        return False
    if paper.date_basis == "participant_provenance":
        # A source a candidate actually cited, whose publication date is absent from the
        # supplied record. Admitted only for quote verification, never as a novelty comparator.
        return paper.source == "native_export" and bool(paper.version and paper.date_source)
    return _pre_cutoff(paper, date.fromisoformat(task.cutoff_date))


def evidence_payload(task: TaskSpec, prior_work: list[PriorWorkItem] | None = None) -> list[dict[str, str]]:
    """Return the exact passages available to the model, with traceable IDs and dates."""
    result = []
    for paper in task.prior_work if prior_work is None else prior_work:
        if not admissible(task, paper):
            continue
        passages = "\n".join(
            p for p in (paper.fulltext_snippet, paper.abstract, paper.important_result, paper.key_contribution)
            if p and p.strip()
        )
        if passages:
            result.append({
                "paper_id": paper.id, "title": paper.title,
                "date": paper.publication_date or paper.year or "",
                "date_basis": paper.date_basis,
                "passages": passages[:_MAX_PASSAGE],
            })
    return result


def _decide(claims: list[ContributionFinding]) -> IdeaGovernanceResult:
    core = [claim for claim in claims if claim.role == "core"]
    if not core:
        return IdeaGovernanceResult(action="seek_evidence", claims=claims,
                                    unresolved_reason="no_grounded_core_span", defect="localization")
    uncredited = any(claim.status == "covered" and not claim.already_credited for claim in claims)
    if all(claim.status == "covered" for claim in core):
        action = "regenerate"
    elif uncredited:
        action = "correct_attribution"
    elif any(claim.status == "distinct_hypothesis" for claim in core):
        action = "retain_and_test"
    else:
        action = "seek_evidence"
    return IdeaGovernanceResult(
        action=action, claims=claims,
        unresolved_reason="core_coverage_unresolved" if action == "seek_evidence" else None,
        defect="coverage" if action == "seek_evidence" else "none",
    )


def audit_idea(
    task: TaskSpec,
    draft: str | dict[str, Any],
    judge: AuditModel,
    *,
    claimnov: ClaimNovAuditResult | None = None,
    prior_work: list[PriorWorkItem] | None = None,
) -> IdeaGovernanceResult:
    """Audit an unmodified idea draft. The judge and optional ClaimNov hints can be replaced."""
    evidence = evidence_payload(task, prior_work)
    if not evidence:
        return IdeaGovernanceResult(action="seek_evidence", unresolved_reason="no_pre_cutoff_evidence", defect="evidence")
    if isinstance(draft, dict):
        clean = filter_proposal_fields(draft)
        text = json.dumps(clean or draft, ensure_ascii=False, indent=2)
    else:
        text = draft
    hints = [
        {"claim": item.get("claim"), "paper_ids": [match["paper_id"] for match in item.get("top_matches", [])
                                                 if match.get("paper_id")]}
        for item in claimnov.claims
    ] if claimnov else []
    user = json.dumps({
        "topic": task.title, "cutoff_date": task.cutoff_date, "draft": text,
        "dated_evidence": evidence, "claimnov_search_hints": hints,
    }, ensure_ascii=False)
    try:
        raw = judge.chat_json(_SYSTEM, user, max_tokens=3600, tag="idea_governance")
        proposed = _ProposedFindings.model_validate(raw)
    except (json.JSONDecodeError, ValidationError):
        return IdeaGovernanceResult(action="seek_evidence", unresolved_reason="invalid_audit_response", defect="localization")

    draft_text = _collapse(text)
    passages = {row["paper_id"]: _collapse(row["passages"]) for row in evidence}
    findings = []
    for finding in proposed.claims:
        span = _collapse(finding.proposal_span)
        if not span or span not in draft_text:
            # An ungrounded reported core is still unresolved and cannot be dropped from the decision.
            if finding.role == "core":
                return IdeaGovernanceResult(
                    action="seek_evidence", claims=findings, unresolved_reason="invalid_core_span",
                    defect="localization",
                )
            continue
        grounded = bool(finding.evidence) and all(
            _collapse(reference.quote)
            and _collapse(reference.quote) in passages.get(reference.paper_id, "")
            for reference in finding.evidence
        )
        if finding.status != "unresolved" and not grounded:
            finding = finding.model_copy(update={
                "status": "unresolved", "evidence": [], "rationale": "No verifiable evidence quote for this finding.",
            })
        elif not grounded:
            finding = finding.model_copy(update={"evidence": []})
        findings.append(finding)
    return _decide(findings)
