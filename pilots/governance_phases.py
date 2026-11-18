"""Audit and one-call regeneration for the fixed-evidence pilot."""

from __future__ import annotations

import json
from dataclasses import asdict

from pydantic import BaseModel, Field, StrictBool, ValidationError

from edara_mvp_eval.epistemic_audit import audit_gap_premises
from edara_mvp_eval.governance import IdeaGovernanceResult, audit_idea, evidence_payload
from edara_mvp_eval.llm_client import extract_json_dict
from edara_mvp_eval.schemas import ClaimNovAuditResult, NormalizedProposal


def audit_case(task, case, baseline, runner):
    prior_audit = baseline["audit"]
    normalized = NormalizedProposal.model_validate(prior_audit["proposal"])
    claimnov = ClaimNovAuditResult.model_validate(prior_audit["claimnov"])
    gap = audit_gap_premises(normalized, task, runner)
    result = audit_idea(task, case["draft"], runner, claimnov=claimnov)
    correct, reason = score_case(case, result)
    return {
        "governance": result.model_dump(mode="json"),
        "gap": gap.model_dump(mode="json"), "matched_expectation": correct, "reason": reason,
        "usage": asdict(runner.usage),
    }


def score_case(case, result):
    if result.unresolved_reason not in (None, "core_coverage_unresolved"):
        return False, f"did not produce a usable core audit ({result.unresolved_reason})"
    core = [claim for claim in result.claims if claim.role == "core"]
    grounded_covered = [
        claim for claim in core if claim.status == "covered"
        and any(q.paper_id == case["decisive_paper_id"] for q in claim.evidence)
    ]
    if case["expected"] == "covered":
        success = bool(grounded_covered) and result.action == "regenerate"
        return success, "core covered by expected paper and routed to regeneration" if success else (
            "expected grounded coverage and regeneration"
        )
    if case["expected"] == "unresolved":
        expected_ids = {paper["id"] for paper in case["evidence"]}
        credited = {
            q.paper_id for claim in result.claims if claim.role == "component" and claim.status == "covered"
            for q in claim.evidence
        }
        success = bool(core) and all(claim.status == "unresolved" for claim in core) and expected_ids <= credited
        return success, "components anchored; combination remains unresolved" if success else (
            "expected both components anchored and unresolved combination"
        )
    success = bool(core) and not any(claim.status == "covered" for claim in core) and result.action in {
        "retain_and_test", "seek_evidence", "correct_attribution",
    }
    return success, "exploration retained against the supplied pool" if success else (
        "expected a grounded core finding without rejecting the candidate"
    )


class RegeneratedIdea(BaseModel):
    title: str = Field(min_length=1)
    literature_gap: str = Field(min_length=1)
    hypothesis: str = Field(min_length=1)
    method: str = Field(min_length=1)
    minimal_test: str = Field(min_length=1)
    decisive_metric: str = Field(min_length=1)
    falsifying_result: str = Field(min_length=1)
    related_work: str = Field(min_length=1)


_GENERATION = """Propose one research idea for the same topic after considering the original draft
and dated evidence. Provide an alternative that can be tested concretely. Attribute existing work,
state predicted results as hypotheses, and limit claims about novelty to the evidence. Treat any optional
diagnosis as advice about the original, not as evidence about your new idea.
Return one JSON object with these string fields: title, literature_gap, hypothesis, method,
minimal_test, decisive_metric, falsifying_result, related_work."""


def regenerate_candidate(task, case, runner, feedback=None, *, highlights=None, system=_GENERATION):
    request = {
        "topic": task.title, "cutoff_date": task.cutoff_date, "original_draft": case["draft"],
        "dated_evidence": evidence_payload(task),
    }
    if highlights is not None:
        request["evidence_highlights"] = highlights
    if feedback is not None:
        request["diagnosis"] = feedback
    response = runner.chat_text(
        system, json.dumps(request, ensure_ascii=False),
        temperature=0.3, max_tokens=4096, tag="governance_regenerate", json_mode=True,
    )
    try:
        idea = RegeneratedIdea.model_validate(extract_json_dict(response))
    except (ValidationError, json.JSONDecodeError):
        return {"idea": None, "response": response, "error": "invalid_generation_response", "usage": asdict(runner.usage)}
    return {"idea": idea.model_dump(), "usage": asdict(runner.usage)}


class BlindVerdict(BaseModel):
    replaces_covered_core: StrictBool
    has_unsupported_novelty: StrictBool
    tests_distinct_hypothesis: StrictBool
    rationale: str = Field(min_length=1)
    prior_paper_ids: list[str] = Field(default_factory=list)


_EVALUATION = """Evaluate the candidate as an alternative to the original research idea. Use only
the supplied pre-cutoff evidence to decide coverage or support. Answer three independent questions:
- replaces_covered_core: does the candidate propose a materially different central mechanism from the
  prior-covered contribution in the original? It can reuse that mechanism as a credited component or
  comparator if the new core is distinct; a renamed mechanism or another dataset alone does not count.
- has_unsupported_novelty: does the candidate claim priority without support, present an established
  mechanism as its own, or pass off an untested result as fact? A bounded untested hypothesis is allowed.
- tests_distinct_hypothesis: does it propose a concrete test and falsifying result for the materially
  different central contribution, in service of the original topic?
Return one JSON object with all three booleans, a brief rationale, and prior_paper_ids (IDs used)."""


def blind_evaluate(task, case, candidate, runner):
    if candidate["idea"] is None:
        return {"verdict": None, "success": False, "error": "invalid_candidate", "usage": asdict(runner.usage)}
    evidence = evidence_payload(task)
    request = {
        "topic": task.title, "cutoff_date": task.cutoff_date, "original_draft": case["draft"],
        "dated_evidence": evidence, "candidate": candidate["idea"],
    }
    try:
        response = runner.chat_json(_EVALUATION, json.dumps(request, ensure_ascii=False),
                                    max_tokens=1350, tag="governance_blind_verdict")
        verdict = BlindVerdict.model_validate(response)
        if set(verdict.prior_paper_ids) - {item["paper_id"] for item in evidence}:
            raise ValueError("verdict cites evidence outside the presented pool")
    except (json.JSONDecodeError, ValueError):
        return {"verdict": None, "success": False, "error": "invalid_verdict", "usage": asdict(runner.usage)}
    return {
        "verdict": verdict.model_dump(),
        "success": verdict.replaces_covered_core and not verdict.has_unsupported_novelty and verdict.tests_distinct_hypothesis,
        "usage": asdict(runner.usage),
    }


def feedback_for(result):
    governance = IdeaGovernanceResult.model_validate(result["governance"])
    if governance.action != "regenerate":
        return None
    return {
        "action": governance.action,
        "findings": [claim.model_dump() for claim in governance.claims],
    }
