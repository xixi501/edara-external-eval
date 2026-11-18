"""Regression cases for evidence binding and unresolved audits."""

import json

import pytest

from edara_mvp_eval.governance import audit_idea
from edara_mvp_eval.epistemic_audit import audit_gap_premises, compute_integrity_gate
from edara_mvp_eval.schemas import ClaimNovAuditResult, NormalizedProposal, PriorWorkItem, TaskSpec, TriStateAuditResult


SPAN = "A predictor coupled to a dynamic budget rule."
QUOTE = "The predictor and dynamic budget rule operate jointly."


class Judge:
    def __init__(self, result):
        self.result = result
        self.requests = []

    def chat_json(self, system, user, **kwargs):
        self.requests.append(user)
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


def task(*papers):
    return TaskSpec(
        task_id="test", title="Research topic", field="test", source_track="aiib",
        prompt_context="Study better budget rules", cutoff_date="2025-01-04", prior_work=list(papers),
    )


def evidence(identifier="paper", passage=QUOTE, date="2024-06-12"):
    return PriorWorkItem(id=identifier, title=identifier, abstract=passage, year=date)


def finding(role="core", status="covered", span=SPAN, paper_id="paper", quote=QUOTE, credited=False):
    return {
        "role": role, "status": status, "proposal_span": span, "rationale": "Audited relation.",
        "evidence": [{"paper_id": paper_id, "quote": quote}] if quote else [],
        "already_credited": credited,
    }


def proposal():
    return NormalizedProposal(
        task_id="test", title="Research topic", literature_gap="No predictor models cost.",
        hypothesis="Prediction will improve.", method=SPAN, minimal_test="Compare under matched budget.",
        decisive_metric="Error rate", supporting_result="Lower error", falsifying_result="No difference",
    )


def test_combination_changes_status_only_with_combination_evidence():
    parts = [evidence("a", "We introduce a predictor."), evidence("b", "We use a dynamic budget rule.")]
    components = [
        finding("component", span="A predictor", paper_id="a", quote=parts[0].abstract, credited=True),
        finding("component", span="a dynamic budget rule", paper_id="b", quote=parts[1].abstract, credited=True),
    ]
    without = audit_idea(task(*parts), SPAN, Judge({"claims": [*components, finding()]}))
    with_prior = audit_idea(task(*parts, evidence()), SPAN, Judge({"claims": [*components, finding()]}))
    assert without.action == "seek_evidence"
    assert without.claims[-1].status == "unresolved"
    assert with_prior.action == "regenerate"
    assert with_prior.claims[-1].evidence[0].paper_id == "paper"


@pytest.mark.parametrize("newer_date", ["2025-01-04", "2025-01-05", "2025", None])
def test_post_cutoff_and_undated_quotes_are_not_shown(newer_date):
    judge = Judge({"claims": [finding()]})
    assessed = audit_idea(task(evidence(date=newer_date)), SPAN, judge)
    assert assessed.action == "seek_evidence"
    assert assessed.unresolved_reason == "no_pre_cutoff_evidence"
    assert not judge.requests


def test_covered_component_and_distinct_hypothesis_are_both_retained():
    raw = {"claims": [
        finding("component", span="A predictor", quote="We introduce a predictor."),
        finding(status="distinct_hypothesis", quote="We introduce a predictor."),
    ]}
    result = audit_idea(task(evidence(passage="We introduce a predictor.")), SPAN, Judge(raw))
    assert result.action == "correct_attribution"
    assert [c.status for c in result.claims] == ["covered", "distinct_hypothesis"]


def test_all_invalid_quote_or_span_cases_are_unresolved():
    invalid_span = audit_idea(task(evidence()), SPAN, Judge({"claims": [finding(span="not in the draft")]}))
    invalid_quote = audit_idea(task(evidence()), SPAN, Judge({"claims": [finding(quote="invented excerpt")]}))
    invalid_response = audit_idea(task(evidence()), SPAN, Judge({"malformed": True}))
    assert invalid_span.unresolved_reason == "invalid_core_span"
    assert invalid_quote.action == "seek_evidence"
    assert invalid_quote.claims[0].status == "unresolved"
    assert invalid_response.unresolved_reason == "invalid_audit_response"


def test_method_passages_take_precedence_over_long_abstracts():
    source = evidence(passage="Research background. " * 150).model_copy(update={"fulltext_snippet": QUOTE})
    judge = Judge({"claims": [finding()]})
    result = audit_idea(task(source), SPAN, judge)
    assert result.action == "regenerate"
    assert json.loads(judge.requests[0])["dated_evidence"][0]["passages"].startswith(QUOTE)


def test_missing_gap_evidence_does_not_call_judge_or_earn_trust():
    judge = Judge({"status": "survived"})
    result = audit_gap_premises(proposal(), task(), judge)
    assert result.items[0].status == "unresolved"
    assert result.unresolved_count == 1
    assert result.survival_rate == 0.0
    assert not result.any_falsified
    assert not judge.requests
    gate = compute_integrity_gate(proposal(), TriStateAuditResult(), result, ClaimNovAuditResult(), {})
    assert gate.gate_pass
    assert not gate.strict_gate_pass
    assert gate.trusted_novelty == 0.0


@pytest.mark.parametrize("response", [
    {}, {"status": "surprisingly_passed"}, {"status": "survived", "counterexample_title": ["invalid"]},
    json.JSONDecodeError("bad", "", 0),
])
def test_invalid_gap_replies_are_unresolved(response, monkeypatch):
    prior = evidence()
    monkeypatch.setattr("edara_mvp_eval.epistemic_audit.gather_evidence_for_query", lambda *args, **kwargs: [prior])
    result = audit_gap_premises(proposal(), task(prior), Judge(response))
    assert result.items[0].status == "unresolved"
    assert result.unresolved_count == 1
    assert result.survival_rate == 0.0
