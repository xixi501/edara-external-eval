import json

from edara_mvp_eval.stage_b_review import (
    candidate_text,
    locate_span,
    normalize,
    quote_in_passage,
    review_candidate,
    review_packet,
)

PASSAGE = "An existing mechanism runs on one benchmark."
SPAN = "The gate reduces selective risk under shift."


class Judge:
    def __init__(self, payload):
        self.source = payload
        self.calls = []

    def chat_json(self, system, user, **kwargs):
        self.calls.append(kwargs.get("tag"))
        return self.source(user) if callable(self.source) else self.source


def packet():
    return {"evidence": [{"paper_id": "p1", "title": "Prior", "date": "2024",
                          "passages": PASSAGE}]}


def verdict(span=SPAN, novelty_extra=None, quote="An existing mechanism runs on one benchmark"):
    novelty = {"status": "pass", "reason": "distinct", "candidate_span": span,
               "aspect": "mechanism",
               "difference": "Adds a shift-stratified abstention rule that the supplied work never tests.",
               "evidence": [{"paper_id": "p1", "quote": quote}]}
    novelty.update(novelty_extra or {})
    return {"honesty": {"status": "pass", "reason": "clear", "candidate_span": span},
            "credibility": {"status": "pass", "reason": "coherent", "candidate_span": span},
            "novelty": novelty}


def test_span_ladder_reports_how_it_matched():
    text = "The gate reduces selective risk under shift."
    assert locate_span(text, SPAN) == (True, "exact")
    assert locate_span(text, f'"{SPAN}"') == (True, "edge_quotes")
    assert locate_span(text, "The gate reduces\n selective risk")[1] == "exact"
    assert locate_span(text, "Something else entirely") == (False, "not_found")


def test_unicode_normalization_matches_typographic_variants():
    assert normalize("\u201cquote\u201d \u2014 dash") == '"quote" - dash'
    assert locate_span("a \u2014 b", "a - b")[0]


def test_quote_matching_allows_ordered_elision():
    passage = "We show that the gate reduces risk on shift and fails on ties in one setting."
    assert quote_in_passage("We show that the gate reduces risk ... fails on ties", passage)
    assert not quote_in_passage("fails on ties ... We show that the gate", passage)
    assert not quote_in_passage("Absent sentence about tides", passage)


def test_edge_quoted_span_is_no_longer_downgraded():
    judge = Judge(verdict(span=f'"{SPAN}"'))
    result = review_candidate(packet(), SPAN, judge)
    honesty = result["axes"]["honesty"]
    assert honesty["status"] == "pass"
    assert honesty["span_mode"] == "edge_quotes"
    assert "validation" not in honesty
    assert result["joint_pass"]


def test_anchor_failure_is_retried_once_then_recorded():
    def responder(user):
        payload = verdict(span="A span that is not in the candidate")
        return payload

    judge = Judge(responder)
    result = review_candidate(packet(), {"hypothesis": SPAN}, judge)
    assert judge.calls == ["stage_b_verdict", "stage_b_verdict_retry"]
    row = result["axes"]["honesty"]
    assert row["status"] == "unresolved"
    assert row["validation"] == "anchor_not_located"
    assert row["raw_status"] == "pass"
    assert row["retry_used"]
    assert not result["joint_pass"]


def test_retry_can_recover_a_verbatim_span():
    calls = {"n": 0}

    def responder(user):
        calls["n"] += 1
        return verdict(span=SPAN if calls["n"] > 1 else "not present at all")

    result = review_candidate(packet(), {"hypothesis": SPAN}, Judge(responder))
    assert result["axes"]["honesty"]["status"] == "pass"
    assert not result["axes"]["honesty"].get("validation")


def test_fabricated_quote_is_rejected():
    result = review_candidate(packet(), {"hypothesis": SPAN},
                              Judge(verdict(quote="An invented sentence about the prior work")))
    novelty = result["axes"]["novelty"]
    assert novelty["status"] == "unresolved"
    assert novelty["validation"] == "evidence_quote_not_found"


def test_novelty_pass_requires_aspect_and_difference():
    missing_aspect = review_candidate(packet(), {"hypothesis": SPAN},
                                      Judge(verdict(novelty_extra={"aspect": None})))
    assert missing_aspect["axes"]["novelty"]["validation"] == "novelty_missing_aspect"
    short = review_candidate(packet(), {"hypothesis": SPAN},
                             Judge(verdict(novelty_extra={"difference": "small tweak"})))
    assert short["axes"]["novelty"]["validation"] == "novelty_missing_difference"
    no_comparator = review_candidate(packet(), {"hypothesis": SPAN},
                                     Judge(verdict(novelty_extra={"evidence": []})))
    assert no_comparator["axes"]["novelty"]["validation"] == "novelty_missing_comparator"


def test_candidate_text_is_order_stable():
    assert candidate_text({"b": 1, "a": 2}) == candidate_text({"a": 2, "b": 1})
    assert candidate_text({"arm": "x", "a": 1}) == candidate_text({"a": 1, "arm": "y"})


def test_review_packet_sorts_and_normalizes_evidence():
    task = type("Task", (), {"title": "t", "prompt_context": "ctx\n\nDated literature:\n[]",
                             "cutoff_date": "2025-01-01", "budget": 1})()
    rows = [{"paper_id": "z", "passages": "b  b\n b"},
            {"paper_id": "a", "passages": "a   a"}]
    packed = review_packet(task, rows)
    assert [row["paper_id"] for row in packed["evidence"]] == ["a", "z"]
    assert packed["evidence"][0]["passages"] == "a a"


def test_invalid_review_response_is_reported_not_scored():
    class Broken:
        def chat_json(self, system, user, **kwargs):
            return {"honesty": {"status": "pass"}}

    result = review_candidate(packet(), {"hypothesis": SPAN}, Broken())
    assert result["status"] == "invalid_review"


def test_rubric_v3_requires_a_falsifying_test():
    payload = verdict(novelty_extra={"falsifying_test": "A within-topic AUROC no higher than chance."})
    result = review_candidate(packet(), SPAN, Judge(payload), rubric="v3")
    assert not result["axes"]["novelty"].get("validation")
    missing = review_candidate(
        packet(), SPAN,
        Judge(verdict(novelty_extra={"falsifying_test": None})), rubric="v3")
    assert missing["axes"]["novelty"]["validation"] == "novelty_missing_falsifying_test"


def test_rubric_v3_requires_credibility_falsifying_test():
    payload = verdict()
    payload["credibility"] = {"status": "pass", "reason": "coherent", "candidate_span": SPAN,
                              "falsifying_test": None}
    result = review_candidate(packet(), SPAN, Judge(payload), rubric="v3")
    assert result["axes"]["credibility"]["validation"] == "credibility_missing_falsifying_test"
    assert not result["joint_pass"]


def test_unknown_rubric_is_rejected():
    import pytest
    with pytest.raises(ValueError):
        review_candidate(packet(), SPAN, Judge(verdict()), rubric="v9")
