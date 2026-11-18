import json
from types import SimpleNamespace

from edara_mvp_eval.feedback import choose_action
from edara_mvp_eval.governance import evidence_payload
from edara_mvp_eval.llm_client import ModelEndpoint
from edara_mvp_eval.schemas import IdeaGovernanceResult, PriorWorkItem, TaskSpec
from edara_mvp_eval.stage_b_meter import RecordedRunner
from edara_mvp_eval.stage_b_review import compare, review_candidate


def test_strict_stage_b_evidence_requires_date_version_and_identity():
    task = TaskSpec(task_id="foo", source_track="lit2test", title="Test", field="ML",
                    prompt_context="Test", cutoff_date="2025-06-10", strict_evidence=True)
    base = PriorWorkItem(id="paper", title="Paper", abstract="An earlier mechanism",
                         date_source="published metadata", version="v1")
    latest = base.model_copy(update={"publication_date": "2025-06-09"})
    same_day = base.model_copy(update={"id": "same", "publication_date": task.cutoff_date})
    missing = base.model_copy(update={"id": "missing", "year": "2025"})
    unknown_version = latest.model_copy(update={"id": "no-version", "version": None})
    assert [x["paper_id"] for x in evidence_payload(task, [same_day, missing, latest, unknown_version])] == ["paper"]


def test_different_actions_for_unknown_and_covered_core():
    for expected in ("seek_evidence", "regenerate"):
        bundle = SimpleNamespace(governance=IdeaGovernanceResult(action=expected))
        assert choose_action(bundle) == expected


def test_blind_verdict_downgrades_invalid_quotes_and_filters_arm():
    class Judge:
        def chat_json(self, system, user, **kwargs):
            request = json.loads(user)
            assert "direct_feedback_mw" not in request["candidate"]
            assert "strict_gate_pass" not in request
            result = {"status": "pass", "reason": "Clear", "candidate_span": "Untested mechanism"}
            wrong = {"paper_id": "prior", "quote": "Invented comparison"}
            return {"honesty": result, "credibility": result, "novelty": {**result, "evidence": [wrong]}}

    packet = {"evidence": [{"paper_id": "prior", "passages": "An existing mechanism"}]}
    candidate = {"arm": "direct_feedback_mw", "hypothesis": "Untested mechanism"}
    verdict = review_candidate(packet, candidate, Judge())
    assert verdict["axes"]["honesty"]["status"] == "pass"
    assert verdict["axes"]["novelty"]["status"] == "unresolved"
    assert not verdict["joint_pass"]


def test_pairwise_keeps_both_reasons_and_folds_disagreement():
    class Judge:
        def chat_json(self, system, user, **kwargs):
            score = {"honesty": 3, "credibility": 3, "novelty": 3}
            return {"winner": "A", "A": score, "B": score, "reason": json.loads(user)["A"]}

    result = compare({"evidence": []}, "First", "Second", Judge())
    assert result["winner"] == "tie"
    assert result["order_sensitive"]
    assert [r["reason"] for r in result["orders"]] == ["First", "Second"]


def test_recorded_request_replays_without_rebilling(tmp_path):
    options = {"task_id": "one", "arm": "direct", "stage": "revision", "dry_run": True}
    runner = RecordedRunner(ModelEndpoint(), tmp_path, **options)
    response = runner.chat_text("system", "prompt", tag="generate_direct")
    assert runner.usage.calls == 1
    replay = RecordedRunner(ModelEndpoint(), tmp_path, **options)
    assert replay.chat_text("system", "prompt", tag="generate_direct") == response
    assert replay.usage.calls == 0
    assert json.loads((tmp_path / "0001.json").read_text())["usage"]["calls"] == 1


def test_related_literature_does_not_match_on_stopwords():
    from edara_mvp_eval.stage_b_evidence import _relevant

    query = "Towards Agentic AI for Science: Hypothesis Generation and Validation"
    assert not _relevant(query, "Extrapolation for bilinear operators", "Towards new bounds.")
    assert _relevant(query, "Agentic hypothesis generation", "A method for research proposals.")
