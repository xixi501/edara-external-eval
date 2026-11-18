from types import SimpleNamespace

from edara_mvp_eval import feedback
from edara_mvp_eval.feedback import (
    FeedbackMiddleware,
    PolicyState,
    append_only_clarify,
    choose_action,
    eligible_actions,
)
from edara_mvp_eval.schemas import (IdeaGovernanceResult, NormalizedProposal, SpecDefectItem,
                                    TaskSpec)

PROPOSAL = NormalizedProposal(
    task_id="t", title="Title", literature_gap="gap", hypothesis="h", method="m",
    minimal_test="test", decisive_metric="metric", supporting_result="support",
    falsifying_result="falsify",
)


def bundle(*, action="seek_evidence", defect="coverage", blocking=0, overclaim=0, falsified=False):
    return SimpleNamespace(
        governance=IdeaGovernanceResult(action=action, defect=defect),
        tri_state=SimpleNamespace(epistemic_overclaim_count=overclaim),
        gap_audit=SimpleNamespace(any_falsified=falsified, items=[]),
        spec_audit=SimpleNamespace(blocking_count=blocking, defects=[]),
        proposal=PROPOSAL,
        arm="direct_feedback_mw",
    )


def test_search_does_not_preempt_a_grounded_clarification():
    actions = [name for name, _ in eligible_actions(bundle(blocking=3))]
    assert actions[0] == "clarify"
    assert "seek_evidence" in actions


def test_exhausted_search_leaves_only_clarification():
    state = PolicyState(searched=True)
    actions = [name for name, _ in eligible_actions(bundle(blocking=3), state)]
    assert actions == ["clarify"]


def test_localization_defect_does_not_trigger_search():
    actions = [name for name, _ in eligible_actions(bundle(defect="localization"))]
    assert "seek_evidence" not in actions
    assert actions == ["retain_and_test"]


def test_overclaim_beats_search_but_not_clarification():
    actions = [name for name, _ in eligible_actions(bundle(overclaim=2, blocking=1))]
    assert actions == ["local_patch", "clarify", "seek_evidence"]


def test_choose_action_stays_backward_compatible():
    assert choose_action(SimpleNamespace(governance=IdeaGovernanceResult(action="seek_evidence"))) == "seek_evidence"
    assert choose_action(SimpleNamespace(governance=IdeaGovernanceResult(action="regenerate"))) == "regenerate"


def test_each_action_runs_at_most_once():
    state = PolicyState(patched=True, clarified=True, regenerated=True, searched=True)
    assert eligible_actions(bundle(overclaim=1, blocking=1), state) == [("retain_and_test", "no_eligible_defect")]


def test_clarification_runs_before_search_and_search_still_runs(monkeypatch):
    calls = {"clarify": 0, "audit": 0}
    audits = [bundle(blocking=3), bundle(blocking=0)]

    def fake_audit(task, arm, draft, runner):
        calls["audit"] += 1
        return audits[min(calls["audit"], len(audits)) - 1]

    monkeypatch.setattr(feedback, "run_full_audit", fake_audit)

    def fake_clarify(proposal, defects, runner):
        calls["clarify"] += 1
        return 1, 0

    monkeypatch.setattr(feedback, "append_only_clarify", fake_clarify)
    task = TaskSpec(task_id="t", source_track="lit2test", title="Title", field="ML",
                    prompt_context="ctx", cutoff_date="2025-01-01", strict_evidence=True)
    current, audited, history, papers, final = FeedbackMiddleware().apply(
        task, {"hypothesis": "h"}, bundle(blocking=3),
        audit_runner=SimpleNamespace(), repair_runner=SimpleNamespace(),
        generator_callback=lambda packet: {}, search=lambda query: [], max_rounds=3)
    chosen = [row["chosen"] for row in history]
    assert chosen[0] == "clarify"
    assert "seek_evidence" in chosen
    assert calls["clarify"] == 1
    assert final == "retain_and_test"


def test_append_only_clarify_passes_only_clarification_payload():
    captured = {}

    class Runner:
        def chat_json(self, system, user, **kwargs):
            captured["system"] = system
            captured["payload"] = user
            return {"spec_clarifications": [{"defect_id": 1, "target_field": "method",
                                             "append_text": "Bound the claim to the tested regime."}]}

    proposal = PROPOSAL.model_copy(deep=True)
    defect = SpecDefectItem(category="metric_or_decision_rule", target_field="method",
                            missing_detail="no decision rule", severity="blocking")
    applied, rejected = append_only_clarify(proposal, [defect], Runner())
    assert "Do not rewrite" in captured["system"]
    assert "defects" in captured["payload"]
    assert isinstance(applied, int) and isinstance(rejected, int)
