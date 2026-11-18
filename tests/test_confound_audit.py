import json

from edara_mvp_eval.confound_audit import audit_confounds, confound_defects, merge_confounds
from edara_mvp_eval.schemas import (ConfoundAuditResult, ConfoundItem, NormalizedProposal,
                                    SpecAuditResult, SpecDefectItem, TaskSpec)


def task():
    return TaskSpec(task_id="t", source_track="lit2test", field="computing", title="T",
                    prompt_context="ctx", cutoff_date="2024-04-01", strict_evidence=True,
                    budget={"gpu_hours": 48})


def proposal():
    return NormalizedProposal(
        task_id="t", title="ARAR", literature_gap="gap", hypothesis="Gating retrieval improves accuracy.",
        method="Gate retrieval on confidence.", minimal_test="Run on HotpotQA and GSM8K.",
        decisive_metric="Exact match.", supporting_result="Gain over CoT.", falsifying_result="No gain.")


class Runner:
    def __init__(self, payload):
        self.payload = payload

    def chat_json(self, system, user, **kwargs):
        return json.loads(json.dumps(self.payload))


def test_absent_confound_becomes_first_blocking_spec_defect():
    raw = {"hypothesis": {"field": "hypothesis", "quote": "Gating retrieval improves accuracy."},
           "predicted_observation": "ARAR beats CoT",
           "confounds": [{"alternative": "Any retrieval is enough",
                          "same_observation_if": "always-retrieve matches ARAR",
                          "discriminating_arm": "always-retrieve with the same context budget",
                          "status": "absent", "anchor": None, "why_not_enough": ""}],
           "decisive_missing": "always-retrieve arm", "priority": "blocking"}
    result = audit_confounds(proposal(), task(), Runner(raw))
    defects = confound_defects(result)
    assert defects[0].category == "confound_control"
    assert defects[0].severity == "blocking"
    assert "always-retrieve" in defects[0].clarification_hint
    spec = merge_confounds(SpecAuditResult(defects=[SpecDefectItem(
        category="dataset_and_split", target_field="minimal_test", missing_detail="split")]), result)
    assert spec.defects[0].category == "confound_control"
    assert spec.blocking_count == 1


def test_present_confound_without_anchor_is_ambiguous_and_not_blocking():
    raw = {"confounds": [{"alternative": "Retrieval amount", "discriminating_arm": "always-retrieve",
                          "status": "present", "anchor": None}], "priority": "blocking"}
    result = audit_confounds(proposal(), task(), Runner(raw))
    assert result.confounds[0].status == "ambiguous"
    assert all(d.severity == "recommended" for d in confound_defects(result))


def test_merge_caps_at_four_and_keeps_unrelated_defects():
    result = ConfoundAuditResult(priority="blocking", confounds=[
        ConfoundItem(alternative="a", discriminating_arm="arm a", status="absent"),
        ConfoundItem(alternative="b", discriminating_arm="arm b", status="absent")])
    spec = SpecAuditResult(defects=[SpecDefectItem(category="dataset_and_split", target_field="minimal_test",
                                                   missing_detail=str(i)) for i in range(4)])
    merged = merge_confounds(spec, result)
    assert len(merged.defects) == 4
    assert [d.category for d in merged.defects[:2]] == ["confound_control", "confound_control"]
