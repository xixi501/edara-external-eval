import json

import pytest

from edara_mvp_eval.artifact_export import ExportError, reconcile

EV_A = "ev-" + "a" * 24
EV_B = "ev-" + "b" * 24


def native_payload():
    return {
        "title": "A proposal",
        "claims": [{"text": f"first claim {EV_A}", "cite": [EV_A]}],
        "method": {"text": f"second claim {EV_B} and again {EV_B}"},
        "rq_coverage": {"dropped": True},
    }


def failed_export():
    payload = {key: value for key, value in native_payload().items() if key != "rq_coverage"}
    payload["task_id"] = "task-1"
    text = json.dumps(payload, ensure_ascii=False)
    text = text.replace(EV_A, "[REF]").replace(EV_B, "[REF]")
    return json.loads(text), text


def evidence_rows():
    return {
        EV_A: {"paper_id": "p1", "quote": "first source sentence", "section_id": "s1"},
        EV_B: {"paper_id": "p2", "quote": "second source sentence", "section_id": "s2"},
    }


def papers():
    return {"p1": {"title": "Paper One"}, "p2": {"title": "Paper Two"}}


def years():
    return {"p1": {"year": "2024", "source": "arxiv", "external_id": "2401.00001"},
            "p2": {"year": "", "source": "uploads", "external_id": "sha"}}


def call(exported, original, **overrides):
    options = dict(task_id="task-1", stage="proposal", version_id="rv-1", source_db="/tmp/db")
    options.update(overrides)
    return reconcile(exported, original, native_payload(), evidence_rows(), papers(), years(), **options)


def test_export_changes_only_citation_tokens():
    exported, original = failed_export()
    result = call(exported, original)
    assert result["text"].count("[S1]") == 2
    assert result["text"].count("[S2]") == 2
    assert "[REF]" not in result["text"]
    assert result["proof"]["refs"] == 4
    assert result["proof"]["sources"] == 2
    assert result["proof"]["inverse_ok"] and result["proof"]["round_trip_ok"]


def test_citation_map_is_keyed_by_first_appearance():
    exported, original = failed_export()
    result = call(exported, original)
    assert list(result["citation_map"]) == ["S1", "S2"]
    assert result["citation_map"]["S1"]["paper_id"] == "p1"
    assert result["citation_map"]["S1"]["quotes"] == {EV_A: "first source sentence"}
    assert result["citation_map"]["S1"]["year"] == "2024"
    assert result["citation_map"]["S2"]["year"] == ""


def test_text_difference_beyond_tokens_is_rejected():
    exported, original = failed_export()
    exported["claims"][0]["text"] = "a different claim [REF]"
    with pytest.raises(ExportError):
        call(exported, original)


def test_reference_count_mismatch_is_rejected():
    exported, original = failed_export()
    exported["claims"][0]["cite"] = []
    with pytest.raises(ExportError):
        call(exported, original)


def test_unknown_evidence_id_is_rejected():
    exported, original = failed_export()
    rows = evidence_rows()
    rows.pop(EV_B)
    with pytest.raises(ExportError):
        reconcile(exported, original, native_payload(), rows, papers(), years(),
                  task_id="task-1", stage="proposal", version_id="rv-1", source_db="/tmp/db")


def test_injected_field_with_references_is_rejected():
    exported, original = failed_export()
    exported["task_id"] = "task [REF]"
    with pytest.raises(ExportError):
        call(exported, original)
