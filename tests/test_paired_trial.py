import json
from types import SimpleNamespace

import pytest

from edara_mvp_eval.claimnov_adapter import run_claimnov_audit
from edara_mvp_eval.paired_calibration import calibration_cases
from edara_mvp_eval.paired_inputs import Evidence, prepare_inputs
from edara_mvp_eval.paired_review import (DESIGN, ask, asymmetric_unknowns, credibility_grade, honesty_score, majority_delta,
                                          novelty_score, review_pair)
from edara_mvp_eval.paired_trial import ROOT, apply_edits, axis_summary, run_arm, shared_full_pool
from edara_mvp_eval.schemas import NormalizedProposal, PriorWorkItem, TaskSpec


def task():
    return TaskSpec(task_id="demonstration", source_track="lit2test", field="computing",
                    title="Cache", prompt_context="Investigate cache misses.", cutoff_date="2024-04-01",
                    strict_evidence=True)


def patch(source, replacement):
    return {"field": "test", "source": source, "replacement": replacement}


def test_append_and_replacement_share_a_budget_and_are_atomic():
    draft = {"test": "Run a fast test on ten workloads."}
    valid = [patch("ten", "twenty"), patch("", "Keep workload order fixed.")]
    updated, _ = apply_edits(draft, {"edits": valid}, scope="local")
    assert updated["test"] == "Run a fast test on twenty workloads.\nKeep workload order fixed."
    with pytest.raises(ValueError, match="invalid_patch_count"):
        apply_edits(draft, {"edits": valid + [patch("", "Clarify.")] * 3}, scope="local")
    overlapped, edits = apply_edits(
        draft, {"edits": [patch("ten", "one"), patch("ten workloads", "two trials")]}, scope="local")
    assert overlapped["test"] == "Run a fast test on one workloads."
    assert [row["status"] for row in edits] == ["applied", "rejected"]
    assert edits[1]["reason"] == "overlapping_patch"
    assert draft["test"] == "Run a fast test on ten workloads."


def test_full_rewrite_can_preserve_untouched_native_fields():
    draft = {"test": "Original", "hypothesis": "Preserved"}
    assert apply_edits(draft, {"draft": {"test": "Revised"}}, scope="full")[0] == {
        "test": "Revised", "hypothesis": "Preserved"
    }
    with pytest.raises(ValueError, match="invalid_patch_count"):
        apply_edits(draft, {"draft": {"test": "Revised"}}, scope="local")


def test_local_uses_sentence_offsets_and_records_rejected_edits():
    original = {"test": "Repeat a test. Repeat a test."}
    output, edits = apply_edits(original, {"edits": [patch("@2", "Change the second test."),
                                                       patch("@9", "No anchor.")]}, scope="local")
    assert output == {"test": "Repeat a test. Change the second test."}
    assert [row["status"] for row in edits] == ["applied", "rejected"]
    assert apply_edits(original, {"draft": {"decision_rules[3]": "A new criterion."}}, scope="full")[0][
        "decision_rules[3]"
    ] == "A new criterion."


def test_attribution_only_version_cannot_be_used_as_a_novelty_comparator():
    pool = Evidence(task())
    dated = PriorWorkItem(id="prior", title="Study", year="2023", abstract="The old experiment used four cases.",
                          version="indexed-abstract", date_source="saved record")
    marker = pool.add(dated)
    undated = dated.model_copy(update={"id": "native", "year": None, "date_source": None,
                                      "abstract": "The native quote has no verifiable date."})
    assert pool.add(undated) == marker
    assert len(pool.packet()[0]["attribution_passages"]) == 2
    assert pool.packet()[0]["novelty_passages"] == [dated.abstract]
    original = {"claim": "Test one case."}
    revised = {"claim": "Test two cases simultaneously."}
    raw = {"label": "gain", "original_span": {"field": "claim", "quote": original["claim"]},
           "span": {"field": "claim", "quote": revised["claim"]}, "reason": "new feature",
           "evidence": [{"paper_id": marker, "quote": undated.abstract}]}
    undecidable = novelty_score(raw, original, revised, pool.packet())
    assert undecidable["signature"] == "unknown"
    assert undecidable["score"] is None
    assert "novelty_evidence_anchor_missing" in undecidable["repairs"]


def test_shared_search_pool_is_independent_of_treatment_visibility():
    task_spec = task()
    seed = Evidence(task_spec).snapshot()
    drafts = {}
    for arm in ("D", "E"):
        evidence = Evidence.restore(task_spec, seed)
        paper = PriorWorkItem(id=arm, title=f"Paper {arm}", abstract=f"Passage {arm}", year="2023",
                              version="indexed-abstract", date_source="saved record")
        marker = evidence.add(paper)
        drafts[arm] = {"draft": {"text": f"[{marker}]"},
                       "retrieved": [{"marker": marker, "paper": paper.model_dump()}]}
        assert len(evidence.packet()) == 1
    packet, variants = shared_full_pool(task_spec, seed, drafts)
    assert len(packet) == 2
    assert variants["D"] != variants["E"]
    assert Evidence.restore(task_spec, seed).packet() == []


def test_credibility_uses_logical_grades_and_keeps_influential_unknowns():
    stronger = dict.fromkeys(DESIGN, "yes")
    assert credibility_grade(stronger) == 3
    assert credibility_grade(stronger | {"decision": "no"}) == 2
    assert credibility_grade(stronger | {"control": "no"}) == 1
    assert credibility_grade(stronger | {"test": "no", "control": "unknown"}) == 0
    assert credibility_grade(stronger | {"decision": "unknown"}) is None


def test_honesty_scores_net_defects_on_fixed_denominator():
    original = {"text": "The source used ten trials.", "coverage": "The plan is preliminary."}
    latest = original | {"text": "The source used six trials. The source published eight models."}
    claims = [{"id": "c1", "field": "text", "quote": original["text"]}]
    baseline = {"c1": {"label": "defect"}}
    evidence = [{"paper_id": "S", "attribution_passages": ["The source used six trials and published one model."]}]
    proof = [{"paper_id": "S", "quote": "The source used six trials and published one model."}]
    raw = {"changes": [{"id": "c1", "label": "supported", "reason": "Correct count", "evidence": proof,
                        "span": {"field": "text", "quote": "The source used six trials."}}],
           "added": [{"label": "defect", "reason": "Wrong count", "evidence": proof,
                      "span": {"field": "text", "quote": "The source published eight models."}}]}
    outcome = honesty_score(raw, original, latest, claims, baseline, evidence)
    assert (outcome["K"], outcome["R"], outcome["E"], outcome["score"]) == (1, 1, 1, 0)


def test_format_controls_and_order_reversal_do_not_invent_effect():
    class Runner:
        def __init__(self):
            self.pairs = []

        def chat_json(self, system, user, **kwargs):
            pair = json.loads(user)
            self.pairs.append(pair)
            return dict.fromkeys(("a", "b"), {"label": "same", "reason": "Spelling only."})

    left, right = {"claim": "Use a fixed rule."}, {"claim": "Use one fixed rule."}
    forward, backward = Runner(), Runner()
    result = review_pair("novelty", left, {"left": left, "right": right}, {}, [],
                         [forward, backward], calibration=True)
    assert result["status"] == "tie"
    assert forward.pairs[0]["a"] == backward.pairs[0]["b"]
    assert forward.pairs[0]["b"] == backward.pairs[0]["a"]


def test_regular_control_only_uses_its_own_evidence_and_diagnosis(tmp_path):
    draft = {"test": "Use two batches."}

    class Caller:
        def __init__(self, answer):
            self.answer = answer
            self.payloads = []

        def chat_json(self, system, user, **kwargs):
            self.payloads.append(json.loads(user))
            return self.answer

    audit = Caller({"findings": ["The order is unclear."]})
    editor = Caller({"edits": [patch("", "Apply the same ordering in both groups.")]})
    task_spec = task()
    evidence = Evidence(task_spec).snapshot()
    result = run_arm("B", draft, task_spec, evidence, audit, editor, tmp_path)
    assert result["status"] == "changed"
    assert audit.payloads[0]["papers"] == editor.payloads[0]["papers"]
    assert editor.payloads[0]["diagnosis"] == audit.answer
    assert not result["queries"]


def test_summary_weights_topics_and_carries_unresolved_bounds():
    compare = "C_vs_B"

    def case(key, status, delta=None):
        return {"task_id": key, "comparisons": {compare: {"honesty": {"status": status, "delta": delta}}}}

    summary = axis_summary([case("x", "win", 0.5), case("x", "win", 0.5),
                           case("y", "loss", -1), case("y", "unresolved")], compare, "honesty")
    assert summary["net_wins"] == 0
    assert summary["bounds"] == [0.0, 0.5]


def test_frozen_cohorts_and_calibration_coverage():
    expected = {"pilot": (3, 9), "core": (15, 38), "extension": (25, 25)}
    for cohort, (topics, cases) in expected.items():
        frozen = ROOT / "data" / f"inputs_{cohort}.json"
        # Standalone handoff repo ships frozen inputs under data/.
        records = json.loads(frozen.read_text(encoding="utf-8")) if frozen.exists() else prepare_inputs(ROOT, cohort)
        ready = [item for topic in records.values() for item in topic["originals"].values()
                 if item["status"] == "ready"]
        assert len(records) == topics and len(ready) == cases
        assert not any(item["unmapped_citations"] for item in ready)
    try:
        cases = calibration_cases(ROOT)
    except FileNotFoundError:
        cases = json.loads((ROOT / "data" / "calibration_cases.json").read_text(encoding="utf-8"))
    assert len(cases) == 54


def test_claimnov_extraction_failure_falls_back_without_crashing_trial():
    class InvalidJSON:
        endpoint = SimpleNamespace(model="offline-model")

        def chat_json(self, *args, **kwargs):
            raise json.JSONDecodeError("invalid", "", 0)

    proposal = NormalizedProposal(task_id="fallback", title="Caching", literature_gap="gap",
                                  hypothesis="test this", method="method", minimal_test="test",
                                  decisive_metric="metric", supporting_result="better",
                                  falsifying_result="worse")
    outcome = run_claimnov_audit(proposal, [PriorWorkItem(id="P", title="Prior", abstract="Excerpt.")],
                                InvalidJSON())
    assert (outcome.claims_count, outcome.novelty_score, outcome.veto) == (0, 0.0, False)


def test_claim_ledger_omissions_and_preexisting_additions_do_not_kill_the_pair():
    original = {"text": "Alpha used ten trials.", "gap": "Nothing measured Beta."}
    candidate = original | {"text": "Alpha used ten trials. Beta reported a new benchmark."}
    claims = [{"id": "c1", "field": "text", "quote": original["text"]},
              {"id": "c2", "field": "gap", "quote": original["gap"]}]
    baseline = {"c1": {"label": "supported"}, "c2": {"label": "unknown"}}
    packet = [{"paper_id": "S", "attribution_passages": ["Alpha used ten trials."],
               "novelty_passages": [], "dates": ["2023"]}]
    raw = {"changes": [{"id": "c1", "label": "same", "reason": "unchanged"}],
           "added": [{"label": "supported", "reason": "new fact",
                      "span": {"field": "text", "quote": "Beta reported a new benchmark."}}]}
    outcome = honesty_score(raw, original, candidate, claims, baseline, packet)
    assert (outcome["K"], outcome["E"], outcome["score"]) == (2, 0, 0.0)
    assert "imputed_same:c2" in outcome["repairs"] or "imputed_unknown:c2" in outcome["repairs"]
    assert "ignored_added_label:supported" in outcome["repairs"]


def test_deleted_claim_still_present_is_scored_as_unchanged():
    original = {"text": "Alpha used ten trials."}
    claims = [{"id": "c1", "field": "text", "quote": original["text"]}]
    baseline = {"c1": {"label": "defect"}}
    raw = {"changes": [{"id": "c1", "label": "deleted", "reason": "judge says removed"}], "added": []}
    outcome = honesty_score(raw, original, original | {"extra": "new"}, claims, baseline, [])
    assert outcome["score"] == 0.0
    assert outcome["signature"] == [("c1", "same")]
    assert "deleted_but_anchored" in outcome["repairs"]


def test_only_asymmetric_unknowns_widen_the_band():
    order = {"B": {"signature": [("c1", "unknown"), ("c2", "same")]},
             "C": {"signature": [("c1", "unknown"), ("c2", "same")]}}
    assert asymmetric_unknowns(order, ("B", "C")) == 0
    order["B"]["signature"] = [("c1", "same"), ("c2", "same")]
    assert asymmetric_unknowns(order, ("B", "C")) == 1


def test_string_passage_reference_resolves_to_verbatim_text():
    packet = [{"paper_id": "S1", "title": "T",
               "attribution_passages": ["Old work measured four cases."],
               "novelty_passages": ["Old work measured four cases."], "dates": ["2023"]}]

    class Runner:
        def chat_json(self, system, user, **kwargs):
            return {"a": {"label": "same", "evidence": ["S1 A1"], "reason": "same"}}

    resolved = ask(Runner(), "prompt", {"a": {"claim": "x"}, "b": {"claim": "y"}, "papers": packet}, "test")
    assert resolved["a"]["evidence"] == [{"paper_id": "S1", "quote": "Old work measured four cases."}]


def test_three_votes_use_the_majority_delta():
    assert majority_delta([1, 1, -1]) == (1, "ok")
    assert majority_delta([0, 0, None]) == (0, "ok")
    assert majority_delta([1, 0, -1]) == (None, "disagreement")
    assert majority_delta([1, None, None]) == (None, "insufficient")


def test_append_cap_is_400_characters():
    draft = {"test": "Anchor sentence."}
    kept = apply_edits(draft, {"edits": [patch("", "a" * 400)]}, scope="local")
    assert kept[0]["test"].endswith("a" * 400)
    dropped = apply_edits(draft, {"edits": [patch("", "a" * 401)]}, scope="local")[1]
    assert dropped[0]["reason"] == "invalid_append"


def test_three_vote_review_wires_three_orders():
    class Runner:
        def chat_json(self, system, user, **kwargs):
            return {"a": {"label": "same", "reason": "unchanged"}, "b": {"label": "same", "reason": "unchanged"}}

    left, right = {"claim": "Alpha rule."}, {"claim": "Alpha rule, restated."}
    result = review_pair("novelty", left, {"left": left, "right": right}, {}, [],
                         [Runner(), Runner(), Runner()], calibration=False)
    assert result["status"] == "tie"
    assert len(result["orders"]) == 3


def test_unanchored_design_check_degrades_to_unknown():
    from edara_mvp_eval.paired_review import credibility_score
    draft = {"hypothesis": "H", "minimal_test": "One test.", "decisive_metric": "M",
             "supporting_result": "S", "falsifying_result": ""}
    raw = {key: {"label": "yes", "span": {"field": "falsifying_result", "quote": "@1"}, "reason": "r"}
           for key in DESIGN}
    raw["hypothesis"] = {"label": "yes", "span": {"field": "hypothesis", "quote": "H"}, "reason": "r"}
    outcome = credibility_score(raw, draft)
    assert outcome["checks"]["falsifying_result" if "falsifying_result" in DESIGN else "decision"] == "unknown"
    assert "decision:span_anchor_missing" in outcome["repairs"]


def test_sentence_marker_span_resolves_in_credibility_payload():
    payload = {"task": {}, "a": {"hypothesis": "First claim. Second claim."},
               "b": {"hypothesis": "Only one claim."}}

    class Runner:
        def chat_json(self, system, user, **kwargs):
            return {"a": {"hypothesis": {"label": "yes", "span": {"field": "hypothesis", "quote": "@2"},
                                          "reason": "r"}}}

    resolved = ask(Runner(), "p", payload, "t")
    assert resolved["a"]["hypothesis"]["span"]["quote"] == "Second claim."


def test_confound_enforcement_retries_once_and_records_missing():
    from edara_mvp_eval.middleware import TrustworthyNoveltyMiddleware

    diag = {"spec_defects": [
        {"defect_id": 1, "category": "confound_control", "severity": "blocking"},
        {"defect_id": 2, "category": "dataset_and_split", "severity": "recommended"}]}
    base = {"spec_clarifications": [{"defect_id": 2, "append_text": "split"}]}

    class Runner:
        def __init__(self, answer):
            self.answer = answer

        def chat_json(self, system, user, **kwargs):
            return self.answer

    good = Runner({"spec_clarifications": [{"defect_id": 1, "append_text": "arm"}, {"defect_id": 2}]})
    outcome = TrustworthyNoveltyMiddleware._enforce_confounds(None, diag, base, "u", good)
    assert outcome["retried"] and outcome["unaddressed"] == []
    assert outcome["payload"] is good.answer if False else outcome["payload"]["spec_clarifications"][0]["defect_id"] == 1

    bad = Runner({"spec_clarifications": [{"defect_id": 2}]})
    outcome = TrustworthyNoveltyMiddleware._enforce_confounds(None, diag, base, "u", bad)
    assert outcome["unaddressed"] == [1]
    assert outcome["payload"] is base


def test_confound_targets_use_the_single_decisive_arm():
    from edara_mvp_eval.paired_trial import confound_addressed, confound_targets
    diag = {"design_confounds": {"priority": "blocking", "decisive_missing": "always-retrieve arm",
                                 "confounds": [{"status": "absent", "discriminating_arm": "other arm"}]}}
    assert confound_targets(diag) == ["always-retrieve arm"]
    assert confound_addressed({"minimal_test": "Add the always-retrieve arm here.", "decisive_metric": ""},
                              ["always-retrieve arm"])
    assert not confound_addressed({"minimal_test": "Nothing relevant.", "decisive_metric": ""},
                                  ["always-retrieve arm"])


def test_reference_without_colon_and_marker_ids_and_paraphrased_same():
    from edara_mvp_eval.paired_review import lookup_pointer
    assert lookup_pointer("A1 The controller was introduced by them.", {"A1": "The controller was introduced by them."}) == \
        "The controller was introduced by them."

    original = {"claim": "We introduce a new controller for memory allocation."}
    candidate = {"claim": "The controller for memory allocation was originally introduced by S001."}
    claims = [{"id": "c1", "field": "claim", "quote": original["claim"]}]
    baseline = {"c1": {"label": "defect"}}
    packet = [{"paper_id": "S001", "attribution_passages": ["The controller was introduced by S001."]}]
    raw = {"changes": [{"id": "@1", "label": "supported", "reason": "fixed attribution",
                        "span": {"field": "claim", "quote": candidate["claim"]},
                        "evidence": [{"paper_id": "S001", "quote": "The controller was introduced by S001."}]}],
           "added": []}
    outcome = honesty_score(raw, original, candidate, claims, baseline, packet)
    assert outcome["R"] == 1 and outcome["score"] == 1.0
    assert "mapped_marker_ids" in outcome["repairs"]


def test_design_field_picks_existing_field_and_strips_preregistration():
    from edara_mvp_eval.paired_trial import design_field, apply_edits
    assert design_field({"validation": "x"}) == "validation"
    assert design_field({"minimal_test": "x", "validation": "y"}) == "minimal_test"
    draft = {"validation": "Baseline: none."}
    _, log = apply_edits(draft, {"edits": [{"field": "validation", "source": "",
                                            "replacement": "with a pre-registered sweep grid"}]},
                         scope="local")
    assert log[0]["status"] == "applied"
    assert "pre-registered" not in log[0]["replacement"].lower()
