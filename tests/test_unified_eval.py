"""Tests for unified bench-grounded evaluation (Lit2Test + AlphaIdeaBench + ClaimNov)."""
from __future__ import annotations

from edara_mvp_eval.llm_client import LLMRunner, ModelEndpoint
from edara_mvp_eval.paired_trial import ROOT
from edara_mvp_eval.schemas import PriorWorkItem, TaskSpec
from edara_mvp_eval.unified_eval import (
    build_lit2test_calibration_suite,
    calibrate_unified_evaluator,
    eval_alphaideabench_pair,
    eval_claimnov_pair,
    eval_lit2test_pair,
    evaluate_unified_pair,
    summarize_unified_rows,
)


def _task() -> TaskSpec:
    return TaskSpec(
        task_id="u-test-1",
        source_track="lit2test",
        field="ml",
        title="Adaptive Eviction",
        prompt_context="Study adaptive KV cache eviction under fixed memory.",
        cutoff_date="2024-06-01",
        strict_evidence=True,
        budget={"gpu": "1x A100", "gpu_hours": 24, "api_usd": 15},
        prior_work=[
            PriorWorkItem(
                id="S1",
                title="Static Eviction",
                abstract="Prior work uses fixed sliding windows with bounded attention mass.",
                year="2023",
                publication_date="2023-05-01",
                date_source="test",
                version="v1",
                source="lit2test_paper",
            )
        ],
    )


def _packet() -> list[dict]:
    return [
        {
            "paper_id": "S1",
            "title": "Static Eviction",
            "attribution_passages": ["Prior work uses fixed sliding windows with bounded attention mass."],
            "novelty_passages": ["Prior work uses fixed sliding windows with bounded attention mass."],
            "dates": ["2023"],
        }
    ]


def test_lit2test_bidirectional_folding_and_position_bias_protection():
    runner = LLMRunner(ModelEndpoint(), dry_run=True)
    weak = {
        "literature_gap": "No prior work studies eviction.",
        "hypothesis": "Eviction helps.",
        "minimal_test": "Run once.",
        "decisive_metric": "Accuracy.",
        "falsifying_result": "No gain.",
    }
    strong = {
        "literature_gap": "Static sliding windows drop rare bridge tokens.",
        "hypothesis": "Retention improves recall under matched budget.",
        "minimal_test": "Held-out split with matched-budget ablation control [Repaired].",
        "decisive_metric": "AURC at 80% coverage (bounded).",
        "falsifying_result": "AURC fails to beat matched control by >= 2%.",
    }
    res = eval_lit2test_pair(weak, strong, _task(), _packet(), [runner, runner])
    assert res["status"] == "win"
    assert res["folded_winner"] == "right"
    assert not res["order_sensitive"]
    assert res["score_sum_delta"] > 0

    # Identical drafts short-circuit with zero calls.
    tie_res = eval_lit2test_pair(weak, weak, _task(), _packet(), [runner, runner])
    assert tie_res["status"] == "tie" and tie_res["detail"] == "identical_text"

    # A position-biased judge (always picking A) must fold to tie.
    class AlwaysARunner:
        def chat_json(self, *args, **kwargs):
            return {
                "dimensions": {
                    d: {"score_a": 3, "score_b": 1, "winner": "A", "reason": "bias"}
                    for d in ("grounding", "hypothesis_specificity", "minimality_feasibility",
                              "decisive_metric", "falsifiability")
                },
                "overall_winner": "A",
                "overall_confidence": "high",
                "overall_reason": "Always A.",
            }

    biased = eval_lit2test_pair(weak, strong, _task(), _packet(), [AlwaysARunner(), AlwaysARunner()])
    assert biased["order_sensitive"] is True
    assert biased["status"] == "tie"


def test_alphaideabench_official_rubric_and_multihost_edara_projection():
    runner = LLMRunner(ModelEndpoint(), dry_run=True)
    edara_left = {
        "title": "Static Baseline",
        "hypothesis": "H1",
        "conditions[1]": "Baseline",
        "analysis.metrics[1]": "Accuracy",
        "decision_rules[1]": "Stop if worse",
    }
    edara_right = {
        "title": "Bounded Adaptive Eviction [Repaired]",
        "hypothesis": "H1 with matched control",
        "conditions[1]": "Baseline",
        "conditions[2]": "Matched-budget ablation control (bounded)",
        "analysis.metrics[1]": "AURC at 80% coverage",
        "decision_rules[1]": "Falsify if AURC gain < 2%",
    }
    aib = eval_alphaideabench_pair(
        edara_left, edara_right, _task(), _packet(), runner,
        dimensions=("context_grounding", "testability", "method_novelty"),
    )
    assert aib["dimension_deltas"]["context_grounding"] == 1.0
    assert aib["dimension_deltas"]["testability"] == 1.0
    assert len(aib["left"]["dimensions"]["context_grounding"]["checklist_notes"]) > 0
    assert not aib["left"]["dimensions"]["context_grounding"]["missing_checklist_items"]

    cn = eval_claimnov_pair(edara_left, edara_right, _task(), _task().prior_work, runner)
    assert "novelty_delta" in cn and "veto_delta" in cn

    combined = evaluate_unified_pair(
        edara_left, edara_right, _task(), _packet(),
        pair_runners=[runner, runner], aib_runner=runner, claimnov_runner=runner,
        prior_corpus=_task().prior_work,
    )
    summary = summarize_unified_rows(
        [{"task_id": "t1", "unified_comparisons": {"C_vs_B": combined}}],
        "C_vs_B",
    )
    assert summary["lit2test_net_wins"] == 1.0
    assert summary["alphaideabench_dimension_deltas"]["context_grounding"] == 1.0


def test_lit2test_subtle_corruption_and_sham_calibration_suite_dry_run():
    cases = build_lit2test_calibration_suite(ROOT, limit_per_kind=2)
    assert len(cases) == 8  # 3 corruption kinds * 2 + 2 shams
    runner = LLMRunner(ModelEndpoint(), dry_run=True)
    report = calibrate_unified_evaluator(cases, lambda _cid: [runner, runner])
    assert report["substantive_correct"] == 6
    assert report["sham_ties"] == 2
    assert report["qualified"] is True
    assert report["mean_target_dimension_deltas"]["grounding"] < 0
    assert report["mean_target_dimension_deltas"]["decisive_metric"] < 0
    assert report["mean_target_dimension_deltas"]["falsifiability"] < 0


def test_paired_trial_unified_eval_mode_integrates_and_summarizes(tmp_path):
    from types import SimpleNamespace
    from edara_mvp_eval.paired_inputs import Evidence
    from edara_mvp_eval.paired_trial import Trial, summarize

    out_dir = tmp_path / "trial_unified"
    out_dir.mkdir()
    args = SimpleNamespace(
        out=out_dir,
        phase="freeze",
        eval_mode="unified",
        calibration_set="original",
        calibration_from=None,
        gen_profile=ROOT.parents[1] / "data" / "p1b-idea-profile.json",
        audit_profile=ROOT.parents[1] / "data" / "p1b-critic-profile.json",
        judge_profile=ROOT.parents[1] / "data" / "p1b-critic-profile.json",
    )
    if not args.gen_profile.is_file():
        return
    trial = Trial(args)
    assert trial.eval_mode == "unified"
    assert (out_dir / "manifest.json").is_file()

    # Check unified evaluate & summarize path offline.
    task = _task()
    ev = Evidence(task)
    ev.add(task.prior_work[0])
    trial.runner = lambda *_a, **_kw: LLMRunner(ModelEndpoint(), dry_run=True)
    versions = {
        "B": {"draft": {"title": "Weak", "minimal_test": "Run once.", "decisive_metric": "Acc"}, "retrieved": []},
        "C": {"draft": {"title": "Bounded [Repaired]", "minimal_test": "Held-out split with matched-budget ablation control.", "decisive_metric": "AURC"}, "retrieved": []},
        "D": {"draft": {"title": "Weak", "minimal_test": "Run once."}, "retrieved": []},
        "E": {"draft": {"title": "Weak", "minimal_test": "Run once."}, "retrieved": []},
    }
    res = trial.evaluate(out_dir, "u-test-1/direct", versions["B"]["draft"], ev.snapshot(), versions, task, [], ("unified",))
    assert res["C_vs_B"]["unified"]["lit2test"]["status"] == "win"
    assert res["E_vs_D"]["unified"]["lit2test"]["status"] == "tie"

    report = {
        "cohort": "pilot",
        "eval_mode": "unified",
        "results": [{
            "task_id": "u-test-1",
            "host": "direct",
            "stage": "proposal",
            "development": False,
            "source_sha256": "abc",
            "versions": {a: {"status": "changed", "sha256": a, "edit_volume": {"tokens_deleted": 0, "tokens_inserted": 5}, "queries": []} for a in "BCDE"},
            "unified_comparisons": {k: v["unified"] for k, v in res.items()},
        }],
        "missing": [],
    }
    summarize(out_dir, report)
    assert (out_dir / "pilot_summary.md").is_file()
    assert "Lit2Test" in (out_dir / "pilot_summary.md").read_text(encoding="utf-8")


def test_extract_json_dict_handles_trailing_extra_braces():
    from edara_mvp_eval.llm_client import extract_json_dict
    parsed = extract_json_dict('{\n  "overall_winner": "B",\n  "score": 3\n}\n\n}')
    assert parsed == {"overall_winner": "B", "score": 3}


def test_batched_aib_fallback_and_concurrent_claimnov():
    from concurrent.futures import ThreadPoolExecutor
    from edara_mvp_eval.claimnov_adapter import run_claimnov_audit
    from edara_mvp_eval.schemas import NormalizedProposal
    from edara_mvp_eval.unified_eval import eval_alphaideabench_draft

    # Verify concurrent ClaimNov runs succeed without shared lock contention or cross-thread bleed.
    base_runner = LLMRunner(ModelEndpoint(), dry_run=True)
    prop = NormalizedProposal(
        task_id="conc",
        title="Adaptive Eviction",
        literature_gap="gap",
        hypothesis="h",
        method="m",
        minimal_test="test",
        decisive_metric="AURC",
        supporting_result="AURC improves",
        falsifying_result="AURC drops",
    )
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda _i: run_claimnov_audit(prop, _task().prior_work, base_runner), range(4)))
    assert all(r.claims_count > 0 for r in results)

    # Verify partial/failed batch falls back to per-dimension judge cleanly.
    calls = []

    class PartialBatchRunner(LLMRunner):
        def chat_json(self, system, user, *, tag="generic", **kwargs):
            calls.append(tag)
            if tag == "aib_batch_judge":
                return {"dimensions": {}}  # empty triggers per-dim fallback
            return super().chat_json(system, user, tag=tag, **kwargs)

    res = eval_alphaideabench_draft(
        {"title": "T [Repaired]", "hypothesis": "H"},
        _task(),
        _packet(),
        PartialBatchRunner(ModelEndpoint(), dry_run=True),
        dimensions=("context_grounding", "testability"),
        batched=True,
    )
    assert calls[0] == "aib_batch_judge"
    assert calls.count("aib_dimension_judge") == 2
    assert res["overall"] == 4.0
