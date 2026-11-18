"""Offline verification tests for the 15-task MVP evaluation harness & middleware."""
from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from edara_mvp_eval.claimnov_adapter import analyze_compositional_remainder, run_claimnov_audit
from edara_mvp_eval.cli import app
from edara_mvp_eval.epistemic_audit import compute_integrity_gate, run_full_audit
from edara_mvp_eval.literature_scout import extract_relevant_excerpt, is_forbidden_or_post_cutoff
from edara_mvp_eval.llm_client import LLMRunner, ModelEndpoint
from edara_mvp_eval.middleware import (
    TrustworthyNoveltyMiddleware,
    append_spec_clarifications,
    apply_bounded_patches,
    compute_retention_metrics,
)
from edara_mvp_eval.pairwise import compare_pair_bidirectional
from edara_mvp_eval.schemas import (
    ClaimNovAuditResult,
    GapAuditResult,
    GapPremiseAuditItem,
    NormalizedProposal,
    PriorFactAudit,
    PriorWorkItem,
    SpecAuditResult,
    SpecDefectItem,
    TriStateAuditResult,
    UnknownDependencyClaim,
)
from edara_mvp_eval.spec_locator import locate_spec_defects
from edara_mvp_eval.tasks_loader import (
    DEFAULT_PILOT3_IDS,
    build_mvp15_tasks,
    build_pilot3_tasks,
    export_pilot3_external_inputs,
    normalize_title_hash,
)


def test_mvp15_task_suite_and_pdf_extraction(tmp_path: Path) -> None:
    cache = tmp_path / "tasks_mvp15.json"
    tasks = build_mvp15_tasks(cache_path=cache)
    assert len(tasks) == 15
    tracks = [t.source_track for t in tasks]
    assert tracks.count("aiib") == 5
    assert tracks.count("lit2test") == 5
    assert tracks.count("alphaideabench") == 5

    aiib_tasks = [t for t in tasks if t.source_track == "aiib"]
    assert all(len(t.prior_work) == 5 for t in aiib_tasks)
    assert all(any(len(p.fulltext_snippet) > 500 for p in t.prior_work) for t in aiib_tasks)
    assert all(len(t.forbidden_title_hashes) > 0 for t in aiib_tasks)


def test_leak_boundary_blocks_post_cutoff_and_forbidden_hash() -> None:
    forbidden = {normalize_title_hash("Secret Target Paper")}
    assert is_forbidden_or_post_cutoff(
        title="Secret   Target: Paper!",
        pub_date="2023-01-01",
        cutoff_date="2024-05-07",
        forbidden_hashes=forbidden,
    )
    # Also enforce E-DARA boundary.py prefixed hashes ("title:", "arxiv:", etc.)
    import hashlib

    edara_style_forbidden = {
        hashlib.sha256(b"title:secret target paper").hexdigest(),
        hashlib.sha256(b"arxiv:2405.05259").hexdigest(),
    }
    assert is_forbidden_or_post_cutoff(
        title="Secret: Target Paper",
        pub_date="2023-01-01",
        cutoff_date="2024-05-07",
        forbidden_hashes=edara_style_forbidden,
    )
    assert is_forbidden_or_post_cutoff(
        title="Any Unrelated Title",
        pub_date="2023-01-01",
        cutoff_date="2024-05-07",
        forbidden_hashes=edara_style_forbidden,
        identifier="https://arxiv.org/abs/2405.05259v1",
    )
    assert is_forbidden_or_post_cutoff(
        title="Post Cutoff Paper",
        pub_date="2024-05-08",
        cutoff_date="2024-05-07",
        forbidden_hashes=forbidden,
    )
    assert not is_forbidden_or_post_cutoff(
        title="Valid Pre Cutoff Paper",
        pub_date="2024-05-07",
        cutoff_date="2024-05-07",
        forbidden_hashes=forbidden,
    )


def test_integrity_gate_zeros_novelty_on_fabrication_or_gap_falsification() -> None:
    proposal = NormalizedProposal(
        task_id="t1",
        title="Test",
        literature_gap="No prior work exists.",
        hypothesis="H",
        method="M",
        minimal_test="T",
        decisive_metric="AUC",
        supporting_result="AUC up",
        falsifying_result="AUC flat",
    )
    tri_clean = TriStateAuditResult(
        prior_fact_audits=[
            PriorFactAudit(claim="x", citation_status="verified_local", smoothing_score=1, epistemic_overclaim=False)
        ],
        untested_hypothesis_count=2,
        honest_unknown_count=1,
    )
    gap_bad = GapAuditResult(
        items=[GapPremiseAuditItem(premise="No prior work exists.", status="falsified")],
        survival_rate=0.0,
        any_falsified=True,
    )
    cn_ok = ClaimNovAuditResult(novelty_score=0.9, veto=False)
    gate = compute_integrity_gate(proposal, tri_clean, gap_bad, cn_ok, {"budget_feasible": True, "budget_feasibility": 4.0, "novelty_score": 4.5})
    assert not gate.gate_pass
    assert "gap_premise_falsified_pre_cutoff" in gate.hard_fail_reasons
    assert gate.novelty_score_raw > 0.5
    assert gate.trusted_novelty == 0.0


def test_claimnov_adapter_bm25_and_middleware_repair_end_to_end(tmp_path: Path) -> None:
    tasks = build_mvp15_tasks(cache_path=tmp_path / "tasks.json")
    task = tasks[0]
    runner = LLMRunner(ModelEndpoint(), dry_run=True)

    direct_raw = runner.chat_json("sys", "user", tag="generate_direct")
    mw = TrustworthyNoveltyMiddleware(live_search=False, enable_embeddings=False)
    repaired, before, after, _ = mw.apply(task, direct_raw, repair_runner=runner, audit_runner=runner)

    assert not before.gate.gate_pass
    assert before.claimnov.retriever_mode == "bm25_fallback"
    assert after.gate.gate_pass
    assert after.gate.trusted_novelty > before.gate.trusted_novelty
    assert repaired["title"].endswith("[Repaired]")

    # Re-applying middleware to an already clean proposal should be a zero-cost no-op.
    _, _, after2, overhead2 = mw.apply(task, repaired, repair_runner=runner, audit_runner=runner)
    assert after2.gate.gate_pass
    assert overhead2 == 0.0

    # Pairwise bidirectional judge should consistently prefer repaired over uncalibrated direct draft.
    pw = compare_pair_bidirectional(task, before.model_copy(update={"arm": "direct"}), after, runner)
    assert not pw.order_sensitive
    assert pw.folded_winner == "Y"


def test_pairwise_order_flip_folds_to_tie() -> None:
    class FlippingRunner(LLMRunner):
        def chat_json(self, system: str, user: str, *, temperature: float = 0.0, max_tokens: int = 1800, tag: str = "generic"):
            # Always picks Answer A regardless of order -> flips when swapped!
            return {"winner": "A", "confidence": "high", "main_reason": "position bias test"}

    tasks = build_mvp15_tasks()
    task = tasks[0]
    runner = LLMRunner(ModelEndpoint(), dry_run=True)
    b1 = run_full_audit(task, "arm1", runner.chat_json("s", "u", tag="generate_direct"), runner)
    b2 = run_full_audit(task, "arm2", "[repaired] clean draft", runner)
    pw = compare_pair_bidirectional(task, b1, b2, FlippingRunner(ModelEndpoint(), dry_run=True))
    assert pw.order_sensitive is True
    assert pw.folded_winner == "tie"


def test_pilot3_selection_export_and_parallel_cli_with_imported_arm_mw(tmp_path: Path) -> None:
    import json

    out_root = tmp_path / "pilot_out"
    runner = CliRunner()
    res_prep = runner.invoke(app, ["prepare-tasks", "--out-root", str(out_root)])
    assert res_prep.exit_code == 0, res_prep.output

    pilot3_cache = out_root / "tasks_pilot3.json"
    pilot3 = build_pilot3_tasks(cache_path=pilot3_cache)
    assert tuple(t.task_id for t in pilot3) == DEFAULT_PILOT3_IDS

    inputs_dir = export_pilot3_external_inputs(pilot3, out_root / "pilot3_inputs")
    for tid in DEFAULT_PILOT3_IDS:
        assert (inputs_dir / "ai_scientist_topics" / f"{tid}.md").is_file()
        assert (inputs_dir / "ai_scientist_topics" / f"{tid}.boundary.json").is_file()
        assert (inputs_dir / "edara_cases" / f"{tid}.json").is_file()
        assert (inputs_dir / "edara_cases" / tid / "literature-boundary.json").is_file()
    assert (inputs_dir / "edara_cases" / "edara_live_config.json").is_file()

    # Verify generation-only Direct command writes 3 proposals without requiring judge credentials.
    direct_out = tmp_path / "direct_only.jsonl"
    res_gen = runner.invoke(
        app,
        [
            "generate-direct",
            "--tasks-cache",
            str(pilot3_cache),
            "--out",
            str(direct_out),
            "--workers",
            "3",
            "--dry-run",
        ],
    )
    assert res_gen.exit_code == 0, res_gen.output
    gen_lines = [json.loads(line) for line in direct_out.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert [r["task_id"] for r in gen_lines] == list(DEFAULT_PILOT3_IDS)

    # Build a mock external arm JSONL covering the 3 pilot topics.
    ext_jsonl = tmp_path / "external_arm.jsonl"
    rows = [
        json.dumps(
            {
                "task_id": tid,
                "title": f"External Draft for {tid}",
                "literature_gap": "No prior work exists anywhere.",
                "hypothesis": "External hypothesis",
                "method": "External method",
                "minimal_test": "Run baseline ablation on dev split",
                "decisive_metric": "AUROC",
                "supporting_result": "AUROC >= 0.80",
                "falsifying_result": "AUROC < 0.72",
            }
        )
        for tid in DEFAULT_PILOT3_IDS
    ]
    ext_jsonl.write_text("\n".join(rows) + "\n", encoding="utf-8")

    eval_out = tmp_path / "eval_run"
    res_run = runner.invoke(
        app,
        [
            "run",
            "--tasks-cache",
            str(pilot3_cache),
            "--out",
            str(eval_out),
            "--workers",
            "3",
            "--import-arm",
            f"ai-scientist-v2={ext_jsonl}",
            "--mw-on-imported",
            "--dry-run",
        ],
    )
    assert res_run.exit_code == 0, res_run.output
    results = json.loads((eval_out / "results.json").read_text(encoding="utf-8"))
    assert results["n_tasks"] == 3
    arm_names = {a["arm"] for a in results["arm_summaries"]}
    assert arm_names == {"direct", "direct_mw", "ai-scientist-v2", "ai-scientist-v2_mw"}
    # C(4, 2) = 6 pairs per task * 3 tasks = 18 pairwise comparisons
    assert len(results["pairwise"]) == 18
    assert (eval_out / "summary.md").is_file()


def test_extract_relevant_excerpt_surfaces_deep_fulltext_when_limitation_present() -> None:
    filler = "Background sentence about generic benchmarks. " * 25  # > 1000 chars
    deep_fact = "LLM-SRBench reports peak equation discovery accuracy of 31% under out-of-distribution shifts."
    item = PriorWorkItem(
        id="doc1",
        title="Agentic Benchmark Survey",
        limitation="Limited to symbolic tasks.",
        fulltext_snippet=filler + deep_fact + " " + filler,
        source="topic_doc",
    )
    excerpt = extract_relevant_excerpt("LLM-SRBench peak 31% accuracy", item, max_chars=900)
    assert "Limited to symbolic tasks." in excerpt
    assert "31%" in excerpt and "LLM-SRBench" in excerpt


def test_spec_defect_locator_and_compositional_remainder() -> None:
    tasks = build_mvp15_tasks()
    task = tasks[0]
    runner = LLMRunner(ModelEndpoint(), dry_run=True)

    vague = NormalizedProposal(
        task_id=task.task_id,
        title="Vague Idea",
        literature_gap="Existing work fails.",
        hypothesis="Adding a gate helps.",
        method="Use a scalar cutoff.",
        minimal_test="Run accuracy check.",
        decisive_metric="",
        supporting_result="Better.",
        falsifying_result="",
        related_work="Builds on standard encoders.",
    )
    spec = locate_spec_defects(vague, task, runner)
    assert not spec.codification_ready
    assert spec.blocking_count >= 2
    assert spec.completeness_score < 0.70

    cn = ClaimNovAuditResult(
        novelty_score=0.72,
        veto=False,
        claims=[
            {
                "claim": "Uses encoder from Paper Alpha.",
                "dimension": "methodology",
                "max_entailment_level": 4,
                "top_matches": [{"paper_id": "pA", "paper_title": "Paper Alpha: Contrastive Encoders"}],
            },
            {
                "claim": "Couples cross-regime uncertainty probe.",
                "dimension": "methodology",
                "max_entailment_level": 2,
                "top_matches": [{"paper_id": "pB", "paper_title": "Paper Beta: Regime Probes"}],
            },
        ],
    )
    comp = analyze_compositional_remainder(vague, cn)
    assert comp.is_compositional_novelty is True
    assert "Paper Alpha: Contrastive Encoders" in comp.attribution_requirements
    assert "Couples cross-regime uncertainty probe." in comp.compositional_remainder


def test_bounded_patches_and_retention_guard_reject_destructive_or_ambiguous_edits() -> None:
    orig = NormalizedProposal(
        task_id="t1",
        title="Original Title",
        literature_gap="No prior work tests X.",
        hypothesis="Regime stratification reduces error.",
        method="Combine calibration intervals with an explicit regime-shift diagnostic probe.",
        minimal_test="Evaluate on one split.",
        decisive_metric="AURC",
        supporting_result="AURC drops",
        falsifying_result="AURC flat",
        unknown_dependencies=[
            UnknownDependencyClaim(
                dependency="Held-out shift variance",
                why_unresolved="Needs pilot run",
                resolution_path="Measure on dev slice",
            )
        ],
    )
    payload = {
        "patches": [
            {
                "target_field": "literature_gap",
                "exact_source_span": "No prior work tests X.",
                "replacement_text": "Cross-regime drift in X remains under-tested.",
                "edit_type": "gap_refinement",
            },
            {
                "target_field": "method",
                "exact_source_span": "Combine calibration intervals with an explicit regime-shift diagnostic probe.",
                "replacement_text": "Use a totally unrelated baseline algorithm instead.",
                "edit_type": "scope_narrowing",
            },
            {
                "target_field": "title",
                "exact_source_span": "Nonexistent span",
                "replacement_text": "Ignored",
                "edit_type": "scope_narrowing",
            },
        ],
        "append_unknown_contracts": [
            {
                "dependency": "Threshold stability across splits",
                "why_unresolved": "Empirical sensitivity unknown",
                "resolution_path": "Calibrate on 10% slice",
                "minimal_calibration_test": "Grid over 5 cutoffs",
                "fallback_branch": "Use abstention fallback",
            }
        ],
    }
    patched, n_app, n_rej, reverted = apply_bounded_patches(orig, payload)
    assert n_app == 1
    assert n_rej == 2
    assert reverted is True
    assert patched.method == orig.method
    assert patched.literature_gap == "Cross-regime drift in X remains under-tested."
    assert len(patched.unknown_dependencies) == 2
    ret = compute_retention_metrics(orig, patched, rounds_used=1, patches_applied=n_app, patches_rejected=n_rej, guard_reverted=reverted)
    assert ret.method_token_retention == 1.0
    assert ret.unknown_retention_ratio == 1.0


@pytest.mark.parametrize(("source", "span", "replacement", "expected"), [
    ("Before. The  reported\n effect was universal. After.", "The reported effect was universal.",
     "The reported effect held under the tested conditions.", "Before. The reported effect held under the tested conditions. After."),
    ("Before. The claimed result is universal. After.", "“The claimed result is universal!”",
     "The result applies to the tested split.", "Before. The result applies to the tested split. After."),
    ("Before. The claimed result is universal. After.", "The claimed result is universal!",
     "The result applies to the tested split", "Before. The result applies to the tested split. After."),
    ("Prior work reports acceleration. Prior  work reports acceleration.",
     "Prior work  reports acceleration.", "The effect is conditional.", None),
    ("The earlier baseline uses cost. The later baseline uses cost.",
     "baseline uses cost!", "baseline held cost fixed", None),
    ("The threshold is 0.55.", "The threshold is 0.5!", "The threshold is calibrated.", None),
])
def test_span_fallback_preserves_exact_boundaries_and_rejects_ambiguity(source, span, replacement, expected) -> None:
    proposal = NormalizedProposal(
        task_id="t", title="Title", literature_gap="", hypothesis="Hypothesis", method="Method",
        minimal_test="Test", decisive_metric="Metric", supporting_result="Supports",
        falsifying_result="Falsifies", related_work=source,
    )
    amended, applied, rejected, _ = apply_bounded_patches(proposal, {"patches": [{
        "target_field": "related_work", "exact_source_span": span, "replacement_text": replacement,
        "edit_type": "scope_narrowing",
    }]})
    assert amended.related_work == (expected if expected is not None else source)
    assert (applied, rejected) == ((1, 0) if expected is not None else (0, 1))


def test_spec_clarifications_keep_four_span_slots_and_existing_fields() -> None:
    proposal = NormalizedProposal(
        task_id="t", title="Title", literature_gap="Gap", hypothesis="Hypothesis",
        method="Use fixed calibration with a baseline comparator.", minimal_test="Test.",
        decisive_metric="Metric", supporting_result="Supports", falsifying_result="Falsifies",
        limitations="Existing limit.", resource_plan={"gpu_hours": 4},
    )
    patches = [{
        "target_field": field, "exact_source_span": getattr(proposal, field), "replacement_text": f"New {field}",
        "edit_type": "scope_narrowing",
    } for field in ("title", "literature_gap", "related_work", "supporting_result", "falsifying_result")]
    patches[2].update(exact_source_span="", edit_type="attribution_credit")
    fields = ("minimal_test", "resource_plan", "method", "decisive_metric")
    defects = [
        SpecDefectItem(category="baseline_alignment", target_field=field, missing_detail="Resolve this defect.")
        for field in fields
    ]
    payload = {
        "patches": patches,
        "spec_clarifications": [
            {"defect_id": 1, "target_field": "minimal_test", "append_text": "Use held-out examples."},
            {"defect_id": 2, "target_field": "resource_plan", "append_text": "Run matched workloads within the cap."},
            {"defect_id": 3, "target_field": "method", "append_text": "Fit the threshold on the calibration split."},
            {"defect_id": 4, "target_field": "title", "append_text": "Invalid target."},
        ],
    }
    updated, applied, rejected, _ = apply_bounded_patches(proposal, payload)
    assert (applied, rejected) == (4, 0)
    assert updated.falsifying_result == "Falsifies"
    assert append_spec_clarifications(updated, payload, defects) == (3, 1)
    assert updated.minimal_test == "Test.\nUse held-out examples."
    assert updated.method.startswith(proposal.method + "\n")
    assert updated.resource_plan == {
        "gpu_hours": 4, "implementation_clarifications": "Run matched workloads within the cap.",
    }
    assert proposal.resource_plan == {"gpu_hours": 4}
    assert compute_retention_metrics(
        proposal, updated, rounds_used=1, patches_applied=4, patches_rejected=0, guard_reverted=False,
    ).method_token_retention == 1.0


def test_spec_only_repair_is_audited_and_retained(monkeypatch, tmp_path: Path) -> None:
    task = build_mvp15_tasks(cache_path=tmp_path / "tasks.json")[0]
    runner = LLMRunner(ModelEndpoint(), dry_run=True)
    baseline = run_full_audit(task, "direct", "[repaired] clean draft", runner)
    defect = SpecDefectItem(
        category="baseline_alignment", target_field="minimal_test",
        missing_detail="Name the matched control", severity="blocking",
    )
    before = baseline.model_copy(update={
        "spec_audit": SpecAuditResult(defects=[defect], blocking_count=1),
        "gate": baseline.gate.model_copy(update={"strict_gate_pass": False}),
    })
    seen = []

    def audit(task, arm, raw, runner, **kwargs):
        seen.append(raw)
        if len(seen) == 1:
            return before
        assert isinstance(raw, NormalizedProposal)
        assert raw.method == before.proposal.method
        assert raw.minimal_test.endswith("Use the same compute cap for the static baseline.")
        return baseline.model_copy(update={"proposal": raw})

    class Clarifier(LLMRunner):
        def chat_json(self, system, user, **kwargs):
            assert '"defect_id": 1' in user
            return {"spec_clarifications": [{
                "defect_id": 1, "target_field": "minimal_test",
                "append_text": "Use the same compute cap for the static baseline.",
            }]}

    monkeypatch.setattr("edara_mvp_eval.middleware.run_full_audit", audit)
    repaired, _, after, _ = TrustworthyNoveltyMiddleware(max_rounds=1).apply(
        task, "[repaired] clean draft", repair_runner=Clarifier(ModelEndpoint(), dry_run=True), audit_runner=runner,
    )
    assert len(seen) == 2
    assert after.gate.strict_gate_pass
    assert after.proposal.minimal_test == repaired["minimal_test"]
    assert after.retention.patches_applied == 0
    assert after.retention.spec_clarifications_applied == 1


def test_ablation_baselines_and_direct_inputs_reuse_dry_run(tmp_path: Path) -> None:
    import json

    out_root = tmp_path / "ablation_prep"
    runner = CliRunner()
    assert runner.invoke(app, ["prepare-tasks", "--out-root", str(out_root)]).exit_code == 0
    pilot3_cache = out_root / "tasks_pilot3.json"
    direct_out = tmp_path / "direct_inputs.jsonl"
    assert runner.invoke(
        app,
        ["generate-direct", "--tasks-cache", str(pilot3_cache), "--out", str(direct_out), "--workers", "3", "--dry-run"],
    ).exit_code == 0

    eval_out = tmp_path / "ablation_out"
    res = runner.invoke(
        app,
        [
            "run",
            "--tasks-cache",
            str(pilot3_cache),
            "--direct-inputs",
            str(direct_out),
            "--with-self-refine",
            "--with-retrieve-revise",
            "--with-pre-brief",
            "--workers",
            "3",
            "--out",
            str(eval_out),
            "--dry-run",
        ],
    )
    assert res.exit_code == 0, res.output
    results = json.loads((eval_out / "results.json").read_text(encoding="utf-8"))
    arm_names = {a["arm"] for a in results["arm_summaries"]}
    assert arm_names == {"direct", "direct_mw", "direct_self_refine", "direct_retrieve_revise", "direct_pre_mw"}
    summary = (eval_out / "summary.md").read_text(encoding="utf-8")
    assert "Spec Comp (↑)" in summary and "Method Ret (↑)" in summary
    assert results["bundles"]["direct_mw"][0]["retention"]["patches_applied"] >= 1


def test_json_recovery_baseline_fallback_and_checkpoint_resume(tmp_path: Path) -> None:
    import json
    from edara_mvp_eval.baselines import run_self_refine
    from edara_mvp_eval.epistemic_audit import run_full_audit
    from edara_mvp_eval.llm_client import LLMRunner, ModelEndpoint, extract_json_dict
    from edara_mvp_eval.schemas import TaskSpec

    parsed = extract_json_dict('```json\n{"a": 1, "b": [2, 3,],}\n```')
    assert parsed == {"a": 1, "b": [2, 3]}

    task = TaskSpec(
        task_id="ckpt-test-1",
        source_track="lit2test",
        field="cs",
        title="Test Task",
        cutoff_date="2025-12-31",
        prompt_context="Evaluate calibration.",
    )
    audit_runner = LLMRunner(ModelEndpoint(), dry_run=True)
    draft = {"title": "Baseline", "hypothesis": "H1", "method": "M1", "minimal_test": "T1"}
    before = run_full_audit(task, "direct", draft, audit_runner)

    class FailingJSONRunner(LLMRunner):
        def chat_json(self, system: str, user: str, **kwargs):  # type: ignore[override]
            raise json.JSONDecodeError("bad json", "{", 1)

    failing_runner = FailingJSONRunner(ModelEndpoint(), dry_run=True)
    revised, sr_bundle, _ = run_self_refine(
        task, draft, before_bundle=before, repair_runner=failing_runner, audit_runner=audit_runner
    )
    assert revised["title"] == "Baseline"
    assert sr_bundle.arm == "direct_self_refine"

    out_root = tmp_path / "ckpt_prep"
    runner = CliRunner()
    assert runner.invoke(app, ["prepare-tasks", "--out-root", str(out_root)]).exit_code == 0
    pilot3_cache = out_root / "tasks_pilot3.json"
    eval_out = tmp_path / "ckpt_eval"
    cmd = [
        "run",
        "--tasks-cache",
        str(pilot3_cache),
        "--limit",
        "2",
        "--skip-pairwise",
        "--workers",
        "2",
        "--out",
        str(eval_out),
        "--dry-run",
    ]
    r1 = runner.invoke(app, cmd)
    assert r1.exit_code == 0, r1.output
    ckpts = list((eval_out / "checkpoints").glob("*.json"))
    assert len(ckpts) == 2

    r2 = runner.invoke(app, cmd)
    assert r2.exit_code == 0, r2.output
    assert "loaded from checkpoint" in r2.output
