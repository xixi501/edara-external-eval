"""Unit tests for independent factual audit, multi-host adaptation, and routing/preservation guards."""
from __future__ import annotations

import json
from types import SimpleNamespace

from edara_mvp_eval.epistemic_audit import run_full_audit
from edara_mvp_eval.factual_audit import audit_independent_claims, clean_replacement_text
from edara_mvp_eval.feedback import FeedbackMiddleware, PolicyState, eligible_actions
from edara_mvp_eval.host_adapter import (
    confound_addressed,
    design_field,
    extract_design_elements,
    preserves_design_elements,
)
from edara_mvp_eval.llm_client import LLMRunner, ModelEndpoint
from edara_mvp_eval.middleware import TrustworthyNoveltyMiddleware
from edara_mvp_eval.paired_inputs import Evidence
from edara_mvp_eval.paired_trial import run_arm
from edara_mvp_eval.schemas import (
    ClaimNovAuditResult,
    GapAuditResult,
    IdeaGovernanceResult,
    PriorWorkItem,
    SpecAuditResult,
    TaskSpec,
    TriStateAuditResult,
)


def _task() -> TaskSpec:
    return TaskSpec(
        task_id="t-plugin",
        source_track="lit2test",
        field="ml",
        title="KV Cache Compression",
        prompt_context="Evaluate dynamic eviction under fixed token budget.",
        cutoff_date="2024-06-01",
        strict_evidence=True,
        budget={"gpu": "1x A100", "api_usd": 20},
        prior_work=[
            PriorWorkItem(
                id="P1",
                title="DynamicKV",
                abstract="DynamicKV operates at a 1.7% cache budget and evaluates on LLM-SRBench.",
                year="2023",
                publication_date="2023-10-01",
                date_source="test",
                version="v1",
                source="lit2test_paper",
            )
        ],
    )


def test_multihost_design_field_and_confound_addressed_on_edara_flattened_drafts():
    edara_proposal = {
        "claims[1].evidence_ids[1]": "S1",
        "title": "Idea",
        "method[1]": "Gate retrieval by entropy.",
        "validation[1]": "Compare against CoT.",
        "validation[2]": "Run an always-retrieve arm under matched context budget.",
    }
    assert design_field(edara_proposal) == "validation[2]"
    assert confound_addressed(edara_proposal, ["always-retrieve arm with matched context budget"])
    assert not confound_addressed({"claims[1].text": "always-retrieve arm"}, ["always-retrieve arm"])

    edara_plan = {
        "alternative_explanations[1]": "Sampling temperature differs.",
        "conditions[1]": "Base model baseline",
        "conditions[2]": "Trajectory diversity with temperature-matched control",
        "analysis.metrics[1]": "AURC at 80% coverage",
        "decision_rules[1]": "Falsify if AURC gain < 2%",
    }
    assert design_field(edara_plan) == "conditions[2]"
    assert confound_addressed(edara_plan, ["temperature-matched control"])
    elements = extract_design_elements(edara_plan)
    assert elements["control"] and elements["metric"] and elements["decision"]


def test_independent_factual_audit_catches_numeric_coverage_and_unattested_process_without_judge_ledger():
    draft = {
        "related_work": "DynamicKV uses a 5% cache budget, and no supplied work covers LLM-SRBench.",
        "minimal_test": "We use a pre-registered, 300-case evaluation split with matched baselines.",
    }

    class ScriptedRunner:
        def chat_json(self, system, user, tag="", **kwargs):
            if tag == "independent_claim_extract":
                return {
                    "claims": [
                        {
                            "field": "related_work",
                            "quote": "DynamicKV uses a 5% cache budget",
                            "claim_kind": "numeric_or_benchmark",
                        },
                        {
                            "field": "related_work",
                            "quote": "no supplied work covers LLM-SRBench",
                            "claim_kind": "coverage_gap",
                        },
                    ]
                }
            if tag == "independent_claim_audit":
                return {
                    "findings": [
                        {
                            "id": "c1",
                            "label": "defect",
                            "defect_type": "numeric_mismatch",
                            "evidence": [{"paper_id": "P1", "quote": "DynamicKV operates at a 1.7% cache budget"}],
                            "basis": "",
                            "reason": "Paper states 1.7%, not 5%.",
                        },
                        {
                            "id": "c2",
                            "label": "defect",
                            "defect_type": "coverage_contradiction",
                            "evidence": [{"paper_id": "P1", "quote": "evaluates on LLM-SRBench"}],
                            "basis": "",
                            "reason": "Supplied paper P1 explicitly covers LLM-SRBench.",
                        },
                    ]
                }
            return {}

    findings = audit_independent_claims(draft, _task(), ScriptedRunner())
    types = {f.defect_type for f in findings if f.label == "defect"}
    assert {"numeric_mismatch", "coverage_contradiction", "unattested_process"} <= types
    assert clean_replacement_text("Run a pre-registered, 300-case split.") == "Run a 300-case split."


def test_factual_audit_downgrades_unanchored_defects_to_unknown():
    draft = {"hypothesis": "Our method improves recall."}

    class BadEvidenceRunner:
        def chat_json(self, system, user, tag="", **kwargs):
            if tag == "independent_claim_extract":
                return {"claims": [{"field": "hypothesis", "quote": draft["hypothesis"], "claim_kind": "prior_fact"}]}
            return {
                "findings": [{
                    "id": "c1",
                    "label": "defect",
                    "defect_type": "attribution",
                    "evidence": [{"paper_id": "P1", "quote": "fabricated quote not in paper"}],
                    "basis": "nonexistent draft sentence",
                    "reason": "unanchored critique",
                }]
            }

    findings = audit_independent_claims(draft, _task(), BadEvidenceRunner())
    assert len(findings) == 1
    assert findings[0].label == "unknown"


def test_build_diag_places_blocking_confounds_then_factual_defects_from_independent_audit():
    task = _task()
    runner = LLMRunner(ModelEndpoint(), dry_run=True)
    raw = runner.chat_json("s", "u", tag="generate_direct")
    bundle = run_full_audit(task, "C", raw, runner)
    assert any(f.label == "defect" for f in bundle.factual_findings)
    diag = TrustworthyNoveltyMiddleware()._build_diag(bundle)
    keys = list(diag.keys())
    assert "factual_findings" in keys
    if bundle.confound_audit.priority == "blocking":
        assert keys.index("design_confounds") < keys.index("factual_findings")


def test_preserves_design_elements_and_routing_unlocks_regenerate_while_blocking_regressions():
    before = {
        "hypothesis": "Gating improves selective risk.",
        "minimal_test": "Compare against matched baseline and ablation control on held-out split.",
        "decisive_metric": "AURC at 80% coverage",
        "falsifying_result": "AURC fails to improve by >= 3%",
    }
    regressed = {
        "hypothesis": "Gating improves selective risk.",
        "minimal_test": "Run the proposed gate on held-out examples.",
        "decisive_metric": "",
        "falsifying_result": "None specified.",
    }
    ok, regressions = preserves_design_elements(before, regressed)
    assert not ok
    assert "lost_control" in regressions and "lost_metric" in regressions

    # Routing triggers local_patch on factual_findings defect and regenerate on ClaimNov veto.
    bundle = SimpleNamespace(
        governance=IdeaGovernanceResult(action="seek_evidence", claims=[]),
        tri_state=TriStateAuditResult(),
        gap_audit=GapAuditResult(any_falsified=False),
        spec_audit=SpecAuditResult(blocking_count=0),
        claimnov=ClaimNovAuditResult(veto=True),
        factual_findings=[SimpleNamespace(label="defect")],
        proposal=None,
        arm="E",
    )
    actions = [a for a, _ in eligible_actions(bundle, PolicyState())]
    assert actions[0] == "local_patch"
    assert "regenerate" in actions

    # FeedbackMiddleware blocks a regenerate candidate that drops control/decision rules.
    regressed_non_empty = {
        "hypothesis": "Gating improves selective risk.",
        "method": "New gating mechanism.",
        "minimal_test": "Evaluate the proposed gate on held-out cases.",
        "decisive_metric": "Latency in ms",
        "falsifying_result": "None specified.",
    }
    full_bundle = run_full_audit(_task(), "E", before, LLMRunner(ModelEndpoint(), dry_run=True))
    full_bundle = full_bundle.model_copy(
        update={
            "governance": IdeaGovernanceResult(action="regenerate", claims=[]),
            "claimnov": ClaimNovAuditResult(veto=True),
            "spec_audit": SpecAuditResult(blocking_count=0),
        }
    )
    audit_runner = LLMRunner(ModelEndpoint(), dry_run=True)
    current, _, history, _, _ = FeedbackMiddleware().apply(
        _task(),
        before,
        full_bundle,
        audit_runner=audit_runner,
        repair_runner=audit_runner,
        generator_callback=lambda _pkt: regressed_non_empty,
        search=lambda _q: [],
        max_rounds=1,
    )
    assert current == before
    assert history[0]["status"] == "design_regression_blocked"


def test_run_arm_e_blocks_full_rewrite_design_regression(tmp_path):
    draft = {
        "hypothesis": "Gating improves selective risk.",
        "method": "Gate on entropy.",
        "minimal_test": "Compare against matched baseline and ablation control on held-out split.",
        "decisive_metric": "AURC at 80% coverage",
        "supporting_result": "AURC drops >= 5%.",
        "falsifying_result": "AURC drops < 2%.",
    }
    task = _task()
    ev = Evidence(task)
    ev.add(task.prior_work[0])

    class FakeAuditor(LLMRunner):
        def __init__(self):
            super().__init__(ModelEndpoint(), dry_run=True)
            self.index = 0
            self.reused = 0

        def chat_json(self, system, user, tag="", **kwargs):
            if tag == "normalize":
                return draft | {"prior_facts": [], "prospective_hypotheses": [], "unknown_dependencies": [], "gap_premises": []}
            if tag == "spec_locate":
                return {"defects": []}
            if tag == "confound_audit":
                return {"confounds": [], "decisive_missing": "", "priority": "none"}
            if tag == "idea_governance":
                return {
                    "claims": [{
                        "role": "core",
                        "status": "covered",
                        "proposal_span": "Gate on entropy.",
                        "evidence": [{"paper_id": ev.packet()[0]["paper_id"], "quote": "DynamicKV operates at a 1.7% cache budget"}],
                        "rationale": "covered",
                        "already_credited": True,
                    }]
                }
            if tag == "feedback_candidate_check":
                return {"aligned": True, "preserves": True, "material_change": True}
            return super().chat_json(system, user, tag=tag, **kwargs)

    class RegressingEditor:
        def chat_json(self, system, user, tag="", **kwargs):
            return {
                "draft": {
                    "hypothesis": "New gating reduces cost.",
                    "method": "New mechanism.",
                    "minimal_test": "Evaluate on test cases.",
                    "decisive_metric": "",
                    "falsifying_result": "",
                }
            }

    out = run_arm("E", draft, task, ev.snapshot(), FakeAuditor(), RegressingEditor(), tmp_path)
    assert out["draft"] == draft
    assert out["history"][0]["action"] == "regenerate"
    assert "design_regression_blocked" in out["history"][0]


def test_host_adapter_filters_boilerplate_and_fills_multihost_cards():
    from edara_mvp_eval.host_adapter import filter_proposal_fields, to_alphaideabench_idea, to_lit2test_card

    edara_plan = {
        "purpose": "Test query-guided eviction under strict KV budgets. Prior methods fail on two-needle jumps.",
        "hypothesis": "Query probes preserve two-needle recall at 1.7% budget.",
        "protocol[1].purpose": "Compare query-probe allocation against DynamicKV baseline.",
        "protocol[1].system": "INTERNAL SYSTEM PROMPT NOISE DO NOT AUDIT",
        "protocol[1].user": "INTERNAL USER TEMPLATE NOISE",
        "conditions[1]": "DynamicKV control baseline",
        "analysis.metrics[1]": "Two-evidence exact match",
        "decision_rules[1]": "Support if gain >= 5 points; falsify if gain < 1 point.",
        "resource_requirements.compute[1].minimum.gpu_hours": "16",
        "resource_requirements.model_api[1].estimated_cost_microusd": "5000000",
    }
    filtered = filter_proposal_fields(edara_plan)
    assert "protocol[1].system" not in filtered and "protocol[1].user" not in filtered
    card = to_lit2test_card(edara_plan)
    assert card["title"] and card["method"] and card["decisive_metric"] and card["falsifying_result"]
    assert "SYSTEM PROMPT NOISE" not in json.dumps(card)
    aib_idea = to_alphaideabench_idea("t1", edara_plan)
    assert aib_idea["resource_plan"]["estimated_gpu_hours"] == 16.0
    assert aib_idea["resource_plan"]["estimated_api_usd"] == 5.0

    # Nested direct fields (method.*, resource_plan.*) and AI-Scientist validation metrics fill cards cleanly.
    nested_direct = {
        "title": "Nested Agent Study",
        "hypothesis": "Verification reduces hallucination.",
        "method.systems": "Implement verification gate.",
        "method.ablations": "Drop gate component.",
        "minimal_test": "Run held-out split.",
        "decisive_metric": "Hallucination rate",
        "resource_plan.gpu_hours_total": "24",
    }
    assert "verification gate" in to_lit2test_card(nested_direct)["method"].lower()
    assert to_alphaideabench_idea("t2", nested_direct)["resource_plan"]["estimated_gpu_hours"] == 24.0

    ai_sci = {
        "title": "EventCLIP-Seg",
        "hypothesis": "Patch alignment bridges events and CLIP; falsify if event-frame CLIP matches.",
        "approach": "Map voxel grids to CLIP space. However, closed-set event segmentation remains limited.",
        "validation[1]": "Use DSEC-Semantic. Metrics: mIoU on unseen classes and harmonic mean.",
    }
    sci_card = to_lit2test_card(ai_sci)
    assert sci_card["decisive_metric"] and sci_card["falsifying_result"] and sci_card["literature_gap"]

    # Strict confound_addressed rejects incidental generic word matches without distinctive arm terms.
    generic_draft = {"validation[1]": "Evaluate text control condition with baseline method prototype audit."}
    assert not confound_addressed(generic_draft, ["image-side text-masked shortcut control"])
    assert confound_addressed(
        {"validation[1]": "Add an image-side text-masked shortcut control condition."},
        ["image-side text-masked shortcut control"],
        baseline_draft=generic_draft,
    )


def test_factual_audit_prereg_scope_deep_numbers_and_consistency():
    from edara_mvp_eval.factual_audit import _evidence_packet_for_audit, audit_draft_consistency

    # Prospective E-DARA rules mentioning pre-registered should NOT falsely fire; completed splits MUST fire.
    prospective_findings = audit_independent_claims(
        {"stop_conditions[1]": "Stop at the pre-registered Stage 2 criterion if gain < 1 point."},
        _task(),
        LLMRunner(ModelEndpoint(), dry_run=True),
        claims=[],
    )
    assert not any(f.defect_type == "unattested_process" for f in prospective_findings)

    # Prioritized evidence keeps deep numbers beyond 2400 chars intact.
    long_filler = "Background filler methodology sentence without percentages. " * 60
    deep_task = _task().model_copy(
        update={
            "prior_work": [
                PriorWorkItem(
                    id="Pdeep",
                    title="Deep Numbers",
                    abstract=long_filler + " Omitting consistency loss causes a 5.56% drop in mIoU.",
                    year="2023",
                    publication_date="2023-05-01",
                    date_source="test",
                    version="v1",
                    source="lit2test_paper",
                )
            ]
        }
    )
    pkt = _evidence_packet_for_audit(
        deep_task,
        claims=[{"quote": "consistency loss leads to a 5.56% drop in mIoU"}],
    )
    assert "5.56%" in pkt[0]["passages"]

    # Internal consistency check catches hypothesis vs metric threshold contradiction and copied falsifier.
    inconsistent = audit_draft_consistency({
        "hypothesis": "DynamicKV will lose at least 10 absolute exact-match points on multi-evidence QA.",
        "decisive_metric": "Support requires G >= 5 absolute EM points.",
        "supporting_result": "G >= 5 points and query probe beats random probe.",
        "falsifying_result": "G >= 5 points and query probe beats random probe.",
    })
    assert sum(1 for f in inconsistent if f.defect_type == "internal_inconsistency") == 2


def test_run_arm_e_guided_revise_and_post_search_revision(tmp_path):
    draft = {
        "hypothesis": "Compression loses at least 10 EM points.",
        "method": "Probe KV cache layers.",
        "minimal_test": "Compare against baseline.",
        "decisive_metric": "Support if G >= 5 EM points.",
        "supporting_result": "G >= 5 EM points.",
        "falsifying_result": "G >= 5 EM points.",
    }
    task = _task()
    ev = Evidence(task)
    ev.add(task.prior_work[0])
    extra_paper = PriorWorkItem(
        id="P2",
        title="Cache Query Probes",
        abstract="Query magnitude correlates with attention mass on retrieval tasks.",
        year="2023",
        publication_date="2023-08-01",
        date_source="test",
        version="v1",
        source="lit2test_paper",
    )
    from edara_mvp_eval.stage_b_evidence import EvidenceStore
    EvidenceStore(tmp_path / "lookup", task).search = lambda _q: [extra_paper]  # ensure search no-op test uses actions

    actions = [
        a
        for a, _ in eligible_actions(
            SimpleNamespace(
                governance=IdeaGovernanceResult(action="seek_evidence", claims=[]),
                tri_state=TriStateAuditResult(),
                gap_audit=GapAuditResult(any_falsified=False),
                spec_audit=SpecAuditResult(blocking_count=1),
                confound_audit=SimpleNamespace(priority="blocking"),
                claimnov=ClaimNovAuditResult(veto=False),
                factual_findings=[],
                proposal=None,
            ),
            PolicyState(),
            allow_guided_revise=True,
        )
    ]
    assert actions[0] == "guided_revise"
    post_search = eligible_actions(
        SimpleNamespace(
            governance=IdeaGovernanceResult(action="seek_evidence", claims=[]),
            tri_state=TriStateAuditResult(),
            gap_audit=GapAuditResult(any_falsified=False),
            spec_audit=SpecAuditResult(blocking_count=0),
            confound_audit=SimpleNamespace(priority="none"),
            claimnov=ClaimNovAuditResult(veto=False),
            factual_findings=[],
            proposal=None,
        ),
        PolicyState(revised=True, searched=True, needs_post_search_revision=True),
        allow_guided_revise=True,
    )
    assert post_search[0] == ("guided_revise", "incorporate_retrieved_evidence")


def test_confound_addressed_matches_design_field_clarify_allows_edits_and_entity_sweep():
    from edara_mvp_eval.factual_audit import audit_unverified_entities
    from edara_mvp_eval.paired_trial import apply_edits, editor_call

    # 1. confound_addressed scans the same design_contract / approach fields that design_field() picks.
    before_edara = {"design_contract": "Run base evaluation."}
    after_edara = {
        "design_contract": "Experimental arm: a presentation-matched control in which gate-derived formatting and hedging are normalized.",
    }
    arm_edara = [
        "A presentation-matched control arm that normalizes gate-derived formatting and hedging so raters cannot distinguish."
    ]
    assert design_field(before_edara) == "design_contract"
    assert confound_addressed(after_edara, arm_edara, baseline_draft=before_edara)

    before_sci = {"approach": "Train adapter."}
    after_sci = {
        "approach": "Control arm: matched post-processing control evaluates event-frame CLIP with the same superpixel aggregation and calibration as the adapter.",
    }
    arm_sci = [
        "A matched post-processing control: event-frame CLIP evaluated with the same superpixel aggregation and calibration as the adapter."
    ]
    assert confound_addressed(after_sci, arm_sci, baseline_draft=before_sci)

    # 2. scope='clarify' allows surgical replacement of contradicting text alongside empty appends.
    draft = {"hypothesis": "Loss is >= 10 EM points.", "minimal_test": "Compare DynamicKV baseline."}
    updated, edits = apply_edits(
        draft,
        {
            "edits": [
                {"field": "hypothesis", "source": "Loss is >= 10 EM points.", "replacement": "Loss is >= 5 EM points."},
                {"field": "minimal_test", "source": "", "replacement": "Include query-probe control."},
            ]
        },
        scope="clarify",
    )
    assert updated["hypothesis"] == "Loss is >= 5 EM points."
    assert "Include query-probe control." in updated["minimal_test"]
    assert [e["status"] for e in edits] == ["applied", "applied"]

    # 3. Deterministic unverified external entity sweep flags external names not in supplied packet,
    # while exempting own method names, packet papers, and hedged mentions.
    draft_with_external = {
        "title": "OurMethod: Bounded Compression",
        "hypothesis": "We introduce OurMethod to reduce KV cache miss rate on SCBench and Event-Cityscapes.",
        "related_work": "Unlike SegCLIP and MADM, DynamicKV operates at a 1.7% cache budget; Baby-AIGS (unverified in supplied packet) was reported elsewhere.",
    }
    unverified = audit_unverified_entities(draft_with_external, _task())
    names = {u["entity"] for u in unverified}
    assert {"SCBench", "Event-Cityscapes", "SegCLIP", "MADM"} <= names
    assert "DynamicKV" not in names and "OurMethod" not in names and "Baby-AIGS" not in names

    # Check diag and editor_call surface unverified_external_entities cleanly.
    bundle = SimpleNamespace(
        gate=SimpleNamespace(hard_fail_reasons=[]),
        tri_state=TriStateAuditResult(),
        gap_audit=GapAuditResult(items=[]),
        spec_audit=SpecAuditResult(defects=[]),
        compositional=SimpleNamespace(model_dump=lambda: {}),
        governance=None,
        claimnov=ClaimNovAuditResult(),
        confound_audit=SimpleNamespace(priority="none", model_dump=lambda: {"priority": "none"}),
        factual_findings=[],
        unverified_entities=unverified,
    )
    diag = TrustworthyNoveltyMiddleware()._build_diag(bundle)
    assert "unverified_external_entities" in diag

    captured = {}

    class InspectRunner:
        def chat_json(self, system, user, **kwargs):
            captured["system"] = system
            return {"edits": []}

    editor_call(InspectRunner(), draft_with_external, diag, Evidence(_task()), {"title": "t"}, scope="local")
    assert "Unverified external entities" in captured["system"]
    assert "SCBench" in captured["system"]
