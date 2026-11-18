"""Canonical Pydantic contracts shared across the evaluation harness and middleware."""
from __future__ import annotations

from typing import Any, Literal
from pydantic import BaseModel, Field


class PriorWorkItem(BaseModel):
    id: str
    title: str
    abstract: str = ""
    fulltext_snippet: str = ""
    key_contribution: str = ""
    limitation: str = ""
    important_result: str = ""
    year: str | None = None
    publication_date: str | None = None
    date_source: str | None = None
    version: str | None = None
    url: str | None = None
    source: Literal["local_pdf", "lit2test_paper", "topic_doc", "openalex", "crossref", "arxiv",
                    "semantic_scholar", "native_export"] = "local_pdf"
    date_basis: Literal["publication", "participant_provenance"] = "publication"


class TaskSpec(BaseModel):
    task_id: str
    source_track: Literal["aiib", "lit2test", "alphaideabench"]
    field: str
    title: str
    prompt_context: str
    cutoff_date: str
    forbidden_title_hashes: list[str] = Field(default_factory=list)
    budget: dict[str, Any] = Field(default_factory=dict)
    strict_evidence: bool = False
    prior_work: list[PriorWorkItem] = Field(default_factory=list)


class PriorFactClaim(BaseModel):
    claim: str
    cited_titles: list[str] = Field(default_factory=list)
    cited_ids: list[str] = Field(default_factory=list)


class HypothesisClaim(BaseModel):
    claim: str
    decisive_metric: str = ""
    falsifying_outcome: str = ""


class UnknownDependencyClaim(BaseModel):
    dependency: str
    why_unresolved: str = ""
    resolution_path: str = ""
    minimal_calibration_test: str = ""
    fallback_branch: str = ""


class NormalizedProposal(BaseModel):
    """Unified blind contract merging Lit2Test 6-field schema, AlphaIdeaBench budget, and Tri-State claims."""

    task_id: str
    title: str
    literature_gap: str
    hypothesis: str
    method: str
    minimal_test: str
    decisive_metric: str
    supporting_result: str
    falsifying_result: str
    related_work: str = ""
    limitations: str = ""
    resource_plan: dict[str, Any] = Field(default_factory=dict)
    prior_facts: list[PriorFactClaim] = Field(default_factory=list)
    prospective_hypotheses: list[HypothesisClaim] = Field(default_factory=list)
    unknown_dependencies: list[UnknownDependencyClaim] = Field(default_factory=list)
    gap_premises: list[str] = Field(default_factory=list)


class PriorFactAudit(BaseModel):
    claim: str
    citation_status: Literal["verified_local", "verified_pre_cutoff", "unverified", "fabricated_or_missing"]
    smoothing_score: int = Field(ge=1, le=5)
    smoothing_type: Literal["none", "unsupported_composition", "scope_expansion", "misattribution", "fabrication"] = "none"
    support_evidence: str = ""
    skeptic_counter_evidence: str = ""
    proposal_span: str = ""
    evidence_quote: str = ""
    critique_grounded: bool = True
    epistemic_overclaim: bool = False


class SpecDefectItem(BaseModel):
    category: Literal[
        "hyperparameter_or_threshold",
        "dataset_and_split",
        "baseline_alignment",
        "failure_regime",
        "metric_or_decision_rule",
        "confound_control",
    ]
    target_field: str
    missing_detail: str
    why_implementation_critical: str = ""
    clarification_hint: str = ""
    severity: Literal["blocking", "recommended"] = "recommended"


class SpecAuditResult(BaseModel):
    defects: list[SpecDefectItem] = Field(default_factory=list)
    completeness_score: float = 1.0
    codification_ready: bool = True
    blocking_count: int = 0


class ConfoundItem(BaseModel):
    alternative: str = Field(min_length=1)
    same_observation_if: str = ""
    discriminating_arm: str = Field(min_length=1)
    status: Literal["present", "absent", "ambiguous"] = "absent"
    anchor: dict[str, Any] | None = None
    why_not_enough: str = ""


class ConfoundAuditResult(BaseModel):
    hypothesis: dict[str, Any] | None = None
    predicted_observation: str = ""
    confounds: list[ConfoundItem] = Field(default_factory=list)
    decisive_missing: str = ""
    priority: Literal["blocking", "recommended", "none"] = "none"


class CompositionalRemainderResult(BaseModel):
    covered_components: list[dict[str, Any]] = Field(default_factory=list)
    compositional_remainder: str = ""
    is_compositional_novelty: bool = False
    attribution_requirements: list[str] = Field(default_factory=list)


class EvidenceQuote(BaseModel):
    paper_id: str = Field(min_length=1)
    quote: str = Field(min_length=1)


class ContributionFinding(BaseModel):
    role: Literal["core", "component"]
    status: Literal["covered", "distinct_hypothesis", "unresolved"]
    proposal_span: str = Field(min_length=1)
    evidence: list[EvidenceQuote] = Field(default_factory=list)
    rationale: str
    already_credited: bool = False


class IdeaGovernanceResult(BaseModel):
    action: Literal["correct_attribution", "retain_and_test", "regenerate", "seek_evidence"]
    claims: list[ContributionFinding] = Field(default_factory=list)
    unresolved_reason: str | None = None
    defect: Literal["none", "localization", "coverage", "evidence"] = "none"


class BoundedPatch(BaseModel):
    target_field: str
    exact_source_span: str
    replacement_text: str
    edit_type: Literal[
        "scope_narrowing",
        "attribution_credit",
        "gap_refinement",
        "spec_clarification",
        "unknown_conversion",
    ]
    rationale: str = ""


class SpecClarification(BaseModel):
    defect_id: int = Field(ge=1, strict=True)
    target_field: str
    append_text: str


class RetentionMetrics(BaseModel):
    method_token_retention: float = 1.0
    hypothesis_retention: float = 1.0
    unknown_retention_ratio: float = 1.0
    edit_locality_ratio: float = 1.0
    guard_reverted: bool = False
    rounds_used: int = 0
    patches_applied: int = 0
    patches_rejected: int = 0
    spec_clarifications_applied: int = 0
    spec_clarifications_rejected: int = 0


class TriStateAuditResult(BaseModel):
    prior_fact_audits: list[PriorFactAudit] = Field(default_factory=list)
    untested_hypothesis_count: int = 0
    honest_unknown_count: int = 0
    unoperationalized_unknown_count: int = 0
    epistemic_overclaim_count: int = 0
    max_smoothing_score: int = 1
    mean_smoothing_score: float = 1.0
    epistemic_overclaim_rate: float = 0.0
    citation_fabrication_detected: bool = False


class GapPremiseAuditItem(BaseModel):
    premise: str
    status: Literal["survived", "partially_eroded", "falsified", "unresolved"]
    counterexample_title: str | None = None
    counterexample_year: str | None = None
    justification: str = ""


class GapAuditResult(BaseModel):
    items: list[GapPremiseAuditItem] = Field(default_factory=list)
    survival_rate: float = 1.0
    any_falsified: bool = False
    unresolved_count: int = 0


class ClaimNovAuditResult(BaseModel):
    retriever_mode: str = "bm25_fallback"
    novelty_score: float = 1.0
    veto: bool = False
    veto_evidence: dict[str, Any] | None = None
    claims_count: int = 0
    claims: list[dict[str, Any]] = Field(default_factory=list)


class IntegrityGateResult(BaseModel):
    gate_pass: bool
    strict_gate_pass: bool = True
    hard_fail_reasons: list[str] = Field(default_factory=list)
    novelty_score_raw: float = 0.0
    trusted_novelty: float = 0.0
    actionability_score: float = 0.0
    rigor_score: float = 0.0
    spec_completeness: float = 1.0
    budget_feasible: bool = True


class FullAuditBundle(BaseModel):
    task_id: str
    arm: str
    proposal: NormalizedProposal
    tri_state: TriStateAuditResult
    gap_audit: GapAuditResult
    claimnov: ClaimNovAuditResult
    spec_audit: SpecAuditResult = Field(default_factory=SpecAuditResult)
    confound_audit: ConfoundAuditResult = Field(default_factory=ConfoundAuditResult)
    compositional: CompositionalRemainderResult = Field(default_factory=CompositionalRemainderResult)
    governance: IdeaGovernanceResult | None = None
    aib_scores: dict[str, Any] = Field(default_factory=dict)
    gate: IntegrityGateResult
    retention: RetentionMetrics | None = None
    audit_cost_usd: float = 0.0


class PairwiseComparison(BaseModel):
    task_id: str
    arm_x: str
    arm_y: str
    verdict_xy: Literal["X", "Y", "tie"]
    verdict_yx_mapped: Literal["X", "Y", "tie"]
    folded_winner: Literal["X", "Y", "tie"]
    order_sensitive: bool
    reason: str = ""
