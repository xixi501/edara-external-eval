"""IdeaAMBIG-inspired Specification Defect Locator for Ideation -> Proposal/Plan artifacts."""
from __future__ import annotations

import json
import re
from typing import Any

from .factual_audit import audit_draft_consistency
from .llm_client import LLMRunner
from .schemas import NormalizedProposal, SpecAuditResult, SpecDefectItem, TaskSpec

_SPEC_LOCATOR_SYSTEM = """You are a Specification Defect Locator (inspired by IdeaAMBIG).
Models struggle to self-locate missing implementation specs (~9.6%), yet once pointed to exact gaps they clarify them reliably (~80.6%).
Inspect the proposal against the Task Budget Envelope and locate up to 4 CONCRETE implementation-critical missing details across:
1. hyperparameter_or_threshold: unspecified key threshold, loss weight, budget ratio, or calibration parameter (ignore if ALREADY declared in unknown_dependencies with a concrete calibration test/resolution_path).
2. dataset_and_split: missing dataset/benchmark name, train/cal/eval split isolation, or sample/trial count.
3. baseline_alignment: missing matched-compute/matched-budget baseline or missing ablation isolating the claimed mechanism.
4. failure_regime: missing explicit regime/boundary where the method is expected NOT to help or degrades.
5. metric_or_decision_rule: vague decisive_metric or non-quantitative supporting/falsifying decision rule.

Be strict on genuine implementation ambiguity, but NEVER invent fake domain requirements unrelated to the proposal.
Return valid JSON only:
{
  "defects": [
    {
      "category": "hyperparameter_or_threshold" | "dataset_and_split" | "baseline_alignment" | "failure_regime" | "metric_or_decision_rule",
      "target_field": "method" | "minimal_test" | "decisive_metric" | "falsifying_result" | "limitations" | "resource_plan",
      "missing_detail": "<what specific detail is missing or underspecified>",
      "why_implementation_critical": "<1 concise sentence>",
      "clarification_hint": "<exact minimal detail or unknown-dependency calibration test to add>",
      "severity": "blocking" | "recommended"
    }
  ]
}"""

_VALID_CATEGORIES = {
    "hyperparameter_or_threshold",
    "dataset_and_split",
    "baseline_alignment",
    "failure_regime",
    "metric_or_decision_rule",
    "confound_control",
}


def _deterministic_spec_defects(proposal: NormalizedProposal) -> list[SpecDefectItem]:
    """Catch unambiguous structural specification gaps before invoking LLM defect locator."""
    defects: list[SpecDefectItem] = []
    mt = (proposal.minimal_test or "").strip()
    if len(mt) < 40 or not re.search(r"\b(baseline|ablation|control|split|benchmark|dataset|held-out)\b", mt, re.I):
        defects.append(
            SpecDefectItem(
                category="baseline_alignment",
                target_field="minimal_test",
                missing_detail="Minimal test lacks explicit matched baseline/control and evaluation split specification.",
                why_implementation_critical="Cannot verify causal isolation or prevent confounded implementation.",
                clarification_hint="Name the concrete baseline(s), held-out split, and ablation control inside minimal_test.",
                severity="blocking",
            )
        )
    if not proposal.decisive_metric.strip() or not proposal.falsifying_result.strip():
        defects.append(
            SpecDefectItem(
                category="metric_or_decision_rule",
                target_field="falsifying_result" if not proposal.falsifying_result.strip() else "decisive_metric",
                missing_detail="Missing operational decisive metric or explicit falsifying threshold.",
                why_implementation_critical="Without a numeric/operational falsification rule, evaluation outcomes cannot refute the hypothesis.",
                clarification_hint="Specify one primary metric and a concrete quantitative falsifying condition.",
                severity="blocking",
            )
        )
    lim = (proposal.limitations or "").strip()
    if len(lim) < 25 and not proposal.unknown_dependencies:
        defects.append(
            SpecDefectItem(
                category="failure_regime",
                target_field="limitations",
                missing_detail="Missing failure regimes or unverified empirical prerequisites.",
                why_implementation_critical="Omits boundary conditions where the proposed mechanism breaks or requires calibration.",
                clarification_hint="State failure regimes in limitations or add structured unknown_dependencies.",
                severity="recommended",
            )
        )
    for finding in audit_draft_consistency(
        {
            "hypothesis": proposal.hypothesis,
            "decisive_metric": proposal.decisive_metric,
            "supporting_result": proposal.supporting_result,
            "falsifying_result": proposal.falsifying_result,
        }
    ):
        defects.append(
            SpecDefectItem(
                category="metric_or_decision_rule",
                target_field=finding.field if finding.field in {"decisive_metric", "falsifying_result"} else "decisive_metric",
                missing_detail=finding.reason[:240],
                why_implementation_critical="Internal conflict between hypothesis and metric/falsification criteria breaks testability.",
                clarification_hint="Align effect-size thresholds across hypothesis, decisive_metric, supporting_result, and falsifying_result.",
                severity="blocking",
            )
        )
    return defects


def locate_spec_defects(
    proposal: NormalizedProposal,
    task: TaskSpec,
    judge_runner: LLMRunner,
) -> SpecAuditResult:
    """Locate implementation-critical specification defects in a normalized proposal."""
    det = _deterministic_spec_defects(proposal)
    user = (
        f"Task ID: {task.task_id}\n"
        f"Budget Envelope: {json.dumps(task.budget, ensure_ascii=False)}\n\n"
        f"Proposal:\n{proposal.model_dump_json(indent=2)}"
    )
    try:
        res = judge_runner.chat_json(_SPEC_LOCATOR_SYSTEM, user, max_tokens=1600, tag="spec_locate")
    except json.JSONDecodeError as exc:
        print(f"[warn] {task.task_id} spec_locate JSON parse failed ({exc}); using deterministic defects only.", flush=True)
        res = {"defects": []}
    llm_defects: list[SpecDefectItem] = []
    seen_keys = {(d.category, d.target_field) for d in det}
    for raw in res.get("defects") or []:
        if not isinstance(raw, dict):
            continue
        cat = str(raw.get("category") or "hyperparameter_or_threshold")
        if cat not in _VALID_CATEGORIES:
            cat = "hyperparameter_or_threshold"
        tf = str(raw.get("target_field") or "minimal_test").strip()
        missing = str(raw.get("missing_detail") or "").strip()
        if not missing:
            continue
        sev = str(raw.get("severity") or "recommended").strip()
        if sev not in {"blocking", "recommended"}:
            sev = "recommended"
        key = (cat, tf)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        llm_defects.append(
            SpecDefectItem(
                category=cat,  # type: ignore[arg-type]
                target_field=tf,
                missing_detail=missing[:260],
                why_implementation_critical=str(raw.get("why_implementation_critical") or "")[:220],
                clarification_hint=str(raw.get("clarification_hint") or "")[:240],
                severity=sev,  # type: ignore[arg-type]
            )
        )
        if len(det) + len(llm_defects) >= 4:
            break

    all_defects = (det + llm_defects)[:4]
    blocking = sum(1 for d in all_defects if d.severity == "blocking")
    recommended = len(all_defects) - blocking
    # Score starts at 1.0; blocking defect deducts 0.22, recommended deducts 0.10.
    completeness = max(0.0, round(1.0 - 0.22 * blocking - 0.10 * recommended, 3))
    return SpecAuditResult(
        defects=all_defects,
        completeness_score=completeness,
        codification_ready=(blocking == 0 and completeness >= 0.70),
        blocking_count=blocking,
    )
