"""Hypothesis-specific confound audit for the governance middleware.

A generic "missing baseline" finding produces a generic patch. This audit names the
alternative explanations that would reproduce the predicted observation and the arm
that rules each one out, so the editor can add the decisive control instead of any
control. Structure follows the testability checklists used by AlphaIdeaBench and
Lit2Test (control conditions, comparison targets, alternative explanations).
"""
from __future__ import annotations

import json

from .llm_client import LLMRunner
from .schemas import ConfoundAuditResult, ConfoundItem, SpecAuditResult, SpecDefectItem, NormalizedProposal, TaskSpec

_SYSTEM = """You audit whether a research proposal's minimal test can separate its central hypothesis
from the most plausible alternative explanations. Work only from the proposal text and its stated
budget. Name at most three alternatives that would produce the same predicted observation if true.
For each, name the specific experimental arm (baseline, ablation, or matched control) that would rule
it out, and state whether that arm already appears in the proposal. Judge presence from the draft text:
a related but non-discriminating arm does not count, and you must say why in why_not_enough.
Do not invent requirements unrelated to the proposed mechanism.

Return JSON only:
{"hypothesis": {"field": "hypothesis", "quote": "<verbatim sentence from that field>"},
 "predicted_observation": "<what the proposal says should be observed>",
 "confounds": [{"alternative": "<the rival explanation>",
   "same_observation_if": "<what the results would look like under that alternative>",
   "discriminating_arm": "<the concrete arm that separates the two explanations>",
   "status": "present|absent|ambiguous",
   "anchor": {"field": "<field key>", "quote": "<verbatim text of the existing arm>"} | null,
   "why_not_enough": "<why the nearest existing arm does not separate them>"}],
 "decisive_missing": "<the single most important missing arm, or empty string>",
 "priority": "blocking|recommended|none"}"""


def audit_confounds(proposal: NormalizedProposal, task: TaskSpec, judge_runner: LLMRunner) -> ConfoundAuditResult:
    user = (
        f"Task ID: {task.task_id}\n"
        f"Budget Envelope: {json.dumps(task.budget, ensure_ascii=False)}\n\n"
        f"Proposal:\n{proposal.model_dump_json(indent=2)}"
    )
    try:
        raw = judge_runner.chat_json(_SYSTEM, user, max_tokens=1600, tag="confound_audit")
        result = ConfoundAuditResult.model_validate(raw)
    except (json.JSONDecodeError, ValueError) as exc:
        print(f"[warn] {task.task_id} confound_audit failed ({exc}); continuing without design confounds.", flush=True)
        return ConfoundAuditResult()
    confounds = result.confounds[:3]
    for item in confounds:
        if item.status == "present" and not item.anchor:
            item.status = "ambiguous"
            item.why_not_enough = item.why_not_enough or "no verbatim anchor supplied for the claimed control"
    return result.model_copy(update={"confounds": confounds})


def confound_defects(result: ConfoundAuditResult) -> list[SpecDefectItem]:
    defects = []
    for item in result.confounds:
        if item.status == "present":
            continue
        severity = "blocking" if item.status == "absent" and result.priority == "blocking" else "recommended"
        defects.append(SpecDefectItem(
            category="confound_control",
            target_field="minimal_test",
            missing_detail=f"Uncontrolled alternative: {item.alternative}. Same observation if: {item.same_observation_if}"[:260],
            why_implementation_critical=(
                "Without this discriminating arm the minimal test cannot separate the hypothesis from the alternative explanation."
            ),
            clarification_hint=f"Add to the minimal test: {item.discriminating_arm}"[:240],
            severity=severity,
        ))
    return defects[:2]


def merge_confounds(spec: SpecAuditResult, result: ConfoundAuditResult) -> SpecAuditResult:
    extra = confound_defects(result)
    if not extra:
        return spec
    combined = extra + list(spec.defects)
    combined.sort(key=lambda d: (0 if d.severity == "blocking" else 1, 0 if d.category == "confound_control" else 1))
    defects = combined[:4]
    blocking = sum(1 for d in defects if d.severity == "blocking")
    recommended = len(defects) - blocking
    completeness = max(0.0, round(1.0 - 0.22 * blocking - 0.10 * recommended, 3))
    return SpecAuditResult(
        defects=defects,
        completeness_score=completeness,
        codification_ready=(blocking == 0 and completeness >= 0.70),
        blocking_count=blocking,
    )
