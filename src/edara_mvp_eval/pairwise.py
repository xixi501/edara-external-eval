"""Lit2Test-style blind bidirectional order-swapped pairwise evaluator."""
from __future__ import annotations

import json
from typing import Literal

from .llm_client import LLMRunner
from .schemas import FullAuditBundle, PairwiseComparison, TaskSpec

_PAIRWISE_SYSTEM = """You are a blind research-program reviewer comparing two normalized proposals (Answer A and Answer B) for the same task and cutoff context.
Do NOT guess system identity or reward vague confidence/length.
First score BOTH answers (1-5) across 4 rubric axes, then decide the overall winner:
1. Honesty & Prior-Fact Grounding (no smoothed/universal overclaims, fabricated citations, or falsified gaps);
2. Implementation Completeness & Actionability (concrete baselines, splits, thresholds, and calibration tests for unknowns);
3. Falsifiability & Budget Realism (sharp decisive metric + explicit falsifying outcome within budget);
4. Grounded Novelty (distinct compositional or mechanistic contribution without contradicting prior work).

Return valid JSON only:
{
  "rubric_scores": {
    "A": {"honesty": 1, "actionability": 1, "falsifiability": 1, "novelty": 1},
    "B": {"honesty": 1, "actionability": 1, "falsifiability": 1, "novelty": 1}
  },
  "winner": "A" | "B" | "tie",
  "confidence": "low" | "medium" | "high",
  "main_reason": "<1-2 sentences>"
}"""


def _compact_card(bundle: FullAuditBundle) -> dict:
    p = bundle.proposal
    return {
        "title": p.title,
        "literature_gap": p.literature_gap,
        "hypothesis": p.hypothesis,
        "method": p.method,
        "minimal_test": p.minimal_test,
        "decisive_metric": p.decisive_metric,
        "supporting_result": p.supporting_result,
        "falsifying_result": p.falsifying_result,
        "unknown_dependencies": [u.model_dump() for u in p.unknown_dependencies],
        "resource_plan": p.resource_plan,
        "evidence_audit_digest": {
            "gate_pass": bundle.gate.gate_pass,
            "strict_gate_pass": bundle.gate.strict_gate_pass,
            "hard_fail_reasons": bundle.gate.hard_fail_reasons,
            "max_smoothing": bundle.tri_state.max_smoothing_score,
            "gap_survival_rate": bundle.gap_audit.survival_rate,
            "gap_unresolved_count": bundle.gap_audit.unresolved_count,
            "spec_completeness": bundle.spec_audit.completeness_score,
            "claimnov_veto": bundle.claimnov.veto,
        },
    }


def _single_pair_call(
    task: TaskSpec,
    card_a: dict,
    card_b: dict,
    runner: LLMRunner,
) -> tuple[Literal["A", "B", "tie"], str]:
    user = (
        f"Task: {task.title} (Cutoff: {task.cutoff_date})\n"
        f"Budget: {json.dumps(task.budget, ensure_ascii=False)}\n\n"
        f"Answer A:\n```json\n{json.dumps(card_a, ensure_ascii=False, indent=2)}\n```\n\n"
        f"Answer B:\n```json\n{json.dumps(card_b, ensure_ascii=False, indent=2)}\n```"
    )
    try:
        res = runner.chat_json(_PAIRWISE_SYSTEM, user, max_tokens=1500, tag="pairwise_judge")
    except json.JSONDecodeError as exc:
        print(f"[warn] {task.task_id} pairwise_judge JSON parse failed ({exc}); folding to tie.", flush=True)
        return "tie", "judge_json_parse_fallback"
    w = str(res.get("winner") or "tie").strip()
    if w not in {"A", "B", "tie"}:
        w = "tie"
    return w, str(res.get("main_reason") or "")  # type: ignore[return-value]


def compare_pair_bidirectional(
    task: TaskSpec,
    bundle_x: FullAuditBundle,
    bundle_y: FullAuditBundle,
    runner: LLMRunner,
) -> PairwiseComparison:
    card_x = _compact_card(bundle_x)
    card_y = _compact_card(bundle_y)

    w_xy_raw, reason_xy = _single_pair_call(task, card_x, card_y, runner)
    v_xy: Literal["X", "Y", "tie"] = "X" if w_xy_raw == "A" else ("Y" if w_xy_raw == "B" else "tie")

    w_yx_raw, _ = _single_pair_call(task, card_y, card_x, runner)
    v_yx_mapped: Literal["X", "Y", "tie"] = "Y" if w_yx_raw == "A" else ("X" if w_yx_raw == "B" else "tie")

    order_sensitive = v_xy != v_yx_mapped
    folded: Literal["X", "Y", "tie"] = "tie" if order_sensitive else v_xy
    return PairwiseComparison(
        task_id=task.task_id,
        arm_x=bundle_x.arm,
        arm_y=bundle_y.arm,
        verdict_xy=v_xy,
        verdict_yx_mapped=v_yx_mapped,
        folded_winner=folded,
        order_sensitive=order_sensitive,
        reason=reason_xy,
    )
