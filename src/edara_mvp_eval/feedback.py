"""Route grounded idea defects to bounded governance actions.

The action set is derived independently from the audit: an unresolved novelty search
must not block an independent, text-grounded correction. Every action runs at most
once, and search exhaustion is recorded as a state flag rather than a reason to stop.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable

from .epistemic_audit import run_full_audit
from .governance import evidence_payload
from .host_adapter import preserves_design_elements
from .middleware import TrustworthyNoveltyMiddleware, append_spec_clarifications

GeneratorCallback = Callable[[dict[str, Any]], dict[str, Any]]


@dataclass
class PolicyState:
    regenerated: bool = False
    patched: bool = False
    clarified: bool = False
    searched: bool = False
    revised: bool = False
    revision_blocked: bool = False
    needs_post_search_revision: bool = False


def _governance(bundle):
    return getattr(bundle, "governance", None)


def _has_overclaim(bundle) -> bool:
    tri = getattr(bundle, "tri_state", None)
    gap = getattr(bundle, "gap_audit", None)
    factual = getattr(bundle, "factual_findings", None) or ()
    return (
        bool(getattr(tri, "epistemic_overclaim_count", 0))
        or bool(getattr(gap, "any_falsified", False))
        or any(getattr(item, "label", "") == "defect" for item in factual)
    )


def _blocking_defects(bundle) -> int:
    spec = getattr(bundle, "spec_audit", None)
    return int(getattr(spec, "blocking_count", 0) or 0)


def _should_regenerate(bundle, state: PolicyState) -> bool:
    governance = _governance(bundle)
    if governance and governance.action == "regenerate":
        return True
    claimnov = getattr(bundle, "claimnov", None)
    if getattr(claimnov, "veto", False):
        claims = getattr(governance, "claims", ()) if governance else ()
        if not any(getattr(c, "status", "") == "distinct_hypothesis" for c in claims):
            return not (governance and governance.action == "correct_attribution")
    gap = getattr(bundle, "gap_audit", None)
    return bool((state.patched or state.revised) and getattr(gap, "any_falsified", False))


def _needs_guided_revise(bundle, state: PolicyState) -> bool:
    if state.revised:
        return False
    if state.needs_post_search_revision:
        return True
    confound = getattr(bundle, "confound_audit", None)
    if getattr(confound, "priority", "none") in {"blocking", "recommended"}:
        return True
    if _blocking_defects(bundle) or _has_overclaim(bundle) or getattr(bundle, "unverified_entities", ()):
        return True
    governance = _governance(bundle)
    return bool(governance and governance.action == "correct_attribution")


def eligible_actions(
    bundle,
    state: PolicyState | None = None,
    *,
    allow_guided_revise: bool = False,
) -> list[tuple[str, str]]:
    """Ordered (action, reason) pairs. Multi-round arms may run governed guided_revise."""
    state = state or PolicyState()
    governance = _governance(bundle)
    actions: list[tuple[str, str]] = []
    if allow_guided_revise:
        if state.needs_post_search_revision:
            actions.append(("guided_revise", "incorporate_retrieved_evidence"))
        if not state.regenerated and _should_regenerate(bundle, state):
            actions.append(("regenerate", "core_covered_by_prior_work"))
        if not state.revised and not state.revision_blocked and _needs_guided_revise(bundle, state):
            actions.append(("guided_revise", "governed_multifield_revision"))
        if (not state.searched and governance and governance.action == "seek_evidence"
                and getattr(governance, "defect", "none") != "localization"):
            actions.append(("seek_evidence", "novelty_evidence_missing"))
        if not state.patched and (_has_overclaim(bundle)
                                  or (governance and governance.action == "correct_attribution")):
            actions.append(("local_patch", "remaining_grounded_defect"))
        if not state.clarified and _blocking_defects(bundle):
            actions.append(("clarify", "remaining_blocking_spec_defect"))
        if not actions:
            actions.append(("retain_and_test", "no_eligible_defect"))
        return actions

    if not state.patched and (_has_overclaim(bundle)
                              or (governance and governance.action == "correct_attribution")):
        actions.append(("local_patch", "grounded_attribution_or_overclaim"))
    if not state.clarified and _blocking_defects(bundle):
        actions.append(("clarify", "blocking_spec_defect"))
    if not state.regenerated and _should_regenerate(bundle, state):
        actions.append(("regenerate", "core_covered_by_prior_work"))
    if (not state.searched and governance and governance.action == "seek_evidence"
            and getattr(governance, "defect", "none") != "localization"):
        actions.append(("seek_evidence", "novelty_evidence_missing"))
    if not actions:
        actions.append(("retain_and_test", "no_eligible_defect"))
    return actions


def choose_action(bundle, state: PolicyState | None = None) -> str:
    return eligible_actions(bundle, state)[0][0]


_CLARIFY_SYSTEM = """You may append short clarifying clauses to existing proposal fields.
Return JSON {"spec_clarifications": [{"defect_id": <1-based index>, "target_field": <field>,
"append_text": <one clause, under 240 characters>}]}.
Only append. Do not rewrite, delete, reorder or summarise existing text. Do not add results."""


def append_only_clarify(proposal, defects, runner) -> tuple[int, int]:
    payload = {"defects": [defect.model_dump() for defect in defects],
               "proposal": proposal.model_dump()}
    raw = runner.chat_json(_CLARIFY_SYSTEM, json.dumps(payload, ensure_ascii=False),
                           max_tokens=1800, tag="spec_clarification")
    candidate = raw.get("spec_clarifications") if isinstance(raw, dict) else None
    return append_spec_clarifications(proposal, {"spec_clarifications": candidate or []}, defects)


class FeedbackMiddleware:
    def apply(self, task, raw, initial_audit, *, audit_runner, repair_runner,
              generator_callback: GeneratorCallback, search, max_rounds: int = 3):
        current, bundle = raw, initial_audit
        history: list[dict[str, Any]] = []
        state = PolicyState()
        for round_id in range(1, max_rounds + 1):
            eligible = eligible_actions(bundle, state)
            action, reason = eligible[0]
            record = {"round": round_id, "eligible": [name for name, _ in eligible],
                      "chosen": action, "reason": reason}
            if action == "retain_and_test":
                history.append({**record, "status": "retained"})
                break

            if action == "seek_evidence":
                state.searched = True
                gaps = [item.premise for item in getattr(bundle.gap_audit, "items", [])
                        if getattr(item, "status", "") == "unresolved"]
                queries = [bundle.proposal.title, *(gaps or [bundle.proposal.hypothesis])]
                previous = {paper.id: paper for paper in task.prior_work}
                found = dict(previous)
                for query in queries[:2]:
                    if query:
                        for paper in search(query)[:4]:
                            found.setdefault(paper.id, paper)
                added = sorted(set(found) - set(previous))
                history.append({**record, "added": added})
                if added:
                    task = task.model_copy(update={"prior_work": list(found.values())})
                    bundle = run_full_audit(task, bundle.arm, current, audit_runner)
                continue

            if action == "clarify":
                state.clarified = True
                proposal = bundle.proposal.model_copy(deep=True)
                applied, rejected = append_only_clarify(proposal, bundle.spec_audit.defects, repair_runner)
                history.append({**record, "applied": applied, "rejected": rejected})
                if applied:
                    current = proposal.model_dump()
                    bundle = run_full_audit(task, bundle.arm, current, audit_runner)
                continue

            if action == "local_patch":
                state.patched = True
                traces: list[dict[str, Any]] = []
                revised, _, audited, _ = TrustworthyNoveltyMiddleware(max_rounds=1).apply(
                    task, current, repair_runner=repair_runner, audit_runner=audit_runner,
                    initial_audit=bundle, history=traces)
                history.append({**record, "edits": traces})
                if revised != current and audited is not bundle:
                    current, bundle = revised, audited
                continue

            state.regenerated = True
            diagnostic = TrustworthyNoveltyMiddleware()._build_diag(bundle)
            callback_input = {"action": action, "topic": task.title, "task": task.prompt_context,
                              "budget": task.budget, "cutoff": task.cutoff_date, "draft": current,
                              "evidence": evidence_payload(task), "diagnostics": diagnostic}
            revised = generator_callback(callback_input)
            if not all(isinstance(revised.get(key), str) and revised[key].strip()
                       for key in ("hypothesis", "method", "minimal_test", "decisive_metric",
                                   "falsifying_result")):
                history.append({**record, "candidate": revised, "status": "invalid_candidate"})
                continue
            if revised == current:
                history.append({**record, "status": "unchanged"})
                continue
            audited = run_full_audit(task, bundle.arm, revised, audit_runner)
            comparison = audit_runner.chat_json(
                "Check candidate task alignment and preservation of the specified valid distinction. "
                "If a new core was requested, require substantive change relative to the covered core. "
                "Return JSON booleans aligned, preserves, material_change, and a reason.",
                json.dumps({"topic": task.title, "action": action, "original": current,
                            "candidate": revised, "old_contribution": diagnostic.get("governance")}),
                tag="feedback_candidate_check", max_tokens=900)
            preserved, regressions = preserves_design_elements(
                current, revised, before_bundle=bundle, after_bundle=audited,
            )
            accepted = (preserved
                        and comparison.get("aligned") is True and comparison.get("preserves") is True
                        and comparison.get("material_change") is True
                        and not audited.tri_state.citation_fabrication_detected
                        and not (audited.governance and audited.governance.action == "regenerate"))
            history.append({**record, "candidate": revised, "audit": audited.model_dump(),
                            "comparison": comparison, "design_regressions": regressions,
                            "status": "accepted" if accepted else (
                                "design_regression_blocked" if not preserved else "returned")})
            if accepted:
                current, bundle = revised, audited

        return current, bundle, history, task.prior_work, choose_action(bundle, state)
