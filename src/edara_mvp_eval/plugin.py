"""Public plug-and-play Python API for the Trustworthy & Novelty Proposal Governance Middleware."""
from __future__ import annotations

import tempfile
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

from .llm_client import LLMRunner, ModelEndpoint
from .paired_inputs import Evidence
from .paired_trial import LIMITS, run_arm
from .schemas import PriorWorkItem, TaskSpec


@dataclass(frozen=True)
class GovernanceOutcome:
    """Structured result returned by ``govern_proposal``."""

    draft: dict[str, Any]
    initial_draft: dict[str, Any]
    status: str
    changed: bool
    rounds: int
    queries: list[str]
    retrieved_papers: list[dict[str, Any]]
    history: list[dict[str, Any]]
    edit_volume: dict[str, int]
    cost_usd: float


class _BoundedPluginRunner(LLMRunner):
    """Lightweight call-bounded LLMRunner wrapper matching paired_trial semantics."""

    def __init__(self, endpoint: ModelEndpoint, *, limit: int, dry_run: bool = False) -> None:
        super().__init__(endpoint, dry_run=dry_run)
        self.limit = limit
        self.index = 0
        self.reused = 0

    def chat_text(self, system: str, user: str, **kwargs: Any) -> str:
        from .paired_trial import BudgetExceeded, OUTPUT_TOKENS

        if self.index >= self.limit:
            raise BudgetExceeded("model_call_limit")
        self.index += 1
        kwargs["max_tokens"] = min(kwargs.get("max_tokens", OUTPUT_TOKENS), OUTPUT_TOKENS)
        return super().chat_text(system, user, **kwargs)


def _coerce_prior_work(items: Sequence[PriorWorkItem | Mapping[str, Any]] | None) -> list[PriorWorkItem]:
    if not items:
        return []
    out: list[PriorWorkItem] = []
    for idx, item in enumerate(items, 1):
        if isinstance(item, PriorWorkItem):
            out.append(item)
        else:
            data = dict(item)
            data.setdefault("id", str(data.get("paper_id") or f"P{idx}"))
            data.setdefault("title", str(data.get("title") or f"Reference {idx}"))
            if data.get("year") and not data.get("date_source"):
                data["date_source"] = "user_supplied"
                data.setdefault("version", "v1")
            out.append(PriorWorkItem.model_validate(data))
    return out


def govern_proposal(
    draft: Mapping[str, Any],
    *,
    task_title: str = "Research Proposal",
    task_context: str = "",
    prior_work: Sequence[PriorWorkItem | Mapping[str, Any]] | None = None,
    cutoff_date: str = "2025-12-31",
    budget: Mapping[str, Any] | None = None,
    task_id: str = "custom-task",
    editor_profile: str | Path | ModelEndpoint | None = None,
    audit_profile: str | Path | ModelEndpoint | None = None,
    mode: str = "full",
    enforce_guards: bool = True,
    lookup_dir: Path | None = None,
    dry_run: bool = False,
) -> GovernanceOutcome:
    """Run host-agnostic governance over any LLM or agent research proposal / experiment plan dict.

    Preserves the host's native JSON keys while eliminating unattested process hype, curbing
    unverified external entities, enforcing discriminating confound controls & falsifiable
    decision boundaries, and guarding compositional novelty.
    """
    if mode not in {"full", "local"}:
        raise ValueError("mode must be 'full' (multi-round closed-loop Arm E) or 'local' (single-round Arm C)")

    def _resolve_ep(spec: str | Path | ModelEndpoint | None, *, disable_thinking: bool) -> ModelEndpoint:
        if isinstance(spec, ModelEndpoint):
            ep = spec
        elif spec is not None:
            ep = ModelEndpoint.from_env_or_profile(profile_path=Path(spec))
        else:
            ep = ModelEndpoint.from_env_or_profile()
        if disable_thinking:
            low = ep.model.lower()
            if "deepseek" in low or "glm" in low:
                extra = dict(ep.extra_body or {})
                extra["thinking"] = {"type": "disabled"}
                if "glm" in low and "autodl" in (ep.base_url or "").lower():
                    extra["reasoning_effort"] = "none"
                ep = replace(ep, extra_body=extra)
        return ep

    editor_ep = _resolve_ep(editor_profile, disable_thinking=True)
    audit_ep = _resolve_ep(audit_profile or editor_profile, disable_thinking=True)

    task = TaskSpec(
        task_id=task_id,
        source_track="custom",
        field="general",
        title=task_title,
        prompt_context=task_context or task_title,
        cutoff_date=cutoff_date,
        budget=dict(budget or {}),
        strict_evidence=True,
        prior_work=_coerce_prior_work(prior_work),
    )
    ev = Evidence(task)
    for paper in task.prior_work:
        ev.add(paper)

    arm = "E" if mode == "full" else "C"
    limits = LIMITS[mode]
    auditor = _BoundedPluginRunner(audit_ep, limit=limits["audit"], dry_run=dry_run)
    editor = _BoundedPluginRunner(editor_ep, limit=limits["editor"], dry_run=dry_run)
    editor_confound = _BoundedPluginRunner(editor_ep, limit=limits["rounds"], dry_run=dry_run)

    raw_dict = {str(k): (v if isinstance(v, str) else str(v)) for k, v in dict(draft).items()}

    def _run_with_dir(work_dir: Path) -> dict[str, Any]:
        return run_arm(
            arm,
            raw_dict,
            task,
            ev.snapshot(),
            auditor,
            editor,
            work_dir / "lookup",
            audit_cache=work_dir / "audits",
            editor_confound=editor_confound,
            enforce_guards=enforce_guards,
        )

    if lookup_dir is not None:
        lookup_dir.mkdir(parents=True, exist_ok=True)
        res = _run_with_dir(lookup_dir)
    else:
        with tempfile.TemporaryDirectory(prefix="edara_govern_") as tmp:
            res = _run_with_dir(Path(tmp))

    total_cost = round(
        auditor.usage.cost_usd + editor.usage.cost_usd + editor_confound.usage.cost_usd,
        6,
    )
    return GovernanceOutcome(
        draft=res["draft"],
        initial_draft=raw_dict,
        status=res["status"],
        changed=res["draft"] != raw_dict,
        rounds=len(res["history"]),
        queries=list(res.get("queries") or []),
        retrieved_papers=list(res.get("retrieved") or []),
        history=list(res.get("history") or []),
        edit_volume=dict(res.get("edit_volume") or {}),
        cost_usd=total_cost,
    )
