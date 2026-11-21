"""Trustworthy & Novelty Middleware and Unified Multi-Benchmark Evaluation Suite."""
from __future__ import annotations

from typing import Any

from .schemas import FullAuditBundle, NormalizedProposal, PriorWorkItem, TaskSpec

__all__ = [
    "FullAuditBundle",
    "GovernanceOutcome",
    "NormalizedProposal",
    "PriorWorkItem",
    "TaskSpec",
    "TrustworthyNoveltyMiddleware",
    "build_pre_gen_brief",
    "govern_proposal",
]


def __getattr__(name: str) -> Any:
    if name in {"TrustworthyNoveltyMiddleware", "build_pre_gen_brief"}:
        from . import middleware

        return getattr(middleware, name)
    if name in {"GovernanceOutcome", "govern_proposal"}:
        from . import plugin

        return getattr(plugin, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

