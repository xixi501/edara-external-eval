"""Render host artifacts into one comparable text template.

The judge should compare research content, not field naming. Every host artifact is
mapped onto the same six sections; sections without content are omitted rather than
filled with placeholders.
"""

from __future__ import annotations

import json

SECTIONS = ("Problem", "Proposed approach", "Contribution claims", "Evidence and unknowns",
            "Validation plan", "Risks and limitations")


def _text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (list, tuple)):
        parts = []
        for item in value:
            rendered = _text(item)
            if rendered:
                parts.append(rendered if rendered.startswith(("-", "*")) else f"- {rendered}")
        return "\n".join(parts)
    if isinstance(value, dict):
        return "\n".join(f"- {key}: {_text(item)}" for key, item in value.items() if _text(item))
    return str(value)


def _render(title: str, sections: list[tuple[str, object]]) -> str:
    lines = [f"# {title}".strip()]
    for name, value in sections:
        body = _text(value)
        if body:
            lines.extend(["", f"## {name}", body])
    return "\n".join(lines).strip() + "\n"


def render_direct(draft: dict) -> str:
    return _render(draft.get("title", ""), [
        ("Problem", draft.get("literature_gap")),
        ("Proposed approach", [draft.get("hypothesis"), draft.get("method")]),
        ("Contribution claims", draft.get("related_work")),
        ("Evidence and unknowns", draft.get("resource_plan")),
        ("Validation plan", [draft.get("minimal_test"), draft.get("decisive_metric"),
                             draft.get("supporting_result"), draft.get("falsifying_result")]),
        ("Risks and limitations", draft.get("limitations")),
    ])


def render_ai_scientist(idea: dict) -> str:
    return _render(idea.get("Title", ""), [
        ("Problem", idea.get("Abstract")),
        ("Proposed approach", [idea.get("Short Hypothesis"), idea.get("Abstract")]),
        ("Contribution claims", idea.get("Related Work")),
        ("Evidence and unknowns", None),
        ("Validation plan", idea.get("Experiments")),
        ("Risks and limitations", idea.get("Risk Factors and Limitations")),
    ])


def render_edara(artifact: dict) -> str:
    is_plan = str(artifact.get("_edara", {}).get("stage")) == "experiment_plan"
    if is_plan:
        return _render(artifact.get("objective", ""), [
            ("Problem", [artifact.get("purpose"), artifact.get("objective"), artifact.get("hypothesis")]),
            ("Proposed approach", [artifact.get("design_contract"), artifact.get("conditions")]),
            ("Contribution claims", artifact.get("rq_ids")),
            ("Evidence and unknowns", [artifact.get("alternative_explanations"), artifact.get("exposed_result_ids")]),
            ("Validation plan", [artifact.get("falsification"), artifact.get("analysis"), artifact.get("decision_rules")]),
            ("Risks and limitations", [artifact.get("limitations"), artifact.get("resource_requirements")]),
        ])
    return _render(artifact.get("title", ""), [
        ("Problem", artifact.get("problem")),
        ("Proposed approach", [artifact.get("idea"), artifact.get("method")]),
        ("Contribution claims", artifact.get("claims")),
        ("Evidence and unknowns", [artifact.get("gap"), artifact.get("unknowns")]),
        ("Validation plan", artifact.get("validation")),
        ("Risks and limitations", artifact.get("risks")),
    ])


def render(host: str, artifact: dict, citation_key: list[str] | None = None) -> str:
    if host == "direct":
        text = render_direct(artifact)
    elif host == "ai_scientist":
        text = render_ai_scientist(artifact)
    elif host == "edara":
        text = render_edara(artifact)
    else:
        raise ValueError(f"unknown host {host!r}")
    if citation_key:
        text = text.rstrip() + "\n\n## Citation key\n" + "\n".join(citation_key) + "\n"
    return text


def sha(text: str) -> str:
    import hashlib
    return hashlib.sha256(text.encode()).hexdigest()


if __name__ == "__main__":
    print(json.dumps({name: len(render(name, {"title": "t", "problem": "p"}) or "")
                      for name in ("direct", "ai_scientist", "edara")}))
