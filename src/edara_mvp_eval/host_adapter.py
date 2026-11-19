"""Multi-host draft field adapters and structural design-preservation guards."""
from __future__ import annotations

import json
import re
from collections.abc import Mapping
from typing import Any

_INDEX_RE = re.compile(r"\[\d+\]")
_BOILERPLATE_SUFFIXES = (
    ".system",
    ".user",
    ".node_id",
    ".profile_version",
    ".kind",
    ".max_calls",
    ".paper_id",
    ".section",
    ".need_id",
    ".require_verified",
    ".distinct_models",
)
_BOILERPLATE_PREFIXES = ("resource_requirements.", "citation_map", "unmapped_citations")
_META_TOP_KEYS = frozenset({"status", "stage", "version", "raw_sha256", "sha256", "task_id", "_edara"})

_DESIGN_PRIORITY = (
    "minimal_test",
    "validation",
    "conditions",
    "design_contract",
    "method",
    "approach",
    "decisive_metric",
    "decision_rules",
    "analysis.metrics",
    "falsification",
)

_ROLE_BASES: dict[str, tuple[str, ...]] = {
    "title": ("title", "name"),
    "hypothesis": ("hypothesis", "idea", "objective", "purpose", "research_question"),
    "gap": ("literature_gap", "gap", "problem", "motivation"),
    "method": ("method", "approach", "design_contract", "protocol.purpose"),
    "test": (
        "minimal_test",
        "validation",
        "conditions",
        "experiments",
        "baselines",
        "ablations",
        "sampling_plan",
        "validation_independence",
        "randomization",
        "sample_size_rationale",
        "units",
        "seeds",
    ),
    "metric": (
        "decisive_metric",
        "analysis.metrics",
        "analysis.estimand",
        "analysis.aggregation",
        "analysis.tolerance",
        "analysis.uncertainty_method",
        "method.statistics",
    ),
    "decision": (
        "supporting_result",
        "falsifying_result",
        "falsification",
        "decision_rules",
        "analysis.interpretation_rules",
        "analysis.exclusion_rules",
        "analysis.missing_rule",
        "stop_conditions",
    ),
    "related_work": (
        "related_work",
        "claims.text",
        "related_work.work",
        "related_work.title",
        "related_work.relevance",
        "related_work.relation",
        "related_work.use",
    ),
    "limitations": ("limitations", "risks", "unknowns", "unknown_dependencies", "alternative_explanations"),
}

_STOPWORDS = frozenset({
    "with", "same", "than", "that", "this", "from", "under", "using", "control", "which", "every",
    "baseline", "baselines", "ablation", "condition", "conditions", "model", "models", "metric",
    "metrics", "test", "testing", "evaluation", "evaluate", "data", "dataset", "text", "protocol",
    "method", "methods", "check", "step", "steps", "experiment", "experiments", "result", "results",
    "arms", "score", "split", "splits", "benchmark", "benchmarks", "matched", "matching", "compare",
    "comparing", "across", "within", "between", "only", "without", "into", "over", "more", "less",
    "cannot", "could", "should", "would", "while", "where", "instead", "rather", "because", "other",
})


def as_draft_dict(draft: Any) -> dict[str, Any]:
    """Return a plain dict view for either a raw Draft mapping, JSON object string, or Pydantic model."""
    if hasattr(draft, "model_dump"):
        return draft.model_dump()
    if isinstance(draft, Mapping):
        return dict(draft)
    if isinstance(draft, str) and draft.lstrip().startswith("{"):
        try:
            parsed = json.loads(draft)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass
    return {"text": str(draft)}


def canonical_field_base(key: str) -> str:
    """Strip array indices so 'validation[4]' -> 'validation' and 'claims[1].text' -> 'claims.text'."""
    return _INDEX_RE.sub("", key).strip()


def _flatten_strings(draft: Mapping[str, Any]) -> dict[str, str]:
    flat: dict[str, str] = {}

    def visit(prefix: str, value: Any) -> None:
        if isinstance(value, Mapping):
            for child, item in value.items():
                visit(f"{prefix}.{child}" if prefix else str(child), item)
        elif isinstance(value, list):
            for idx, item in enumerate(value, 1):
                visit(f"{prefix}[{idx}]", item)
        elif isinstance(value, str):
            flat[prefix] = value
        elif value is not None:
            flat[prefix] = json.dumps(value, ensure_ascii=False)

    for k, v in draft.items():
        visit(str(k), v)
    return flat


def filter_proposal_fields(draft: Any) -> dict[str, str]:
    """Return a clean, read-only flattened string view excluding agent boilerplate/prompt metadata."""
    flat = _flatten_strings(as_draft_dict(draft))
    cleaned: dict[str, str] = {}
    for key, text in flat.items():
        if not text.strip():
            continue
        if key in _META_TOP_KEYS:
            continue
        if key.startswith(_BOILERPLATE_PREFIXES):
            continue
        if key.endswith(_BOILERPLATE_SUFFIXES):
            continue
        if ".dependencies[" in key or ".alternatives[" in key or ".evidence_ids[" in key:
            continue
        cleaned[key] = text
    return cleaned


def design_field(draft: Mapping[str, Any]) -> str:
    """Pick the best existing field for appending experimental arms across all host schemas."""
    for name in _DESIGN_PRIORITY:
        if name in draft:
            return name
    by_base: dict[str, list[str]] = {}
    for key in draft:
        by_base.setdefault(canonical_field_base(str(key)), []).append(str(key))
    for base in _DESIGN_PRIORITY:
        matches = by_base.get(base)
        if matches:
            return matches[-1]
    return next(iter(draft), "minimal_test")


def _matches_role(base: str, role: str, wanted_bases: set[str]) -> bool:
    if base in wanted_bases or base.split(".")[-1] in wanted_bases:
        return True
    if role == "method" and (base.startswith("method.") or (base.startswith("protocol.") and base.endswith(".purpose"))):
        return True
    if role == "related_work" and base.startswith("related_work.") and not base.endswith((".paper_id", ".role")):
        return True
    return False


def collect_role_text(draft: Mapping[str, Any], roles: tuple[str, ...]) -> str:
    """Collect text belonging to logical proposal roles across flat and flattened multi-host keys."""
    flat = filter_proposal_fields(draft)
    chunks: list[str] = []
    seen_keys: set[str] = set()
    for role in roles:
        wanted = set(_ROLE_BASES.get(role, (role,)))
        for key, text in flat.items():
            if key in seen_keys:
                continue
            base = canonical_field_base(key)
            if _matches_role(base, role, wanted):
                seen_keys.add(key)
                chunks.append(text.strip())
    return "\n".join(chunks)


def collect_design_text(draft: Mapping[str, Any]) -> str:
    """Collect experimental design, control, metric, and decision text (excluding limitations)."""
    return collect_role_text(draft, ("test", "method", "metric", "decision"))


def _distinctive_words(arm: str) -> list[str]:
    words = [
        word
        for word in re.findall(r"[a-z0-9][a-z0-9_-]{3,}", arm.lower())
        if word not in _STOPWORDS
    ]
    if words:
        return list(dict.fromkeys(words))
    # Fallback when all words were short/common: keep non-stopword alphanumeric tokens >= 3 chars.
    return list(dict.fromkeys(w for w in re.findall(r"[a-z0-9]{3,}", arm.lower()) if w not in _STOPWORDS))


def _token_matched(word: str, text: str) -> bool:
    if word in text:
        return True
    if "-" in word and word.replace("-", " ") in text:
        return True
    if len(word) >= 7 and word[:6] in text:
        return True
    return False


def confound_addressed(
    draft: Mapping[str, Any],
    arms: list[str],
    *,
    baseline_draft: Mapping[str, Any] | None = None,
) -> bool:
    """Check whether every required discriminating arm appears in the same design sections design_field writes to."""
    # Search full collect_design_text (test, method, metric, decision) so approach/design_contract
    # selected by design_field() stay aligned with verification, while still excluding claims/related_work.
    text = collect_design_text(draft).lower()
    base_text = collect_design_text(baseline_draft).lower() if baseline_draft is not None else ""
    for arm in arms:
        words = _distinctive_words(arm)
        if not words:
            continue
        hyphenated = [w for w in words if "-" in w]
        if hyphenated and not any(_token_matched(h, text) for h in hyphenated):
            return False
        hits = sum(_token_matched(word, text) for word in words)
        if hits / len(words) < 0.60:
            return False
        if baseline_draft is not None and text == base_text:
            return False
    return True


def extract_design_elements(draft: Any) -> dict[str, bool]:
    """Deterministically detect key structural elements of a falsifiable experimental design."""
    data = as_draft_dict(draft)
    hyp = collect_role_text(data, ("hypothesis",)).strip()
    test = collect_role_text(data, ("test",)).strip()
    method_and_test = collect_role_text(data, ("test", "method")).strip()
    metric_text = collect_role_text(data, ("metric",)).strip()
    decision_text = collect_role_text(data, ("decision",)).strip()
    all_design = collect_design_text(data)

    has_control = bool(
        re.search(
            r"\b(baseline|control|ablation|compar\w*|matched|placebo|sham|oracle|condition|arm)\b",
            method_and_test,
            re.I,
        )
    )
    has_metric = len(metric_text) >= 4 or bool(
        re.search(
            r"\b(accuracy|auroc|aurc|miou|iou|exact\s*match|em|recall|f1|ece|brier|loss|error|rate|score|metric|latency|risk|pass@\d+)\b",
            all_design,
            re.I,
        )
    )
    has_decision = len(decision_text) >= 12 or bool(
        re.search(
            r"\b(falsif\w*|refut\w*|reject\w*|threshold|margin|criteri\w*|decision|>=|<=|percentage point|p\s*<)\b",
            all_design,
            re.I,
        )
    )
    return {
        "hypothesis": len(hyp) >= 5,
        "test": len(test) >= 15 or len(method_and_test) >= 30,
        "control": has_control,
        "metric": has_metric,
        "decision": has_decision,
    }


_GENERIC_TITLE_HYPHEN_RE = re.compile(
    r"^(?:multi-agent|multi-hop|single-hop|multi-step|multi-turn|multi-task|zero-shot|few-shot|one-shot|"
    r"end-to-end|real-time|large-scale|open-vocabulary|open-domain|closed-set|cross-domain|cross-modal|"
    r"pre-trained|fine-tuned|test-time|in-context|self-consistency|black-box|white-box|"
    r"task-aware|query-aware|layer-wise|long-context|high-level|low-cost)$",
    re.I,
)


def _title_anchor_tokens(title: str) -> list[str]:
    clean = str(title or "").strip()
    if not clean:
        return []
    head = clean.split(":", 1)[0].strip()
    source = head if any("-" in t or any(c.isupper() for c in t[1:]) or t.isupper() for t in re.findall(r"\b[A-Za-z][A-Za-z0-9-]{2,}\b", head)) else clean
    tokens = re.findall(r"\b[A-Za-z][A-Za-z0-9-]{2,}\b", source)
    strong = [
        tok for tok in tokens
        if not _GENERIC_TITLE_HYPHEN_RE.match(tok)
        and (
            any(c.isupper() for c in tok[1:])
            or tok.isupper()
            or ("-" in tok and any(part.isupper() or any(c.isupper() for c in part[1:]) for part in tok.split("-")))
        )
        and len(tok) >= 4
    ]
    return list(dict.fromkeys(strong))


def preserves_related_work_anchors(
    before: Any,
    after: Any,
    *,
    prior_work_titles: Sequence[str] = (),
) -> tuple[bool, list[str]]:
    """Reject rewrites that wipe non-empty related_work fields or delete supplied-paper citations present before."""
    prev = as_draft_dict(before)
    curr = as_draft_dict(after)
    regressions: list[str] = []

    prev_blob = "\n".join(str(v) for v in prev.values() if isinstance(v, str)).lower()
    curr_blob = "\n".join(str(v) for v in curr.values() if isinstance(v, str)).lower()

    for title in prior_work_titles or ():
        for tok in _title_anchor_tokens(title):
            low = tok.lower()
            if low in prev_blob and low not in curr_blob:
                regressions.append(f"dropped_supplied_reference:{tok}")
                break

    for key, val in prev.items():
        if key.startswith("related_work") and isinstance(val, str) and val.strip():
            if not str(curr.get(key) or "").strip():
                val_low = val.lower()
                if any(
                    tok.lower() in val_low
                    for title in (prior_work_titles or ())
                    for tok in _title_anchor_tokens(title)
                ):
                    regressions.append(f"emptied_supplied_related_work:{key}")

    return len(regressions) == 0, regressions


def preserves_design_elements(
    before: Any,
    after: Any,
    *,
    before_bundle: Any = None,
    after_bundle: Any = None,
    prior_work_titles: Sequence[str] = (),
) -> tuple[bool, list[str]]:
    """Reject rewrites that silently drop existing structural design elements, present controls, or supplied anchors."""
    prev = extract_design_elements(before)
    curr = extract_design_elements(after)
    regressions = [f"lost_{key}" for key in ("hypothesis", "test", "control", "metric", "decision")
                   if prev[key] and not curr[key]]
    if before_bundle is not None and after_bundle is not None:
        prev_present = {
            c.alternative.strip().lower()[:60]
            for c in getattr(getattr(before_bundle, "confound_audit", None), "confounds", ())
            if c.status == "present"
        }
        curr_absent = {
            c.alternative.strip().lower()[:60]
            for c in getattr(getattr(after_bundle, "confound_audit", None), "confounds", ())
            if c.status == "absent"
        }
        if prev_present & curr_absent:
            regressions.append("lost_present_confound_control")
    if prior_work_titles:
        _, ref_regressions = preserves_related_work_anchors(
            before, after, prior_work_titles=prior_work_titles
        )
        regressions.extend(ref_regressions)
    return len(regressions) == 0, regressions


def _extract_matching_sentences(text: str, pattern: str, *, limit: int = 3) -> str:
    if not text.strip():
        return ""
    hits: list[str] = []
    seen: set[str] = set()
    rx = re.compile(pattern, re.I)
    for chunk in re.split(r"(?<=[.!?])\s+|\n+", text):
        s = chunk.strip()
        if not s or not rx.search(s):
            continue
        norm = " ".join(s.lower().split())
        if norm not in seen:
            seen.add(norm)
            hits.append(s)
            if len(hits) >= limit:
                break
    return " ".join(hits)


def to_lit2test_card(draft: Any) -> dict[str, str]:
    """Project any host draft into a uniform Lit2Test-compatible evaluation card."""
    data = as_draft_dict(draft)
    clean_flat = filter_proposal_fields(data)

    title = collect_role_text(clean_flat, ("title",)) or clean_flat.get("title", "")
    gap = collect_role_text(clean_flat, ("gap",))
    hyp = collect_role_text(clean_flat, ("hypothesis",))
    method = collect_role_text(clean_flat, ("method",))
    minimal_test = collect_role_text(clean_flat, ("test",))
    decisive_metric = collect_role_text(clean_flat, ("metric",))
    decision = collect_role_text(clean_flat, ("decision",))
    supporting_result = clean_flat.get("supporting_result", "")
    related_work = collect_role_text(clean_flat, ("related_work",))
    limitations = collect_role_text(clean_flat, ("limitations",))

    if not title:
        seed = hyp or gap or method
        first = re.split(r"[.\n!?]", seed.strip(), maxsplit=1)[0].strip()
        title = first[:160]
    if not gap:
        gap = _extract_matching_sentences(
            "\n".join(p for p in (method, hyp) if p),
            r"\b(remain\w*|lack\w*|limit\w*|gap|however|fails?\b|challenge|open problem|closed-set|does not|cannot)\b",
            limit=2,
        )
    if not decisive_metric:
        decisive_metric = _extract_matching_sentences(
            "\n".join(p for p in (minimal_test, method, hyp) if p),
            r"\b(metrics?|primary|miou|iou|accuracy|auroc|aurc|exact match|f1|ece|brier|loss|latency|recall|perplexity|harmonic mean|bootstrap)\b",
            limit=3,
        )
    if not supporting_result and decision:
        supporting_result = _extract_matching_sentences(
            decision,
            r"\b(support\w*|confirm\w*|pass|require\w*|positive|outperform|exceed|above)\b(?!.*\b(falsif|refut|reject)\b)",
            limit=3,
        )
    falsifying_result = decision
    if supporting_result and falsifying_result:
        # Remove duplicated supporting text when extracting falsifying/decision criteria.
        filtered_chunks = [
            s.strip()
            for s in re.split(r"\n+", falsifying_result)
            if s.strip() and s.strip() != supporting_result.strip()
        ]
        falsifying_result = "\n".join(filtered_chunks) or falsifying_result
    if not falsifying_result:
        falsifying_result = _extract_matching_sentences(
            "\n".join(p for p in (minimal_test, hyp, method) if p),
            r"\b(falsif\w*|refut\w*|reject\w*|overturn\w*|inconclusive|underperform|fail\w* to|drop\w* below|null|decision rule|stop condition|threshold)\b",
            limit=3,
        )
    if not related_work:
        related_work = _extract_matching_sentences(
            "\n".join(p for p in (method, gap, hyp, minimal_test) if p),
            r"\[S[0-9a-f]+\]|\b(prior work|existing methods?|previously|et al\.|baseline studies)\b",
            limit=4,
        )

    return {
        "title": title,
        "literature_gap": gap,
        "hypothesis": hyp,
        "method": method,
        "minimal_test": minimal_test,
        "decisive_metric": decisive_metric,
        "supporting_result": supporting_result,
        "falsifying_result": falsifying_result,
        "related_work": related_work,
        "limitations": limitations,
    }


def _to_float(val: Any) -> float:
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, str):
        m = re.search(r"[-+]?\d+(?:\.\d+)?", val)
        if m:
            return float(m.group(0))
    return 0.0


def to_alphaideabench_idea(task_id: str, draft: Any) -> dict[str, Any]:
    """Project any host draft into AlphaIdeaBench ResearchIdea-compatible fields."""
    data = as_draft_dict(draft)
    flat = _flatten_strings(data)
    card = to_lit2test_card(data)
    raw_rp = data.get("resource_plan") if isinstance(data.get("resource_plan"), Mapping) else {}

    gpu_hours = _to_float(raw_rp.get("estimated_gpu_hours") or raw_rp.get("gpu_hours_total") or flat.get("resource_plan.gpu_hours_total") or flat.get("resource_plan.estimated_gpu_hours"))
    if gpu_hours == 0.0:
        gpu_hours = sum(
            _to_float(v)
            for k, v in flat.items()
            if k.startswith("resource_requirements.compute") and k.endswith(".gpu_hours")
        )
    if gpu_hours == 0.0:
        gpu_hours = 24.0 * sum(
            _to_float(v)
            for k, v in flat.items()
            if k.startswith("resource_requirements.compute") and k.endswith("gpu_days")
        )
    api_usd = _to_float(raw_rp.get("estimated_api_usd") or raw_rp.get("api_usd") or flat.get("resource_plan.api_usd") or flat.get("resource_plan.estimated_api_usd"))
    if api_usd == 0.0:
        api_usd = round(
            sum(
                _to_float(v)
                for k, v in flat.items()
                if k.startswith("resource_requirements.") and k.endswith(".estimated_cost_microusd")
            ) / 1_000_000.0,
            2,
        )
    human_hours = _to_float(raw_rp.get("estimated_human_hours") or raw_rp.get("annotator_hours") or flat.get("resource_plan.annotator_hours") or flat.get("resource_plan.estimated_human_hours"))
    calendar_weeks = _to_float(raw_rp.get("calendar_weeks") or flat.get("resource_plan.calendar_weeks"))
    notes = str(
        raw_rp.get("notes")
        or flat.get("resource_plan.notes")
        or flat.get("resource_plan.timeline")
        or flat.get("resource_plan.compute_notes")
        or flat.get("resource_plan.feasibility_notes")
        or flat.get("resource_plan.api_plan")
        or "; ".join(
            v.strip()
            for k, v in flat.items()
            if k.startswith("resource_requirements.") and k.endswith(".reason") and v.strip()
        )
        or ""
    )
    gpu_desc = str(raw_rp.get("gpu") or flat.get("resource_plan.gpu") or flat.get("resource_plan.compute") or "")

    rp: dict[str, Any] = {
        "gpu": gpu_desc,
        "estimated_gpu_hours": gpu_hours,
        "estimated_api_usd": api_usd,
        "estimated_human_hours": human_hours,
        "calendar_weeks": calendar_weeks,
        "notes": notes,
    }
    return {
        "idea_id": task_id,
        "topic_id": task_id,
        "title": card["title"] or task_id,
        "research_question": card["hypothesis"],
        "hypothesis": card["hypothesis"],
        "problem": card["literature_gap"],
        "motivation": card["literature_gap"],
        "method": card["method"],
        "experiments": card["minimal_test"],
        "baselines": card["minimal_test"],
        "success_criteria": (
            f"Decisive metric: {card['decisive_metric']}\n"
            f"Decision / falsification rules: {card['falsifying_result']}"
        ).strip(),
        "related_work": card["related_work"],
        "limitations": card["limitations"],
        "resource_plan": rp,
    }
