"""Independent claim-level factual, numeric, coverage, and status-boundary auditing."""
from __future__ import annotations

import json
import re
from typing import Any

from .governance import admissible
from .host_adapter import as_draft_dict, filter_proposal_fields
from .llm_client import LLMRunner
from .schemas import EvidenceQuote, FactualFinding, PriorWorkItem, TaskSpec
from .stage_b_review import locate_span, quote_in_passage

_PREREG_RE = re.compile(r"(?i)\bpre-?registered\b[\s,-]*")
_PROCESS_HYPE_PATTERNS = (
    (
        "unattested_process",
        re.compile(
            r"(?i)(?:\b(?:we\s+(?:use|have|ran|already\s+\w+)|uses?|already|was|were)\b[^.!?\n]{0,40}\bpre-?registered\b"
            r"|\bpre-?registered\b[^.!?\n]{0,40}\b(?:split|cohort|dataset|data|sample|subset|examples?|benchmarks?)\b)"
        ),
    ),
    (
        "unattested_process",
        re.compile(
            r"(?i)\b(?:we\s+have\s+already|already\s+(?:validated|calibrated|verified|benchmarked|observed|confirmed))\b"
        ),
    ),
)

_DEFECT_PRIORITY = {
    "fabrication": 0,
    "prediction_as_observed": 1,
    "internal_inconsistency": 2,
    "numeric_mismatch": 3,
    "coverage_contradiction": 4,
    "attribution": 5,
    "scope_overclaim": 6,
    "unattested_process": 7,
    "none": 8,
}

_EXTRACT_SYSTEM = """Extract up to 12 distinct, verifiable factual, attribution, numeric, coverage, or result-status claims from the research proposal.
Focus on claims that can be wrong or misleading against the proposal's own framing or supplied papers:
1. prior_fact: statements attributing methods, limitations, or findings to prior work.
2. numeric_or_benchmark: statements quoting numbers, percentages, budgets, sample sizes, or benchmark results.
3. coverage_gap: statements asserting that prior literature lacks, omits, or leaves unresolved a method, dataset, or finding.
4. result_or_process_status: statements phrasing unexecuted/proposed outcomes as already observed or established, or claiming unattested procedures (e.g. 'pre-registered', 'we have already verified', 'calibrated on N').

Copy each field key and an exact contiguous quote from that field. Omit purely prospective plans explicitly marked as 'we will', 'we plan to', or 'we hypothesize'.
Return JSON only:
{"claims": [{"field": "exact field key", "quote": "exact substring from draft", "claim_kind": "prior_fact|numeric_or_benchmark|coverage_gap|result_or_process_status"}]}."""

_AUDIT_SYSTEM = """Audit each extracted claim against the CURRENT proposal draft and the supplied pre-cutoff evidence papers.
Classify every claim as:
- supported: accurately grounded in the supplied papers, or explicitly framed in the current draft as a hypothesis or planned experiment with a clear basis.
- defect: violates factual or epistemic integrity in ONE of these specific ways (set defect_type):
  * numeric_mismatch: reported number, budget, percentage, or parameter contradicts the supplied paper passage.
  * coverage_contradiction: claims prior work/materials lack or leave unresolved something that IS already present in the supplied papers.
  * prediction_as_observed: states an unexecuted/predicted outcome as an already-observed or established fact without support.
  * unattested_process: asserts an unverified procedural status (e.g. 'pre-registered', 'already validated') absent from the task or evidence.
  * attribution: misattributes a method, benchmark, or finding to the wrong source or omits required source credit.
  * scope_overclaim: extrapolates a narrow/conditional result into a universal claim contradicted by caveats in the paper.
  * fabrication: cites a non-existent or unverified entity/result not present in the supplied sources.
- unknown: materials are insufficient to verify or refute the claim (set defect_type to "none").

CRITICAL CHECKS:
- Check exact numbers, percentages, and budgets word-for-word against the supplied paper passages before marking numeric_or_benchmark claims supported.
- Check all supplied paper titles and passages before marking coverage_gap claims supported; if the supplied packet already covers the cited benchmark or finding, mark defect (coverage_contradiction).
- Every defect or supported verdict citing a paper MUST copy an exact verbatim quote from that paper's passages, or cite a verbatim internal draft sentence in basis.

Return JSON only:
{"findings": [{"id": "c1", "label": "supported|defect|unknown", "defect_type": "none|attribution|numeric_mismatch|coverage_contradiction|prediction_as_observed|unattested_process|scope_overclaim|fabrication", "evidence": [{"paper_id": "...", "quote": "..."}], "basis": "...", "reason": "..."}]}."""


def clean_replacement_text(text: str) -> str:
    """Strip ungrounded process wording such as 'pre-registered' from edited text."""
    cleaned = _PREREG_RE.sub("", text or "")
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    cleaned = re.sub(r"\s+([,.;:!?])", r"\1", cleaned)
    return cleaned.strip() if text and text == text.strip() else cleaned


def _draft_text_fields(draft: Any) -> dict[str, str]:
    filtered = filter_proposal_fields(as_draft_dict(draft))
    out: dict[str, str] = {}
    for key, value in filtered.items():
        base = key.split("[", 1)[0].split(".", 1)[0]
        if base in {"prior_facts", "prospective_hypotheses", "unknown_dependencies", "gap_premises"}:
            continue
        if value.strip():
            out[key] = value
    return out


def _anchor_claim_field(fields: dict[str, str], field: str, quote: str) -> str | None:
    if field in fields and locate_span(fields[field], quote)[0]:
        return field
    matches = [k for k, text in fields.items() if locate_span(text, quote)[0]]
    return matches[0] if len(matches) == 1 else None


def extract_independent_claims(draft: Any, runner: LLMRunner) -> list[dict[str, str]]:
    """Independently extract verifiable factual/status claims from the draft without judge state."""
    fields = _draft_text_fields(draft)
    if not fields:
        return []
    try:
        raw = runner.chat_json(
            _EXTRACT_SYSTEM,
            json.dumps({"draft": fields}, ensure_ascii=False),
            max_tokens=2048,
            tag="independent_claim_extract",
        )
    except (json.JSONDecodeError, ValueError):
        return []
    rows = raw.get("claims") if isinstance(raw, dict) else None
    if not isinstance(rows, list):
        return []
    claims: list[dict[str, str]] = []
    seen: set[str] = set()
    valid_kinds = {"prior_fact", "numeric_or_benchmark", "coverage_gap", "result_or_process_status"}
    for item in rows[:16]:
        if not isinstance(item, dict):
            continue
        quote = str(item.get("quote") or "").strip()
        field = str(item.get("field") or "").strip()
        anchored_field = _anchor_claim_field(fields, field, quote)
        if not anchored_field:
            continue
        norm_q = " ".join(quote.lower().split())
        if norm_q in seen:
            continue
        seen.add(norm_q)
        kind = str(item.get("claim_kind") or "prior_fact")
        if kind not in valid_kinds:
            kind = "prior_fact"
        claims.append({
            "id": f"c{len(claims) + 1}",
            "field": anchored_field,
            "quote": quote,
            "claim_kind": kind,
        })
        if len(claims) >= 12:
            break
    return claims


_COMMON_TECH_TOKENS = frozenset({
    "a100", "h100", "v100", "rtx", "cpu", "cpus", "gpu", "gpus", "tpu", "tpus", "cuda", "pytorch", "tensorflow",
    "huggingface", "json", "yaml", "sql", "api", "apis", "sdk", "cli", "auroc", "aurc",
    "aupr", "rmse", "mae", "bleu", "rouge", "iou", "miou", "ece", "nll", "rgb", "hdr",
    "dvs", "uda", "sota", "nlp", "llm", "llms", "vlm", "vlms", "vit", "vit-s", "vit-b",
    "vit-l", "convnext", "convnext-t", "resnet", "clip", "lora", "slic", "bm25", "rag",
    "cot", "sde", "iid", "ood", "llama", "llama-3", "mistral", "qwen", "gpt", "gpt-4",
    "claude", "deepseek", "glm", "coco", "imagenet", "dsec", "cityscapes", "longbench",
    "kv", "qa", "em", "f1", "ci", "mc", "id", "ids", "usd", "stage", "mlp", "eig", "icc",
    "sha-256", "ram", "vram", "flops", "flashattention",
})
_CAMEL_ENTITY_RE = re.compile(r"\b([A-Z][A-Za-z0-9]*(?:[A-Z][a-z0-9]+|[A-Z]{2,}|-[A-Z][A-Za-z0-9]+)[A-Za-z0-9-]*)\b")
_ACRONYM_CITE_RE = re.compile(
    r"\b(?:such as|including|like|vs\.?|versus|beyond|than|from|on|baseline[s]?|method[s]?|dataset[s]?)\s+"
    r"([A-Z][A-Z0-9-]{2,9})\b"
)
_OWN_INTRO_RE = re.compile(
    r"(?i)\b(?:we\s+(?:introduce|propose|present|name|call|design|develop)|proposed\s+(?:method|framework|system|adapter|approach)\s+)"
    r"([A-Z][A-Za-z0-9-]{2,})\b"
)
_PAREN_ACRONYM_RE = re.compile(r"\b[A-Za-z][A-Za-z0-9\s-]{3,50}\(([A-Z][A-Za-z0-9-]{2,12})\)")
_HYPHEN_MODIFIER_RE = re.compile(
    r"^(.+?)-(?:style|based|aligned|centric|filtering|only|compatible|hours?|hr|minutes?|min|seconds?|level|aware|guided|free|biased|memory|bound|masked|driven|backed)$",
    re.I,
)
_HEDGE_RE = re.compile(
    r"(?i)\b(?:unverified|not verified|cannot be verified|could not verify|absent from supplied|"
    r"not in supplied|not among (?:the )?supplied|external|outside (?:the )?supplied|"
    r"if publicly available|if confirmed|if available|conditional on|contingent on|to-verify|verified before)\b"
)


def audit_unverified_entities(
    draft: Any,
    task: TaskSpec,
    *,
    prior_work: list[PriorWorkItem] | None = None,
) -> list[dict[str, str]]:
    """Deterministically list named external systems/datasets absent from task prompt and supplied evidence."""
    fields = _draft_text_fields(draft)
    if not fields:
        return []
    papers = task.prior_work if prior_work is None else prior_work
    verified_chunks = [task.title, task.prompt_context]
    for p in papers:
        verified_chunks.extend(
            x for x in (p.title, p.abstract, p.key_contribution, p.important_result, p.limitation, p.fulltext_snippet)
            if x
        )
    verified_blob = "\n".join(verified_chunks).lower()
    verified_spaced = verified_blob.replace("-", " ")

    # Exempt the proposal's own title name, "We introduce X", and inline defined acronyms like "(QPR)".
    own_names: set[str] = set()
    title = fields.get("title", "")
    if ":" in title:
        prefix = title.split(":", 1)[0].strip()
        for tok in re.findall(r"[A-Za-z0-9-]+", prefix):
            own_names.add(tok.lower())
    for k, text in fields.items():
        if k in {"hypothesis", "method", "approach", "idea", "title"} or k.startswith("method"):
            for m in _OWN_INTRO_RE.finditer(text):
                own_names.add(m.group(1).lower())
        for m in _PAREN_ACRONYM_RE.finditer(text):
            own_names.add(m.group(1).lower())

    found: dict[str, dict[str, str]] = {}
    for field, text in fields.items():
        if field == "title" or field.startswith(("resource_plan", "resource_requirements")):
            continue
        for sent in re.split(r"(?<=[.!?;])\s+|\n+", text):
            s = sent.strip()
            if not s or _HEDGE_RE.search(s):
                continue
            candidates = list(_CAMEL_ENTITY_RE.findall(s)) + list(_ACRONYM_CITE_RE.findall(s))
            for ent in candidates:
                mod = _HYPHEN_MODIFIER_RE.match(ent)
                if mod:
                    ent = mod.group(1)
                if "-" in ent:
                    prefix, suffix = ent.split("-", 1)
                    if prefix.lower() in _COMMON_TECH_TOKENS and suffix.islower():
                        ent = prefix
                low = ent.lower().strip("-")
                if len(low) < 3 or low in _COMMON_TECH_TOKENS or low in own_names:
                    continue
                if re.fullmatch(r"[A-Z]?\d+[A-Za-z0-9-]*", ent):
                    continue
                if low in verified_blob or low.replace("-", " ") in verified_spaced:
                    continue
                if low not in found:
                    found[low] = {"entity": ent, "field": field, "quote": s[:200]}
    return list(found.values())[:10]


def _prioritize_passages(parts: list[str], claim_terms: set[str], max_chars: int = 4500) -> str:
    joined = "\n".join(parts)
    if len(joined) <= max_chars:
        return joined
    sentences: list[str] = []
    seen: set[str] = set()
    for part in parts:
        for s in re.split(r"(?<=[.!?])\s+|\n+", part):
            st = s.strip()
            if not st:
                continue
            norm = " ".join(st.lower().split())
            if norm not in seen:
                seen.add(norm)
                sentences.append(st)
    if not sentences:
        return joined[:max_chars]

    def _score(idx_s: tuple[int, str]) -> tuple[int, int, int]:
        idx, sent = idx_s
        toks = set(re.findall(r"[a-z0-9%.]+", sent.lower()))
        overlap = len(toks & claim_terms) if claim_terms else 0
        has_num = 1 if re.search(r"\d+(?:\.\d+)?%?", sent) else 0
        return (-(overlap * 3 + has_num), idx, len(sent))

    chosen: list[tuple[int, str]] = []
    total = 0
    for idx, sent in sorted(enumerate(sentences), key=_score):
        need = len(sent) + (1 if chosen else 0)
        if total + need > max_chars:
            continue
        chosen.append((idx, sent))
        total += need
    chosen.sort(key=lambda pair: pair[0])
    return "\n".join(s for _, s in chosen) if chosen else joined[:max_chars]


def _evidence_packet_for_audit(
    task: TaskSpec,
    prior_work: list[PriorWorkItem] | None = None,
    *,
    claims: list[dict[str, str]] | None = None,
) -> list[dict[str, str]]:
    papers = task.prior_work if prior_work is None else prior_work
    claim_terms = {
        tok
        for c in (claims or [])
        for tok in re.findall(r"[a-z0-9%.]+", (c.get("quote") or "").lower())
        if len(tok) >= 2
    }
    packet: list[dict[str, str]] = []
    for paper in papers:
        if task.strict_evidence and not admissible(task, paper):
            continue
        parts = [
            p.strip()
            for p in (paper.key_contribution, paper.important_result, paper.limitation,
                      paper.fulltext_snippet, paper.abstract)
            if (p or "").strip()
        ]
        if parts:
            packet.append({
                "paper_id": paper.id,
                "title": paper.title,
                "date": paper.publication_date or paper.year or "",
                "date_basis": paper.date_basis,
                "passages": _prioritize_passages(parts, claim_terms, max_chars=4500),
            })
    return packet


_THRESHOLD_RE = re.compile(
    r"(?i)(?:at\s+least|>=|<=|>|<|drop\s+of|gain\s+of|loss\s+of|margin\s+of|improve\w*\s+by)\s*"
    r"([-+]?\d+(?:\.\d+)?)\s*(%|pp|percentage\s+points?|absolute\s+(?:[a-z-]+\s+){0,2}points?|points?|em|miou|iou|f1|auroc|aurc)\b"
)


def _unit_family(unit: str) -> str:
    u = unit.lower()
    if "%" in u or "percent" in u or "pp" in u:
        return "percent_or_pp"
    if "point" in u or u == "em":
        return "points"
    return u


def _threshold_values(text: str) -> dict[str, set[float]]:
    by_unit: dict[str, set[float]] = {}
    for num_s, unit in _THRESHOLD_RE.findall(text or ""):
        try:
            val = float(num_s)
        except ValueError:
            continue
        by_unit.setdefault(_unit_family(unit), set()).add(val)
    return by_unit


def audit_draft_consistency(fields: dict[str, str], start_idx: int = 1) -> list[FactualFinding]:
    """Catch unambiguous internal inconsistencies such as copied falsifiers or mismatched effect thresholds."""
    extra: list[FactualFinding] = []
    idx = start_idx

    sup = (fields.get("supporting_result") or "").strip()
    fal = (fields.get("falsifying_result") or "").strip()
    if sup and fal:
        s_toks = set(re.findall(r"[a-z0-9]{3,}", sup.lower()))
        f_toks = set(re.findall(r"[a-z0-9]{3,}", fal.lower()))
        if s_toks and f_toks:
            jaccard = len(s_toks & f_toks) / len(s_toks | f_toks)
            if jaccard >= 0.72:
                extra.append(
                    FactualFinding(
                        id=f"k{idx}",
                        field="falsifying_result",
                        quote=fal[:320],
                        claim_kind="internal_consistency",
                        label="defect",
                        defect_type="internal_inconsistency",
                        basis=sup[:320],
                        reason="supporting_result and falsifying_result overlap excessively instead of defining opposite decision outcomes.",
                    )
                )
                idx += 1

    hyp = (fields.get("hypothesis") or "").strip()
    if hyp:
        hyp_vals = _threshold_values(hyp)
        decision_blob = "\n".join(
            fields.get(k) or ""
            for k in ("decisive_metric", "supporting_result", "falsifying_result", "decision_rules[1]", "analysis.interpretation_rules[1]")
            if fields.get(k)
        ).strip()
        if hyp_vals and decision_blob:
            dec_vals = _threshold_values(decision_blob)
            for unit_fam, h_nums in hyp_vals.items():
                d_nums = dec_vals.get(unit_fam)
                if d_nums and h_nums.isdisjoint(d_nums):
                    extra.append(
                        FactualFinding(
                            id=f"k{idx}",
                            field="hypothesis",
                            quote=hyp[:320],
                            claim_kind="internal_consistency",
                            label="defect",
                            defect_type="internal_inconsistency",
                            basis=decision_blob[:320],
                            reason=(
                                f"Quantitative effect threshold mismatch between hypothesis ({sorted(h_nums)}) "
                                f"and metric/decision rules ({sorted(d_nums)}) for {unit_fam}."
                            ),
                        )
                    )
                    idx += 1
                    break
    return extra


def _deterministic_process_defects(
    fields: dict[str, str],
    task: TaskSpec,
    packet: list[dict[str, str]],
    existing: list[FactualFinding],
) -> list[FactualFinding]:
    """Catch unambiguous unattested completed-process claims not present in task or evidence."""
    allowed_blob = (task.prompt_context + "\n" + " ".join(p["passages"] for p in packet)).lower()
    extra: list[FactualFinding] = []
    existing_quotes = {" ".join(f.quote.lower().split()) for f in existing}
    idx = len(existing) + 1
    for defect_type, pattern in _PROCESS_HYPE_PATTERNS:
        if pattern.search(allowed_blob):
            continue
        for field, text in fields.items():
            for match in pattern.finditer(text):
                start = max(0, text.rfind(".", 0, match.start()) + 1)
                end = text.find(".", match.end())
                clause = text[start : (end + 1 if end >= 0 else len(text))].strip()
                norm = " ".join(clause.lower().split())
                if not clause or norm in existing_quotes:
                    continue
                existing_quotes.add(norm)
                extra.append(
                    FactualFinding(
                        id=f"d{idx}",
                        field=field,
                        quote=clause[:320],
                        claim_kind="result_or_process_status",
                        label="defect",
                        defect_type=defect_type,
                        basis=clause[:320],
                        reason="Asserts an unattested completed process status not established by task or evidence.",
                    )
                )
                idx += 1
                break
    return extra


def audit_independent_claims(
    draft: Any,
    task: TaskSpec,
    runner: LLMRunner,
    *,
    prior_work: list[PriorWorkItem] | None = None,
    claims: list[dict[str, str]] | None = None,
) -> list[FactualFinding]:
    """Run independent claim extraction + evidence/numeric/coverage/status verification."""
    fields = _draft_text_fields(draft)
    if not fields:
        return []
    extracted = list(claims) if claims is not None else extract_independent_claims(fields, runner)
    packet = _evidence_packet_for_audit(task, prior_work, claims=extracted)

    findings: list[FactualFinding] = []
    if extracted:
        payload = {
            "task": {"title": task.title, "cutoff_date": task.cutoff_date, "budget": task.budget},
            "draft": fields,
            "claims": extracted,
            "papers": packet,
        }
        try:
            raw = runner.chat_json(
                _AUDIT_SYSTEM,
                json.dumps(payload, ensure_ascii=False),
                max_tokens=3072,
                tag="independent_claim_audit",
            )
        except (json.JSONDecodeError, ValueError):
            raw = {}
        rows = raw.get("findings") if isinstance(raw, dict) else None
        by_id = {c["id"]: c for c in extracted}
        passages_by_id = {p["paper_id"]: p["passages"] for p in packet}
        draft_blob = "\n".join(fields.values())
        valid_defects = set(_DEFECT_PRIORITY)
        if isinstance(rows, list):
            for row in rows:
                if not isinstance(row, dict) or row.get("id") not in by_id:
                    continue
                claim_meta = by_id[row["id"]]
                label = str(row.get("label") or "unknown")
                if label not in ("supported", "defect", "unknown"):
                    continue
                defect_type = str(row.get("defect_type") or "none")
                if defect_type not in valid_defects:
                    defect_type = "none"
                valid_ev: list[EvidenceQuote] = []
                for ev in row.get("evidence") or []:
                    if not isinstance(ev, dict):
                        continue
                    pid = str(ev.get("paper_id") or "").strip()
                    q = str(ev.get("quote") or "").strip()
                    if pid in passages_by_id and q and quote_in_passage(q, passages_by_id[pid]):
                        valid_ev.append(EvidenceQuote(paper_id=pid, quote=q))
                basis = str(row.get("basis") or "").strip()
                if basis and not locate_span(draft_blob, basis)[0]:
                    basis = ""
                if label == "defect" and not valid_ev and not basis:
                    if defect_type in {"prediction_as_observed", "unattested_process"}:
                        basis = claim_meta["quote"]
                    else:
                        label = "unknown"
                        defect_type = "none"
                if label != "defect":
                    defect_type = "none"
                elif defect_type == "none":
                    defect_type = "attribution" if valid_ev else "prediction_as_observed"
                findings.append(
                    FactualFinding(
                        id=claim_meta["id"],
                        field=claim_meta["field"],
                        quote=claim_meta["quote"],
                        claim_kind=claim_meta.get("claim_kind", "prior_fact"),  # type: ignore[arg-type]
                        label=label,  # type: ignore[arg-type]
                        defect_type=defect_type,  # type: ignore[arg-type]
                        evidence=valid_ev,
                        basis=basis,
                        reason=str(row.get("reason") or "")[:300],
                    )
                )

    findings.extend(_deterministic_process_defects(fields, task, packet, findings))
    findings.extend(audit_draft_consistency(fields, start_idx=len(findings) + 1))
    rank = {"defect": 0, "unknown": 1, "supported": 2}
    findings.sort(
        key=lambda f: (
            rank[f.label],
            _DEFECT_PRIORITY.get(f.defect_type, 8) if f.label == "defect" else 9,
            0 if (f.evidence or f.basis) else 1,
            f.id,
        )
    )
    return findings
