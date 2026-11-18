"""Leak-proof Pre-Cutoff literature scout for citation verification and Gap-Premise counterexample retrieval."""
from __future__ import annotations

from contextvars import ContextVar

import hashlib
import json
import os
import re
import unicodedata
import urllib.parse
import urllib.request
from typing import Any

from rank_bm25 import BM25Okapi

from .schemas import PriorWorkItem, TaskSpec
from .tasks_loader import normalize_title_hash


stage_search = ContextVar("stage_search", default=None)


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", (text or "").lower())


def _candidate_identity_hashes(title: str, identifier: str = "") -> set[str]:
    out: set[str] = set()
    if title:
        out.add(normalize_title_hash(title))
        unicode_norm = " ".join(re.findall(r"\w+", unicodedata.normalize("NFKC", title).casefold()))
        if unicode_norm:
            out.add(hashlib.sha256(f"title:{unicode_norm}".encode("utf-8")).hexdigest())
    if identifier:
        text = urllib.parse.unquote(identifier).strip().casefold()
        if text:
            out.add(hashlib.sha256(f"identity:{text}".encode("utf-8")).hexdigest())
            doi = re.search(r"10\.\d{4,9}/[^\s?#]+", text)
            if doi:
                out.add(hashlib.sha256(f"doi:{doi.group(0).rstrip('./')}".encode("utf-8")).hexdigest())
            arxiv = re.search(
                r"(?:arxiv[:/]|abs/|pdf/|^)(\d{4}\.\d{4,5}|[a-z.-]+/\d{7})(?:v\d+)?(?:\.pdf)?(?:$|[?#/])",
                text,
            )
            if arxiv:
                out.add(hashlib.sha256(f"arxiv:{arxiv.group(1)}".encode("utf-8")).hexdigest())
    return out


def is_forbidden_or_post_cutoff(
    *,
    title: str,
    pub_date: str | None,
    cutoff_date: str,
    forbidden_hashes: set[str],
    identifier: str = "",
) -> bool:
    if _candidate_identity_hashes(title, identifier) & forbidden_hashes:
        return True
    if pub_date and cutoff_date:
        # Compare YYYY-MM-DD prefixes lexicographically when date string is valid ISO/year.
        d = pub_date.strip()[:10]
        if len(d) == 4 and d.isdigit():
            d = f"{d}-12-31"
        if re.match(r"^\d{4}-\d{2}-\d{2}$", d) and d > cutoff_date:
            return True
    return False


def _local_bm25_search(query: str, corpus: list[PriorWorkItem], *, top_k: int = 5) -> list[PriorWorkItem]:
    if not corpus:
        return []
    docs = [
        _tokenize(f"{p.title} {p.abstract} {p.key_contribution} {p.limitation} {p.important_result} {p.fulltext_snippet[:1500]}")
        for p in corpus
    ]
    q_toks = _tokenize(query)
    if not q_toks or not any(docs):
        return corpus[:top_k]
    bm25 = BM25Okapi(docs)
    scores = bm25.get_scores(q_toks)
    ranked = sorted(zip(scores, corpus), key=lambda x: -float(x[0]))
    return [p for _, p in ranked[:top_k]]


def _fetch_openalex_pre_cutoff(
    query: str,
    *,
    cutoff_date: str,
    forbidden_hashes: set[str],
    limit: int = 5,
    timeout: float = 12.0,
) -> list[PriorWorkItem]:
    """Key-free OpenAlex lookup strictly bounded to publication_date <= cutoff_date."""
    q = urllib.parse.quote(query[:220])
    url = (
        f"https://api.openalex.org/works?search={q}"
        f"&filter=to_publication_date:{cutoff_date}"
        f"&per-page={max(limit * 2, 8)}"
        f"&select=id,title,publication_date,abstract_inverted_index"
    )
    req = urllib.request.Request(url, headers={"User-Agent": "EDARA-MVP-Eval/0.1 (research benchmark)"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception:
        return []

    out: list[PriorWorkItem] = []
    for item in data.get("results") or []:
        title = str(item.get("title") or "").strip()
        pub_date = str(item.get("publication_date") or "")
        ident = str(item.get("id") or "")
        if not title or is_forbidden_or_post_cutoff(
            title=title, pub_date=pub_date, cutoff_date=cutoff_date, forbidden_hashes=forbidden_hashes, identifier=ident
        ):
            continue
        inv = item.get("abstract_inverted_index") or {}
        abstract = _reconstruct_openalex_abstract(inv)
        out.append(
            PriorWorkItem(
                id=str(item.get("id") or f"oa_{len(out)}"),
                title=title,
                abstract=abstract,
                year=pub_date[:4] if pub_date else None,
                source="openalex",
            )
        )
        if len(out) >= limit:
            break
    return out


def _reconstruct_openalex_abstract(inv: dict[str, list[int]]) -> str:
    if not isinstance(inv, dict) or not inv:
        return ""
    pos_word: list[tuple[int, str]] = []
    for word, positions in inv.items():
        if isinstance(positions, list):
            for pos in positions:
                pos_word.append((int(pos), str(word)))
    pos_word.sort()
    return " ".join(w for _, w in pos_word)[:1800]


def _fetch_s2_pre_cutoff(
    query: str,
    *,
    cutoff_date: str,
    forbidden_hashes: set[str],
    limit: int = 5,
    timeout: float = 12.0,
) -> list[PriorWorkItem]:
    """Optional Semantic Scholar lookup (uses SEMANTIC_SCHOLAR_API_KEY / S2_API_KEY if set)."""
    q = urllib.parse.quote(query[:220])
    url = (
        f"https://api.semanticscholar.org/graph/v1/paper/search?query={q}"
        f"&publicationDateOrYear=:{cutoff_date}&limit={max(limit * 2, 8)}"
        f"&fields=paperId,title,abstract,publicationDate,year"
    )
    headers = {"User-Agent": "EDARA-MVP-Eval/0.1"}
    key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY") or os.environ.get("S2_API_KEY")
    if key:
        headers["x-api-key"] = key
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception:
        return []

    out: list[PriorWorkItem] = []
    for item in data.get("data") or []:
        title = str(item.get("title") or "").strip()
        pub_date = str(item.get("publicationDate") or item.get("year") or "")
        ident = str(item.get("paperId") or "")
        if not title or is_forbidden_or_post_cutoff(
            title=title, pub_date=pub_date, cutoff_date=cutoff_date, forbidden_hashes=forbidden_hashes, identifier=ident
        ):
            continue
        out.append(
            PriorWorkItem(
                id=str(item.get("paperId") or f"s2_{len(out)}"),
                title=title,
                abstract=str(item.get("abstract") or "")[:1800],
                year=str(item.get("year") or ""),
                source="semantic_scholar",
            )
        )
        if len(out) >= limit:
            break
    return out


def gather_evidence_for_query(
    query: str,
    task: TaskSpec,
    *,
    live_search: bool = False,
    top_k: int = 5,
) -> list[PriorWorkItem]:
    """Retrieve local evidence first, optionally enriched with leak-checked pre-cutoff OpenAlex/S2 hits."""
    search = stage_search.get()
    if task.strict_evidence:
        from .governance import admissible

        local = [paper for paper in task.prior_work if admissible(task, paper)]
        extra = search(query) if live_search and search else []
        unique = {normalize_title_hash(paper.title): paper for paper in [*extra, *local]}
        return _local_bm25_search(query, list(unique.values()), top_k=top_k)
    forbidden = set(task.forbidden_title_hashes)
    local_hits = _local_bm25_search(query, task.prior_work, top_k=top_k)
    if not live_search:
        return local_hits

    seen_hashes = {normalize_title_hash(p.title) for p in local_hits if p.title}
    extra: list[PriorWorkItem] = []
    for cand in _fetch_openalex_pre_cutoff(query, cutoff_date=task.cutoff_date, forbidden_hashes=forbidden, limit=top_k):
        h = normalize_title_hash(cand.title)
        if h not in seen_hashes:
            seen_hashes.add(h)
            extra.append(cand)
    if len(extra) < 2:
        for cand in _fetch_s2_pre_cutoff(query, cutoff_date=task.cutoff_date, forbidden_hashes=forbidden, limit=top_k):
            h = normalize_title_hash(cand.title)
            if h not in seen_hashes:
                seen_hashes.add(h)
                extra.append(cand)
    return (local_hits + extra)[: max(top_k, 6)]


def extract_relevant_excerpt(query: str, item: PriorWorkItem, *, max_chars: int = 900) -> str:
    """Select the highest-overlap passages across limitation/key fields and fulltext chunks."""
    parts: list[str] = []
    for label, text in (
        ("Contribution", item.key_contribution),
        ("ImportantResult", item.important_result),
        ("Limitation", item.limitation),
    ):
        clean = (text or "").strip()
        if clean:
            parts.append(f"[{label}] {clean[:350]}")

    body = (item.fulltext_snippet or "").strip()
    if body:
        # Slide overlapping windows so evidence deep inside PDFs / topic_docs is never clipped at char 400.
        windows: list[str] = []
        win_size = 360
        stride = 200
        for start in range(0, len(body), stride):
            chunk = body[start : start + win_size].strip()
            if chunk:
                windows.append(chunk)
            if start + win_size >= len(body):
                break
        q_toks = set(_tokenize(query))
        if q_toks and len(windows) > 1:
            docs = [_tokenize(w) for w in windows]
            if any(docs):
                scores = BM25Okapi(docs).get_scores(list(q_toks))
                ranked_idx = sorted(range(len(windows)), key=lambda idx: -float(scores[idx]))
            else:
                ranked_idx = list(range(len(windows)))
        else:
            ranked_idx = list(range(len(windows)))
        selected = [windows[idx] for idx in ranked_idx[:2]]
        parts.append("[MatchedExcerpt] " + " ... ".join(selected))

    combined = " | ".join(parts).strip()
    return combined[:max_chars]


def make_counterexample_query(gap_premise: str) -> str:
    """Strip negation boilerplate from a gap assertion to search for pre-cutoff counterexamples."""
    cleaned = re.sub(
        r"\b(no prior work|existing methods fail to|prior methods ignore|never considered|remain[s]? untested|lacks?)\b",
        "",
        gap_premise,
        flags=re.I,
    )
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" .,:;")
    return cleaned or gap_premise
