"""Runtime adapter around UIUC ClaimNov with chat-only gateway routing and BM25 fallback."""
from __future__ import annotations

import json
import os
import re
import sys
import threading
from pathlib import Path
from typing import Any

import numpy as np
from rank_bm25 import BM25Okapi

from .llm_client import LLMRunner, extract_json_dict
from .schemas import ClaimNovAuditResult, CompositionalRemainderResult, NormalizedProposal, PriorWorkItem

CLAIMNOV_SRC = Path("/workspace/edara/staging/04_external_eval_refs/claimnov/src")
# Serialize ClaimNov module monkey-patching across --workers threads while preserving internal thread pools.
_CLAIMNOV_LOCK = threading.Lock()


def _ensure_claimnov_imported():
    src_str = str(CLAIMNOV_SRC)
    if src_str not in sys.path:
        sys.path.insert(0, src_str)
    from novelty import config, entail, extract, gate, llm as cn_llm, pipeline, retrieve

    return config, entail, extract, gate, cn_llm, pipeline, retrieve


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", (text or "").lower())


def run_claimnov_audit(
    proposal: NormalizedProposal,
    prior_corpus: list[PriorWorkItem],
    judge_runner: LLMRunner,
    *,
    enable_embeddings: bool = False,
) -> ClaimNovAuditResult:
    """Invoke ClaimNov typed claim extraction, same-dimension retrieval, entailment, and profile_veto."""
    with _CLAIMNOV_LOCK:
        config, _, extract, gate, cn_llm, pipeline, retrieve = _ensure_claimnov_imported()

        # Bind ClaimNov LLM hooks strictly in-memory so upstream ClaimNov files remain unmodified.
        orig_chat_json = cn_llm.chat_json
        orig_chat_text = cn_llm.chat_text
        orig_gate_chat = cn_llm.gate_chat
        orig_build_pool = retrieve.build_pool
        orig_top_k = retrieve.top_k_works

        retriever_mode = "openai_embeddings" if enable_embeddings else "bm25_fallback"

        def _patched_chat_json(system: str, user: str, model: str = config.EXTRACT_MODEL, max_tokens: int = 800) -> dict:
            try:
                return judge_runner.chat_json(system, user, max_tokens=max(max_tokens, 2048), tag="claimnov_extract")
            except json.JSONDecodeError as exc:
                print(f"[warn] {proposal.task_id} claimnov_extract JSON parse failed ({exc}); returning empty claims.", flush=True)
                return {"claims": []}

        def _patched_chat_text(prompt: str, model: str = config.JUDGE_MODEL, max_tokens: int = 8) -> str:
            return judge_runner.chat_text(
                "You output only one integer from 1 to 5.",
                prompt,
                max_tokens=max_tokens,
                tag="claimnov_entail",
            )

        def _patched_gate_chat(prompt: str, model: str = config.GATE_JUDGE_MODEL, max_tokens: int = 300) -> str:
            return judge_runner.chat_text(
                "Return valid JSON only.",
                prompt,
                max_tokens=max_tokens,
                tag="claimnov_veto",
            )

        def _bm25_build_pool(corpus: list[dict], max_workers: int = 4) -> list[dict]:
            facets: list[dict] = []
            for paper in corpus:
                claims = extract.extract_claims(paper)
                for c in claims:
                    facets.append(
                        {
                            **c,
                            "paper_id": paper.get("id") or paper.get("title", "?"),
                            "paper_title": paper.get("title", ""),
                            "paper_abstract": paper.get("abstract", ""),
                        }
                    )
            return facets

        def _bm25_top_k_works(target_facet: dict, pool: list[dict], k: int = config.TOP_K) -> list[dict]:
            cands = [f for f in pool if config.ANY_DIMENSION or f["dimension"] == target_facet["dimension"]]
            if not cands:
                return []
            if k == config.TOP_K:
                k = config.top_k_for(len({f["paper_id"] for f in cands}))
            docs = [_tokenize(f"{f.get('attribute', '')} {f.get('value', '')} {f.get('claim', '')}") for f in cands]
            q = _tokenize(f"{target_facet.get('attribute', '')} {target_facet.get('value', '')} {target_facet.get('claim', '')}")
            if q and any(docs):
                scores = BM25Okapi(docs).get_scores(q)
                mx = float(np.max(scores)) if len(scores) else 1.0
                norm = (scores / (mx + 1e-9)) if mx > 0 else scores
            else:
                norm = np.zeros(len(cands), dtype=float)
            best_per_paper: dict[str, dict] = {}
            for f, sc in zip(cands, norm):
                f_copy = dict(f)
                f_copy["_cos"] = float(sc)
                pid = str(f_copy["paper_id"])
                prev = best_per_paper.get(pid)
                if prev is None or f_copy["_cos"] > prev["_cos"]:
                    best_per_paper[pid] = f_copy
            ranked = sorted(best_per_paper.values(), key=lambda x: -x["_cos"])
            return ranked[:k]

        cn_llm.chat_json = _patched_chat_json
        cn_llm.chat_text = _patched_chat_text
        cn_llm.gate_chat = _patched_gate_chat

        try:
            target_doc = {
                "id": proposal.task_id,
                "title": proposal.title,
                "abstract": (
                    f"Gap: {proposal.literature_gap}\n"
                    f"Hypothesis: {proposal.hypothesis}\n"
                    f"Method: {proposal.method}\n"
                    f"Validation: {proposal.minimal_test} Metric: {proposal.decisive_metric}"
                ),
            }
            prior_docs = [
                {
                    "id": p.id,
                    "title": p.title,
                    "abstract": p.abstract or p.key_contribution or p.fulltext_snippet[:1200],
                }
                for p in prior_corpus
                if (p.abstract or p.key_contribution or p.fulltext_snippet)
            ]
            if not prior_docs:
                return ClaimNovAuditResult(retriever_mode="empty_corpus", novelty_score=0.0, veto=False)

            if enable_embeddings and os.environ.get(judge_runner.endpoint.credential_env):
                try:
                    os.environ.setdefault("OPENAI_API_KEY", judge_runner.endpoint.resolve_api_key())
                    if judge_runner.endpoint.base_url:
                        os.environ.setdefault("OPENAI_BASE_URL", judge_runner.endpoint.base_url)
                    pool = orig_build_pool(prior_docs)
                except Exception:
                    retriever_mode = "bm25_fallback"
                    retrieve.build_pool = _bm25_build_pool
                    retrieve.top_k_works = _bm25_top_k_works
                    pool = _bm25_build_pool(prior_docs)
            else:
                retriever_mode = "bm25_fallback"
                retrieve.build_pool = _bm25_build_pool
                retrieve.top_k_works = _bm25_top_k_works
                pool = _bm25_build_pool(prior_docs)

            profile = pipeline.score_paper(target_doc, pool)
            if profile.get("novelty") is None:
                return ClaimNovAuditResult(retriever_mode=retriever_mode, novelty_score=0.0, veto=False)
            verdict = gate.profile_veto(
                target_doc,
                prior_docs,
                profile,
                model=judge_runner.endpoint.model,
                chat=_patched_gate_chat,
            )
            return ClaimNovAuditResult(
                retriever_mode=retriever_mode,
                novelty_score=float(profile.get("novelty", 1.0)),
                veto=bool(verdict.get("veto", False)),
                veto_evidence=verdict.get("evidence"),
                claims_count=int(profile.get("n_claims", len(profile.get("claims", [])))),
                claims=list(profile.get("claims", [])),
            )
        finally:
            cn_llm.chat_json = orig_chat_json
            cn_llm.chat_text = orig_chat_text
            cn_llm.gate_chat = orig_gate_chat
            retrieve.build_pool = orig_build_pool
            retrieve.top_k_works = orig_top_k


def analyze_compositional_remainder(
    proposal: NormalizedProposal,
    claimnov: ClaimNovAuditResult,
) -> CompositionalRemainderResult:
    """Separate multi-paper covered components from the distinct compositional remainder."""
    covered: list[dict[str, Any]] = []
    uncovered_claims: list[str] = []
    attribution_titles: list[str] = []
    seen_titles: set[str] = set()

    for c in claimnov.claims:
        level = c.get("max_entailment_level")
        if level is None:
            continue
        max_lvl = int(level)
        top_matches = c.get("top_matches") or []
        best = top_matches[0] if top_matches else {}
        if max_lvl >= 4 and best:
            title = str(best.get("paper_title") or "").strip()
            covered.append(
                {
                    "claim": str(c.get("claim") or ""),
                    "dimension": str(c.get("dimension") or ""),
                    "covering_paper": title,
                    "paper_id": str(best.get("paper_id") or ""),
                    "entailment_level": max_lvl,
                }
            )
            if title and title.lower() not in seen_titles:
                seen_titles.add(title.lower())
                attribution_titles.append(title)
        elif max_lvl <= 2:
            uncovered_claims.append(f"[{c.get('dimension', 'claim')}] {c.get('claim', '')}")

    # Check if already credited in proposal text (related_work / method / hypothesis).
    proposal_text = f"{proposal.related_work}\n{proposal.method}\n{proposal.hypothesis}".lower()
    missing_attributions = [t for t in attribution_titles if t.lower()[:28] not in proposal_text]

    remainder_text = ""
    if uncovered_claims:
        remainder_text = "; ".join(uncovered_claims[:3])
    elif not claimnov.veto and proposal.hypothesis:
        remainder_text = proposal.hypothesis[:280]

    is_comp = bool(covered and uncovered_claims and not claimnov.veto)
    return CompositionalRemainderResult(
        covered_components=covered,
        compositional_remainder=remainder_text,
        is_compositional_novelty=is_comp,
        attribution_requirements=missing_attributions,
    )
