"""Runtime adapter around UIUC ClaimNov with chat-only gateway routing and BM25 fallback."""
from __future__ import annotations

import contextvars
import hashlib
import os
import re
import sys
import threading
from pathlib import Path
from typing import Any

import numpy as np
from rank_bm25 import BM25Okapi

from .llm_client import LLMRunner
from .schemas import ClaimNovAuditResult, CompositionalRemainderResult, NormalizedProposal, PriorWorkItem

CLAIMNOV_SRC = Path(
    os.environ.get("CLAIMNOV_SRC")
    or next(
        (str(p) for p in (Path(__file__).resolve().parents[3] / "claimnov" / "src", Path("/workspace/edara/staging/04_external_eval_refs/claimnov/src")) if p.is_dir()),
        "/workspace/edara/staging/04_external_eval_refs/claimnov/src",
    )
)
# Install ContextVar-aware hooks once so --workers threads run ClaimNov concurrently without global lock blocking.
_INIT_LOCK = threading.Lock()
_HOOKS_INSTALLED = False
_ORIG_HOOKS: dict[str, Any] = {}
_ACTIVE_RUNNER: contextvars.ContextVar[LLMRunner | None] = contextvars.ContextVar("claimnov_runner", default=None)
_ACTIVE_EMBED: contextvars.ContextVar[bool] = contextvars.ContextVar("claimnov_embed", default=False)
_MEMO_LOCK = threading.Lock()
_FACET_MEMO: dict[tuple[str, str], list[dict[str, Any]]] = {}


def _ensure_claimnov_imported():
    src_str = str(CLAIMNOV_SRC)
    if src_str not in sys.path:
        sys.path.insert(0, src_str)
    from novelty import config, entail, extract, gate, llm as cn_llm, pipeline, retrieve

    return config, entail, extract, gate, cn_llm, pipeline, retrieve


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", (text or "").lower())


def _paper_cache_key(paper: dict[str, Any]) -> tuple[str, str]:
    pid = str(paper.get("id") or paper.get("title") or "?")
    sig = hashlib.sha256(f"{paper.get('title', '')}\n{paper.get('abstract', '')}".encode("utf-8")).hexdigest()[:16]
    return pid, sig


def _install_hooks():
    global _HOOKS_INSTALLED
    config, entail, extract, gate, cn_llm, pipeline, retrieve = _ensure_claimnov_imported()
    if _HOOKS_INSTALLED:
        return config, entail, extract, gate, cn_llm, pipeline, retrieve
    with _INIT_LOCK:
        if _HOOKS_INSTALLED:
            return config, entail, extract, gate, cn_llm, pipeline, retrieve
        _ORIG_HOOKS.update({
            "chat_json": cn_llm.chat_json,
            "chat_text": cn_llm.chat_text,
            "gate_chat": cn_llm.gate_chat,
            "build_pool": retrieve.build_pool,
            "top_k_works": retrieve.top_k_works,
            "judge_all": entail.judge_all,
        })

        def _ctx_chat_json(system: str, user: str, model: str = config.EXTRACT_MODEL, max_tokens: int = 800) -> dict:
            runner = _ACTIVE_RUNNER.get(None)
            if runner is None:
                return _ORIG_HOOKS["chat_json"](system, user, model=model, max_tokens=max_tokens)
            try:
                return runner.chat_json(system, user, max_tokens=max(max_tokens, 2048), tag="claimnov_extract")
            except (ValueError, Exception) as exc:
                if type(exc).__name__ == "JSONDecodeError" or isinstance(exc, ValueError):
                    print(f"[warn] claimnov_extract JSON parse failed ({exc}); returning empty claims.", flush=True)
                    return {"claims": []}
                raise

        def _ctx_chat_text(prompt: str, model: str = config.JUDGE_MODEL, max_tokens: int = 8) -> str:
            runner = _ACTIVE_RUNNER.get(None)
            if runner is None:
                return _ORIG_HOOKS["chat_text"](prompt, model=model, max_tokens=max_tokens)
            return runner.chat_text(
                "You output only one integer from 1 to 5.",
                prompt,
                max_tokens=max_tokens,
                tag="claimnov_entail",
            )

        def _ctx_gate_chat(prompt: str, model: str = config.GATE_JUDGE_MODEL, max_tokens: int = 300) -> str:
            runner = _ACTIVE_RUNNER.get(None)
            if runner is None:
                return _ORIG_HOOKS["gate_chat"](prompt, model=model, max_tokens=max_tokens)
            return runner.chat_text(
                "Return valid JSON only.",
                prompt,
                max_tokens=max_tokens,
                tag="claimnov_veto",
            )

        def _ctx_build_pool(corpus: list[dict], max_workers: int = 4) -> list[dict]:
            runner = _ACTIVE_RUNNER.get(None)
            if runner is None or _ACTIVE_EMBED.get(False):
                return _ORIG_HOOKS["build_pool"](corpus, max_workers=max_workers)
            live_memo = getattr(runner, "dry_run", True) is False
            facets: list[dict[str, Any]] = []
            for paper in corpus:
                ck = _paper_cache_key(paper)
                claims = None
                if live_memo:
                    with _MEMO_LOCK:
                        claims = _FACET_MEMO.get(ck)
                if claims is None:
                    claims = extract.extract_claims(paper)
                    if live_memo and claims:
                        with _MEMO_LOCK:
                            _FACET_MEMO[ck] = claims
                for c in claims:
                    facets.append({
                        **c,
                        "paper_id": paper.get("id") or paper.get("title", "?"),
                        "paper_title": paper.get("title", ""),
                        "paper_abstract": paper.get("abstract", ""),
                    })
            return facets

        def _ctx_top_k_works(target_facet: dict, pool: list[dict], k: int = config.TOP_K) -> list[dict]:
            if _ACTIVE_RUNNER.get(None) is None or _ACTIVE_EMBED.get(False):
                return _ORIG_HOOKS["top_k_works"](target_facet, pool, k=k)
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

        def _ctx_judge_all(target: dict, cands: list[dict], max_workers: int = 4) -> list[dict]:
            if _ACTIVE_RUNNER.get(None) is None:
                return _ORIG_HOOKS["judge_all"](target, cands, max_workers=max_workers)
            # Run sequentially within one runner so RecordedRunner sequences stay deterministic
            # while top-level --workers threads run completely concurrently.
            out = []
            for c in cands:
                verdict = entail.judge(target, c)
                cleaned = {k: v for k, v in c.items() if not k.startswith("_emb")}
                out.append({**cleaned, **verdict})
            return sorted(out, key=lambda x: -x["e"])

        cn_llm.chat_json = _ctx_chat_json
        cn_llm.chat_text = _ctx_chat_text
        cn_llm.gate_chat = _ctx_gate_chat
        retrieve.build_pool = _ctx_build_pool
        retrieve.top_k_works = _ctx_top_k_works
        entail.judge_all = _ctx_judge_all
        _HOOKS_INSTALLED = True
    return config, entail, extract, gate, cn_llm, pipeline, retrieve


def run_claimnov_audit(
    proposal: NormalizedProposal,
    prior_corpus: list[PriorWorkItem],
    judge_runner: LLMRunner,
    *,
    enable_embeddings: bool = False,
) -> ClaimNovAuditResult:
    """Invoke ClaimNov typed claim extraction, same-dimension retrieval, entailment, and profile_veto."""
    _, _, _, gate, cn_llm, pipeline, retrieve = _install_hooks()

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

    ep = getattr(judge_runner, "endpoint", None)
    model_name = getattr(ep, "model", "offline-model")
    use_embed = False
    retriever_mode = "bm25_fallback"
    if enable_embeddings and ep and os.environ.get(getattr(ep, "credential_env", "")):
        os.environ.setdefault("OPENAI_API_KEY", ep.resolve_api_key())
        if getattr(ep, "base_url", None):
            os.environ.setdefault("OPENAI_BASE_URL", ep.base_url)
        use_embed = True
        retriever_mode = "openai_embeddings"

    tok_runner = _ACTIVE_RUNNER.set(judge_runner)
    tok_embed = _ACTIVE_EMBED.set(use_embed)
    try:
        if use_embed:
            try:
                pool = retrieve.build_pool(prior_docs)
            except Exception:
                _ACTIVE_EMBED.set(False)
                retriever_mode = "bm25_fallback"
                pool = retrieve.build_pool(prior_docs)
        else:
            pool = retrieve.build_pool(prior_docs)

        profile = pipeline.score_paper(target_doc, pool)
        if profile.get("novelty") is None:
            return ClaimNovAuditResult(retriever_mode=retriever_mode, novelty_score=0.0, veto=False)
        verdict = gate.profile_veto(
            target_doc,
            prior_docs,
            profile,
            model=model_name,
            chat=lambda prompt: cn_llm.gate_chat(prompt, model=model_name),
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
        _ACTIVE_EMBED.reset(tok_embed)
        _ACTIVE_RUNNER.reset(tok_runner)


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
