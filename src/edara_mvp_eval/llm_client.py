"""OpenAI-compatible LLM wrapper supporting E-DARA ModelProfile JSONs, env vars, and deterministic dry-run."""
from __future__ import annotations

import hashlib
import json
import os
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from openai import OpenAI


@dataclass
class ModelEndpoint:
    model: str = "dry-run-model"
    base_url: str | None = None
    credential_env: str = "OPENAI_API_KEY"
    input_usd_per_m: float = 0.5
    output_usd_per_m: float = 2.0
    extra_body: dict[str, Any] = field(default_factory=dict)
    timeout_seconds: float = 90.0

    @classmethod
    def from_env_or_profile(
        cls,
        *,
        profile_path: str | Path | None = None,
        prefix: str = "EDARA_GEN",
        fallback_model: str = "gpt-4o-mini",
    ) -> "ModelEndpoint":
        if profile_path:
            data = json.loads(Path(profile_path).read_text(encoding="utf-8"))
            pricing = data.get("pricing") or {}
            in_m = float(pricing.get("input_microusd_per_million", 500_000)) / 1_000_000.0
            out_m = float(pricing.get("output_microusd_per_million", 2_000_000)) / 1_000_000.0
            params = data.get("parameters") or {}
            extra_body = {k: params[k] for k in ("reasoning_effort", "thinking") if k in params}
            timeout_s = min(float(data.get("timeout_seconds") or 90.0), 120.0)
            return cls(
                model=str(data["model"]),
                base_url=data.get("base_url"),
                credential_env=str(data.get("credential_env") or "OPENAI_API_KEY"),
                input_usd_per_m=in_m,
                output_usd_per_m=out_m,
                extra_body=extra_body,
                timeout_seconds=timeout_s,
            )
        model = os.environ.get(f"{prefix}_MODEL") or os.environ.get("OPENAI_MODEL") or fallback_model
        base_url = os.environ.get(f"{prefix}_BASE_URL") or os.environ.get("OPENAI_BASE_URL")
        cred_env = os.environ.get(f"{prefix}_CREDENTIAL_ENV") or (
            f"{prefix}_API_KEY" if os.environ.get(f"{prefix}_API_KEY") else "OPENAI_API_KEY"
        )
        return cls(model=model, base_url=base_url, credential_env=cred_env)

    def resolve_api_key(self) -> str:
        return os.environ.get(self.credential_env, "").strip()


def extract_json_dict(text: str) -> dict[str, Any]:
    cleaned = (text or "").strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        obj = json.loads(cleaned)
        return obj if isinstance(obj, dict) else {"value": obj}
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start >= 0 and end > start:
            snippet = cleaned[start : end + 1]
            for candidate in (snippet, re.sub(r",(\s*[}\]])", r"\1", snippet)):
                try:
                    obj = json.loads(candidate)
                    return obj if isinstance(obj, dict) else {"value": obj}
                except json.JSONDecodeError:
                    try:
                        obj = json.loads(candidate, strict=False)
                        return obj if isinstance(obj, dict) else {"value": obj}
                    except json.JSONDecodeError:
                        pass
                # Recover when trailing extra braces or commentary follow a valid top-level object.
                for strict_flag in (True, False):
                    try:
                        obj, _ = json.JSONDecoder(strict=strict_flag).raw_decode(candidate)
                        return obj if isinstance(obj, dict) else {"value": obj}
                    except json.JSONDecodeError:
                        continue
            if candidate.count("{") == candidate.count("}") + 1 and '"dimensions"' in candidate and '"overall_winner"' in candidate:
                fixed = re.sub(r'\},\s*"overall_winner"\s*:', '}},"overall_winner":', candidate, count=1)
                if fixed != candidate:
                    try:
                        obj = json.loads(fixed)
                        return obj if isinstance(obj, dict) else {"value": obj}
                    except json.JSONDecodeError:
                        pass
        raise


@dataclass
class MeteredUsage:
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0


_FAST_STRUCTURED_TAGS = frozenset({
    "claimnov_entail",
    "claimnov_extract",
    "claimnov_veto",
    "aib_batch_judge",
    "aib_dimension_judge",
    "tri_state_fact_audit",
    "gap_falsify",
    "spec_locate",
    "confound_audit",
    "independent_claim_extract",
    "independent_claim_audit",
    "normalize",
    "aib_eval",
})


class LLMRunner:
    """Thin Chat-only client with token/cost tracking and realistic offline dry-run fixtures."""

    def __init__(self, endpoint: ModelEndpoint, *, dry_run: bool = False) -> None:
        self.endpoint = endpoint
        self.dry_run = dry_run
        self.usage = MeteredUsage()
        self._client: OpenAI | None = None
        self._lock = threading.Lock()
        self._supports_json_mode = True

    def get_openai_client(self) -> OpenAI:
        with self._lock:
            if self._client is None:
                key = self.endpoint.resolve_api_key()
                if not key:
                    raise RuntimeError(
                        f"Missing API key in env var {self.endpoint.credential_env!r}. "
                        f"Run with --dry-run to test offline or export {self.endpoint.credential_env}."
                    )
                kwargs: dict[str, Any] = {
                    "api_key": key,
                    "timeout": self.endpoint.timeout_seconds,
                    "max_retries": 2,
                }
                if self.endpoint.base_url:
                    kwargs["base_url"] = self.endpoint.base_url
                self._client = OpenAI(**kwargs)
            return self._client

    def chat_text(
        self,
        system: str,
        user: str,
        *,
        temperature: float = 0.0,
        max_tokens: int = 1600,
        tag: str = "generic",
        json_mode: bool = False,
    ) -> str:
        if self.dry_run:
            with self._lock:
                self.usage.calls += 1
                self.usage.prompt_tokens += 450
                self.usage.completion_tokens += 220
                self.usage.cost_usd += 0.0007
            return _dry_run_response(tag=tag, system=system, user=user)

        client = self.get_openai_client()
        effective_extra_body = dict(self.endpoint.extra_body) if self.endpoint.extra_body else {}
        if "glm" in self.endpoint.model.lower() and tag in _FAST_STRUCTURED_TAGS:
            effective_extra_body["thinking"] = {"type": "disabled"}
        thinking_cfg = effective_extra_body.get("thinking")
        thinking_disabled = isinstance(thinking_cfg, dict) and thinking_cfg.get("type") == "disabled"
        has_thinking = (not thinking_disabled) and (bool(effective_extra_body) or "glm" in self.endpoint.model.lower())
        effective_max_tokens = max(max_tokens + 2048, 4096) if has_thinking else max_tokens
        create_kwargs: dict[str, Any] = {
            "model": self.endpoint.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
            "max_tokens": effective_max_tokens,
        }
        if effective_extra_body:
            create_kwargs["extra_body"] = effective_extra_body
        if json_mode and self._supports_json_mode:
            create_kwargs["response_format"] = {"type": "json_object"}
        try:
            resp = client.chat.completions.create(**create_kwargs)
        except Exception as exc:
            if "response_format" in create_kwargs and "response_format" in str(exc).lower():
                with self._lock:
                    self._supports_json_mode = False
                create_kwargs.pop("response_format", None)
                resp = client.chat.completions.create(**create_kwargs)
            else:
                raise
        u = getattr(resp, "usage", None)
        pt = int(getattr(u, "prompt_tokens", 0) or 0)
        ct = int(getattr(u, "completion_tokens", 0) or 0)
        with self._lock:
            self.usage.calls += 1
            self.usage.prompt_tokens += pt
            self.usage.completion_tokens += ct
            self.usage.cost_usd += (
                (pt / 1_000_000.0) * self.endpoint.input_usd_per_m
                + (ct / 1_000_000.0) * self.endpoint.output_usd_per_m
            )
        return (resp.choices[0].message.content or "").strip()

    def chat_json(self, system: str, user: str, *, temperature: float = 0.0, max_tokens: int = 1800, tag: str = "generic") -> dict[str, Any]:
        last_err: json.JSONDecodeError | None = None
        for attempt in range(3):
            cur_max_tokens = max_tokens if attempt == 0 else max(max_tokens * (2 ** attempt), 6144 * attempt)
            cur_user = (
                user
                if attempt == 0
                else user + "\n\nIMPORTANT: Return ONE complete, compact, strictly valid JSON object only (no markdown fences, escape all double quotes inside strings, concise string values so nothing truncates)."
            )
            cur_temp = temperature if attempt == 0 else (0.2 if temperature == 0.0 else temperature)
            raw = self.chat_text(
                system,
                cur_user,
                temperature=cur_temp,
                max_tokens=cur_max_tokens,
                tag=tag,
                json_mode=True,
            )
            try:
                return extract_json_dict(raw)
            except json.JSONDecodeError as exc:
                last_err = exc
        assert last_err is not None
        raise last_err


def _dry_run_response(*, tag: str, system: str, user: str) -> str:
    """Deterministic realistic responses exercising every gate, smoothing, gap-falsification, and pairwise branch."""
    low = (tag + " " + system + " " + user[:600]).lower()

    if tag == "stage_b_verdict":
        return json.dumps({name: {"status": "unresolved", "reason": "offline fixture"}
                           for name in ("honesty", "credibility", "novelty")})
    if tag == "stage_b_pairwise":
        score = {name: 3 for name in ("honesty", "credibility", "novelty")}
        return json.dumps({"winner": "tie", "A": score, "B": score, "reason": "offline fixture"})
    if tag == "feedback_candidate_check":
        return '{"aligned":true,"preserves":true,"material_change":true,"reason":"offline fixture"}'
    if tag == "stage_b_feedback":
        tag = "generate_direct"
    if tag == "generate_direct":
        # Deliberately includes one smoothed/overclaimed premise to verify the audit + middleware catch it.
        return json.dumps({
            "title": "Universal Zero-Shot Calibration Wrapper for Foundation Models",
            "literature_gap": "No prior work addresses calibrated failure prediction under shifting open-world constraints.",
            "hypothesis": "Adding a single lightweight entropy gate universally eliminates distribution-shift failures across all domains without task-specific data.",
            "method": "Apply an off-the-shelf entropy threshold calibrated on validation logits and replace baseline pipelines end-to-end.",
            "minimal_test": "Evaluate accuracy on one standard benchmark split and compare against baseline averages.",
            "decisive_metric": "Overall accuracy improvement >= 2%",
            "supporting_result": "Accuracy rises across benchmarks.",
            "falsifying_result": "Accuracy drops slightly.",
            "related_work": "Prior papers prove entropy alone guarantees domain robustness in all deployed systems.",
            "limitations": "Assumes clean distribution overlap.",
            "resource_plan": {"gpu": "2x A100 40GB", "estimated_gpu_hours": 40, "estimated_api_usd": 25},
        })

    if tag == "normalize":
        # Deterministic structured extraction respecting whether draft has explicit unknowns/overclaims.
        is_direct = "universal zero-shot calibration" in user.lower() or "universally eliminates" in user.lower()
        is_repaired = "[repaired" in user.lower() or "bounded uncertainty" in user.lower() or "explicit unknown" in user.lower()
        if is_direct and not is_repaired:
            return json.dumps({
                "title": "Universal Zero-Shot Calibration Wrapper for Foundation Models",
                "literature_gap": "No prior work addresses calibrated failure prediction under shifting open-world constraints.",
                "hypothesis": "Adding a lightweight entropy gate eliminates shift failures across domains.",
                "method": "Threshold validation logits with a single scalar cutoff.",
                "minimal_test": "Evaluate accuracy on one benchmark split.",
                "decisive_metric": "Accuracy delta >= 2%",
                "supporting_result": "Accuracy improves by >= 2% under shift.",
                "falsifying_result": "Accuracy drops or shows no significant shift resilience.",
                "related_work": "Prior papers show entropy guarantees robustness in all settings.",
                "limitations": "Assumes overlap.",
                "resource_plan": {"gpu": "2x A100 40GB", "estimated_gpu_hours": 40, "estimated_api_usd": 25},
                "prior_facts": [
                    {"claim": "Prior work proves entropy thresholding guarantees domain robustness in all deployed systems.", "cited_titles": [], "cited_ids": ["r01"]},
                    {"claim": "Existing methods never consider calibrated uncertainty estimation.", "cited_titles": [], "cited_ids": []},
                ],
                "prospective_hypotheses": [
                    {"claim": "Entropy gating reduces shift error.", "decisive_metric": "Accuracy delta >= 2%", "falsifying_outcome": "Accuracy drops under shift."}
                ],
                "unknown_dependencies": [],
                "gap_premises": ["No prior work addresses calibrated failure prediction under open-world shift."],
            })
        return json.dumps({
            "title": "Boundary-Scoped Failure Diagnosis with Explicit Pre-Shift Calibration (Repaired)",
            "literature_gap": "While prior work evaluates calibration on static benchmarks, cross-regime threshold drift under uncalibrated priors remains under-tested.",
            "hypothesis": "Stratifying predictions by regime-mismatch indicators reduces silent failure rate relative to unstratified confidence gating.",
            "method": "Combine source-anchored confidence intervals with an explicit regime-shift diagnostic probe and fallback abstention rule.",
            "minimal_test": "Hold out one shifted evaluation split alongside in-domain controls with paired ablations.",
            "decisive_metric": "AURC and selective risk reduction at 80% coverage under shift (p < 0.05)",
            "supporting_result": "Selective risk drops >= 15% relative to static confidence baseline at matched coverage.",
            "falsifying_result": "Selective risk fails to beat static confidence or threshold variance exceeds baseline.",
            "related_work": "Builds explicitly on cited seed papers while attributing established calibration baselines.",
            "limitations": "Requires empirical threshold validation on a held-out calibration slice.",
            "resource_plan": {"gpu": "2x A100 40GB", "estimated_gpu_hours": 45, "estimated_api_usd": 30},
            "prior_facts": [
                {"claim": "Seed references report gains under fixed benchmark splits and note degradation under shift.", "cited_titles": [], "cited_ids": ["r01"]},
            ],
            "prospective_hypotheses": [
                {"claim": "Regime-stratified abstention reduces selective risk at matched coverage.", "decisive_metric": "AURC at 80% coverage", "falsifying_outcome": "No selective risk improvement over baseline."}
            ],
            "unknown_dependencies": [
                {"dependency": "Optimal shift-threshold stability across unseen task sub-regimes", "why_unresolved": "Depends on empirical shift magnitude prior to running held-out probes", "resolution_path": "Measure on pilot validation slice before main evaluation"}
            ],
            "gap_premises": ["Cross-regime threshold drift under explicit abstention constraints lacks controlled falsification diagnostics."],
        })

    if tag == "independent_claim_extract":
        is_direct = "universal zero-shot" in user.lower() or "universally eliminates" in user.lower()
        is_repaired = "[repaired" in user.lower() or "bounded uncertainty" in user.lower()
        if is_direct and not is_repaired:
            return json.dumps({
                "claims": [
                    {
                        "field": "related_work",
                        "quote": "Prior papers prove entropy alone guarantees domain robustness in all deployed systems.",
                        "claim_kind": "prior_fact",
                    }
                ]
            })
        return json.dumps({"claims": []})

    if tag == "independent_claim_audit":
        if "guarantees domain robustness" in user.lower():
            return json.dumps({
                "findings": [
                    {
                        "id": "c1",
                        "label": "defect",
                        "defect_type": "scope_overclaim",
                        "evidence": [],
                        "basis": "Prior papers prove entropy alone guarantees domain robustness in all deployed systems.",
                        "reason": "Offline dry-run: claims universal guarantee unsupported by seed caveats.",
                    }
                ]
            })
        return json.dumps({"findings": []})

    if tag == "tri_state_fact_audit":
        is_direct_claim = "guarantees" in low or "all deployed" in low or "never consider" in low
        m_claim = re.search(r"Prior-Fact Claim:\s*(.+)", user)
        claim_snip = (m_claim.group(1).strip()[:120] if m_claim else "Prior work claim")
        return json.dumps({
            "citation_status": "unverified" if "never consider" in low else "verified_local",
            "smoothing_score": 4 if is_direct_claim else 1,
            "smoothing_type": "scope_expansion" if is_direct_claim else "none",
            "proposal_span": claim_snip,
            "evidence_quote": "Source explicitly restricts scope and reports degradation outside training regimes." if is_direct_claim else "Source evaluates fixed-split calibration.",
            "support_evidence": "Source evaluates fixed-split calibration." if not is_direct_claim else "Mentioned entropy metric in narrow setting.",
            "skeptic_counter_evidence": "Source explicitly restricts scope and reports degradation outside training regimes." if is_direct_claim else "Scoped accurately to fixed benchmarks.",
            "epistemic_overclaim": is_direct_claim,
        })

    if tag == "gap_falsify":
        # Check ONLY the Gap Premise line in user text (not example wording in system prompt).
        m = re.search(r"Gap Premise:\s*(.+)", user)
        premise_low = (m.group(1) if m else user[:200]).lower()
        is_absolute = "no prior work" in premise_low or "never " in premise_low
        return json.dumps({
            "status": "falsified" if is_absolute else "survived",
            "counterexample_title": "Matched Prior Seed / Pre-Cutoff Study" if is_absolute else None,
            "counterexample_year": "2024" if is_absolute else None,
            "justification": (
                "Pre-cutoff literature already explores calibrated uncertainty under distribution shift."
                if is_absolute
                else "No retrieved pre-cutoff counterexample directly tests this scoped abstention boundary."
            ),
        })

    if tag == "claimnov_extract":
        return json.dumps({
            "claims": [
                {
                    "claim": "Proposes an epistemic stratification and bounded falsification mechanism.",
                    "dimension": "methodology",
                    "attribute": "falsification mechanism",
                    "value": "regime-stratified abstention and diagnostic probe",
                    "abstract_span": "method",
                },
                {
                    "claim": "Formalizes shift-regime evaluation with matched coverage control.",
                    "dimension": "problem",
                    "attribute": "evaluation setting",
                    "value": "controlled shift falsification",
                    "abstract_span": "hypothesis",
                },
            ]
        })

    if tag == "claimnov_entail":
        return "2"

    if tag == "claimnov_veto":
        return json.dumps({
            "decision": "novel",
            "anticipating_paper": None,
            "established_claims": [],
            "why": "Proposed diagnostic mechanism and controlled test differ substantively from retrieved prior work.",
        })

    if tag == "aib_eval":
        is_direct = "universal zero-shot" in low
        return json.dumps({
            "budget_feasible": True,
            "motivation_method_consistency": 3 if is_direct else 4,
            "technical_feasibility": 3 if is_direct else 4,
            "data_feasibility": 3 if is_direct else 4,
            "budget_feasibility": 4,
            "verification_feasibility": 2 if is_direct else 4,
            "novelty_score": 2.5 if is_direct else 4.0,
        })

    if tag == "spec_locate":
        is_clean = (
            "held-out" in user.lower()
            and ("matched" in user.lower() or "ablation" in user.lower() or "baseline" in user.lower())
            and "unknown_dependencies" in user
            and ("calibration" in user.lower() or "failure" in user.lower())
        )
        if is_clean and "repaired" in user.lower():
            return json.dumps({"defects": []})
        return json.dumps({
            "defects": [
                {
                    "category": "hyperparameter_or_threshold",
                    "target_field": "method",
                    "missing_detail": "Scalar threshold selection rule and calibration slice protocol are unspecified.",
                    "why_implementation_critical": "Unbound threshold prevents reproducible execution.",
                    "clarification_hint": "Add bounded calibration protocol or declare an unknown_dependency with calibration test.",
                    "severity": "blocking" if not is_clean else "recommended",
                },
                {
                    "category": "failure_regime",
                    "target_field": "limitations",
                    "missing_detail": "Failure regime under severe distribution shift is omitted.",
                    "why_implementation_critical": "Without explicit failure bounds, overclaiming occurs.",
                    "clarification_hint": "Specify held-out shift failure regime.",
                    "severity": "recommended",
                },
            ]
        })

    if tag == "bounded_patch":
        patches = []
        if "Universal Zero-Shot Calibration Wrapper for Foundation Models" in user:
            patches.append({
                "target_field": "title",
                "exact_source_span": "Universal Zero-Shot Calibration Wrapper for Foundation Models",
                "replacement_text": "Boundary-Scoped Calibration Wrapper with Explicit Unknowns [Repaired]",
                "edit_type": "scope_narrowing",
                "rationale": "Remove universal overclaim from title.",
            })
        if "No prior work addresses calibrated failure prediction under shifting open-world constraints." in user:
            patches.append({
                "target_field": "literature_gap",
                "exact_source_span": "No prior work addresses calibrated failure prediction under shifting open-world constraints.",
                "replacement_text": "While prior work evaluates calibration on static benchmarks, cross-regime threshold drift remains under-tested.",
                "edit_type": "gap_refinement",
                "rationale": "Narrow falsified universal gap to surviving regime boundary.",
            })
        if "Prior papers show entropy guarantees robustness in all settings." in user:
            patches.append({
                "target_field": "related_work",
                "exact_source_span": "Prior papers show entropy guarantees robustness in all settings.",
                "replacement_text": "Seed papers evaluate entropy on fixed splits and report degradation under distribution shift.",
                "edit_type": "attribution_credit",
                "rationale": "Ground prior claim in source caveats.",
            })
        if "Evaluate accuracy on one benchmark split." in user:
            patches.append({
                "target_field": "minimal_test",
                "exact_source_span": "Evaluate accuracy on one benchmark split.",
                "replacement_text": "Evaluate held-out shifted benchmark split against matched baseline and ablation control (bounded uncertainty calibration).",
                "edit_type": "spec_clarification",
                "rationale": "Address IdeaAMBIG baseline_alignment and split defects.",
            })
        return json.dumps({
            "patches": patches[:4],
            "append_unknown_contracts": [
                {
                    "dependency": "Optimal shift-threshold stability across unseen task sub-regimes",
                    "why_unresolved": "Depends on empirical shift magnitude prior to running held-out probes",
                    "resolution_path": "Measure on pilot calibration slice before main evaluation",
                    "minimal_calibration_test": "Sweep 5 thresholds on 10% held-out calibration split",
                    "fallback_branch": "Revert to abstention-weighted selective prediction if AURC delta < 5%",
                }
            ],
            "updated_gap_premises": [
                "Cross-regime threshold drift under explicit abstention constraints lacks controlled falsification diagnostics."
            ],
            "repair_notes": ["Applied bounded surgical patches and appended structured unknown contract."],
        })

    if tag == "self_refine_critique":
        return json.dumps({
            "needs_revision": True,
            "feedback": ["Clarify evaluation metric and tone down broad claims."],
        })

    if tag in {"self_refine_revise", "retrieve_revise", "middleware_repair"}:
        # Retrieve-and-revise / self-refine do unconstrained full rewrite (dropping some original phrasing).
        if tag == "self_refine_revise":
            return json.dumps({
                "title": "Universal Zero-Shot Calibration Wrapper (Self-Refined)",
                "literature_gap": "No prior work addresses calibrated failure prediction under shifting open-world constraints.",
                "hypothesis": "Entropy gating improves robustness across domains.",
                "method": "Use validation logits thresholded globally across benchmarks.",
                "minimal_test": "Evaluate accuracy on one benchmark split.",
                "decisive_metric": "Accuracy delta >= 2%",
                "supporting_result": "Accuracy increases.",
                "falsifying_result": "Accuracy does not increase.",
                "related_work": "Prior papers show entropy guarantees robustness in all settings.",
                "limitations": "Assumes clean overlap.",
                "resource_plan": {"gpu": "2x A100 40GB", "estimated_gpu_hours": 40, "estimated_api_usd": 25},
            })
        return json.dumps({
            "title": "Boundary-Scoped Failure Diagnosis with Explicit Pre-Shift Calibration [Repaired]",
            "literature_gap": "While prior work evaluates calibration on static benchmarks, cross-regime threshold drift under uncalibrated priors remains under-tested.",
            "hypothesis": "Stratifying predictions by regime-mismatch indicators reduces silent failure rate relative to unstratified confidence gating.",
            "method": "Combine source-anchored confidence intervals with an explicit regime-shift diagnostic probe and fallback abstention rule (bounded uncertainty calibration).",
            "minimal_test": "Hold out one shifted evaluation split alongside in-domain controls with paired ablations.",
            "decisive_metric": "AURC and selective risk reduction at 80% coverage under shift (p < 0.05)",
            "supporting_result": "Selective risk drops >= 15% relative to static confidence baseline at matched coverage.",
            "falsifying_result": "Selective risk fails to beat static confidence or threshold variance exceeds baseline.",
            "related_work": "Explicitly attributes calibration baselines and prior shift caveats from supplied sources.",
            "limitations": "Requires empirical threshold validation on a held-out calibration slice (explicit unknown dependency).",
            "resource_plan": {"gpu": "2x A100 40GB", "estimated_gpu_hours": 45, "estimated_api_usd": 30},
            "repair_notes": [
                "Scoped universal overclaim into explicit regime-boundary framing.",
                "Converted implicit threshold assumption into an explicit unknown dependency with resolution criterion.",
            ],
        })

    if tag == "lit2test_decomposed_pair":
        try:
            payload = json.loads(user)
        except json.JSONDecodeError:
            payload = {}
        card_a = json.dumps(payload.get("answer_a") or {}, ensure_ascii=False).lower()
        card_b = json.dumps(payload.get("answer_b") or {}, ensure_ascii=False).lower()

        def _score_card(text: str, dim: str) -> int:
            if dim == "grounding" and "published trials already prove" in text:
                return 1
            if dim == "decisive_metric" and "aggregate overall task accuracy" in text:
                return 1
            if dim == "falsifiability" and "failure to reach statistical significance" in text:
                return 1
            if any(k in text for k in ("repaired", "bounded", "aurc", "ablation", "matched", "held-out")):
                return 3
            return 2

        dims = {}
        tot_a = 0
        tot_b = 0
        for dim in ("grounding", "hypothesis_specificity", "minimality_feasibility", "decisive_metric", "falsifiability"):
            sa = _score_card(card_a, dim)
            sb = _score_card(card_b, dim)
            tot_a += sa
            tot_b += sb
            w = "A" if sa > sb else "B" if sb > sa else "tie"
            dims[dim] = {"score_a": sa, "score_b": sb, "winner": w, "reason": "offline lit2test fixture"}
        overall = "A" if tot_a > tot_b else "B" if tot_b > tot_a else "tie"
        return json.dumps({
            "dimensions": dims,
            "overall_winner": overall,
            "overall_confidence": "high",
            "overall_reason": "Offline Lit2Test decomposed fixture.",
        })

    if tag == "aib_batch_judge":
        idea_section = user.split("## Idea", 1)[-1].lower()
        is_repaired = "repaired" in idea_section or "bounded" in idea_section or "aurc" in idea_section
        score = 4 if is_repaired else 3
        dim_blocks = re.findall(r"### \d+\.\s+([^\n`]+)\(`([a-z_]+)`\)(.*?)(?=\n### \d+\.|\Z)", system, re.S)
        dims = {}
        for _, dim_id, block in dim_blocks:
            items = re.findall(r"^- (.+?)  \[yes =", block, re.M)
            dims[dim_id] = {
                "dimension_id": dim_id,
                "score": score,
                "reasoning": "Offline AlphaIdeaBench batched fixture.",
                "checklist_notes": [
                    {"item": item, "judgment": "yes" if is_repaired else "partial", "evidence": "offline fixture"}
                    for item in items
                ],
                "confidence": 0.9,
                "flags": [],
            }
        return json.dumps({"dimensions": dims})

    if tag == "aib_dimension_judge":
        m_dim = re.search(r"\(`([a-z_]+)`\)", user)
        dim_id = m_dim.group(1) if m_dim else "context_grounding"
        items = re.findall(r"^- (.+?)  \[yes =", system, re.M)
        idea_section = user.split("## Idea", 1)[-1].lower()
        is_repaired = "repaired" in idea_section or "bounded" in idea_section or "aurc" in idea_section
        score = 4 if is_repaired else 3
        notes = [
            {"item": item, "judgment": "yes" if is_repaired else "partial", "evidence": "offline fixture"}
            for item in items
        ]
        return json.dumps({
            "dimension_id": dim_id,
            "score": score,
            "reasoning": "Offline AlphaIdeaBench dimension fixture.",
            "checklist_notes": notes,
            "confidence": 0.9,
            "flags": [],
        })

    if tag == "pairwise_judge":
        # Prefer the boundary-scoped / repaired answer over the universal ungrounded one in both orders.
        a_marker = user.find("Answer A:")
        b_marker = user.find("Answer B:")
        block_a = user[a_marker:b_marker].lower() if a_marker >= 0 and b_marker > a_marker else ""
        block_b = user[b_marker:].lower() if b_marker >= 0 else ""
        a_good = "repaired" in block_a or "bounded" in block_a or "aurc" in block_a
        b_good = "repaired" in block_b or "bounded" in block_b or "aurc" in block_b
        if a_good and not b_good:
            winner = "A"
        elif b_good and not a_good:
            winner = "B"
        else:
            winner = "tie"
        return json.dumps({
            "rubric_scores": {
                "A": {"honesty": 4 if a_good else 2, "actionability": 4 if a_good else 2, "falsifiability": 4 if a_good else 2, "novelty": 4},
                "B": {"honesty": 4 if b_good else 2, "actionability": 4 if b_good else 2, "falsifiability": 4 if b_good else 2, "novelty": 4},
            },
            "winner": winner,
            "confidence": "high",
            "main_reason": "Prefers operationalized falsification, scoped prior attribution, and explicit unresolved dependencies.",
        })

    # Fallback hash-varied JSON
    digest = hashlib.sha1(user.encode("utf-8")).hexdigest()[:8]
    return json.dumps({"ok": True, "dry_run_id": digest})
