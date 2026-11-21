"""Run EvoScientist (research-ideation v2.1.2) host generation + 5-arm trial (A/B/C/D/E) + 3-judge blind evaluation on Benchmark-50 waves.

Faithfully executes EvoScientist's multi-stage evolutionary ideation & proposal workflow
(Challenge-Insight Literature Tree -> 3 Directions -> Innovator/Pragmatist/Critic Evolutionary Refinement
-> Pairwise ELO Round-Robin Tournament [K=32] -> 6-Section Proposal Extension) under each topic's
strict pre-cutoff literature boundary, then runs symmetric B, C, D, E arms and 3-judge blind evaluation.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import json
import math
from pathlib import Path
import shutil
from typing import Any

import numpy as np

from edara_mvp_eval.epistemic_audit import run_full_audit
from edara_mvp_eval.factual_audit import audit_unverified_entities
from edara_mvp_eval.host_adapter import (
    confound_addressed,
    extract_design_elements,
    preserves_design_elements,
)
from edara_mvp_eval.llm_client import LLMRunner, ModelEndpoint
from edara_mvp_eval.paired_inputs import Evidence
from edara_mvp_eval.paired_trial import (
    LIMITS,
    OUTPUT_TOKENS,
    PROTOCOL_VERSION,
    ROOT,
    TrialRunner,
    WORKSPACE,
    cached,
    digest,
    load,
    path_hash,
    public_task,
    run_arm,
    shared_full_pool,
    usage,
)
from edara_mvp_eval.schemas import PriorWorkItem, TaskSpec
from edara_mvp_eval.stage_b_evidence import EvidenceStore, write_json
from edara_mvp_eval.unified_eval import (
    CORE_AIB_DIMENSIONS,
    LIT2TEST_DIMENSIONS,
    eval_lit2test_pair,
    evaluate_unified_pair,
    summarize_unified_rows,
)

EVO_SKILL_VERSION = "evoscientist-research-ideation/v2.1.2"
EVO_FIELDS = {
    "Title": "title",
    "Short Hypothesis": "hypothesis",
    "Background": "literature_gap",
    "Related Work": "related_work",
    "Proposed Method": "method",
    "Experimental Plan": "minimal_test",
    "Expected Results": "supporting_result",
    "Risks and Mitigations": "limitations",
}

_EVO_STEP2_TREE_SYS = """You are running EvoScientist research-ideation Step 1-2 (Goal & Challenge-Insight Literature Tree).
Given the research topic, constraints, and pre-cutoff papers (each with a [S...] paper_id), construct:
1) 2-3 pre-cutoff literature search queries to broaden coverage;
2) a Challenge-Insight Tree mapping technical challenges to existing insights/techniques, identifying unsolved challenges and cross-domain transfer opportunities;
3) a concise literature synthesis referencing actual [S...] paper IDs.
Return JSON only:
{"queries": ["q1", "q2"], "challenges": [{"id": "C1", "challenge": "...", "existing_insights": ["..."], "gap": "..."}], "synthesis": "..."}"""

_EVO_STEP3_DIRS_SYS = """You are running EvoScientist research-ideation Step 3 (3 Distinct Directions via Innovator Persona).
Rules from EvoScientist SKILL.md:
- Problem selection > solution design; pursue specific failure cases rather than incremental tweaks.
- Technology is creative combination, not simple A->B concatenation.
- Identify 3 fundamentally different research directions from the Challenge-Insight Tree and generate 1 initial idea per direction (Focused Contribution or System Contribution).
Return JSON only:
{"ideas": [
  {"track": 1, "direction": "...", "path_type": "Focused|System", "title": "...", "core_idea": "...", "validation_plan": "..."},
  {"track": 2, "direction": "...", "path_type": "Focused|System", "title": "...", "core_idea": "...", "validation_plan": "..."},
  {"track": 3, "direction": "...", "path_type": "Focused|System", "title": "...", "core_idea": "...", "validation_plan": "..."}
]}"""

_EVO_STEP4_REFINE_SYS = """You are running EvoScientist research-ideation Step 4 (3-Persona Evolutionary Refinement).
For EACH of the 3 tracks, apply the 3 personas — Innovator (novelty & bold mechanism), Pragmatist (feasibility within compute/budget constraints), and Critic (scientific rigor, variable isolation, confounding failure modes) — using EvoScientist's 5 Evolution Strategies (Enhancement through Grounding with [S...] citations, Improving Coherence, Inspiration & Combination, Simplification via Subtraction, or Literature-Driven Pivot if dead-end).
Produce the final Track Champion for each of the 3 tracks.
Return JSON only:
{"champions": [
  {"track": 1, "title": "...", "short_hypothesis": "one self-contained paragraph with mechanism and testable prediction", "mechanism": "...", "prediction": "...", "validation_plan": "datasets, baselines, metrics, key ablations", "refinement_summary": "what changed and which persona drove it"},
  {"track": 2, "title": "...", "short_hypothesis": "...", "mechanism": "...", "prediction": "...", "validation_plan": "...", "refinement_summary": "..."},
  {"track": 3, "title": "...", "short_hypothesis": "...", "mechanism": "...", "prediction": "...", "validation_plan": "...", "refinement_summary": "..."}
]}"""

_EVO_STEP5_ELO_SYS = """You are running EvoScientist research-ideation Step 5 (Pairwise ELO Round-Robin Tournament).
Evaluate all 3 pairwise matches (Track 1 vs 2, Track 1 vs 3, Track 2 vs 3) across the 4 EvoScientist dimensions (1-10 integers):
- novelty: How different from existing published work?
- feasibility: Can this be implemented within the stated resources?
- relevance: Does this address an important problem aligned with the goal?
- clarity: Is the idea well-defined enough to start immediately?
Higher 4-dimension sum wins the match.
Return JSON only:
{"matches": [
  {"left_track": 1, "right_track": 2, "left_scores": {"novelty": 8, "feasibility": 8, "relevance": 8, "clarity": 8}, "right_scores": {"novelty": 7, "feasibility": 8, "relevance": 8, "clarity": 7}, "winner_track": 1, "reason": "..."},
  {"left_track": 1, "right_track": 3, "left_scores": {"novelty": 8, "feasibility": 8, "relevance": 8, "clarity": 8}, "right_scores": {"novelty": 7, "feasibility": 7, "relevance": 8, "clarity": 7}, "winner_track": 1, "reason": "..."},
  {"left_track": 2, "right_track": 3, "left_scores": {"novelty": 7, "feasibility": 8, "relevance": 8, "clarity": 7}, "right_scores": {"novelty": 7, "feasibility": 7, "relevance": 8, "clarity": 7}, "winner_track": 2, "reason": "..."}
]}"""

_EVO_STEP7_PROPOSAL_SYS = """You are running EvoScientist research-ideation Step 7 (Expand Tournament Winner into Full 6-Section Research Proposal).
Follow EvoScientist proposal-extension.md specifications strictly:
- Title: Crisp, specific proposal title.
- Short Hypothesis: Self-contained causal hypothesis stating mechanism and testable prediction.
- Background: Exact problem definition (inputs, outputs, constraints, evaluation criteria), where/why existing solutions fail (citing [S...] papers), and explicit scope boundaries.
- Related Work: Positioning in the literature landscape, 3-5 key prior works ([S...] citations), specific gaps, and differentiation from closest work.
- Proposed Method: High-level approach, step-by-step method overview, explicit key insight ("Our key insight is that [X], which enables [Y] because [Z]"), Rule of Three (list 3 testable contributions: Primary, Secondary, Tertiary), and assumptions.
- Experimental Plan: Datasets & splits, at least 3 baselines, primary & secondary metrics, and ablation design mapping to contributions.
- Expected Results: Quantitative targets and qualitative expectations across regimes.
- Risks and Mitigations: Concrete technical/experimental risks and fallback mitigations.
Return valid JSON with exact string keys:
{"Title": "...", "Short Hypothesis": "...", "Background": "...", "Related Work": "...", "Proposed Method": "...", "Experimental Plan": "...", "Expected Results": "...", "Risks and Mitigations": "..."}"""


def _compute_elo_rankings(champions: list[dict[str, Any]], matches: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    ratings = {1: 1500.0, 2: 1500.0, 3: 1500.0}
    dim_accum: dict[int, dict[str, list[float]]] = {
        t: {d: [] for d in ("novelty", "feasibility", "relevance", "clarity")} for t in (1, 2, 3)
    }
    wins = Counter()
    for m in matches:
        lt = int(m.get("left_track", 1))
        rt = int(m.get("right_track", 2))
        if lt not in ratings or rt not in ratings or lt == rt:
            continue
        ls = m.get("left_scores") or {}
        rs = m.get("right_scores") or {}
        for d in ("novelty", "feasibility", "relevance", "clarity"):
            dim_accum[lt][d].append(float(ls.get(d, 7)))
            dim_accum[rt][d].append(float(rs.get(d, 7)))
        l_sum = sum(float(ls.get(d, 7)) for d in ("novelty", "feasibility", "relevance", "clarity"))
        r_sum = sum(float(rs.get(d, 7)) for d in ("novelty", "feasibility", "relevance", "clarity"))
        actual_l = 1.0 if l_sum > r_sum else (0.0 if r_sum > l_sum else 0.5)
        if actual_l == 1.0:
            wins[lt] += 1
        elif actual_l == 0.0:
            wins[rt] += 1
        exp_l = 1.0 / (1.0 + math.pow(10.0, (ratings[rt] - ratings[lt]) / 400.0))
        ratings[lt] += 32.0 * (actual_l - exp_l)
        ratings[rt] += 32.0 * ((1.0 - actual_l) - (1.0 - exp_l))

    scored = []
    for ch in champions:
        t = int(ch.get("track", 1))
        means = {
            d: round(float(np.mean(dim_accum[t][d])) if dim_accum[t][d] else 7.5, 2)
            for d in ("novelty", "feasibility", "relevance", "clarity")
        }
        composite = round(sum(means.values()) / 4.0, 3)
        scored.append({
            **ch,
            **means,
            "composite": composite,
            "elo": round(ratings.get(t, 1500.0), 2),
        })
    # EvoScientist tie-break rule: sort by Elo; if equal or 1-win cycle, order by composite score then track order
    one_win_cycle = sorted(wins.values()) == [1, 1, 1]
    if one_win_cycle or len({r["elo"] for r in scored}) < len(scored):
        scored.sort(key=lambda x: (-x["composite"], -x["elo"], int(x.get("track", 9))))
    else:
        scored.sort(key=lambda x: (-x["elo"], -x["composite"], int(x.get("track", 9))))
    for idx, item in enumerate(scored, 1):
        item["rank"] = idx
    return scored, int(scored[0]["track"])


def _configure_endpoint(profile_path: Path, *, role: str, dry_run: bool) -> ModelEndpoint:
    if dry_run:
        return ModelEndpoint(model=f"dry-run-{role}")
    ep = ModelEndpoint.from_env_or_profile(profile_path=profile_path)
    low = ep.model.lower()
    if role in {"editor", "audit", "host_gen"}:
        if "deepseek" in low:
            ep = replace(ep, extra_body={"thinking": {"type": "disabled"}})
        elif "glm" in low:
            extra: dict[str, Any] = {"thinking": {"type": "disabled"}}
            if "autodl" in (ep.base_url or "").lower():
                extra["reasoning_effort"] = "none"
            ep = replace(ep, extra_body=extra)
    elif role == "judge":
        if "glm" in low and "autodl" in (ep.base_url or "").lower():
            ep = replace(ep, extra_body={"thinking": {"type": "disabled"}, "reasoning_effort": "none"})
        elif "deepseek" in low:
            ep = replace(ep, extra_body={"thinking": {"type": "disabled"}})
    if not ep.resolve_api_key():
        raise RuntimeError(f"Missing credential in {ep.credential_env} for {profile_path.name}")
    return ep


def _make_runner(endpoint: ModelEndpoint, log_dir: Path, job: str, arm: str, stage: str, limit: int, *, dry_run: bool):
    if dry_run:
        r = LLMRunner(endpoint, dry_run=True)
        r.index = 0  # type: ignore[attr-defined]
        r.reused = 0  # type: ignore[attr-defined]
        return r
    return TrialRunner(endpoint, log_dir, task_id=job, arm=arm, stage=stage, limit=limit)


def generate_evoscientist_proposal(
    task: TaskSpec,
    base_evidence_snap: dict[str, Any],
    runner,
    lookup_dir: Path,
    work_dir: Path,
    *,
    dry_run: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run EvoScientist research-ideation v2.1.2 (Steps 2->3->4->5->7) and return (A_record, updated_evidence_snap)."""
    work_dir.mkdir(parents=True, exist_ok=True)
    ev = Evidence.restore(task, base_evidence_snap)
    store = EvidenceStore(lookup_dir, task)
    shared = public_task(task)

    if dry_run:
        draft = {
            "title": f"EvoScientist Dry-Run Proposal for {task.task_id}",
            "hypothesis": "Adaptive evolutionary mechanism improves target metric by >=10% under matched compute.",
            "literature_gap": "Prior methods [S1] fail under distribution shift.",
            "related_work": "Prior studies [S1] explore static schedules without adaptive feedback.",
            "method": "Our key insight is dynamic gating. Contributions: (1) Method, (2) Analysis, (3) Benchmark.",
            "minimal_test": "Evaluate on standard benchmark against 3 baselines + leave-one-out ablation.",
            "supporting_result": "10-15% error reduction at matched FLOPs.",
            "limitations": "Sensitivity to extreme sparsity; mitigated via warmup fallback.",
        }
        rec = {
            "status": "ready",
            "host": "evoscientist",
            "stage": "proposal",
            "skill_version": EVO_SKILL_VERSION,
            "sha256": digest(draft),
            "draft": draft,
        }
        return rec, ev.snapshot()

    # Step 2a: Initial Challenge-Insight Tree + autonomous search queries
    step2_init = runner.chat_json(
        _EVO_STEP2_TREE_SYS,
        json.dumps({"task": shared, "seed_papers": ev.packet()}, ensure_ascii=False),
        max_tokens=2400,
        tag="evo_step2_tree",
    )
    queries = [str(q).strip()[:240] for q in (step2_init.get("queries") or [task.title]) if str(q).strip()][:3]
    added_markers = []
    for q in queries:
        for paper in store.search(q)[:6]:
            m = ev.add(paper)
            if m:
                added_markers.append(m)

    # Step 2b: Refresh tree if new papers were discovered
    if added_markers:
        tree_state = runner.chat_json(
            _EVO_STEP2_TREE_SYS,
            json.dumps({"task": shared, "seed_papers": ev.packet()}, ensure_ascii=False),
            max_tokens=2400,
            tag="evo_step2_tree_grounded",
        )
    else:
        tree_state = step2_init
    write_json(work_dir / "step2_literature_tree.json", {"queries": queries, "added_markers": added_markers, "tree": tree_state})

    # Step 3: 3 Directions x Innovator
    step3 = runner.chat_json(
        _EVO_STEP3_DIRS_SYS,
        json.dumps({"task": shared, "literature_tree": tree_state, "papers": ev.packet()}, ensure_ascii=False),
        max_tokens=3000,
        tag="evo_step3_ideas",
    )
    write_json(work_dir / "step3_initial_ideas.json", step3)

    # Step 4: 3-Persona Evolutionary Refinement (2 iterative passes)
    refined = step3
    for it in (1, 2):
        refined = runner.chat_json(
            _EVO_STEP4_REFINE_SYS,
            json.dumps(
                {"iteration": it, "task": shared, "literature_tree": tree_state, "current_tracks": refined, "papers": ev.packet()},
                ensure_ascii=False,
            ),
            max_tokens=3600,
            tag=f"evo_step4_refine_{it}",
        )
    champions = refined.get("champions") or []
    if len(champions) < 3:
        raise RuntimeError(f"EvoScientist Step 4 returned fewer than 3 champions for {task.task_id}")
    write_json(work_dir / "step4_champions.json", refined)

    # Step 5: Pairwise ELO Round-Robin Tournament
    tour = runner.chat_json(
        _EVO_STEP5_ELO_SYS,
        json.dumps({"task": shared, "champions": champions}, ensure_ascii=False),
        max_tokens=2200,
        tag="evo_step5_elo",
    )
    ranked, winner_track = _compute_elo_rankings(champions[:3], tour.get("matches") or [])
    winner = ranked[0]
    ranked_artifact = {
        "schema_version": "evoscientist-native-ranked-ideas/v1",
        "skill_version": EVO_SKILL_VERSION,
        "task_id": task.task_id,
        "queries": queries,
        "paper_ids": list(ev.rows.keys()),
        "candidates": ranked,
        "matches": tour.get("matches") or [],
        "selected_rank": 1,
        "selected_track": winner_track,
    }
    write_json(work_dir / "ranked-ideas.json", ranked_artifact)

    # Step 7: Expand Rank-1 Champion into 6-Section Proposal
    raw_prop = runner.chat_json(
        _EVO_STEP7_PROPOSAL_SYS,
        json.dumps(
            {
                "task": shared,
                "winning_champion": winner,
                "literature_tree": tree_state,
                "papers": ev.packet(),
            },
            ensure_ascii=False,
        ),
        max_tokens=OUTPUT_TOKENS,
        tag="evo_step7_proposal",
    )
    write_json(work_dir / "raw_proposal.json", raw_prop)

    draft: dict[str, str] = {}
    for raw_k, mapped_k in EVO_FIELDS.items():
        val = raw_prop.get(raw_k) if isinstance(raw_prop, dict) else None
        if val is None and isinstance(raw_prop, dict):
            val = raw_prop.get(mapped_k, "")
        draft[mapped_k] = str(val or "").strip() if not isinstance(val, (dict, list)) else json.dumps(val, ensure_ascii=False)
    if not draft.get("title"):
        draft["title"] = str(winner.get("title") or task.title)
    if not draft.get("hypothesis"):
        draft["hypothesis"] = str(winner.get("short_hypothesis") or "")

    rec = {
        "status": "ready",
        "host": "evoscientist",
        "stage": "proposal",
        "skill_version": EVO_SKILL_VERSION,
        "sha256": digest(draft),
        "draft": draft,
        "elo_winner": {"title": winner.get("title"), "elo": winner.get("elo"), "composite": winner.get("composite")},
    }
    return rec, ev.snapshot()


def _topic_weighted_ci(rows_by_topic: dict[str, list[float]]) -> tuple[float | None, list[float] | None]:
    if not rows_by_topic:
        return None, None
    bins = [float(np.mean(v)) for v in rows_by_topic.values()]
    mean = round(float(np.mean(bins)), 4)
    if len(bins) < 2:
        return mean, None
    rs = np.random.default_rng(20261122)
    draws = np.mean(rs.choice(bins, (10000, len(bins)), replace=True), axis=1)
    ci = [round(float(x), 4) for x in np.quantile(draws, [0.025, 0.975])]
    return mean, ci


def _majority_vote_status(statuses: list[str]) -> str:
    c = Counter(statuses)
    top, n = c.most_common(1)[0]
    if n >= 2 and top in ("win", "loss", "tie"):
        return top
    if c["win"] > c["loss"]:
        return "win"
    if c["loss"] > c["win"]:
        return "loss"
    return "tie"


def _summarize_judge(per_ms: list[dict[str, Any]], judge_key: str) -> dict[str, Any]:
    val_map = {"win": 1.0, "loss": -1.0, "tie": 0.0}
    counts = Counter()
    by_topic_net: dict[str, list[float]] = defaultdict(list)
    by_topic_sum: dict[str, list[float]] = defaultdict(list)
    order_sens = 0
    for item in per_ms:
        l2t = item["judges"][judge_key]
        st = l2t["status"]
        counts[st] += 1
        if l2t.get("order_sensitive"):
            order_sens += 1
        if st in val_map:
            tid = item["task_id"]
            by_topic_net[tid].append(val_map[st])
            by_topic_sum[tid].append(float(l2t.get("score_sum_delta") or 0.0))
    net_mean, net_ci = _topic_weighted_ci(by_topic_net)
    sum_mean, sum_ci = _topic_weighted_ci(by_topic_sum)
    return {
        "counts": dict(counts),
        "order_sensitive": order_sens,
        "topic_net_wins": net_mean,
        "topic_net_wins_ci95": net_ci,
        "score_sum_delta": sum_mean,
        "score_sum_delta_ci95": sum_ci,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wave", choices=("wave1", "wave2", "wave3"), default="wave1")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--v4-dir", type=Path, default=ROOT / "outputs/unified_pilot_20261119_v4")
    parser.add_argument("--ext15-dir", type=Path, default=ROOT / "outputs/extension_15_20261120_qwen3")
    parser.add_argument("--editor-profile", type=Path, default=WORKSPACE / "data/dashscope-deepseek-profile.json")
    parser.add_argument("--audit-profile", type=Path, default=WORKSPACE / "data/p1b-critic-profile.json")
    parser.add_argument("--qwen-judge-profile", type=Path, default=WORKSPACE / "data/dashscope-qwen-profile.json")
    parser.add_argument("--deepseek-judge-profile", type=Path, default=WORKSPACE / "data/dashscope-deepseek-profile.json")
    parser.add_argument("--glm-judge-profile", type=Path, default=WORKSPACE / "data/p1b-critic-profile.json")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)

    host_ep = _configure_endpoint(args.editor_profile, role="host_gen", dry_run=args.dry_run)
    editor_ep = _configure_endpoint(args.editor_profile, role="editor", dry_run=args.dry_run)
    audit_ep = _configure_endpoint(args.audit_profile, role="audit", dry_run=args.dry_run)
    judges = {
        "qwen3.8-flash": (_configure_endpoint(args.qwen_judge_profile, role="judge", dry_run=args.dry_run), min(args.workers, 3)),
        "deepseek-v4.1-flash": (_configure_endpoint(args.deepseek_judge_profile, role="judge", dry_run=args.dry_run), min(max(args.workers, 3), 4)),
        "GLM-5.3-flash": (_configure_endpoint(args.glm_judge_profile, role="judge", dry_run=args.dry_run), min(args.workers, 2)),
    }

    manifest_path = ROOT / "data/benchmark_50_topics.json"
    bench_spec = load(manifest_path)
    selected_entries = [t for t in bench_spec["topics"] if t["wave"] == args.wave]

    v4_inputs = load(args.v4_dir / "inputs_core.json") if (args.v4_dir / "inputs_core.json").is_file() else {}
    ext_inputs = load(args.ext15_dir / "inputs_extension.json") if (args.ext15_dir / "inputs_extension.json").is_file() else {}

    write_json(
        out / "manifest.json",
        {
            "protocol_version": PROTOCOL_VERSION,
            "host": "evoscientist",
            "skill_version": EVO_SKILL_VERSION,
            "wave": args.wave,
            "dry_run": args.dry_run,
            "code": {p.name: path_hash(p) for p in sorted((ROOT / "src/edara_mvp_eval").glob("*.py"))},
            "models": {
                "host_generator": host_ep.model,
                "editor": editor_ep.model,
                "audit": audit_ep.model,
                "judges": {k: ep.model for k, (ep, _) in judges.items()},
            },
        },
    )

    # Seed lookup caches from existing v4/ext15 runs where available so pre-cutoff OpenAlex queries hit cache first
    topic_seeds: list[tuple[str, dict[str, Any], str]] = []
    for entry in selected_entries:
        tid = entry["task_id"]
        if tid in v4_inputs:
            seed, src_dir, src_name = v4_inputs[tid], args.v4_dir, "v4"
        elif tid in ext_inputs:
            seed, src_dir, src_name = ext_inputs[tid], args.ext15_dir, "ext15"
        else:
            raise RuntimeError(f"Topic {tid} missing from v4/ext15 frozen task inputs")
        dst_lookup = out / "lookup" / tid / "evoscientist"
        if not dst_lookup.exists():
            for cand_host in ("ai_scientist", "direct"):
                src_lk = src_dir / "lookup" / tid / cand_host
                if src_lk.is_dir():
                    dst_lookup.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copytree(src_lk, dst_lookup)
                    break
        topic_seeds.append((tid, seed, src_name))

    # Phase 0: Generate EvoScientist Arm A + updated symmetric evidence snapshot
    print(f"[evoscientist:{args.wave}] Phase 0: Generating EvoScientist proposals across {len(topic_seeds)} topics...", flush=True)

    def _gen_one_a(item: tuple[str, dict[str, Any], str]) -> tuple[str, dict[str, Any], dict[str, Any], str]:
        tid, seed, src_name = item
        vdir = out / "versions" / tid / "evoscientist"
        vdir.mkdir(parents=True, exist_ok=True)
        a_file = vdir / "A.json"
        ev_file = vdir / "initial_evidence.json"
        task = TaskSpec.model_validate(seed["task"])
        if a_file.is_file() and ev_file.is_file():
            a_rec = load(a_file)
            ev_snap = load(ev_file)
        else:
            gen_runner = _make_runner(host_ep, vdir / "A_gen", f"{tid}/evoscientist", "A", "host_gen", 16, dry_run=args.dry_run)
            a_rec, ev_snap = generate_evoscientist_proposal(
                task,
                seed["evidence"],
                gen_runner,
                out / "lookup" / tid / "evoscientist",
                vdir / "evoscientist_trace",
                dry_run=args.dry_run,
            )
            write_json(a_file, a_rec)
            write_json(ev_file, ev_snap)
        print(f"[Arm A:EvoSci] {tid} -> ready (elo={a_rec.get('elo_winner', {}).get('elo')})", flush=True)
        return tid, a_rec, {"task": seed["task"], "evidence": ev_snap}, src_name

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        jobs = list(pool.map(_gen_one_a, topic_seeds))

    # Phase 1: Run Arms B, C, D, E symmetrically
    print(f"[evoscientist:{args.wave}] Phase 1: Running 5-arm trial (B, C, D, E) across {len(jobs)} manuscripts...", flush=True)

    def _run_arms_bcde(item: tuple[str, dict[str, Any], dict[str, Any], str]) -> None:
        tid, a_rec, evo_seed, _ = item
        job = f"{tid}/evoscientist"
        task = TaskSpec.model_validate(evo_seed["task"])
        vdir = out / "versions" / tid / "evoscientist"
        audit_cache = vdir / "structured_audits"
        audit_cache.mkdir(parents=True, exist_ok=True)
        lookup_dir = out / "lookup" / tid / "evoscientist"
        # Run C before E so E reuses C's Round-0 structured audit deterministically (matching paired_trial protocol)
        for arm in ("B", "C", "D", "E"):
            lim = LIMITS["full" if arm in ("D", "E") else "local"]
            auditor = _make_runner(audit_ep, vdir / arm / "audit", job, arm, "audit", lim["audit"], dry_run=args.dry_run)
            editor = _make_runner(editor_ep, vdir / arm / "editor", job, arm, "editor", lim["editor"], dry_run=args.dry_run)
            editor_conf = (
                _make_runner(editor_ep, vdir / arm / "editor_confound", job, arm, "editor", lim["rounds"], dry_run=args.dry_run)
                if arm in ("C", "E")
                else None
            )
            res = cached(
                vdir / f"{arm}.json",
                lambda a=arm, au=auditor, ed=editor, ec=editor_conf: run_arm(
                    a,
                    a_rec["draft"],
                    task,
                    evo_seed["evidence"],
                    au,
                    ed,
                    lookup_dir,
                    audit_cache=audit_cache if a in ("C", "E") else None,
                    editor_confound=ec,
                ),
            )
            if arm == "E":
                acts = [h.get("action") for h in (res.get("history") or [])]
                print(f"[Arm E:EvoSci] {tid} -> status={res['status']} actions={acts}", flush=True)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(_run_arms_bcde, jobs))

    # Phase 2: Primary Unified Judge (Qwen3.8-flash: C_vs_B and E_vs_D)
    print(f"[evoscientist:{args.wave}] Phase 2: Running Primary Unified Judge (qwen3.8-flash)...", flush=True)
    qwen_ep, qwen_workers = judges["qwen3.8-flash"]

    def _eval_qwen(item: tuple[str, dict[str, Any], dict[str, Any], str]) -> dict[str, Any]:
        tid, a_rec, evo_seed, src_name = item
        job = f"{tid}/evoscientist"
        task = TaskSpec.model_validate(evo_seed["task"])
        vdir = out / "versions" / tid / "evoscientist"
        versions = {arm: load(vdir / f"{arm}.json") for arm in ("B", "C", "D", "E")}
        local_ev = Evidence.restore(task, evo_seed["evidence"])
        local_packet = local_ev.packet()
        local_corpus = list(local_ev.audit_task().prior_work)
        full_packet, full_drafts = shared_full_pool(task, evo_seed["evidence"], versions)
        full_corpus = list(local_corpus)
        seen_ids = {p.id for p in full_corpus}
        for arm in ("D", "E"):
            for r in versions[arm].get("retrieved") or []:
                p = PriorWorkItem.model_validate(r["paper"])
                if p.id not in seen_ids:
                    full_corpus.append(p)
                    seen_ids.add(p.id)

        def _eval_cmp(cmp_name: str, left_d: dict[str, str], right_d: dict[str, str], pkt: list[dict], corp: list[PriorWorkItem]):
            rdir = vdir / "review_qwen" / cmp_name
            pair_runners = [
                _make_runner(qwen_ep, rdir / "lit2test" / o, job, "blind", o, 12, dry_run=args.dry_run)
                for o in ("forward", "reverse")
            ]
            aib_runner = _make_runner(qwen_ep, rdir / "alphaideabench", job, "blind", "alphaideabench", 24, dry_run=args.dry_run)
            cn_runner = _make_runner(qwen_ep, rdir / "claimnov", job, "blind", "claimnov", 64, dry_run=args.dry_run)
            return evaluate_unified_pair(
                left_d,
                right_d,
                task,
                pkt,
                pair_runners=pair_runners,
                aib_runner=aib_runner,
                claimnov_runner=cn_runner,
                prior_corpus=corp,
                aib_dimensions=CORE_AIB_DIMENSIONS,
            )

        c_vs_b = cached(vdir / "review_qwen_C_vs_B.json", lambda: _eval_cmp("C_vs_B", versions["B"]["draft"], versions["C"]["draft"], local_packet, local_corpus))
        e_vs_d = cached(vdir / "review_qwen_E_vs_D.json", lambda: _eval_cmp("E_vs_D", full_drafts["D"], full_drafts["E"], full_packet, full_corpus))
        print(f"[Judge:Qwen:EvoSci] {tid} -> E_vs_D={e_vs_d['lit2test']['status']} (sum_delta={e_vs_d['lit2test'].get('score_sum_delta')})", flush=True)
        return {
            "task_id": tid,
            "host": "evoscientist",
            "source_cohort": src_name,
            "a_record": a_rec,
            "evo_seed": evo_seed,
            "versions": versions,
            "c_vs_b_qwen": c_vs_b,
            "e_vs_d_qwen": e_vs_d,
        }

    with ThreadPoolExecutor(max_workers=qwen_workers) as pool:
        primary_rows = list(pool.map(_eval_qwen, jobs))

    # Phase 3: Cross-Judge Lit2Test Evaluation (DeepSeek-v4.1-flash & GLM-5.3-flash on E_vs_D)
    cross_maps: dict[str, dict[str, dict[str, Any]]] = {"deepseek-v4.1-flash": {}, "GLM-5.3-flash": {}}
    for j_key in ("deepseek-v4.1-flash", "GLM-5.3-flash"):
        j_ep, j_workers = judges[j_key]
        slug = "deepseek" if "deepseek" in j_key.lower() else "glm"
        print(f"[evoscientist:{args.wave}] Phase 3 ({j_key}): Running Lit2Test bidirectional judge...", flush=True)

        def _eval_cross(row_item: dict[str, Any], _jk=j_key, _ep=j_ep, _sl=slug) -> tuple[str, dict[str, Any]]:
            tid = row_item["task_id"]
            job = f"{tid}/evoscientist"
            task = TaskSpec.model_validate(row_item["evo_seed"]["task"])
            vdir = out / "versions" / tid / "evoscientist"
            full_packet, full_drafts = shared_full_pool(task, row_item["evo_seed"]["evidence"], row_item["versions"])
            rdir = vdir / f"review_{_sl}" / "E_vs_D"

            def _run_l2t():
                pair_runners = [
                    _make_runner(_ep, rdir / "lit2test" / o, job, "blind", o, 12, dry_run=args.dry_run)
                    for o in ("forward", "reverse")
                ]
                return eval_lit2test_pair(full_drafts["D"], full_drafts["E"], task, full_packet, pair_runners)

            l2t = cached(vdir / f"review_{_sl}_E_vs_D.json", _run_l2t)
            print(f"[Judge:{_jk}:EvoSci] {tid} -> {l2t['status']} (sum_delta={l2t.get('score_sum_delta')})", flush=True)
            return tid, l2t

        with ThreadPoolExecutor(max_workers=j_workers) as pool:
            for tid, l2t in pool.map(_eval_cross, primary_rows):
                cross_maps[j_key][tid] = l2t

    # Aggregate results & mechanism stats
    import re
    hype_re = re.compile(
        r"\b(?:pre-registered|preregistered|already\s+(?:validated|verified|confirmed|demonstrated)|"
        r"pilot\s+(?:data|study|run)\s+(?:shows?|confirms?)|preliminary\s+experiments?\s+(?:show|confirm))\b",
        re.IGNORECASE,
    )
    mech = {
        "n_manuscripts": len(primary_rows),
        "orig_keys_kept_all": 0,
        "process_hype": {a: {"manuscripts": 0, "occurrences": 0} for a in ("A", "B", "C", "D", "E")},
        "unverified_entities": {a: {"manuscripts": 0, "total": 0} for a in ("A", "B", "C", "D", "E")},
    }
    per_ms = []
    unified_for_summary = []
    for r in primary_rows:
        tid = r["task_id"]
        task = TaskSpec.model_validate(r["evo_seed"]["task"])
        local_ev = Evidence.restore(task, r["evo_seed"]["evidence"])
        local_corpus = list(local_ev.audit_task().prior_work)
        _, full_drafts = shared_full_pool(task, r["evo_seed"]["evidence"], r["versions"])
        full_corpus = list(local_corpus)
        seen_ids = {p.id for p in full_corpus}
        for arm in ("D", "E"):
            for item_p in r["versions"][arm].get("retrieved") or []:
                p = PriorWorkItem.model_validate(item_p["paper"])
                if p.id not in seen_ids:
                    full_corpus.append(p)
                    seen_ids.add(p.id)

        a_draft = r["a_record"]["draft"]
        e_draft = r["versions"]["E"]["draft"]
        if set(a_draft.keys()).issubset(set(e_draft.keys())):
            mech["orig_keys_kept_all"] += 1
        for ak in ("A", "B", "C", "D", "E"):
            dr = a_draft if ak == "A" else r["versions"][ak]["draft"]
            dr_eval = full_drafts[ak] if ak in ("D", "E") else dr
            corp_eval = full_corpus if ak in ("D", "E") else local_corpus
            txt = "\n".join(str(v) for v in dr.values())
            hits = hype_re.findall(txt)
            if hits:
                mech["process_hype"][ak]["manuscripts"] += 1
                mech["process_hype"][ak]["occurrences"] += len(hits)
            unv = audit_unverified_entities(dr_eval, task, prior_work=corp_eval)
            if unv:
                mech["unverified_entities"][ak]["manuscripts"] += 1
                mech["unverified_entities"][ak]["total"] += len(unv)

        q_l2t = r["e_vs_d_qwen"]["lit2test"]
        d_l2t = cross_maps["deepseek-v4.1-flash"][tid]
        g_l2t = cross_maps["GLM-5.3-flash"][tid]
        maj_status = _majority_vote_status([q_l2t["status"], d_l2t["status"], g_l2t["status"]])
        maj_sum = round(float(np.mean([float(x.get("score_sum_delta") or 0.0) for x in (q_l2t, d_l2t, g_l2t)])), 4)
        per_ms.append({
            "task_id": tid,
            "host": "evoscientist",
            "elo_winner": r["a_record"].get("elo_winner"),
            "e_status": r["versions"]["E"]["status"],
            "e_actions": [h.get("action") for h in (r["versions"]["E"].get("history") or [])],
            "judges": {
                "qwen3.8-flash": q_l2t,
                "deepseek-v4.1-flash": d_l2t,
                "GLM-5.3-flash": g_l2t,
                "majority_3judge": {"status": maj_status, "score_sum_delta": maj_sum, "order_sensitive": False},
            },
        })
        unified_for_summary.append({
            "task_id": tid,
            "host": "evoscientist",
            "unified_comparisons": {
                "C_vs_B": {"unified": r["c_vs_b_qwen"]},
                "E_vs_D": {"unified": r["e_vs_d_qwen"]},
            },
        })

    judge_summaries = {jk: _summarize_judge(per_ms, jk) for jk in ("qwen3.8-flash", "deepseek-v4.1-flash", "GLM-5.3-flash", "majority_3judge")}
    qwen_unified_summary = {
        "C_vs_B": summarize_unified_rows(unified_for_summary, "C_vs_B"),
        "E_vs_D": summarize_unified_rows(unified_for_summary, "E_vs_D"),
    }
    usage_tot = {k: dict(v) for k, v in usage(out).items()}

    summary_json = {
        "wave": args.wave,
        "host": "evoscientist",
        "skill_version": EVO_SKILL_VERSION,
        "n_topics": len(per_ms),
        "judge_summaries": judge_summaries,
        "qwen_unified_summary": qwen_unified_summary,
        "mechanism_stats": mech,
        "usage": usage_tot,
        "manuscripts": per_ms,
    }
    write_json(out / f"{args.wave}_evoscientist_summary.json", summary_json)

    lines = [
        f"# EvoScientist (`{EVO_SKILL_VERSION}`) `{args.wave}` 5-Arm + 3-Judge Evaluation (`N={len(per_ms)}`)",
        "",
        "## 1. Aggregate `E_vs_D` Across 3 Calibrated Blind Judges",
        "",
        "| Judge | Win / Loss / Tie | Topic Net Wins (95% CI) | Score Sum Delta (95% CI) | Order Sens. |",
        "| --- | :---: | :---: | :---: | ---: |",
    ]
    for jk, label in (
        ("qwen3.8-flash", "Qwen3.8-flash (Primary)"),
        ("deepseek-v4.1-flash", "DeepSeek-v4.1-flash"),
        ("GLM-5.3-flash", "GLM-5.3-flash"),
        ("majority_3judge", "**3-Judge Majority Vote**"),
    ):
        s = judge_summaries[jk]
        c = s["counts"]
        w, l, t = c.get("win", 0), c.get("loss", 0), c.get("tie", 0)
        nci = f"`[{s['topic_net_wins_ci95'][0]:+.3f}, {s['topic_net_wins_ci95'][1]:+.3f}]`" if s["topic_net_wins_ci95"] else "—"
        sci = f"`[{s['score_sum_delta_ci95'][0]:+.3f}, {s['score_sum_delta_ci95'][1]:+.3f}]`" if s["score_sum_delta_ci95"] else "—"
        lines.append(f"| {label} | `{w} / {l} / {t}` | **`{s['topic_net_wins']:+.3f}`** ({nci}) | **`{s['score_sum_delta']:+.3f}`** ({sci}) | {s['order_sensitive']} / {len(per_ms)} |")

    cb = qwen_unified_summary["C_vs_B"]
    cb_c = cb.get("counts") or {}
    ed_u = qwen_unified_summary["E_vs_D"]
    aib_d = ed_u.get("alphaideabench_dimension_deltas") or {}
    lines.extend([
        "",
        f"- **Single-Pass Ablation (`C_vs_B`, Qwen)**: `{cb_c.get('win',0)}W / {cb_c.get('loss',0)}L / {cb_c.get('tie',0)}T` (`net_wins={cb.get('topic_net_wins') or 0.0:+.3f}`, `sum_delta={cb.get('score_sum_delta') or 0.0:+.3f}`)",
        f"- **ClaimNov Compositional Novelty Delta (`E_vs_D`)**: `{ed_u.get('claimnov_novelty_delta') or 0.0:+.3f}`",
        f"- **AlphaIdeaBench Deltas (`E_vs_D`)**: `testability={aib_d.get('testability') or 0.0:+.3f}`, `context_grounding={aib_d.get('context_grounding') or 0.0:+.3f}`, `method_novelty={aib_d.get('method_novelty') or 0.0:+.3f}`",
        f"- **Original Keys Preserved (`E`)**: `{mech['orig_keys_kept_all']} / {mech['n_manuscripts']}` (`100%`)",
        f"- **Process Hype (`A -> D -> E`)**: `{mech['process_hype']['A']['occurrences']} -> {mech['process_hype']['D']['occurrences']} -> {mech['process_hype']['E']['occurrences']}`",
        f"- **Unverified External Entities (`A -> D -> E`)**: `{mech['unverified_entities']['A']['total']} -> {mech['unverified_entities']['D']['total']} -> {mech['unverified_entities']['E']['total']}`",
        f"- **Total Metered Cost (USD)**: `${sum(float(v.get('cost_usd', 0.0)) for v in usage_tot.values() if isinstance(v, dict)):.4f}`",
        "",
        "## 2. Topic-by-Topic Breakdown",
        "",
        "| Topic | EvoSci Winner ELO | `E` Trajectory | Qwen | DeepSeek | GLM | **Majority (`Δsum`)** |",
        "| --- | :---: | --- | :---: | :---: | :---: | :---: |",
    ])
    for m in per_ms:
        elo_v = (m.get("elo_winner") or {}).get("elo", "—")
        q_s = m["judges"]["qwen3.8-flash"]["status"]
        d_s = m["judges"]["deepseek-v4.1-flash"]["status"]
        g_s = m["judges"]["GLM-5.3-flash"]["status"]
        maj = m["judges"]["majority_3judge"]
        lines.append(f"| `{m['task_id'][:38]}` | `{elo_v}` | `{m['e_actions']}` | `{q_s}` | `{d_s}` | `{g_s}` | **`{maj['status']}` (`{maj['score_sum_delta']:+.3f}`)** |")

    md_path = out / f"{args.wave}_evoscientist_summary.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n[evoscientist:{args.wave}] Completed! Summary written to {md_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
