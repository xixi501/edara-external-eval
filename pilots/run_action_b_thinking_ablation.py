"""Action B: Slow-Thinking / Test-Time Reasoning Scaling + Symmetric 8192-Token Budget Ablation (N=30 Manuscripts).

Reuses the exact frozen Wave-1 `Arm A` proposals across all 10 domains and all 3 external host paradigms
(`Direct LLM`, `AI-Scientist-v2`, `EvoScientist v2.1.2`), executes both `Arm D_think` (unconstrained
retrieve-and-revise) and `Arm E_think` (governed middleware v0.2) symmetrically with native slow-thinking
(`thinking: {"type": "enabled"}`) and a raised `8192`-token content cap (`16384` total API max_tokens),
evaluates `E_think vs D_think` under the exact frozen 3-judge blind panel, runs a 10-topic `Direct`
slow-thinking generation hygiene probe (`Arm A_direct_think`), and produces a 2x2 factorial comparison
against Wave-1 (`Fast 4096` vs `Slow-Thinking 8192`).
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import json
from pathlib import Path
import shutil
import sys
from typing import Any

import numpy as np

PILOTS_DIR = Path(__file__).resolve().parent
if str(PILOTS_DIR) not in sys.path:
    sys.path.insert(0, str(PILOTS_DIR))

from edara_mvp_eval.cli import _generate_direct
from edara_mvp_eval.factual_audit import audit_unverified_entities
from edara_mvp_eval.host_adapter import (
    confound_addressed,
    extract_design_elements,
    preserves_design_elements,
)
from edara_mvp_eval.llm_client import LLMRunner, ModelEndpoint
from edara_mvp_eval.paired_inputs import Evidence, direct_aliases, project
from edara_mvp_eval.paired_trial import (
    LIMITS,
    OUTPUT_TOKENS,
    PROTOCOL_VERSION,
    ROOT,
    TrialRunner,
    WORKSPACE,
    cached,
    confound_targets,
    digest,
    load,
    path_hash,
    run_arm,
    shared_full_pool,
    usage,
)
from edara_mvp_eval.schemas import PriorWorkItem, TaskSpec
from edara_mvp_eval.stage_b_evidence import write_json
from edara_mvp_eval.unified_eval import (
    CORE_AIB_DIMENSIONS,
    LIT2TEST_DIMENSIONS,
    eval_lit2test_pair,
    evaluate_unified_pair,
    summarize_unified_rows,
)
from run_benchmark50_wave import _count_process_hype

THINKING_OUTPUT_TOKENS = 8192


def _configure_endpoint(profile_path: Path, *, role: str, dry_run: bool) -> ModelEndpoint:
    if dry_run:
        return ModelEndpoint(model=f"dry-run-{role}")
    ep = ModelEndpoint.from_env_or_profile(profile_path=profile_path)
    low = ep.model.lower()
    if role in {"editor_think", "host_gen_think"}:
        if "deepseek" in low:
            ep = replace(ep, extra_body={"thinking": {"type": "enabled"}})
        elif "qwen" in low:
            ep = replace(ep, extra_body={"enable_thinking": True})
        elif "glm" in low:
            ep = replace(ep, extra_body={"thinking": {"type": "enabled"}})
    else:
        # Audit & 3-Judge blind panel strictly frozen (`thinking: disabled`) to preserve exact rubric scale
        if "qwen" in low:
            ep = replace(ep, extra_body={"enable_thinking": False})
        elif "deepseek" in low:
            ep = replace(ep, extra_body={"thinking": {"type": "disabled"}})
        elif "glm" in low:
            extra: dict[str, Any] = {"thinking": {"type": "disabled"}}
            if "autodl" in (ep.base_url or "").lower():
                extra["reasoning_effort"] = "none"
            ep = replace(ep, extra_body=extra)
    if not ep.resolve_api_key():
        raise RuntimeError(f"Missing credential in {ep.credential_env} for {profile_path.name}")
    return ep


def _make_runner(
    endpoint: ModelEndpoint,
    log_dir: Path,
    job: str,
    arm: str,
    stage: str,
    limit: int,
    *,
    output_tokens: int = OUTPUT_TOKENS,
    dry_run: bool,
):
    if dry_run:
        r = LLMRunner(endpoint, dry_run=True)
        r.index = 0  # type: ignore[attr-defined]
        r.reused = 0  # type: ignore[attr-defined]
        return r
    return TrialRunner(endpoint, log_dir, task_id=job, arm=arm, stage=stage, limit=limit, output_tokens=output_tokens)


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
    return "win" if c["win"] > c["loss"] else ("loss" if c["loss"] > c["win"] else "tie")


def _summarize_judge(per_ms: list[dict[str, Any]], judge_key: str) -> dict[str, Any]:
    val_map = {"win": 1.0, "loss": -1.0, "tie": 0.0}
    counts = Counter()
    by_topic_net: dict[str, list[float]] = defaultdict(list)
    by_topic_sum: dict[str, list[float]] = defaultdict(list)
    by_topic_dim: dict[str, dict[str, list[float]]] = {d: defaultdict(list) for d in LIT2TEST_DIMENSIONS}
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
            for d in LIT2TEST_DIMENSIONS:
                by_topic_dim[d][tid].append(float((l2t.get("dimension_deltas") or {}).get(d, 0.0)))
    net_mean, net_ci = _topic_weighted_ci(by_topic_net)
    sum_mean, sum_ci = _topic_weighted_ci(by_topic_sum)
    dim_means = {d: _topic_weighted_ci(by_topic_dim[d])[0] for d in LIT2TEST_DIMENSIONS}
    return {
        "counts": dict(counts),
        "order_sensitive": order_sens,
        "topic_net_wins": net_mean,
        "topic_net_wins_ci95": net_ci,
        "score_sum_delta": sum_mean,
        "score_sum_delta_ci95": sum_ci,
        "dimension_deltas": dim_means,
    }


def _compute_mechanism_stats(jobs: list[tuple[str, str, dict[str, Any], dict[str, Any], dict[str, Any]]], out_dir: Path) -> dict[str, Any]:
    stats: dict[str, Any] = {
        "n_manuscripts": len(jobs),
        "E1_orig_keys_kept_all": 0,
        "E1_new_keys_added": 0,
        "E2_process_hype": {arm: {"manuscripts": 0, "occurrences": 0} for arm in ("A", "D", "E")},
        "E2_unverified_entities": {arm: {"manuscripts": 0, "total": 0} for arm in ("A", "D", "E")},
        "E3_confounds": {"targets_required_n": 0, "A_addressed": 0, "D_addressed": 0, "E_addressed": 0},
        "E7_trajectory": {
            "E_changed_n": 0,
            "E_error_n": 0,
            "E_clean_closed_loop_n": 0,
            "E_consolidated_n": 0,
            "A_vs_E_design_violation": 0,
        },
        "design_elements_all_three": {arm: 0 for arm in ("A", "D", "E")},
    }
    for tid, host, a_rec, seed, _ in jobs:
        vdir = out_dir / "versions" / tid / host
        task = TaskSpec.model_validate(seed["task"])
        ev = Evidence.restore(task, seed["evidence"])
        a_draft = a_rec["draft"]
        versions = {arm: load(vdir / f"{arm}.json") for arm in ("D", "E")}
        arm_drafts = {"A": a_draft, "D": versions["D"]["draft"], "E": versions["E"]["draft"]}

        e_draft = arm_drafts["E"]
        if set(a_draft.keys()) <= set(e_draft.keys()):
            stats["E1_orig_keys_kept_all"] += 1
        stats["E1_new_keys_added"] += len(set(e_draft.keys()) - set(a_draft.keys()))

        e_hist = versions["E"].get("history") or []
        targets = confound_targets(e_hist[0]["diagnosis"]) if (e_hist and "diagnosis" in e_hist[0]) else []
        if targets:
            stats["E3_confounds"]["targets_required_n"] += 1
            if confound_addressed(a_draft, targets):
                stats["E3_confounds"]["A_addressed"] += 1
            if confound_addressed(arm_drafts["D"], targets, baseline_draft=a_draft) or confound_addressed(arm_drafts["D"], targets):
                stats["E3_confounds"]["D_addressed"] += 1
            if confound_addressed(e_draft, targets, baseline_draft=a_draft) or confound_addressed(e_draft, targets):
                stats["E3_confounds"]["E_addressed"] += 1

        full_packet, full_drafts = shared_full_pool(task, seed["evidence"], versions)
        local_corpus = list(ev.audit_task().prior_work)
        full_corpus = list(local_corpus)
        seen_p = {p.id for p in full_corpus}
        for arm_r in ("D", "E"):
            for r_item in versions[arm_r].get("retrieved") or []:
                pw = PriorWorkItem.model_validate(r_item["paper"])
                if pw.id not in seen_p:
                    full_corpus.append(pw)
                    seen_p.add(pw.id)

        for arm_k, dr in arm_drafts.items():
            hype = _count_process_hype(dr)
            if hype > 0:
                stats["E2_process_hype"][arm_k]["manuscripts"] += 1
                stats["E2_process_hype"][arm_k]["occurrences"] += hype
            dr_eval = full_drafts[arm_k] if arm_k in ("D", "E") else dr
            corpus_eval = full_corpus if arm_k in ("D", "E") else local_corpus
            unver = audit_unverified_entities(dr_eval, task, prior_work=corpus_eval)
            if unver:
                stats["E2_unverified_entities"][arm_k]["manuscripts"] += 1
                stats["E2_unverified_entities"][arm_k]["total"] += len(unver)
            elems = extract_design_elements(dr)
            if elems["control"] and elems["metric"] and elems["decision"]:
                stats["design_elements_all_three"][arm_k] += 1

        e_row = versions["E"]
        if e_row["status"] == "changed":
            stats["E7_trajectory"]["E_changed_n"] += 1
        elif e_row["status"] != "retained":
            stats["E7_trajectory"]["E_error_n"] += 1
        actions = [h.get("action") for h in e_hist]
        if e_row["status"] in ("changed", "retained") and (
            not actions or actions[-1] in ("stop", "retain_and_test", "guided_revise")
        ):
            stats["E7_trajectory"]["E_clean_closed_loop_n"] += 1
        if len([a for a in actions if a == "guided_revise"]) >= 2:
            stats["E7_trajectory"]["E_consolidated_n"] += 1
        ok, _ = preserves_design_elements(a_draft, e_draft)
        if not ok:
            stats["E7_trajectory"]["A_vs_E_design_violation"] += 1
    return stats


def _compute_fast_wave1_reference(
    jobs: list[tuple[str, str, dict[str, Any], dict[str, Any], dict[str, Any]]],
    mh_dir: Path,
    evo_dir: Path,
) -> dict[str, Any]:
    """Compute exact 1:1 paired Fast-4096 Wave-1 metrics on the identical 30 manuscripts."""
    fast_ms = [fast_rec for _, _, _, _, fast_rec in jobs if fast_rec]
    if not fast_ms:
        return {}
    fast_judges = {}
    for jk in ("qwen3.8-flash", "deepseek-v4.1-flash", "GLM-5.3-flash"):
        fast_judges[jk] = _summarize_judge(fast_ms, jk)
    # Normalize majority key name across mh (`majority_3way`) and evosci (`majority_3judge`)
    norm_maj_ms = []
    for fm in fast_ms:
        mj = fm["judges"].get("majority_3way") or fm["judges"].get("majority_3judge") or {}
        norm_maj_ms.append({**fm, "judges": {"majority_3way": mj}})
    fast_judges["majority_3way"] = _summarize_judge(norm_maj_ms, "majority_3way")

    hype_d = hype_e = unver_d = unver_e = conf_req = conf_d = conf_e = 0
    for tid, host, a_rec, seed, _ in jobs:
        src_root = evo_dir if host == "evoscientist" else mh_dir
        sv = src_root / "versions" / tid / host
        if not ((sv / "D.json").is_file() and (sv / "E.json").is_file()):
            continue
        task = TaskSpec.model_validate(seed["task"])
        ev = Evidence.restore(task, seed["evidence"])
        vers = {"D": load(sv / "D.json"), "E": load(sv / "E.json")}
        _, full_drafts = shared_full_pool(task, seed["evidence"], vers)
        full_corpus = list(ev.audit_task().prior_work)
        seen_p = {p.id for p in full_corpus}
        for arm_r in ("D", "E"):
            for r_item in vers[arm_r].get("retrieved") or []:
                pw = PriorWorkItem.model_validate(r_item["paper"])
                if pw.id not in seen_p:
                    full_corpus.append(pw)
                    seen_p.add(pw.id)
        hype_d += _count_process_hype(vers["D"]["draft"])
        hype_e += _count_process_hype(vers["E"]["draft"])
        unver_d += len(audit_unverified_entities(full_drafts["D"], task, prior_work=full_corpus))
        unver_e += len(audit_unverified_entities(full_drafts["E"], task, prior_work=full_corpus))
        eh = vers["E"].get("history") or []
        tgs = confound_targets(eh[0]["diagnosis"]) if (eh and "diagnosis" in eh[0]) else []
        if tgs:
            conf_req += 1
            conf_d += int(confound_addressed(vers["D"]["draft"], tgs, baseline_draft=a_rec["draft"]) or confound_addressed(vers["D"]["draft"], tgs))
            conf_e += int(confound_addressed(vers["E"]["draft"], tgs, baseline_draft=a_rec["draft"]) or confound_addressed(vers["E"]["draft"], tgs))

    return {
        "judge_summaries": fast_judges,
        "hygiene": {
            "D_process_hype": hype_d,
            "E_process_hype": hype_e,
            "D_unverified_entities": unver_d,
            "E_unverified_entities": unver_e,
            "confounds_required": conf_req,
            "D_confounds_addressed": conf_d,
            "E_confounds_addressed": conf_e,
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--wave1-mh-dir", type=Path, default=ROOT / "outputs/benchmark50_wave1_20261122_r2")
    parser.add_argument("--wave1-evosci-dir", type=Path, default=ROOT / "outputs/benchmark50_evosci_wave1_20261122")
    parser.add_argument("--editor-profile", type=Path, default=WORKSPACE / "data/dashscope-deepseek-profile.json")
    parser.add_argument("--audit-profile", type=Path, default=WORKSPACE / "data/p1b-critic-profile.json")
    parser.add_argument("--qwen-judge-profile", type=Path, default=WORKSPACE / "data/dashscope-qwen-profile.json")
    parser.add_argument("--deepseek-judge-profile", type=Path, default=WORKSPACE / "data/dashscope-deepseek-profile.json")
    parser.add_argument("--glm-judge-profile", type=Path, default=WORKSPACE / "data/p1b-critic-profile.json")
    parser.add_argument("--output-tokens", type=int, default=THINKING_OUTPUT_TOKENS)
    parser.add_argument("--limit-topics", type=int, default=0, help="Optional topic cap for offline dry-run tests")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    out_toks = int(args.output_tokens)

    editor_think_ep = _configure_endpoint(args.editor_profile, role="editor_think", dry_run=args.dry_run)
    gen_think_ep = _configure_endpoint(args.editor_profile, role="host_gen_think", dry_run=args.dry_run)
    audit_ep = _configure_endpoint(args.audit_profile, role="audit", dry_run=args.dry_run)
    judges = {
        "qwen3.8-flash": (_configure_endpoint(args.qwen_judge_profile, role="judge", dry_run=args.dry_run), min(args.workers, 3)),
        "deepseek-v4.1-flash": (_configure_endpoint(args.deepseek_judge_profile, role="judge", dry_run=args.dry_run), min(max(args.workers, 3), 4)),
        "GLM-5.3-flash": (_configure_endpoint(args.glm_judge_profile, role="judge", dry_run=args.dry_run), min(args.workers, 2)),
    }

    b50 = load(ROOT / "data/benchmark_50_topics.json")["topics"]
    b50_tasks_map = {t["task_id"]: TaskSpec.model_validate(t) for t in load(ROOT / "data/tasks_benchmark50.json")}
    selected_entries = [e for e in b50 if e["wave"] == "wave1"]
    if args.limit_topics > 0:
        selected_entries = selected_entries[: args.limit_topics]

    write_json(
        out / "manifest.json",
        {
            "protocol_version": PROTOCOL_VERSION,
            "experiment": "action_b_slow_thinking_8192_ablation",
            "wave": "wave1",
            "dry_run": args.dry_run,
            "output_tokens": out_toks,
            "editor_extra_body": editor_think_ep.extra_body,
            "code": {p.name: path_hash(p) for p in sorted((ROOT / "src/edara_mvp_eval").glob("*.py"))},
            "models": {
                "editor_think": editor_think_ep.model,
                "audit": audit_ep.model,
                "judges": {k: ep.model for k, (ep, _) in judges.items()},
            },
        },
    )

    mh_inputs = load(args.wave1_mh_dir / "inputs_wave1.json") if (args.wave1_mh_dir / "inputs_wave1.json").is_file() else {}
    mh_fast_map = (
        {(m["task_id"], m["host"]): m for m in load(args.wave1_mh_dir / "wave1_crossjudge_3way.json").get("manuscripts", [])}
        if (args.wave1_mh_dir / "wave1_crossjudge_3way.json").is_file()
        else {}
    )
    evo_fast_map = (
        {(m["task_id"], "evoscientist"): m for m in load(args.wave1_evosci_dir / "wave1_evoscientist_summary.json").get("manuscripts", [])}
        if (args.wave1_evosci_dir / "wave1_evoscientist_summary.json").is_file()
        else {}
    )

    # Seed frozen Wave 1 Arm A, initial Round-0 structured audits, and lookup caches (read-only from Wave 1)
    jobs: list[tuple[str, str, dict[str, Any], dict[str, Any], dict[str, Any]]] = []
    for entry in selected_entries:
        tid = entry["task_id"]
        raw_task = b50_tasks_map[tid].model_copy(update={"strict_evidence": True})
        for host in ("direct", "ai_scientist", "evoscientist"):
            src_root = args.wave1_evosci_dir if host == "evoscientist" else args.wave1_mh_dir
            src_vdir = src_root / "versions" / tid / host
            dst_vdir = out / "versions" / tid / host
            dst_vdir.mkdir(parents=True, exist_ok=True)

            if (src_vdir / "A.json").is_file() and not args.dry_run:
                if not (dst_vdir / "A.json").is_file():
                    shutil.copy2(src_vdir / "A.json", dst_vdir / "A.json")
                a_rec = load(dst_vdir / "A.json")
                if host == "evoscientist":
                    seed = {"task": raw_task.model_dump(), "evidence": load(src_vdir / "initial_evidence.json")}
                else:
                    seed = mh_inputs[tid]
                task_obj = TaskSpec.model_validate(seed["task"])
                init_ev = Evidence.restore(task_obj, seed["evidence"])
                init_key = digest([a_rec["draft"], init_ev.packet()])
                src_sa = src_vdir / "structured_audits" / f"{init_key}.json"
                dst_sa = dst_vdir / "structured_audits"
                dst_sa.mkdir(parents=True, exist_ok=True)
                if src_sa.is_file() and not (dst_sa / f"{init_key}.json").is_file():
                    shutil.copy2(src_sa, dst_sa / f"{init_key}.json")
                src_lk = src_root / "lookup" / tid / host
                dst_lk = out / "lookup" / tid / host
                if src_lk.is_dir() and not dst_lk.exists():
                    dst_lk.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copytree(src_lk, dst_lk)
                fast_rec = evo_fast_map.get((tid, host), {}) if host == "evoscientist" else mh_fast_map.get((tid, host), {})
            else:
                ev_dry = Evidence(raw_task)
                for p in raw_task.prior_work:
                    ev_dry.add(p)
                a_draft = {
                    "title": f"Dry-Run {host} Proposal for {tid}",
                    "hypothesis": "Targeted regularizer improves metric by >=5%.",
                    "method": "Apply pre-specified invariant gate.",
                    "minimal_test": "Compare against matched baseline across 5 seeds.",
                    "decisive_metric": "Macro-F1 across 5 seeds.",
                    "supporting_result": ">=5% gain.",
                    "falsifying_result": "<1% gain.",
                    "related_work": "Builds on supplied literature.",
                    "limitations": "Evaluated on standard academic splits.",
                }
                a_rec = {"status": "ready", "host": host, "stage": "proposal", "sha256": digest(a_draft), "draft": a_draft}
                write_json(dst_vdir / "A.json", a_rec)
                seed = {"task": raw_task.model_dump(), "evidence": ev_dry.snapshot()}
                fast_rec = {}
            jobs.append((tid, host, a_rec, seed, fast_rec))

    # Phase 0b: 10-Topic Direct Slow-Thinking Initial Generation Probe (`Arm A_direct_think`)
    print(f"[ActionB:Think] Phase 0b: Running {len(selected_entries)}-topic Direct slow-thinking initial generation probe...", flush=True)

    def _run_direct_think_probe(entry: dict[str, Any]) -> dict[str, Any]:
        tid = entry["task_id"]
        pdir = out / "direct_think_probe" / tid
        pdir.mkdir(parents=True, exist_ok=True)
        res_file = pdir / "probe_result.json"
        if res_file.is_file() and not args.dry_run:
            return load(res_file)

        raw_task = b50_tasks_map[tid].model_copy(update={"strict_evidence": True})
        exp_task_dir = ROOT / "outputs/repair_v3/expansion/tasks" / tid
        if (exp_task_dir / "prepared.json").is_file() and not args.dry_run:
            prepared_task = TaskSpec.model_validate(load(exp_task_dir / "prepared.json")["task"])
        elif tid in mh_inputs and not args.dry_run:
            prepared_task = TaskSpec.model_validate(mh_inputs[tid]["task"])
        else:
            prepared_task = raw_task

        if tid in mh_inputs and not args.dry_run:
            task_obj = TaskSpec.model_validate(mh_inputs[tid]["task"])
            ev_obj = Evidence.restore(task_obj, mh_inputs[tid]["evidence"])
            fast_a = load(args.wave1_mh_dir / "versions" / tid / "direct" / "A.json")["draft"]
        else:
            task_obj = raw_task
            ev_obj = Evidence(raw_task)
            for p in raw_task.prior_work:
                ev_obj.add(p)
            fast_a = {}

        probe_runner = _make_runner(
            gen_think_ep, pdir / "gen_logs", f"{tid}/direct_think_probe", "A_think", "host_gen", 6,
            output_tokens=out_toks, dry_run=args.dry_run,
        )
        raw_payload, _ = _generate_direct(prepared_task, probe_runner, max_tokens=out_toks)
        t_draft = project(raw_payload)
        t_alias = direct_aliases(raw_payload, ev_obj)
        for old_m, new_m in t_alias.items():
            t_draft = {k: v.replace(old_m, new_m) for k, v in t_draft.items()}

        local_corpus = list(ev_obj.audit_task().prior_work)
        think_hype = _count_process_hype(t_draft)
        think_unver = len(audit_unverified_entities(t_draft, task_obj, prior_work=local_corpus))
        fast_hype = _count_process_hype(fast_a) if fast_a else 0
        fast_unver = len(audit_unverified_entities(fast_a, task_obj, prior_work=local_corpus)) if fast_a else 0

        out_row = {
            "task_id": tid,
            "fast_A_process_hype": fast_hype,
            "think_A_process_hype": think_hype,
            "fast_A_unverified_entities": fast_unver,
            "think_A_unverified_entities": think_unver,
            "draft": t_draft,
        }
        if not args.dry_run:
            write_json(res_file, out_row)
        return out_row

    with ThreadPoolExecutor(max_workers=min(args.workers, 3)) as pool:
        probe_rows = list(pool.map(_run_direct_think_probe, selected_entries))

    # Phase 1: Symmetric Slow-Thinking (`thinking: enabled`, `output_tokens=8192`) Arm D_think and Arm E_think
    print(f"[ActionB:Think] Phase 1: Running symmetric D_think & E_think (output_tokens={out_toks}) across {len(jobs)} manuscripts...", flush=True)

    def _run_arms_de_think(item: tuple[str, str, dict[str, Any], dict[str, Any], dict[str, Any]]) -> None:
        tid, host, a_rec, seed, _ = item
        job = f"{tid}/{host}"
        task = TaskSpec.model_validate(seed["task"])
        vdir = out / "versions" / tid / host
        audit_cache = vdir / "structured_audits"
        audit_cache.mkdir(parents=True, exist_ok=True)
        lookup_dir = out / "lookup" / tid / host
        lim = LIMITS["full"]
        for arm in ("D", "E"):
            auditor = _make_runner(audit_ep, vdir / arm / "audit", job, arm, "audit", lim["audit"], output_tokens=OUTPUT_TOKENS, dry_run=args.dry_run)
            editor = _make_runner(editor_think_ep, vdir / arm / "editor", job, arm, "editor", lim["editor"], output_tokens=out_toks, dry_run=args.dry_run)
            editor_conf = (
                _make_runner(editor_think_ep, vdir / arm / "editor_confound", job, arm, "editor", lim["rounds"], output_tokens=out_toks, dry_run=args.dry_run)
                if arm == "E"
                else None
            )
            res = cached(
                vdir / f"{arm}.json",
                lambda a=arm, au=auditor, ed=editor, ec=editor_conf: run_arm(
                    a,
                    a_rec["draft"],
                    task,
                    seed["evidence"],
                    au,
                    ed,
                    lookup_dir,
                    audit_cache=audit_cache if a == "E" else None,
                    editor_confound=ec,
                    output_tokens=out_toks,
                ),
            )
            if arm == "E":
                acts = [h.get("action") for h in (res.get("history") or [])]
                print(f"[ActionB:ArmE_think] {job} -> status={res['status']} actions={acts}", flush=True)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(_run_arms_de_think, jobs))

    # Phase 2: Primary Unified Blind Judge (Qwen3.8-flash, frozen fast scale matching Wave 1)
    qwen_ep, qwen_workers = judges["qwen3.8-flash"]
    print(f"[ActionB:Think] Phase 2: Running Primary Unified Judge (qwen3.8-flash)...", flush=True)

    def _eval_qwen(item: tuple[str, str, dict[str, Any], dict[str, Any], dict[str, Any]]) -> dict[str, Any]:
        tid, host, a_rec, seed, _ = item
        job = f"{tid}/{host}"
        task = TaskSpec.model_validate(seed["task"])
        vdir = out / "versions" / tid / host
        versions = {arm: load(vdir / f"{arm}.json") for arm in ("D", "E")}
        full_packet, full_drafts = shared_full_pool(task, seed["evidence"], versions)
        local_ev = Evidence.restore(task, seed["evidence"])
        full_corpus = list(local_ev.audit_task().prior_work)
        seen_ids = {p.id for p in full_corpus}
        for arm in ("D", "E"):
            for r in versions[arm].get("retrieved") or []:
                p = PriorWorkItem.model_validate(r["paper"])
                if p.id not in seen_ids:
                    full_corpus.append(p)
                    seen_ids.add(p.id)

        rdir = vdir / "review_qwen" / "E_vs_D"

        def _compute_q():
            pair_runners = [
                _make_runner(qwen_ep, rdir / "lit2test" / o, job, "blind", o, 12, dry_run=args.dry_run)
                for o in ("forward", "reverse")
            ]
            aib_runner = _make_runner(qwen_ep, rdir / "alphaideabench", job, "blind", "alphaideabench", 24, dry_run=args.dry_run)
            cn_runner = _make_runner(qwen_ep, rdir / "claimnov", job, "blind", "claimnov", 64, dry_run=args.dry_run)
            return evaluate_unified_pair(
                full_drafts["D"],
                full_drafts["E"],
                task,
                full_packet,
                pair_runners=pair_runners,
                aib_runner=aib_runner,
                claimnov_runner=cn_runner,
                prior_corpus=full_corpus,
                aib_dimensions=CORE_AIB_DIMENSIONS,
            )

        unified_e_vs_d = cached(vdir / "review_qwen_E_vs_D.json", _compute_q)
        l2t = unified_e_vs_d["lit2test"]
        print(f"[ActionB:Judge:Qwen] {job} -> {l2t['status']} (sum_delta={l2t.get('score_sum_delta')})", flush=True)
        return {
            "task_id": tid,
            "host": host,
            "source_sha256": a_rec["sha256"],
            "versions": {arm: {k: versions[arm][k] for k in ("status", "sha256", "edit_volume", "queries")} for arm in ("D", "E")},
            "e_actions": [h.get("action") for h in (versions["E"].get("history") or [])],
            "unified_comparisons": {"E_vs_D": unified_e_vs_d},
        }

    with ThreadPoolExecutor(max_workers=qwen_workers) as pool:
        primary_rows = list(pool.map(_eval_qwen, jobs))

    # Phase 3: Cross-Judges (DeepSeek-v4.1-flash & GLM-5.3-flash)
    extra_judges: dict[str, dict[tuple[str, str], dict[str, Any]]] = {}
    for j_name in ("deepseek-v4.1-flash", "GLM-5.3-flash"):
        j_ep, j_workers = judges[j_name]
        slug = "deepseek" if "deepseek" in j_name else "glm"
        print(f"[ActionB:Think] Phase 3 ({j_name}): Running Lit2Test bidirectional judge...", flush=True)

        def _eval_extra(item: tuple[str, str, dict[str, Any], dict[str, Any], dict[str, Any]], _ep=j_ep, _slug=slug, _jname=j_name):
            tid, host, _, seed, _ = item
            job = f"{tid}/{host}"
            task = TaskSpec.model_validate(seed["task"])
            vdir = out / "versions" / tid / host
            versions = {arm: load(vdir / f"{arm}.json") for arm in ("D", "E")}
            full_packet, full_drafts = shared_full_pool(task, seed["evidence"], versions)
            rdir = vdir / f"review_{_slug}" / "E_vs_D"

            def _compute():
                runners = [
                    _make_runner(_ep, rdir / "lit2test" / o, job, "blind", o, 12, dry_run=args.dry_run)
                    for o in ("forward", "reverse")
                ]
                return eval_lit2test_pair(full_drafts["D"], full_drafts["E"], task, full_packet, runners)

            res = cached(vdir / f"review_{_slug}_E_vs_D.json", _compute)
            print(f"[ActionB:Judge:{_jname}] {job} -> {res['status']} (sum_delta={res.get('score_sum_delta')})", flush=True)
            return (tid, host), res

        with ThreadPoolExecutor(max_workers=j_workers) as pool:
            extra_judges[j_name] = dict(pool.map(_eval_extra, jobs))

    per_ms: list[dict[str, Any]] = []
    for row_p in primary_rows:
        tid, host = row_p["task_id"], row_p["host"]
        q_l2t = row_p["unified_comparisons"]["E_vs_D"]["lit2test"]
        d_l2t = extra_judges["deepseek-v4.1-flash"][(tid, host)]
        g_l2t = extra_judges["GLM-5.3-flash"][(tid, host)]
        maj_status = _majority_vote_status([q_l2t["status"], d_l2t["status"], g_l2t["status"]])
        maj_sum = round(
            float(np.mean([
                float(q_l2t.get("score_sum_delta") or 0.0),
                float(d_l2t.get("score_sum_delta") or 0.0),
                float(g_l2t.get("score_sum_delta") or 0.0),
            ])),
            3,
        )
        maj_dims = {
            dim: round(
                float(np.mean([
                    float((q_l2t.get("dimension_deltas") or {}).get(dim, 0.0)),
                    float((d_l2t.get("dimension_deltas") or {}).get(dim, 0.0)),
                    float((g_l2t.get("dimension_deltas") or {}).get(dim, 0.0)),
                ])),
                3,
            )
            for dim in LIT2TEST_DIMENSIONS
        }
        per_ms.append({
            "task_id": tid,
            "host": host,
            "e_status": row_p["versions"]["E"]["status"],
            "e_actions": row_p["e_actions"],
            "judges": {
                "qwen3.8-flash": q_l2t,
                "deepseek-v4.1-flash": d_l2t,
                "GLM-5.3-flash": g_l2t,
                "majority_3way": {
                    "status": maj_status,
                    "order_sensitive": False,
                    "score_sum_delta": maj_sum,
                    "dimension_deltas": maj_dims,
                },
            },
        })

    judge_keys = ("qwen3.8-flash", "deepseek-v4.1-flash", "GLM-5.3-flash", "majority_3way")
    judge_summaries = {jk: _summarize_judge(per_ms, jk) for jk in judge_keys}
    by_host_summaries = {
        h: {jk: _summarize_judge([m for m in per_ms if m["host"] == h], jk) for jk in judge_keys}
        for h in ("ai_scientist", "evoscientist", "direct")
        if any(m["host"] == h for m in per_ms)
    }
    unified_qwen_summary = summarize_unified_rows(primary_rows, "E_vs_D")
    mech_stats = _compute_mechanism_stats(jobs, out)
    fast_ref = _compute_fast_wave1_reference(jobs, args.wave1_mh_dir, args.wave1_evosci_dir) if not args.dry_run else {}

    probe_summary = {
        "n_topics": len(probe_rows),
        "fast_A_process_hype_total": sum(r["fast_A_process_hype"] for r in probe_rows),
        "think_A_process_hype_total": sum(r["think_A_process_hype"] for r in probe_rows),
        "fast_A_unverified_entities_total": sum(r["fast_A_unverified_entities"] for r in probe_rows),
        "think_A_unverified_entities_total": sum(r["think_A_unverified_entities"] for r in probe_rows),
    }

    metered = {k: dict(v) for k, v in usage(out).items()}
    tot_cost = round(sum(float(v.get("cost_usd", 0.0)) for v in metered.values() if isinstance(v, dict)), 4)

    summary_payload = {
        "experiment": "action_b_slow_thinking_8192_ablation",
        "editor_model": editor_think_ep.model,
        "output_tokens": out_toks,
        "n_topics": len(selected_entries),
        "n_manuscripts": len(per_ms),
        "judge_summaries": judge_summaries,
        "by_host_summaries": by_host_summaries,
        "qwen_unified_summary": unified_qwen_summary,
        "mechanism_stats": mech_stats,
        "direct_think_generation_probe": probe_summary,
        "wave1_fast_4096_paired_reference": fast_ref,
        "total_cost_usd": tot_cost,
        "usage": metered,
        "manuscripts": per_ms,
    }
    write_json(out / "action_b_thinking_ablation_summary.json", summary_payload)

    def _fmt(v: Any) -> str:
        if v is None:
            return "—"
        if isinstance(v, list):
            return f"[{v[0]:+.3f}, {v[1]:+.3f}]"
        return f"{v:+.3f}"

    md = [
        f"# Action B: Slow-Thinking (`thinking: enabled`) + Symmetric `{out_toks}`-Token Budget Ablation (`N={len(per_ms)}` Manuscripts)",
        "",
        "## 1. Aggregate `E_think vs D_think` Across 3 Calibrated Blind Judges",
        "",
        "| Judge | Win / Loss / Tie | Topic Net Wins (95% CI) | Score Sum Delta (95% CI) | Order Sens. |",
        "| --- | :---: | :---: | :---: | ---: |",
    ]
    for label, jk in (
        ("Qwen3.8-flash (Primary Independent)", "qwen3.8-flash"),
        ("DeepSeek-v4.1-flash", "deepseek-v4.1-flash"),
        ("GLM-5.3-flash", "GLM-5.3-flash"),
        ("**3-Judge Majority Vote**", "majority_3way"),
    ):
        s = judge_summaries[jk]
        c = s["counts"]
        md.append(
            f"| {label} | `{c.get('win',0)} / {c.get('loss',0)} / {c.get('tie',0)}` | "
            f"**`{_fmt(s['topic_net_wins'])}`** (`{_fmt(s['topic_net_wins_ci95'])}`) | "
            f"**`{_fmt(s['score_sum_delta'])}`** (`{_fmt(s['score_sum_delta_ci95'])}`) | {s['order_sensitive']} / {len(per_ms)} |"
        )

    md += [
        "",
        "## 2. By Upstream Host (`Qwen3.8-flash Primary` & `3-Judge Majority`)",
        "",
        "| Host | $N$ | Qwen W/L/T (`Net`, `ΔSum`) | **3-Judge Majority W/L/T (`Net`, `ΔSum`)** |",
        "| --- | :---: | :---: | :---: |",
    ]
    for h_name in ("ai_scientist", "evoscientist", "direct"):
        if h_name not in by_host_summaries:
            continue
        hm = by_host_summaries[h_name]
        sq, sm = hm["qwen3.8-flash"], hm["majority_3way"]
        cq, cm = sq["counts"], sm["counts"]
        n_h = sum(cq.values())
        md.append(
            f"| `{h_name}` | {n_h} | `{cq.get('win',0)}/{cq.get('loss',0)}/{cq.get('tie',0)}` (`{_fmt(sq['topic_net_wins'])}`, `{_fmt(sq['score_sum_delta'])}`) | "
            f"**`{cm.get('win',0)}/{cm.get('loss',0)}/{cm.get('tie',0)}`** (`{_fmt(sm['topic_net_wins'])}`, `{_fmt(sm['score_sum_delta'])}`) |"
        )

    d_rt = int(metered.get("D", {}).get("reasoning_tokens", 0))
    e_rt = int(metered.get("E", {}).get("reasoning_tokens", 0))
    md += [
        "",
        "## 3. 2×2 Epistemic Hygiene Comparison (`Fast 4096` vs. `Slow-Thinking 8192`) & Reasoning Token Telemetry",
        "",
        f"- **Schema Keys Preserved (`E_think`)**: `{mech_stats['E1_orig_keys_kept_all']} / {len(per_ms)}`",
        f"- **Slow-Thinking Process Hype (`A -> D_think -> E_think`)**: `{mech_stats['E2_process_hype']['A']['occurrences']} -> {mech_stats['E2_process_hype']['D']['occurrences']} -> {mech_stats['E2_process_hype']['E']['occurrences']}`",
        f"- **Slow-Thinking Unverified Entities (`A -> D_think -> E_think`)**: `{mech_stats['E2_unverified_entities']['A']['total']} -> {mech_stats['E2_unverified_entities']['D']['total']} -> {mech_stats['E2_unverified_entities']['E']['total']}`",
        f"- **Slow-Thinking Blocking Confounds Addressed (`A -> D_think -> E_think`)**: `{mech_stats['E3_confounds']['A_addressed']}/{mech_stats['E3_confounds']['targets_required_n']} -> {mech_stats['E3_confounds']['D_addressed']}/{mech_stats['E3_confounds']['targets_required_n']} -> {mech_stats['E3_confounds']['E_addressed']}/{mech_stats['E3_confounds']['targets_required_n']}`",
        f"- **10-Topic Direct Initial Generation Probe (`Fast A` vs `Slow-Thinking A_think`)**: Process Hype `{probe_summary['fast_A_process_hype_total']} -> {probe_summary['think_A_process_hype_total']}`, Unverified Entities `{probe_summary['fast_A_unverified_entities_total']} -> {probe_summary['think_A_unverified_entities_total']}`",
        f"- **Metered Reasoning Tokens (`D_think` vs `E_think`)**: `D_think = {d_rt:,}`, `E_think = {e_rt:,}`",
        f"- **Total Metered Cost (USD)**: `${tot_cost:.4f}`",
    ]
    if fast_ref:
        fq = fast_ref["judge_summaries"]["qwen3.8-flash"]
        fm = fast_ref["judge_summaries"]["majority_3way"]
        fh = fast_ref["hygiene"]
        md += [
            "",
            "## 4. Paired Reference: Identical 30 Wave-1 Manuscripts under `Fast 4096`",
            f"- **Wave-1 Fast-4096 Qwen Primary**: `{fq['counts'].get('win',0)}W/{fq['counts'].get('loss',0)}L/{fq['counts'].get('tie',0)}T` (`Net = {_fmt(fq['topic_net_wins'])}`, `ΔSum = {_fmt(fq['score_sum_delta'])}`)",
            f"- **Wave-1 Fast-4096 3-Judge Majority**: `{fm['counts'].get('win',0)}W/{fm['counts'].get('loss',0)}L/{fm['counts'].get('tie',0)}T` (`Net = {_fmt(fm['topic_net_wins'])}`, `ΔSum = {_fmt(fm['score_sum_delta'])}`)",
            f"- **Wave-1 Fast-4096 Hygiene**: Hype `D={fh['D_process_hype']} -> E={fh['E_process_hype']}`, Unverified `D={fh['D_unverified_entities']} -> E={fh['E_unverified_entities']}`, Confounds `D={fh['D_confounds_addressed']}/{fh['confounds_required']} -> E={fh['E_confounds_addressed']}/{fh['confounds_required']}`",
        ]

    (out / "action_b_thinking_ablation_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"[ActionB:Think] Completed! Cost=${tot_cost:.4f} -> {out / 'action_b_thinking_ablation_summary.md'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
