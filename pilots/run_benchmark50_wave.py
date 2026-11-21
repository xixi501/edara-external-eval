"""Execute a locked Benchmark-50 wave (wave1/wave2/wave3) reusing frozen A/B/C/D drafts where available.

Runs upgraded Arm E (closed-loop governance + post-patch consolidation + prerequisite isolation)
and evaluates E_vs_D across all 3 calibrated blind judges (Qwen3.8-flash, DeepSeek-v4.1-flash, GLM-5.3-flash),
producing unified metrics, 3-judge majority vote, deterministic mechanism audit stats, and old-vs-new Arm E diffs.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import json
import shutil
from pathlib import Path
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
from edara_mvp_eval.middleware import TrustworthyNoveltyMiddleware
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
    run_arm,
    shared_full_pool,
    usage,
)
from edara_mvp_eval.schemas import FullAuditBundle, PriorWorkItem, TaskSpec
from edara_mvp_eval.stage_b_evidence import write_json
from edara_mvp_eval.unified_eval import (
    CORE_AIB_DIMENSIONS,
    LIT2TEST_DIMENSIONS,
    eval_lit2test_pair,
    evaluate_unified_pair,
    summarize_unified_rows,
)


def _configure_endpoint(profile_path: Path, *, role: str, dry_run: bool) -> ModelEndpoint:
    if dry_run:
        return ModelEndpoint(model=f"dry-run-{role}")
    ep = ModelEndpoint.from_env_or_profile(profile_path=profile_path)
    low = ep.model.lower()
    if role in {"editor", "audit"}:
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


def _seed_cached_manuscript(
    src_out: Path,
    dst_out: Path,
    task_id: str,
    host: str,
    record: dict[str, Any],
    seed: dict[str, Any],
) -> None:
    src_vdir = src_out / "versions" / task_id / host
    dst_vdir = dst_out / "versions" / task_id / host
    dst_vdir.mkdir(parents=True, exist_ok=True)

    for arm in ("A", "B", "C", "D"):
        s_file = src_vdir / f"{arm}.json"
        d_file = dst_vdir / f"{arm}.json"
        if not d_file.is_file():
            shutil.copy2(s_file, d_file)

    a_loaded = load(dst_vdir / "A.json")
    b_loaded = load(dst_vdir / "B.json")
    c_loaded = load(dst_vdir / "C.json")
    if a_loaded["sha256"] != record["sha256"]:
        raise RuntimeError(f"SHA-256 mismatch seeding A.json for {task_id}/{host}")
    if b_loaded["initial_evidence_sha256"] != c_loaded["initial_evidence_sha256"]:
        raise RuntimeError(f"Evidence SHA-256 mismatch between B and C for {task_id}/{host}")

    # Copy C's usage logs for shared initial audit accounting
    src_c_audit = src_vdir / "C" / "audit"
    dst_c_audit = dst_vdir / "C" / "audit"
    if src_c_audit.is_dir() and not dst_c_audit.exists():
        shutil.copytree(src_c_audit, dst_c_audit)

    # Copy ONLY the initial Round-0 structured audit for (original_A, initial_packet)
    task = TaskSpec.model_validate(seed["task"])
    init_ev = Evidence.restore(task, seed["evidence"])
    init_key = digest([record["draft"], init_ev.packet()])
    src_sa = src_vdir / "structured_audits" / f"{init_key}.json"
    dst_sa_dir = dst_vdir / "structured_audits"
    dst_sa_dir.mkdir(parents=True, exist_ok=True)
    if src_sa.is_file() and not (dst_sa_dir / f"{init_key}.json").is_file():
        shutil.copy2(src_sa, dst_sa_dir / f"{init_key}.json")

    # Copy deterministic pre-cutoff OpenAlex lookup cache
    src_lookup = src_out / "lookup" / task_id / host
    dst_lookup = dst_out / "lookup" / task_id / host
    if src_lookup.is_dir() and not dst_lookup.exists():
        shutil.copytree(src_lookup, dst_lookup)


def _count_process_hype(draft: dict[str, Any]) -> int:
    import re

    pat = re.compile(r"\b(?:pre-registered|preregistered|blinded\s+clinical\s+trial|irb-approved)\b", re.I)
    text = "\n".join(str(v) for v in draft.values())
    return len(pat.findall(text))


def _compute_mechanism_stats(jobs: list[tuple[str, str, dict, dict, str]], out_dir: Path) -> dict[str, Any]:
    stats: dict[str, Any] = {
        "n_manuscripts": len(jobs),
        "E1_orig_keys_kept_all": 0,
        "E1_new_keys_added": 0,
        "E2_process_hype": {arm: {"manuscripts": 0, "occurrences": 0} for arm in ("A", "B", "C", "D", "E")},
        "E2_unverified_entities": {arm: {"manuscripts": 0, "total": 0} for arm in ("A", "B", "C", "D", "E")},
        "E3_confounds": {"targets_required_n": 0, "A_addressed": 0, "D_addressed": 0, "E_addressed": 0},
        "E7_trajectory": {
            "E_changed_n": 0,
            "E_error_n": 0,
            "E_clean_closed_loop_n": 0,
            "E_consolidated_n": 0,
            "A_vs_E_design_violation": 0,
        },
        "design_elements_all_three": {arm: 0 for arm in ("A", "B", "C", "D", "E")},
    }
    for task_id, host, record, seed, _ in jobs:
        vdir = out_dir / "versions" / task_id / host
        task = TaskSpec.model_validate(seed["task"])
        ev = Evidence.restore(task, seed["evidence"])
        a_draft = record["draft"]
        versions = {arm: load(vdir / f"{arm}.json") for arm in ("B", "C", "D", "E")}
        arm_drafts = {"A": a_draft, **{arm: versions[arm]["draft"] for arm in ("B", "C", "D", "E")}}

        e_draft = arm_drafts["E"]
        if set(a_draft.keys()) <= set(e_draft.keys()):
            stats["E1_orig_keys_kept_all"] += 1
        stats["E1_new_keys_added"] += len(set(e_draft.keys()) - set(a_draft.keys()))

        # Check initial structured audit for blocking confounds
        init_key = digest([a_draft, ev.packet()])
        sa_path = vdir / "structured_audits" / f"{init_key}.json"
        targets: list[str] = []
        if sa_path.is_file():
            bundle = FullAuditBundle.model_validate(load(sa_path)["bundle"])
            diag = TrustworthyNoveltyMiddleware()._build_diag(bundle)
            targets = [c["discriminating_control"] for c in (diag.get("confounds") or []) if c.get("severity") == "blocking" and c.get("discriminating_control")]
        if targets:
            stats["E3_confounds"]["targets_required_n"] += 1
            for arm_k, field_k in (("A", "A_addressed"), ("D", "D_addressed"), ("E", "E_addressed")):
                if confound_addressed(arm_drafts[arm_k], targets):
                    stats["E3_confounds"][field_k] += 1

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
            if arm_k in ("D", "E"):
                dr_eval = full_drafts[arm_k]
                corpus_eval = full_corpus
            else:
                dr_eval = dr
                corpus_eval = local_corpus
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
        hist = e_row.get("history") or []
        actions = [h.get("action") for h in hist]
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


def _summarize_judge_lit2test(per_ms: list[dict[str, Any]], judge_key: str) -> dict[str, Any]:
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wave", choices=("wave1", "wave2", "wave3", "wave1_2"), default="wave1")
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

    editor_ep = _configure_endpoint(args.editor_profile, role="editor", dry_run=args.dry_run)
    audit_ep = _configure_endpoint(args.audit_profile, role="audit", dry_run=args.dry_run)
    judges = {
        "qwen3.8-flash": (_configure_endpoint(args.qwen_judge_profile, role="judge", dry_run=args.dry_run), min(args.workers, 3)),
        "deepseek-v4.1-flash": (_configure_endpoint(args.deepseek_judge_profile, role="judge", dry_run=args.dry_run), min(max(args.workers, 3), 4)),
        "GLM-5.3-flash": (_configure_endpoint(args.glm_judge_profile, role="judge", dry_run=args.dry_run), min(args.workers, 2)),
    }

    manifest_meta = {
        "protocol_version": PROTOCOL_VERSION,
        "wave": args.wave,
        "dry_run": args.dry_run,
        "code": {p.name: path_hash(p) for p in sorted((ROOT / "src/edara_mvp_eval").glob("*.py"))},
        "models": {
            "editor": editor_ep.model,
            "audit": audit_ep.model,
            "judges": {k: ep.model for k, (ep, _) in judges.items()},
        },
        "limits": LIMITS,
        "output_tokens": OUTPUT_TOKENS,
    }
    write_json(out / "manifest.json", manifest_meta)

    b50 = load(ROOT / "data/benchmark_50_topics.json")["topics"]
    allowed_waves = {"wave1", "wave2"} if args.wave == "wave1_2" else {args.wave}
    selected_entries = [e for e in b50 if e["wave"] in allowed_waves]

    v4_inputs = load(args.v4_dir / "inputs_core.json") if (args.v4_dir / "inputs_core.json").is_file() else {}
    ext_inputs = load(args.ext15_dir / "inputs_extension.json") if (args.ext15_dir / "inputs_extension.json").is_file() else {}

    # Load old results for direct old-vs-new E comparison
    old_e_lookup: dict[tuple[str, str], dict[str, Any]] = {}
    if (args.v4_dir / "core_results.json").is_file():
        for r in load(args.v4_dir / "core_results.json").get("results", []):
            old_e_lookup[(r["task_id"], r["host"])] = {
                "cohort": "v4",
                "e_status": r["versions"]["E"]["status"],
                "l2t": r["unified_comparisons"]["E_vs_D"]["lit2test"],
            }
    if (args.ext15_dir / "extension_results.json").is_file():
        for r in load(args.ext15_dir / "extension_results.json").get("results", []):
            old_e_lookup[(r["task_id"], r["host"])] = {
                "cohort": "ext15",
                "e_status": r["versions"]["E"]["status"],
                "l2t": r["unified_comparisons"]["E_vs_D"]["lit2test"],
            }

    wave_inputs: dict[str, Any] = {}
    jobs: list[tuple[str, str, dict[str, Any], dict[str, Any], str]] = []
    for entry in selected_entries:
        tid = entry["task_id"]
        if tid in v4_inputs:
            src_name, src_dir, seed = "v4", args.v4_dir, v4_inputs[tid]
        elif tid in ext_inputs:
            src_name, src_dir, seed = "ext15", args.ext15_dir, ext_inputs[tid]
        else:
            raise FileNotFoundError(f"Topic {tid} has no frozen input snapshot yet; generate host inputs first.")
        wave_inputs[tid] = seed
        for host, rec in seed["originals"].items():
            if rec["status"] != "ready":
                continue
            _seed_cached_manuscript(src_dir, out, tid, host, rec, seed)
            jobs.append((tid, host, rec, seed, src_name))

    write_json(out / f"inputs_{args.wave}.json", wave_inputs)
    print(f"[{args.wave}] Seeded {len(selected_entries)} topics, {len(jobs)} ready manuscripts.", flush=True)

    # Phase 1: Run upgraded Arm E
    def _run_one_e(item: tuple[str, str, dict[str, Any], dict[str, Any], str]) -> dict[str, Any]:
        tid, host, rec, seed, src_name = item
        job = f"{tid}/{host}"
        task = TaskSpec.model_validate(seed["task"])
        vdir = out / "versions" / tid / host
        lim = LIMITS["full"]
        auditor = _make_runner(audit_ep, vdir / "E" / "audit", job, "E", "audit", lim["audit"], dry_run=args.dry_run)
        editor = _make_runner(editor_ep, vdir / "E" / "editor", job, "E", "editor", lim["editor"], dry_run=args.dry_run)
        editor_conf = _make_runner(editor_ep, vdir / "E" / "editor_confound", job, "E", "editor", lim["rounds"], dry_run=args.dry_run)
        e_row = cached(
            vdir / "E.json",
            lambda: run_arm(
                "E",
                rec["draft"],
                task,
                seed["evidence"],
                auditor,
                editor,
                out / "lookup" / tid / host,
                audit_cache=vdir / "structured_audits",
                editor_confound=editor_conf,
            ),
        )
        acts = [h.get("action") for h in (e_row.get("history") or [])]
        print(f"[Arm E] {job} ({src_name}) -> status={e_row['status']} actions={acts}", flush=True)
        return e_row

    print(f"[{args.wave}] Phase 1: Running upgraded Arm E across {len(jobs)} manuscripts (workers={args.workers})...", flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(_run_one_e, jobs))

    # Phase 2: Primary Unified Evaluation (Qwen3.8-flash: Lit2Test + AlphaIdeaBench + ClaimNov)
    qwen_ep, qwen_workers = judges["qwen3.8-flash"]

    def _eval_primary(item: tuple[str, str, dict[str, Any], dict[str, Any], str]) -> dict[str, Any]:
        tid, host, rec, seed, src_name = item
        job = f"{tid}/{host}"
        task = TaskSpec.model_validate(seed["task"])
        vdir = out / "versions" / tid / host
        versions = {arm: load(vdir / f"{arm}.json") for arm in ("B", "C", "D", "E")}
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

        def _compute_qwen():
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

        unified_e_vs_d = cached(vdir / "review_qwen_E_vs_D.json", _compute_qwen)
        l2t = unified_e_vs_d["lit2test"]
        print(f"[Judge:Qwen] {job} -> {l2t['status']} (sum_delta={l2t.get('score_sum_delta')})", flush=True)
        return {
            "task_id": tid,
            "host": host,
            "source_cohort": src_name,
            "stage": rec.get("stage", "proposal"),
            "development": seed.get("development", False),
            "source_sha256": rec["sha256"],
            "versions": {arm: {k: versions[arm][k] for k in ("status", "sha256", "edit_volume", "queries")} for arm in ("B", "C", "D", "E")},
            "e_actions": [h.get("action") for h in (versions["E"].get("history") or [])],
            "unified_comparisons": {"E_vs_D": unified_e_vs_d},
        }

    print(f"[{args.wave}] Phase 2: Running Primary Unified Judge (qwen3.8-flash, workers={qwen_workers})...", flush=True)
    with ThreadPoolExecutor(max_workers=qwen_workers) as pool:
        primary_rows = list(pool.map(_eval_primary, jobs))

    # Phase 3: Cross-Judge Lit2Test Blind Evaluation (DeepSeek-v4.1-flash & GLM-5.3-flash)
    extra_judges: dict[str, dict[tuple[str, str], dict[str, Any]]] = {}
    for j_name in ("deepseek-v4.1-flash", "GLM-5.3-flash"):
        j_ep, j_workers = judges[j_name]
        slug = "deepseek" if "deepseek" in j_name else "glm"
        print(f"[{args.wave}] Phase 3 ({j_name}): Running Lit2Test bidirectional judge (workers={j_workers})...", flush=True)

        def _eval_extra(item: tuple[str, str, dict[str, Any], dict[str, Any], str], _ep=j_ep, _slug=slug, _jname=j_name):
            tid, host, _, seed, _ = item
            job = f"{tid}/{host}"
            task = TaskSpec.model_validate(seed["task"])
            vdir = out / "versions" / tid / host
            versions = {arm: load(vdir / f"{arm}.json") for arm in ("B", "C", "D", "E")}
            full_packet, full_drafts = shared_full_pool(task, seed["evidence"], versions)
            rdir = vdir / f"review_{_slug}" / "E_vs_D"

            def _compute():
                runners = [
                    _make_runner(_ep, rdir / "lit2test" / o, job, "blind", o, 12, dry_run=args.dry_run)
                    for o in ("forward", "reverse")
                ]
                return eval_lit2test_pair(full_drafts["D"], full_drafts["E"], task, full_packet, runners)

            res = cached(vdir / f"review_{_slug}_E_vs_D.json", _compute)
            print(f"[Judge:{_jname}] {job} -> {res['status']} (sum_delta={res.get('score_sum_delta')})", flush=True)
            return (tid, host), res

        with ThreadPoolExecutor(max_workers=j_workers) as pool:
            extra_judges[j_name] = dict(pool.map(_eval_extra, jobs))

    # Assemble 3-Judge Cross-Evaluation + Old-vs-New Diff
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
        old_rec = old_e_lookup.get((tid, host), {})
        per_ms.append({
            "task_id": tid,
            "host": host,
            "source_cohort": row_p["source_cohort"],
            "e_status": row_p["versions"]["E"]["status"],
            "e_actions": row_p["e_actions"],
            "old_e_status": old_rec.get("e_status"),
            "old_primary_status": (old_rec.get("l2t") or {}).get("status"),
            "old_primary_sum_delta": (old_rec.get("l2t") or {}).get("score_sum_delta"),
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

    judge_summaries = {
        j_key: _summarize_judge_lit2test(per_ms, j_key)
        for j_key in ("qwen3.8-flash", "deepseek-v4.1-flash", "GLM-5.3-flash", "majority_3way")
    }
    unified_qwen_summary = summarize_unified_rows(primary_rows, "E_vs_D")
    by_host_summaries = {}
    for h_name in ("direct", "ai_scientist", "edara"):
        sub = [m for m in per_ms if m["host"] == h_name]
        if sub:
            by_host_summaries[h_name] = {
                j_key: _summarize_judge_lit2test(sub, j_key)
                for j_key in ("qwen3.8-flash", "deepseek-v4.1-flash", "GLM-5.3-flash", "majority_3way")
            }

    mech_stats = _compute_mechanism_stats(jobs, out)
    metered = {k: dict(v) for k, v in usage(out).items()}

    write_json(out / f"{args.wave}_results.json", {"wave": args.wave, "results": primary_rows})
    write_json(out / f"{args.wave}_mechanism_stats.json", mech_stats)
    cross_payload = {
        "wave": args.wave,
        "n_topics": len(selected_entries),
        "n_manuscripts": len(per_ms),
        "judge_summaries": judge_summaries,
        "by_host_summaries": by_host_summaries,
        "qwen_unified_summary": unified_qwen_summary,
        "mechanism_stats": mech_stats,
        "usage": metered,
        "manuscripts": per_ms,
    }
    write_json(out / f"{args.wave}_crossjudge_3way.json", cross_payload)

    # Render clean Markdown report
    def _fmt(v: Any) -> str:
        if v is None:
            return "—"
        if isinstance(v, list):
            return f"[{v[0]:+.3f}, {v[1]:+.3f}]"
        return f"{v:+.3f}"

    md = [
        f"# Benchmark-50 `{args.wave}` Canary Evaluation Report (`N={len(per_ms)}` Manuscripts across `{len(selected_entries)}` Domains)",
        "",
        "## 1. Aggregate `E_vs_D` Across 3 Calibrated Blind Judges",
        "",
        "| Judge | Win / Loss / Tie / Unres | Topic Net Wins (95% CI) | Score Sum Delta (95% CI) | Order Sens. |",
        "| --- | :---: | :---: | :---: | ---: |",
    ]
    for j_label, j_key in (
        ("Qwen3.8-flash (Primary)", "qwen3.8-flash"),
        ("DeepSeek-v4.1-flash", "deepseek-v4.1-flash"),
        ("GLM-5.3-flash", "GLM-5.3-flash"),
        ("**3-Judge Majority Vote**", "majority_3way"),
    ):
        s = judge_summaries[j_key]
        c = s["counts"]
        wltu = f"{c.get('win',0)} / {c.get('loss',0)} / {c.get('tie',0)} / {c.get('unresolved',0)}"
        md.append(
            f"| {j_label} | `{wltu}` | **`{_fmt(s['topic_net_wins'])}`** (`{_fmt(s['topic_net_wins_ci95'])}`) "
            f"| **`{_fmt(s['score_sum_delta'])}`** (`{_fmt(s['score_sum_delta_ci95'])}`) | {s['order_sensitive']} / {len(per_ms)} |"
        )

    aib_d = unified_qwen_summary.get("alphaideabench_dimension_deltas") or {}
    md += [
        "",
        "## 2. Unified Secondary Axes (Qwen3.8-flash + ClaimNov) & Deterministic Mechanisms",
        "",
        f"- **ClaimNov Compositional Novelty Delta**: `{_fmt(unified_qwen_summary.get('claimnov_novelty_delta'))}`",
        f"- **AlphaIdeaBench Deltas**: `testability={_fmt(aib_d.get('testability'))}`, `context_grounding={_fmt(aib_d.get('context_grounding'))}`, `verification_feasibility={_fmt(aib_d.get('verification_feasibility'))}`, `method_novelty={_fmt(aib_d.get('method_novelty'))}`, `insight_novelty={_fmt(aib_d.get('insight_novelty'))}`",
        f"- **Process Hype Defects**: `D={mech_stats['E2_process_hype']['D']['manuscripts']} ms ({mech_stats['E2_process_hype']['D']['occurrences']})` -> `E={mech_stats['E2_process_hype']['E']['manuscripts']} ms ({mech_stats['E2_process_hype']['E']['occurrences']})`",
        f"- **Unverified External Entities**: `D={mech_stats['E2_unverified_entities']['D']['total']}` -> `E={mech_stats['E2_unverified_entities']['E']['total']}`",
        f"- **Blocking Confounds Addressed**: `D={mech_stats['E3_confounds']['D_addressed']}/{mech_stats['E3_confounds']['targets_required_n']}` -> `E={mech_stats['E3_confounds']['E_addressed']}/{mech_stats['E3_confounds']['targets_required_n']}`",
        f"- **Trajectory Integrity (`E`)**: `errors={mech_stats['E7_trajectory']['E_error_n']}`, `clean_closed_loop={mech_stats['E7_trajectory']['E_clean_closed_loop_n']}/{len(per_ms)}`, `consolidated={mech_stats['E7_trajectory']['E_consolidated_n']}`, `orig_keys_kept={mech_stats['E1_orig_keys_kept_all']}/{len(per_ms)}`",
        "",
        "## 3. By Host (`3-Judge Majority`)",
        "",
        "| Host | n | Win / Loss / Tie | Topic Net Wins | Score Sum Delta |",
        "| --- | :---: | :---: | ---: | ---: |",
    ]
    for h_name, h_map in by_host_summaries.items():
        ms_h = [m for m in per_ms if m["host"] == h_name]
        s = h_map["majority_3way"]
        c = s["counts"]
        md.append(
            f"| `{h_name}` | {len(ms_h)} | `{c.get('win',0)}/{c.get('loss',0)}/{c.get('tie',0)}` | `{_fmt(s['topic_net_wins'])}` | `{_fmt(s['score_sum_delta'])}` |"
        )

    md += [
        "",
        "## 4. Manuscript-by-Manuscript Old `E` vs Upgraded `E` Comparison",
        "",
        "| Topic | Host | Old `E` Status / Verdict (`Δsum`) | New `E` Trajectory | Qwen | DeepSeek | GLM | **Majority (`Δsum`)** |",
        "| --- | :---: | --- | --- | :---: | :---: | :---: | :---: |",
    ]
    for m in per_ms:
        old_str = f"`{m['old_e_status']}` / `{m['old_primary_status']}` (`{_fmt(m['old_primary_sum_delta'])}`)"
        traj_str = f"`{m['e_status']}` `{m['e_actions']}`"
        q_st = m["judges"]["qwen3.8-flash"]["status"]
        d_st = m["judges"]["deepseek-v4.1-flash"]["status"]
        g_st = m["judges"]["GLM-5.3-flash"]["status"]
        maj = m["judges"]["majority_3way"]
        md.append(
            f"| `{m['task_id'][:34]}` | `{m['host']}` | {old_str} | {traj_str} | `{q_st}` | `{d_st}` | `{g_st}` | **`{maj['status']}` (`{_fmt(maj['score_sum_delta'])}`)** |"
        )

    e_cost = metered.get("E", {}).get("cost_usd", 0.0)
    j_cost = metered.get("blind", {}).get("cost_usd", 0.0)
    md += [
        "",
        f"- **Metered Cost (USD)**: `Arm E = ${e_cost:.4f}`, `3-Judge Blind Eval = ${j_cost:.4f}`, `Total = ${e_cost + j_cost:.4f}`",
    ]
    (out / f"{args.wave}_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"[{args.wave}] Completed! Summary written to {out / f'{args.wave}_summary.md'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
