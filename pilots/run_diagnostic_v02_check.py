"""Balanced 8-manuscript diagnostic micro-validation (4 Wave-2 Losses + 4 Wave-2 Wins/Ties across all 4 hosts).

Reuses frozen A, B, C, D drafts + Round-0 structured audits from Wave 2, executes ONLY upgraded Arm E (v0.2),
and evaluates E_vs_D under the EXACT SAME FROZEN 3-judge blind panel (Qwen3.8-flash, DeepSeek-v4.1-flash, GLM-5.3-flash)
into an isolated output directory without modifying any Wave 2 archive.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from edara_mvp_eval.llm_client import LLMRunner, ModelEndpoint
from edara_mvp_eval.paired_inputs import Evidence
from edara_mvp_eval.paired_trial import (
    LIMITS,
    ROOT,
    TrialRunner,
    WORKSPACE,
    cached,
    digest,
    load,
    run_arm,
    shared_full_pool,
    usage,
)
from edara_mvp_eval.schemas import PriorWorkItem, TaskSpec
from edara_mvp_eval.stage_b_evidence import write_json
from edara_mvp_eval.unified_eval import eval_lit2test_pair

# Balanced 8-manuscript panel: 4 losses (covering testbed-erasure, tail-patch fragmentation, spurious regenerate)
# + 4 wins/ties across all 4 hosts (to verify zero regression on existing strong cases).
PANEL = (
    # 4 Wave-2 Losses
    ("evosci", "smallari_expansion40_030_pruning_language_llms_however_layer_wise__v02_expansion40", "evoscientist", "loss_testbed_erasure"),
    ("mh", "smallari_expansion40_025_audio_visual_multimodal_aligned_language_visual__v02_expansion40", "direct", "loss_tail_clarify"),
    ("evosci", "smallari_expansion40_001_cache_compression_llms_long_context__v02_expansion40", "evoscientist", "loss_tail_local_patch"),
    ("mh", "smallari_expansion40_009_detection_out_of_distribution_ood_post_hoc_in_di__v02_expansion40", "ai_scientist", "loss_spurious_regenerate"),
    # 4 Wave-2 Wins/Ties (regression guard across all 4 hosts)
    ("mh", "survey_14_diffusion_3d_generation", "ai_scientist", "guard_win_ai_sci"),
    ("evosci", "smallari_expansion40_013_motion_generation_human_editing_flow__v02_expansion40", "evoscientist", "guard_win_evosci"),
    ("mh", "ws_09_world_models_simulation", "direct", "guard_win_direct"),
    ("mh", "smallari_expansion40_029_bottleneck_concept_concepts_information_cbms__v02_expansion40", "edara", "guard_win_edara"),
)


def _cfg_ep(profile_path: Path, role: str) -> ModelEndpoint:
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
    return ep


def _maj(sts: list[str]) -> str:
    from collections import Counter
    c = Counter(sts)
    top, n = c.most_common(1)[0]
    if n >= 2:
        return top
    return "win" if c["win"] > c["loss"] else ("loss" if c["loss"] > c["win"] else "tie")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "outputs/diagnostic_8ms_v02_20261122")
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()

    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)

    mh_dir = ROOT / "outputs/benchmark50_wave2_20261122"
    evo_dir = ROOT / "outputs/benchmark50_evosci_wave2_20261122"
    mh_inputs = load(mh_dir / "inputs_wave2.json")
    mh_old = {(m["task_id"], m["host"]): m for m in load(mh_dir / "wave2_crossjudge_3way.json")["manuscripts"]}
    evo_old = {(m["task_id"], m["host"]): m for m in load(evo_dir / "wave2_evoscientist_summary.json")["manuscripts"]}

    editor_ep = _cfg_ep(WORKSPACE / "data/dashscope-deepseek-profile.json", "editor")
    audit_ep = _cfg_ep(WORKSPACE / "data/p1b-critic-profile.json", "audit")
    judges = {
        "qwen3.8-flash": _cfg_ep(WORKSPACE / "data/dashscope-qwen-profile.json", "judge"),
        "deepseek-v4.1-flash": _cfg_ep(WORKSPACE / "data/dashscope-deepseek-profile.json", "judge"),
        "GLM-5.3-flash": _cfg_ep(WORKSPACE / "data/p1b-critic-profile.json", "judge"),
    }

    jobs = []
    for cohort, tid, host, role_tag in PANEL:
        src_root = mh_dir if cohort == "mh" else evo_dir
        src_vdir = src_root / "versions" / tid / host
        dst_vdir = out / "versions" / tid / host
        dst_vdir.mkdir(parents=True, exist_ok=True)
        for arm in ("A", "B", "C", "D"):
            if not (dst_vdir / f"{arm}.json").is_file():
                shutil.copy2(src_vdir / f"{arm}.json", dst_vdir / f"{arm}.json")

        if cohort == "mh":
            seed = mh_inputs[tid]
        else:
            b50_map = {t["task_id"]: t for t in load(ROOT / "data/tasks_benchmark50.json")}
            seed = {"task": b50_map[tid], "evidence": load(src_vdir / "initial_evidence.json")}

        task = TaskSpec.model_validate(seed["task"])
        a_rec = load(dst_vdir / "A.json")
        init_ev = Evidence.restore(task, seed["evidence"])
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

        old_m = mh_old[(tid, host)] if cohort == "mh" else evo_old[(tid, host)]
        jobs.append((cohort, tid, host, role_tag, a_rec, seed, old_m))

    def _run_one(item):
        cohort, tid, host, role_tag, a_rec, seed, old_m = item
        job = f"{tid}/{host}"
        task = TaskSpec.model_validate(seed["task"])
        vdir = out / "versions" / tid / host
        lim = LIMITS["full"]
        auditor = TrialRunner(audit_ep, vdir / "E" / "audit", task_id=job, arm="E", stage="audit", limit=lim["audit"])
        editor = TrialRunner(editor_ep, vdir / "E" / "editor", task_id=job, arm="E", stage="editor", limit=lim["editor"])
        editor_conf = TrialRunner(editor_ep, vdir / "E" / "editor_confound", task_id=job, arm="E", stage="editor", limit=lim["rounds"])
        e_row = cached(
            vdir / "E.json",
            lambda: run_arm(
                "E",
                a_rec["draft"],
                task,
                seed["evidence"],
                auditor,
                editor,
                out / "lookup" / tid / host,
                audit_cache=vdir / "structured_audits",
                editor_confound=editor_conf,
            ),
        )
        versions = {arm: load(vdir / f"{arm}.json") for arm in ("B", "C", "D", "E")}
        full_packet, full_drafts = shared_full_pool(task, seed["evidence"], versions)
        j_res = {}
        for jk, jep in judges.items():
            slug = jk.split("-")[0].lower()
            rdir = vdir / f"review_{slug}"
            j_res[jk] = cached(
                vdir / f"review_{slug}_E_vs_D.json",
                lambda _ep=jep, _rd=rdir: eval_lit2test_pair(
                    full_drafts["D"],
                    full_drafts["E"],
                    task,
                    full_packet,
                    [TrialRunner(_ep, _rd / o, task_id=job, arm="blind", stage=o, limit=12) for o in ("forward", "reverse")],
                ),
            )
        new_maj_st = _maj([j_res[k]["status"] for k in ("qwen3.8-flash", "deepseek-v4.1-flash", "GLM-5.3-flash")])
        new_maj_sum = round(float(np.mean([float(j_res[k].get("score_sum_delta") or 0.0) for k in ("qwen3.8-flash", "deepseek-v4.1-flash", "GLM-5.3-flash")])), 3)

        old_maj = old_m["judges"].get("majority_3way") or old_m["judges"].get("majority_3judge")
        old_q = old_m["judges"]["qwen3.8-flash"]
        acts = [h.get("action") for h in (e_row.get("history") or [])]
        print(
            f"[Diag] {role_tag:24s} | {job[:46]:46s} | old_acts={old_m.get('e_actions')} -> new_acts={acts} | "
            f"Qwen: {old_q['status']}({old_q.get('score_sum_delta'):+.1f}) -> {j_res['qwen3.8-flash']['status']}({j_res['qwen3.8-flash'].get('score_sum_delta'):+.1f}) | "
            f"Maj: {old_maj['status']}({old_maj.get('score_sum_delta'):+.2f}) -> {new_maj_st}({new_maj_sum:+.2f})",
            flush=True,
        )
        return {
            "role_tag": role_tag,
            "task_id": tid,
            "host": host,
            "old_actions": old_m.get("e_actions"),
            "new_actions": acts,
            "old_qwen": {"status": old_q["status"], "sum_delta": old_q.get("score_sum_delta")},
            "new_qwen": {"status": j_res["qwen3.8-flash"]["status"], "sum_delta": j_res["qwen3.8-flash"].get("score_sum_delta")},
            "old_majority": {"status": old_maj["status"], "sum_delta": old_maj.get("score_sum_delta")},
            "new_majority": {"status": new_maj_st, "sum_delta": new_maj_sum},
            "new_judges": {k: {"status": v["status"], "sum_delta": v.get("score_sum_delta")} for k, v in j_res.items()},
        }

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(_run_one, jobs))

    u = {k: dict(v) for k, v in usage(out).items()}
    cost = sum(float(v.get("cost_usd") or 0.0) for v in u.values())
    write_json(out / "diagnostic_v02_results.json", {"rows": rows, "cost_usd": round(cost, 4), "usage": u})
    print(f"\n[Diag] Completed 8-manuscript check! Total metered cost = ${cost:.4f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
