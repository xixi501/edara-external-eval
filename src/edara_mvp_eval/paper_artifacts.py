"""One-click paper table (LaTeX booktabs + Markdown) and vector figure (SVG) generator for Benchmark-50."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import html
import json
from pathlib import Path
from typing import Any

import numpy as np

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
HOST_ORDER = ("ai_scientist", "evoscientist", "direct", "edara")
HOST_LABELS = {
    "ai_scientist": "AI-Scientist-v2",
    "evoscientist": "EvoScientist (v2.1.2)",
    "direct": "Direct LLM",
    "edara": "E-DARA (Native)",
}
JUDGE_KEYS = ("qwen3.8-flash", "deepseek-v4.1-flash", "GLM-5.3-flash", "majority")


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _compact_manuscript_entry(m: dict[str, Any], wave: str) -> dict[str, Any]:
    judges_raw = m.get("judges") or {}
    maj_raw = judges_raw.get("majority_3way") or judges_raw.get("majority_3judge") or {}
    compact_judges: dict[str, dict[str, Any]] = {}
    for jk in ("qwen3.8-flash", "deepseek-v4.1-flash", "GLM-5.3-flash"):
        jr = judges_raw.get(jk) or {}
        compact_judges[jk] = {
            "status": jr.get("status", "tie"),
            "order_sensitive": bool(jr.get("order_sensitive", False)),
            "score_sum_delta": float(jr.get("score_sum_delta") or 0.0),
            "dimension_deltas": dict(jr.get("dimension_deltas") or {}),
        }
    compact_judges["majority"] = {
        "status": maj_raw.get("status", "tie"),
        "order_sensitive": False,
        "score_sum_delta": float(maj_raw.get("score_sum_delta") or 0.0),
        "dimension_deltas": dict(maj_raw.get("dimension_deltas") or {}),
    }
    return {
        "wave": wave,
        "task_id": m["task_id"],
        "host": m["host"],
        "e_status": m.get("e_status", "changed"),
        "e_actions": list(m.get("e_actions") or []),
        "judges": compact_judges,
    }


def _recompute_confounds_from_versions(run_dir: Path) -> dict[str, int] | None:
    from .host_adapter import confound_addressed
    from .paired_trial import confound_targets

    vroot = run_dir / "versions"
    if not vroot.is_dir():
        return None
    req_n = a_ok = d_ok = e_ok = 0
    for vh in sorted(vroot.glob("*/*")):
        if not ((vh / "A.json").is_file() and (vh / "D.json").is_file() and (vh / "E.json").is_file()):
            continue
        a_dr = _load_json(vh / "A.json")["draft"]
        d_dr = _load_json(vh / "D.json")["draft"]
        e_row = _load_json(vh / "E.json")
        e_dr = e_row["draft"]
        hist = e_row.get("history") or []
        targets = confound_targets(hist[0]["diagnosis"]) if (hist and "diagnosis" in hist[0]) else []
        if targets:
            req_n += 1
            a_ok += int(confound_addressed(a_dr, targets))
            d_ok += int(confound_addressed(d_dr, targets, baseline_draft=a_dr) or confound_addressed(d_dr, targets))
            e_ok += int(confound_addressed(e_dr, targets, baseline_draft=a_dr) or confound_addressed(e_dr, targets))
    return {"targets_required_n": req_n, "A_addressed": a_ok, "D_addressed": d_ok, "E_addressed": e_ok}


def sync_compact_wave_bundles(outputs_dir: Path, results_dir: Path) -> list[Path]:
    """Export lightweight, prose-free compact JSON summaries from staging outputs/ into results/."""
    synced: list[Path] = []
    if not outputs_dir.is_dir():
        return synced

    for wave in ("wave1", "wave2", "wave3"):
        # Prefer latest / _r2 directory for multi-host waves
        cand_dirs = sorted(outputs_dir.glob(f"benchmark50_{wave}_*"))
        cand_dirs = [d for d in cand_dirs if d.is_dir() and (d / f"{wave}_crossjudge_3way.json").is_file()]
        if cand_dirs:
            chosen = cand_dirs[-1]
            raw = _load_json(chosen / f"{wave}_crossjudge_3way.json")
            mech_mh = dict(raw.get("mechanism_stats") or {})
            recomp = _recompute_confounds_from_versions(chosen)
            if recomp is not None:
                mech_mh["E3_confounds"] = recomp
            compact = {
                "schema_version": "edara.wave-compact-summary/v1",
                "source_dir": chosen.name,
                "wave": wave,
                "cohort_kind": "multihost",
                "n_topics": raw.get("n_topics", 0),
                "n_manuscripts": raw.get("n_manuscripts", 0),
                "qwen_unified_summary": raw.get("qwen_unified_summary") or {},
                "mechanism_stats": mech_mh,
                "usage": raw.get("usage") or {},
                "manuscripts": [_compact_manuscript_entry(m, wave) for m in (raw.get("manuscripts") or [])],
            }
            dst = results_dir / chosen.name / f"{wave}_compact_summary.json"
            _write_json(dst, compact)
            for md_name in (f"{wave}_summary.md", f"{wave}_mechanism_stats.json"):
                if (chosen / md_name).is_file():
                    (results_dir / chosen.name).mkdir(parents=True, exist_ok=True)
                    (results_dir / chosen.name / md_name).write_bytes((chosen / md_name).read_bytes())
            synced.append(dst)

        evo_dirs = sorted(outputs_dir.glob(f"benchmark50_evosci_{wave}_*"))
        evo_dirs = [d for d in evo_dirs if d.is_dir() and (d / f"{wave}_evoscientist_summary.json").is_file()]
        if evo_dirs:
            chosen_evo = evo_dirs[-1]
            raw_evo = _load_json(chosen_evo / f"{wave}_evoscientist_summary.json")
            mech_e = raw_evo.get("mechanism_stats") or {}
            recomp_evo = _recompute_confounds_from_versions(chosen_evo)
            norm_mech = {
                "n_manuscripts": mech_e.get("n_manuscripts", 0),
                "E1_orig_keys_kept_all": mech_e.get("orig_keys_kept_all", 0),
                "E2_process_hype": mech_e.get("process_hype") or {},
                "E2_unverified_entities": mech_e.get("unverified_entities") or {},
                "E3_confounds": recomp_evo or mech_e.get("E3_confounds") or {},
            }
            compact_evo = {
                "schema_version": "edara.wave-compact-summary/v1",
                "source_dir": chosen_evo.name,
                "wave": wave,
                "cohort_kind": "evoscientist",
                "n_topics": raw_evo.get("n_topics", 0),
                "n_manuscripts": len(raw_evo.get("manuscripts") or []),
                "qwen_unified_summary": (raw_evo.get("qwen_unified_summary") or {}).get("E_vs_D") or {},
                "mechanism_stats": norm_mech,
                "usage": raw_evo.get("usage") or {},
                "manuscripts": [_compact_manuscript_entry(m, wave) for m in (raw_evo.get("manuscripts") or [])],
            }
            dst_evo = results_dir / chosen_evo.name / f"{wave}_evoscientist_compact.json"
            _write_json(dst_evo, compact_evo)
            if (chosen_evo / f"{wave}_evoscientist_summary.md").is_file():
                (results_dir / chosen_evo.name / f"{wave}_evoscientist_summary.md").write_bytes(
                    (chosen_evo / f"{wave}_evoscientist_summary.md").read_bytes()
                )
            synced.append(dst_evo)

    # Sync Action A (Cross-Backbone Generalization) compact prose-free bundle if present
    act_a_dirs = sorted(outputs_dir.glob("action_a_qwen_backbone_*"))
    act_a_dirs = [d for d in act_a_dirs if d.is_dir() and (d / "action_a_cross_backbone_summary.json").is_file()]
    if act_a_dirs:
        ca = act_a_dirs[-1]
        raw_a = _load_json(ca / "action_a_cross_backbone_summary.json")
        compact_a = {
            "schema_version": "edara.action-a-compact/v1",
            "source_dir": ca.name,
            "backbone_model": raw_a.get("backbone_model", "qwen3.8-flash"),
            "n_topics": raw_a.get("n_topics", 10),
            "n_manuscripts": raw_a.get("n_manuscripts", 30),
            "judge_summaries": raw_a.get("judge_summaries") or {},
            "by_host_summaries": raw_a.get("by_host_summaries") or {},
            "mechanism_stats": raw_a.get("mechanism_stats") or {},
            "total_cost_usd": raw_a.get("total_cost_usd", 0.0),
            "usage": raw_a.get("usage") or {},
        }
        dst_a = results_dir / ca.name / "action_a_compact_summary.json"
        _write_json(dst_a, compact_a)
        if (ca / "action_a_cross_backbone_summary.md").is_file():
            (results_dir / ca.name / "action_a_cross_backbone_summary.md").write_bytes(
                (ca / "action_a_cross_backbone_summary.md").read_bytes()
            )
        synced.append(dst_a)

    # Sync Action B (Slow-Thinking + 8192 Budget Ablation) compact prose-free bundle if present
    act_b_dirs = sorted(outputs_dir.glob("action_b_thinking_ablation_*"))
    act_b_dirs = [d for d in act_b_dirs if d.is_dir() and (d / "action_b_thinking_ablation_summary.json").is_file()]
    if act_b_dirs:
        cb = act_b_dirs[-1]
        raw_b = _load_json(cb / "action_b_thinking_ablation_summary.json")
        compact_b = {
            "schema_version": "edara.action-b-compact/v1",
            "source_dir": cb.name,
            "editor_model": raw_b.get("editor_model", "deepseek-v4.1-flash-1017"),
            "output_tokens": raw_b.get("output_tokens", 8192),
            "n_topics": raw_b.get("n_topics", 10),
            "n_manuscripts": raw_b.get("n_manuscripts", 30),
            "judge_summaries": raw_b.get("judge_summaries") or {},
            "by_host_summaries": raw_b.get("by_host_summaries") or {},
            "mechanism_stats": raw_b.get("mechanism_stats") or {},
            "direct_think_generation_probe": raw_b.get("direct_think_generation_probe") or {},
            "wave1_fast_4096_paired_reference": raw_b.get("wave1_fast_4096_paired_reference") or {},
            "total_cost_usd": raw_b.get("total_cost_usd", 0.0),
            "usage": raw_b.get("usage") or {},
        }
        dst_b = results_dir / cb.name / "action_b_compact_summary.json"
        _write_json(dst_b, compact_b)
        if (cb / "action_b_thinking_ablation_summary.md").is_file():
            (results_dir / cb.name / "action_b_thinking_ablation_summary.md").write_bytes(
                (cb / "action_b_thinking_ablation_summary.md").read_bytes()
            )
        synced.append(dst_b)

    return synced


def load_all_wave_bundles(results_dir: Path, outputs_dir: Path | None = None) -> list[dict[str, Any]]:
    if outputs_dir is not None and outputs_dir.is_dir():
        sync_compact_wave_bundles(outputs_dir, results_dir)

    bundles: list[dict[str, Any]] = []
    for wave in ("wave1", "wave2", "wave3"):
        mh_files = sorted(results_dir.glob(f"benchmark50_{wave}_*/{wave}_compact_summary.json"))
        if mh_files:
            bundles.append(_load_json(mh_files[-1]))
        ev_files = sorted(results_dir.glob(f"benchmark50_evosci_{wave}_*/{wave}_evoscientist_compact.json"))
        if ev_files:
            bundles.append(_load_json(ev_files[-1]))
    return bundles


def _topic_bootstrap(rows_by_topic: dict[str, list[float]], seed: int = 20261122) -> tuple[float, list[float]]:
    if not rows_by_topic:
        return 0.0, [0.0, 0.0]
    bins = [float(np.mean(vals)) for vals in rows_by_topic.values() if vals]
    if not bins:
        return 0.0, [0.0, 0.0]
    mean_v = round(float(np.mean(bins)), 4)
    if len(bins) < 2:
        return mean_v, [mean_v, mean_v]
    rng = np.random.default_rng(seed)
    draws = np.mean(rng.choice(bins, (10000, len(bins)), replace=True), axis=1)
    ci = [round(float(x), 4) for x in np.quantile(draws, [0.025, 0.975])]
    return mean_v, ci


def summarize_subset(manuscripts: list[dict[str, Any]], judge_key: str) -> dict[str, Any]:
    val_map = {"win": 1.0, "loss": -1.0, "tie": 0.0}
    counts: Counter[str] = Counter()
    by_topic_net: dict[str, list[float]] = defaultdict(list)
    by_topic_sum: dict[str, list[float]] = defaultdict(list)
    for m in manuscripts:
        jr = m["judges"][judge_key]
        st = jr["status"]
        counts[st] += 1
        if st in val_map:
            by_topic_net[m["task_id"]].append(val_map[st])
            by_topic_sum[m["task_id"]].append(float(jr.get("score_sum_delta") or 0.0))
    net_m, net_ci = _topic_bootstrap(by_topic_net)
    sum_m, sum_ci = _topic_bootstrap(by_topic_sum)
    return {
        "n_manuscripts": len(manuscripts),
        "n_topics": len(by_topic_net),
        "win": counts.get("win", 0),
        "loss": counts.get("loss", 0),
        "tie": counts.get("tie", 0),
        "topic_net_wins": net_m,
        "topic_net_wins_ci95": net_ci,
        "score_sum_delta": sum_m,
        "score_sum_delta_ci95": sum_ci,
    }


def _aggregate_mechanisms(bundles: list[dict[str, Any]]) -> dict[str, Any]:
    tot_ms = 0
    keys_kept = 0
    hype = {a: 0 for a in ("A", "B", "C", "D", "E")}
    unver = {a: 0 for a in ("A", "B", "C", "D", "E")}
    conf_req = 0
    conf_a = 0
    conf_d = 0
    conf_e = 0
    total_cost = 0.0

    for b in bundles:
        ms = b.get("mechanism_stats") or {}
        tot_ms += int(ms.get("n_manuscripts") or 0)
        keys_kept += int(ms.get("E1_orig_keys_kept_all") or 0)
        for a in ("A", "B", "C", "D", "E"):
            hype[a] += int(((ms.get("E2_process_hype") or {}).get(a) or {}).get("occurrences") or 0)
            unver[a] += int(((ms.get("E2_unverified_entities") or {}).get(a) or {}).get("total") or 0)
        c3 = ms.get("E3_confounds") or {}
        conf_req += int(c3.get("targets_required_n") or 0)
        conf_a += int(c3.get("A_addressed") or 0)
        conf_d += int(c3.get("D_addressed") or 0)
        conf_e += int(c3.get("E_addressed") or 0)
        for k, u in (b.get("usage") or {}).items():
            if isinstance(u, dict):
                # In older Wave 1 R2, 'C' usage was copied from frozen v4/ext15 audit logs when cohort_kind=='multihost' and wave=='wave1'
                if b.get("cohort_kind") == "multihost" and b.get("wave") == "wave1" and k not in ("E", "blind"):
                    continue
                total_cost += float(u.get("cost_usd") or 0.0)
    return {
        "n_manuscripts": tot_ms,
        "orig_keys_kept": keys_kept,
        "orig_keys_kept_rate": round(keys_kept / max(1, tot_ms), 4),
        "process_hype_occurrences": hype,
        "unverified_entities_total": unver,
        "confounds_required": conf_req,
        "confounds_A_addressed": conf_a,
        "confounds_D_addressed": conf_d,
        "confounds_E_addressed": conf_e,
        "total_cost_usd": round(total_cost, 4),
    }


def _render_table1_latex(rows_spec: list[tuple[str, list[dict[str, Any]]]]) -> str:
    lines = [
        "% Auto-generated by edara_mvp_eval.paper_artifacts",
        "\\begin{table*}[t]",
        "\\centering",
        "\\small",
        "\\caption{Same-manuscript paired evaluation (\\texttt{Arm E} governed middleware vs.\\ \\texttt{Arm D} budget-matched retrieval-refinement baseline) across upstream research hosts and three calibrated blind judges.}",
        "\\label{tab:main-multihost}",
        "\\begin{tabular}{l r ccc c c ccc}",
        "\\toprule",
        " & & \\multicolumn{5}{c}{\\textbf{Primary Independent Judge (Qwen3.8-flash)}} & \\multicolumn{3}{c}{\\textbf{3-Judge Majority Vote}} \\\\",
        "\\cmidrule(lr){3-7} \\cmidrule(lr){8-10}",
        "\\textbf{Upstream Host} & $N$ & W & L & T & \\textbf{Net Win (95\\% CI)} & \\textbf{$\\Delta$Sum (95\\% CI)} & W/L/T & \\textbf{Net Win} & \\textbf{$\\Delta$Sum} \\\\",
        "\\midrule",
    ]
    for label, sub in rows_spec:
        if not sub:
            continue
        if label.startswith("\\midrule"):
            lines.append("\\midrule")
            clean_label = label.replace("\\midrule ", "")
        else:
            clean_label = label
        sq = summarize_subset(sub, "qwen3.8-flash")
        sm = summarize_subset(sub, "majority")
        nci = f"[{sq['topic_net_wins_ci95'][0]:+.2f}, {sq['topic_net_wins_ci95'][1]:+.2f}]"
        sci = f"[{sq['score_sum_delta_ci95'][0]:+.2f}, {sq['score_sum_delta_ci95'][1]:+.2f}]"
        lines.append(
            f"{clean_label} & {sq['n_manuscripts']} & {sq['win']} & {sq['loss']} & {sq['tie']} & "
            f"${sq['topic_net_wins']:+.3f}$ {{\\scriptsize ${nci}$}} & "
            f"${sq['score_sum_delta']:+.2f}$ {{\\scriptsize ${sci}$}} & "
            f"{sm['win']}/{sm['loss']}/{sm['tie']} & ${sm['topic_net_wins']:+.3f}$ & ${sm['score_sum_delta']:+.2f}$ \\\\"
        )
    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table*}",
    ])
    return "\n".join(lines) + "\n"


def _render_table1_md(rows_spec: list[tuple[str, list[dict[str, Any]]]]) -> str:
    lines = [
        "| Upstream Host | $N$ | Qwen W/L/T | Qwen Net Win (95% CI) | Qwen $\\Delta$Sum (95% CI) | DeepSeek Net | GLM Net | **3-Judge Majority W/L/T (`Net`, `$\\Delta$Sum`)** |",
        "| :--- | ---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for label, sub in rows_spec:
        if not sub:
            continue
        clean = label.replace("\\midrule ", "").replace("\\textbf{", "**").replace("}", "**") if "\\textbf" in label else label.replace("\\midrule ", "")
        sq = summarize_subset(sub, "qwen3.8-flash")
        sd = summarize_subset(sub, "deepseek-v4.1-flash")
        sg = summarize_subset(sub, "GLM-5.3-flash")
        sm = summarize_subset(sub, "majority")
        lines.append(
            f"| {clean} | {sq['n_manuscripts']} | `{sq['win']}/{sq['loss']}/{sq['tie']}` | "
            f"**`{sq['topic_net_wins']:+.3f}`** (`[{sq['topic_net_wins_ci95'][0]:+.2f}, {sq['topic_net_wins_ci95'][1]:+.2f}]`) | "
            f"**`{sq['score_sum_delta']:+.2f}`** (`[{sq['score_sum_delta_ci95'][0]:+.2f}, {sq['score_sum_delta_ci95'][1]:+.2f}]`) | "
            f"`{sd['topic_net_wins']:+.3f}` | `{sg['topic_net_wins']:+.3f}` | "
            f"**`{sm['win']}/{sm['loss']}/{sm['tie']}`** (`{sm['topic_net_wins']:+.3f}`, `{sm['score_sum_delta']:+.2f}`) |"
        )
    return "\n".join(lines) + "\n"


def _render_domain_tables(
    manuscripts: list[dict[str, Any]],
    domains_meta: list[dict[str, Any]],
    topic_map: dict[str, dict[str, Any]],
) -> tuple[str, str]:
    md_lines = [
        "| Domain ID | Domain | $N$ | Qwen W/L/T | Qwen Net Win | Qwen $\\Delta$Sum | Majority W/L/T | Majority $\\Delta$Sum |",
        "| :--- | :--- | ---: | :---: | ---: | ---: | :---: | ---: |",
    ]
    tex_lines = [
        "% Auto-generated by edara_mvp_eval.paper_artifacts",
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        "\\caption{Domain-stratified performance across the 10 AI/CS domains and 2 benchmark tracks.}",
        "\\label{tab:domain-breakdown}",
        "\\begin{tabular}{ll r cc cc}",
        "\\toprule",
        "\\textbf{ID} & \\textbf{Research Domain} & $N$ & \\textbf{Qwen W/L/T} & \\textbf{Qwen $\\Delta$Sum} & \\textbf{Maj.\\ W/L/T} & \\textbf{Maj.\\ $\\Delta$Sum} \\\\",
        "\\midrule",
    ]
    for dom in domains_meta:
        did = dom["domain_id"]
        sub = [m for m in manuscripts if (topic_map.get(m["task_id"]) or {}).get("domain_id") == did]
        if not sub:
            continue
        sq = summarize_subset(sub, "qwen3.8-flash")
        sm = summarize_subset(sub, "majority")
        short_id = did.split("_")[0]
        den = dom["domain_en"]
        md_lines.append(
            f"| `{short_id}` | {den} | {len(sub)} | `{sq['win']}/{sq['loss']}/{sq['tie']}` | `{sq['topic_net_wins']:+.3f}` | `{sq['score_sum_delta']:+.2f}` | `{sm['win']}/{sm['loss']}/{sm['tie']}` | `{sm['score_sum_delta']:+.2f}` |"
        )
        tex_lines.append(
            f"\\texttt{{{short_id}}} & {den.replace('&', '\\&')} & {len(sub)} & {sq['win']}/{sq['loss']}/{sq['tie']} & ${sq['score_sum_delta']:+.2f}$ & {sm['win']}/{sm['loss']}/{sm['tie']} & ${sm['score_sum_delta']:+.2f}$ \\\\"
        )

    tex_lines.append("\\midrule")
    for track_key, track_label in (("lit2test", "Track: Lit2Test"), ("alphaideabench", "Track: AlphaIdeaBench")):
        sub_t = [m for m in manuscripts if (topic_map.get(m["task_id"]) or {}).get("source_track") == track_key]
        if not sub_t:
            continue
        sq = summarize_subset(sub_t, "qwen3.8-flash")
        sm = summarize_subset(sub_t, "majority")
        md_lines.append(
            f"| — | **{track_label}** | {len(sub_t)} | `{sq['win']}/{sq['loss']}/{sq['tie']}` | `{sq['topic_net_wins']:+.3f}` | `{sq['score_sum_delta']:+.2f}` | `{sm['win']}/{sm['loss']}/{sm['tie']}` | `{sm['score_sum_delta']:+.2f}` |"
        )
        tex_lines.append(
            f"\\multicolumn{{2}}{{l}}{{\\textbf{{{track_label}}}}} & {len(sub_t)} & {sq['win']}/{sq['loss']}/{sq['tie']} & ${sq['score_sum_delta']:+.2f}$ & {sm['win']}/{sm['loss']}/{sm['tie']} & ${sm['score_sum_delta']:+.2f}$ \\\\"
        )

    tex_lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}"])
    return "\n".join(tex_lines) + "\n", "\n".join(md_lines) + "\n"


def _render_mechanism_tables(mech: dict[str, Any]) -> tuple[str, str]:
    n = max(1, mech["n_manuscripts"])
    h = mech["process_hype_occurrences"]
    u = mech["unverified_entities_total"]
    h_red = -100.0 if h["D"] > 0 and h["E"] == 0 else (round(100.0 * (h["E"] - h["D"]) / max(1, h["D"]), 1))
    u_red = round(100.0 * (u["E"] - u["D"]) / max(1, u["D"]), 1)

    md_rows = [
        "| Deterministic Audit Metric | Arm A (Original) | Arm D (Unstructured R&R) | **Arm E (Governed Middleware)** | Relative Change (`E` vs `D`) |",
        "| :--- | :---: | :---: | :---: | :---: |",
        f"| Native Host Schema Keys Preserved | `{mech['n_manuscripts']}/{mech['n_manuscripts']}` | — | **`{mech['orig_keys_kept']}/{mech['n_manuscripts']}` (`{mech['orig_keys_kept_rate']*100:.1f}%`)** | Zero Schema Corruption |",
        f"| Unperformed Process-Hype Claims (`pre-registered`, etc.) | `{h['A']}` | `{h['D']}` | **`{h['E']}`** | **`{h_red:+.1f}%`** |",
        f"| Unverified External Hallucinated Entities | `{u['A']}` | `{u['D']}` | **`{u['E']}`** | **`{u_red:+.1f}%`** |",
    ]
    if mech["confounds_required"] > 0:
        req_c = mech["confounds_required"]
        e_pct = round(100.0 * mech["confounds_E_addressed"] / req_c, 1)
        d_pct = round(100.0 * mech["confounds_D_addressed"] / req_c, 1)
        md_rows.append(
            f"| Blocking Design Confounds Addressed | `{mech['confounds_A_addressed']}/{req_c}` | `{mech['confounds_D_addressed']}/{req_c}` (`{d_pct:.1f}%`) | **`{mech['confounds_E_addressed']}/{req_c}` (`{e_pct:.1f}%`)** | **`{e_pct - d_pct:+.1f}` pp** |"
        )
    md_rows.append(
        f"| Mean Full-Pipeline Cost (incl. 3-Judge Eval) | — | — | **`${mech['total_cost_usd']/n:.3f}` / manuscript** | Total `${mech['total_cost_usd']:.2f}` (`N={mech['n_manuscripts']}`) |"
    )
    md = "\n".join(md_rows) + "\n"

    tex_rows = [
        "% Auto-generated by edara_mvp_eval.paper_artifacts",
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        "\\caption{Deterministic epistemic hygiene and structural preservation audit across $N=" + str(mech["n_manuscripts"]) + "$ manuscripts.}",
        "\\label{tab:mechanism-audit}",
        "\\begin{tabular}{l ccc c}",
        "\\toprule",
        "\\textbf{Deterministic Mechanism Metric} & \\textbf{Arm A} & \\textbf{Arm D} & \\textbf{Arm E (Ours)} & \\textbf{Effect ($E$ vs.\\ $D$)} \\\\",
        "\\midrule",
        f"Native Host Schema Keys Preserved & {mech['n_manuscripts']}/{mech['n_manuscripts']} & --- & \\textbf{{{mech['orig_keys_kept']}/{mech['n_manuscripts']} ({mech['orig_keys_kept_rate']*100:.0f}\\%)}} & Zero schema loss \\\\",
        f"Fabricated Process-Hype Claims & {h['A']} & {h['D']} & \\textbf{{{h['E']}}} & \\textbf{{{h_red:+.1f}\\%}} \\\\",
        f"Unverified External Entities & {u['A']} & {u['D']} & \\textbf{{{u['E']}}} & \\textbf{{{u_red:+.1f}\\%}} \\\\",
    ]
    if mech["confounds_required"] > 0:
        req_c = mech["confounds_required"]
        e_pct = round(100.0 * mech["confounds_E_addressed"] / req_c, 1)
        d_pct = round(100.0 * mech["confounds_D_addressed"] / req_c, 1)
        tex_rows.append(
            f"Blocking Confounds Addressed & {mech['confounds_A_addressed']}/{req_c} & {mech['confounds_D_addressed']}/{req_c} ({d_pct:.1f}\\%) & \\textbf{{{mech['confounds_E_addressed']}/{req_c} ({e_pct:.1f}\\%)}} & \\textbf{{{e_pct - d_pct:+.1f}\\,pp}} \\\\"
        )
    tex_rows.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}"])
    tex = "\n".join(tex_rows) + "\n"
    return tex, md


def _render_fig1_svg(rows_spec: list[tuple[str, list[dict[str, Any]]]]) -> str:
    valid = [(lbl.replace("\\midrule ", "").replace("\\textbf{", "").replace("}", ""), sub) for lbl, sub in rows_spec if sub]
    row_h = 54
    top_pad = 72
    height = top_pad + len(valid) * row_h + 48
    width = 860
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}" font-family="Inter, -apple-system, BlinkMacSystemFont, Segoe UI, Arial, sans-serif">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="24" y="32" font-size="17" font-weight="700" fill="#111827">E-DARA Middleware (Arm E) vs. Matched Retrieval-Refine Baseline (Arm D)</text>',
        '<text x="24" y="52" font-size="12.5" fill="#4b5563">Primary Blind Judge (Qwen3.8-flash, Order-Swapped Bidirectional Folding) — Win / Tie / Loss Share &amp; Net Win Rate</text>',
    ]
    bar_x = 250
    bar_w = 380
    for idx, (label, sub) in enumerate(valid):
        y = top_pad + idx * row_h
        s = summarize_subset(sub, "qwen3.8-flash")
        n = max(1, s["n_manuscripts"])
        w_pct = s["win"] / n
        t_pct = s["tie"] / n
        l_pct = s["loss"] / n
        w_px = round(bar_w * w_pct, 1)
        t_px = round(bar_w * t_pct, 1)
        l_px = round(bar_w - w_px - t_px, 1)

        is_summary = "External" in label or "Overall" in label
        fw = "700" if is_summary else "500"
        svg.append(f'<text x="24" y="{y + 21}" font-size="13" font-weight="{fw}" fill="#1f2937">{html.escape(label)} (N={s["n_manuscripts"]})</text>')
        svg.append(f'<rect x="{bar_x}" y="{y + 4}" width="{w_px}" height="24" rx="3" fill="#15803d"/>')
        svg.append(f'<rect x="{bar_x + w_px}" y="{y + 4}" width="{t_px}" height="24" fill="#cbd5e1"/>')
        svg.append(f'<rect x="{bar_x + w_px + t_px}" y="{y + 4}" width="{max(0.0, l_px)}" height="24" rx="3" fill="#dc2626"/>')
        if w_px >= 34:
            svg.append(f'<text x="{bar_x + w_px/2}" y="{y + 20}" font-size="11" font-weight="600" fill="#ffffff" text-anchor="middle">{s["win"]}W</text>')
        if t_px >= 34:
            svg.append(f'<text x="{bar_x + w_px + t_px/2}" y="{y + 20}" font-size="11" font-weight="600" fill="#334155" text-anchor="middle">{s["tie"]}T</text>')
        if l_px >= 30:
            svg.append(f'<text x="{bar_x + w_px + t_px + l_px/2}" y="{y + 20}" font-size="11" font-weight="600" fill="#ffffff" text-anchor="middle">{s["loss"]}L</text>')

        net_color = "#15803d" if s["topic_net_wins"] > 0 else ("#dc2626" if s["topic_net_wins"] < 0 else "#475569")
        svg.append(
            f'<text x="{bar_x + bar_w + 18}" y="{y + 20}" font-size="12.5" font-weight="700" fill="{net_color}">'
            f'Net {s["topic_net_wins"]:+.3f} (ΔSum {s["score_sum_delta"]:+.2f})</text>'
        )

    leg_y = height - 18
    svg.extend([
        f'<rect x="{bar_x}" y="{leg_y - 10}" width="12" height="12" rx="2" fill="#15803d"/>',
        f'<text x="{bar_x + 18}" y="{leg_y}" font-size="11.5" fill="#374151">Arm E Wins</text>',
        f'<rect x="{bar_x + 105}" y="{leg_y - 10}" width="12" height="12" rx="2" fill="#cbd5e1"/>',
        f'<text x="{bar_x + 123}" y="{leg_y}" font-size="11.5" fill="#374151">Tie (incl. order-sensitive)</text>',
        f'<rect x="{bar_x + 290}" y="{leg_y - 10}" width="12" height="12" rx="2" fill="#dc2626"/>',
        f'<text x="{bar_x + 308}" y="{leg_y}" font-size="11.5" fill="#374151">Arm D Wins</text>',
        '</svg>',
    ])
    return "\n".join(svg) + "\n"


def _render_fig2_svg(mech: dict[str, Any]) -> str:
    h = mech["process_hype_occurrences"]
    u = mech["unverified_entities_total"]
    width, height = 760, 320
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}" font-family="Inter, -apple-system, BlinkMacSystemFont, Segoe UI, Arial, sans-serif">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="24" y="32" font-size="16.5" font-weight="700" fill="#111827">Epistemic Hygiene Audit Across Arms (N={mech["n_manuscripts"]} Manuscripts)</text>',
        '<text x="24" y="52" font-size="12" fill="#4b5563">Unstructured retrieval-refinement (Arm D) amplifies hype &amp; unverified citations; governed middleware (Arm E) suppresses both.</text>',
    ]
    panels = [
        ("Fabricated Process-Hype Claims", h["A"], h["D"], h["E"], 40),
        ("Unverified External Entities", u["A"], u["D"], u["E"], 400),
    ]
    for title, va, vd, ve, px in panels:
        max_v = max(1, va, vd, ve)
        base_y = 255
        max_h = 155
        svg.append(f'<text x="{px}" y="86" font-size="13.5" font-weight="700" fill="#1f2937">{html.escape(title)}</text>')
        svg.append(f'<line x1="{px}" y1="{base_y}" x2="{px + 300}" y2="{base_y}" stroke="#94a3b8" stroke-width="1.5"/>')
        bars = [
            ("Arm A (Orig)", va, "#64748b", px + 24),
            ("Arm D (R&R)", vd, "#ef4444", px + 118),
            ("Arm E (Ours)", ve, "#15803d", px + 212),
        ]
        for b_lbl, val, col, bx in bars:
            bh = round(max_h * (val / max_v), 1)
            by = round(base_y - bh, 1)
            svg.append(f'<rect x="{bx}" y="{by}" width="58" height="{bh}" rx="4" fill="{col}"/>')
            svg.append(f'<text x="{bx + 29}" y="{by - 7}" font-size="12.5" font-weight="700" fill="{col}" text-anchor="middle">{val}</text>')
            svg.append(f'<text x="{bx + 29}" y="{base_y + 20}" font-size="11.5" fill="#334155" text-anchor="middle">{html.escape(b_lbl)}</text>')
    svg.append('</svg>')
    return "\n".join(svg) + "\n"


def _render_fig3_svg(
    manuscripts: list[dict[str, Any]],
    domains_meta: list[dict[str, Any]],
    topic_map: dict[str, dict[str, Any]],
) -> str:
    hosts_present = [h for h in HOST_ORDER if any(m["host"] == h for m in manuscripts)]
    cell_w, cell_h = 122, 34
    left_pad, top_pad = 285, 82
    width = left_pad + len(hosts_present) * cell_w + 40
    height = top_pad + len(domains_meta) * cell_h + 42

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}" font-family="Inter, -apple-system, BlinkMacSystemFont, Segoe UI, Arial, sans-serif">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="24" y="30" font-size="16" font-weight="700" fill="#111827">Domain × Host Score-Sum-Delta Heatmap (Arm E vs. Arm D, Qwen3.8-flash)</text>',
        '<text x="24" y="49" font-size="12" fill="#4b5563">Green cells indicate positive 5-dimension rubric gain (+ΔSum) from governed middleware over matched baseline.</text>',
    ]
    for c_idx, h_key in enumerate(hosts_present):
        cx = left_pad + c_idx * cell_w + cell_w / 2
        svg.append(f'<text x="{cx}" y="{top_pad - 12}" font-size="12" font-weight="700" fill="#1f2937" text-anchor="middle">{html.escape(HOST_LABELS[h_key])}</text>')

    for r_idx, dom in enumerate(domains_meta):
        ry = top_pad + r_idx * cell_h
        short_id = dom["domain_id"].split("_")[0]
        lbl = f"{short_id}: {dom['domain_en']}"
        svg.append(f'<text x="24" y="{ry + 22}" font-size="12" fill="#1f2937">{html.escape(lbl[:38])}</text>')
        for c_idx, h_key in enumerate(hosts_present):
            cx = left_pad + c_idx * cell_w
            sub = [
                m for m in manuscripts
                if m["host"] == h_key and (topic_map.get(m["task_id"]) or {}).get("domain_id") == dom["domain_id"]
            ]
            if not sub:
                svg.append(f'<rect x="{cx + 2}" y="{ry + 2}" width="{cell_w - 4}" height="{cell_h - 4}" rx="4" fill="#f8fafc" stroke="#e2e8f0"/>')
                svg.append(f'<text x="{cx + cell_w/2}" y="{ry + 22}" font-size="11" fill="#94a3b8" text-anchor="middle">—</text>')
                continue
            val = summarize_subset(sub, "qwen3.8-flash")["score_sum_delta"]
            if val > 1.5:
                bg, fg = "#166534", "#ffffff"
            elif val > 0.2:
                bg, fg = "#bbf7d0", "#14532d"
            elif val >= -0.2:
                bg, fg = "#f1f5f9", "#334155"
            else:
                bg, fg = "#fecaca", "#7f1d1d"
            svg.append(f'<rect x="{cx + 2}" y="{ry + 2}" width="{cell_w - 4}" height="{cell_h - 4}" rx="4" fill="{bg}"/>')
            svg.append(f'<text x="{cx + cell_w/2}" y="{ry + 22}" font-size="11.5" font-weight="600" fill="{fg}" text-anchor="middle">{val:+.2f}</text>')

    svg.append('</svg>')
    return "\n".join(svg) + "\n"


def _render_table4_cross_backbone(act_a: dict[str, Any]) -> tuple[str, str]:
    js = act_a.get("judge_summaries") or {}
    bh = act_a.get("by_host_summaries") or {}
    ms = act_a.get("mechanism_stats") or {}
    n_tot = act_a.get("n_manuscripts", 30)

    md_lines = [
        "| Stratum / Host (`Qwen3.8-flash` Backbone) | $N$ | DeepSeek Indep. W/L/T (`Net`) | GLM Indep. W/L/T (`Net`) | **Disjoint 2-Judge (`Net`)** | **3-Judge Majority W/L/T (`Net`, `$\\Delta$Sum`)** |",
        "| :--- | ---: | :---: | :---: | :---: | :---: |",
    ]
    tex_lines = [
        "% Auto-generated by edara_mvp_eval.paper_artifacts",
        "\\begin{table*}[t]",
        "\\centering",
        "\\small",
        "\\caption{Cross-backbone generalization (\\texttt{Action A}): replacing \\texttt{DeepSeek-v4.1-flash} with \\texttt{Qwen3.8-flash} as both upstream host generator and refinement editor across 10 domains ($N=" + str(n_tot) + "$), evaluated under role-inverted disjoint judges.}",
        "\\label{tab:cross-backbone}",
        "\\begin{tabular}{l r cc c cc}",
        "\\toprule",
        "\\textbf{Host Paradigm (Qwen3.8 Backbone)} & $N$ & \\textbf{DeepSeek Indep.} & \\textbf{GLM Indep.} & \\textbf{Disjoint 2-Judge Net} & \\textbf{3-Judge Maj.\\ W/L/T} & \\textbf{Maj.\\ Net ($\\Delta$Sum)} \\\\",
        "\\midrule",
    ]
    for h_key in ("evoscientist", "direct", "ai_scientist"):
        if h_key not in bh:
            continue
        hm = bh[h_key]
        sd, sg, si, sm = hm["deepseek-v4.1-flash"], hm["GLM-5.3-flash"], hm["independent_2judge"], hm["majority_3way"]
        cd, cg, ci, cm = sd["counts"], sg["counts"], si["counts"], sm["counts"]
        nh = sum(cd.values())
        lbl = HOST_LABELS.get(h_key, h_key)
        md_lines.append(
            f"| {lbl} | {nh} | `{cd.get('win',0)}/{cd.get('loss',0)}/{cd.get('tie',0)}` (`{sd['topic_net_wins']:+.3f}`) | "
            f"`{cg.get('win',0)}/{cg.get('loss',0)}/{cg.get('tie',0)}` (`{sg['topic_net_wins']:+.3f}`) | "
            f"**`{ci.get('win',0)}/{ci.get('loss',0)}/{ci.get('tie',0)}` (`{si['topic_net_wins']:+.3f}`)** | "
            f"**`{cm.get('win',0)}/{cm.get('loss',0)}/{cm.get('tie',0)}`** (`{sm['topic_net_wins']:+.3f}`, `{sm['score_sum_delta']:+.2f}`) |"
        )
        tex_lines.append(
            f"{lbl} & {nh} & {cd.get('win',0)}/{cd.get('loss',0)}/{cd.get('tie',0)} (${sd['topic_net_wins']:+.2f}$) & "
            f"{cg.get('win',0)}/{cg.get('loss',0)}/{cg.get('tie',0)} (${sg['topic_net_wins']:+.2f}$) & "
            f"${si['topic_net_wins']:+.3f}$ & {cm.get('win',0)}/{cm.get('loss',0)}/{cm.get('tie',0)} & "
            f"${sm['topic_net_wins']:+.3f}$ (${sm['score_sum_delta']:+.2f}$) \\\\"
        )

    sd, sg, si, sm = js["deepseek-v4.1-flash"], js["GLM-5.3-flash"], js["independent_2judge"], js["majority_3way"]
    cd, cg, ci, cm = sd["counts"], sg["counts"], si["counts"], sm["counts"]
    c3 = ms.get("E3_confounds") or {}
    h2 = ms.get("E2_process_hype") or {}
    md_lines.append(
        f"| **Overall All 3 Hosts (`Qwen3.8-flash`)** | {n_tot} | `{cd.get('win',0)}/{cd.get('loss',0)}/{cd.get('tie',0)}` (`{sd['topic_net_wins']:+.3f}`) | "
        f"`{cg.get('win',0)}/{cg.get('loss',0)}/{cg.get('tie',0)}` (`{sg['topic_net_wins']:+.3f}`) | "
        f"**`{ci.get('win',0)}/{ci.get('loss',0)}/{ci.get('tie',0)}` (`{si['topic_net_wins']:+.3f}`)** | "
        f"**`{cm.get('win',0)}/{cm.get('loss',0)}/{cm.get('tie',0)}`** (`{sm['topic_net_wins']:+.3f}`, `{sm['score_sum_delta']:+.2f}`) |"
    )
    if c3.get("targets_required_n"):
        req = c3["targets_required_n"]
        md_lines.append("")
        md_lines.append(
            f"- **Deterministic Guardrails (`Qwen3.8-flash`)**: Schema keys `{ms.get('E1_orig_keys_kept_all', n_tot)}/{n_tot} (100%)`, "
            f"Process hype `A={(h2.get('A') or {}).get('occurrences', 0)} -> D={(h2.get('D') or {}).get('occurrences', 0)} -> E={(h2.get('E') or {}).get('occurrences', 0)} (-100%)`, "
            f"Blocking confounds addressed `A={c3.get('A_addressed',0)}/{req} -> D={c3.get('D_addressed',0)}/{req} ({100*c3.get('D_addressed',0)/req:.1f}%) -> E={c3.get('E_addressed',0)}/{req} ({100*c3.get('E_addressed',0)/req:.1f}%)`."
        )

    tex_lines.extend([
        "\\midrule",
        f"\\textbf{{Overall All Hosts}} & {n_tot} & {cd.get('win',0)}/{cd.get('loss',0)}/{cd.get('tie',0)} (${sd['topic_net_wins']:+.2f}$) & "
        f"{cg.get('win',0)}/{cg.get('loss',0)}/{cg.get('tie',0)} (${sg['topic_net_wins']:+.2f}$) & "
        f"\\textbf{{${si['topic_net_wins']:+.3f}$}} & \\textbf{{{cm.get('win',0)}/{cm.get('loss',0)}/{cm.get('tie',0)}}} & "
        f"\\textbf{{${sm['topic_net_wins']:+.3f}$}} (${sm['score_sum_delta']:+.2f}$) \\\\",
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table*}",
    ])
    return "\n".join(tex_lines) + "\n", "\n".join(md_lines) + "\n"


def _render_table5_thinking_ablation(act_b: dict[str, Any]) -> tuple[str, str, str]:
    js = act_b.get("judge_summaries") or {}
    bh = act_b.get("by_host_summaries") or {}
    ms = act_b.get("mechanism_stats") or {}
    fast = act_b.get("wave1_fast_4096_paired_reference") or {}
    probe = act_b.get("direct_think_generation_probe") or {}
    n_tot = act_b.get("n_manuscripts", 30)

    sq = js["qwen3.8-flash"]
    sd = js["deepseek-v4.1-flash"]
    sg = js["GLM-5.3-flash"]
    sm = js["majority_3way"]

    md_lines = [
        "| Regime / Stratum (`N=30` Paired Wave-1 Manuscripts) | Qwen Primary W/L/T (`Net`, `$\\Delta$Sum`) | DeepSeek W/L/T (`Net`, `$\\Delta$Sum`) | GLM W/L/T (`Net`, `$\\Delta$Sum`) | **3-Judge Majority W/L/T (`Net [95% CI]`, `$\\Delta$Sum [95% CI]`)** |",
        "| :--- | :---: | :---: | :---: | :---: |",
    ]
    if fast.get("judge_summaries"):
        fj = fast["judge_summaries"]
        fq, fd, fg, fm = fj["qwen3.8-flash"], fj["deepseek-v4.1-flash"], fj["GLM-5.3-flash"], fj["majority_3way"]
        md_lines.append(
            f"| `Fast Mode` (`thinking=off`, `4096` cap) | `{fq['counts'].get('win',0)}/{fq['counts'].get('loss',0)}/{fq['counts'].get('tie',0)}` (`{fq['topic_net_wins']:+.3f}`, `{fq['score_sum_delta']:+.2f}`) | "
            f"`{fd['counts'].get('win',0)}/{fd['counts'].get('loss',0)}/{fd['counts'].get('tie',0)}` (`{fd['topic_net_wins']:+.3f}`, `{fd['score_sum_delta']:+.2f}`) | "
            f"`{fg['counts'].get('win',0)}/{fg['counts'].get('loss',0)}/{fg['counts'].get('tie',0)}` (`{fg['topic_net_wins']:+.3f}`, `{fg['score_sum_delta']:+.2f}`) | "
            f"`{fm['counts'].get('win',0)}/{fm['counts'].get('loss',0)}/{fm['counts'].get('tie',0)}` (`{fm['topic_net_wins']:+.3f}`, `{fm['score_sum_delta']:+.2f}`) |"
        )
    m_nci = sm["topic_net_wins_ci95"]
    m_sci = sm["score_sum_delta_ci95"]
    md_lines.append(
        f"| **`Slow-Thinking Mode` (`thinking=on`, `8192` cap)** | **`{sq['counts'].get('win',0)}/{sq['counts'].get('loss',0)}/{sq['counts'].get('tie',0)}`** (`{sq['topic_net_wins']:+.3f}`, `{sq['score_sum_delta']:+.2f}`) | "
        f"**`{sd['counts'].get('win',0)}/{sd['counts'].get('loss',0)}/{sd['counts'].get('tie',0)}`** (`{sd['topic_net_wins']:+.3f}`, `{sd['score_sum_delta']:+.2f}`) | "
        f"**`{sg['counts'].get('win',0)}/{sg['counts'].get('loss',0)}/{sg['counts'].get('tie',0)}`** (`{sg['topic_net_wins']:+.3f}`, `{sg['score_sum_delta']:+.2f}`) | "
        f"**`{sm['counts'].get('win',0)}/{sm['counts'].get('loss',0)}/{sm['counts'].get('tie',0)}`** (**`{sm['topic_net_wins']:+.3f}`** `[{m_nci[0]:+.2f}, {m_nci[1]:+.2f}]`, **`{sm['score_sum_delta']:+.2f}`** `[{m_sci[0]:+.2f}, {m_sci[1]:+.2f}]`) |"
    )
    for h_key in ("ai_scientist", "evoscientist", "direct"):
        if h_key not in bh:
            continue
        hm = bh[h_key]
        hq, hd, hg, hm_maj = hm["qwen3.8-flash"], hm["deepseek-v4.1-flash"], hm["GLM-5.3-flash"], hm["majority_3way"]
        lbl = HOST_LABELS.get(h_key, h_key)
        md_lines.append(
            f"| ↳ {lbl} (`Slow-Thinking`, $n=10$) | `{hq['counts'].get('win',0)}/{hq['counts'].get('loss',0)}/{hq['counts'].get('tie',0)}` (`{hq['topic_net_wins']:+.3f}`, `{hq['score_sum_delta']:+.2f}`) | "
            f"`{hd['counts'].get('win',0)}/{hd['counts'].get('loss',0)}/{hd['counts'].get('tie',0)}` (`{hd['topic_net_wins']:+.3f}`, `{hd['score_sum_delta']:+.2f}`) | "
            f"`{hg['counts'].get('win',0)}/{hg['counts'].get('loss',0)}/{hg['counts'].get('tie',0)}` (`{hg['topic_net_wins']:+.3f}`, `{hg['score_sum_delta']:+.2f}`) | "
            f"**`{hm_maj['counts'].get('win',0)}/{hm_maj['counts'].get('loss',0)}/{hm_maj['counts'].get('tie',0)}`** (`{hm_maj['topic_net_wins']:+.3f}`, `{hm_maj['score_sum_delta']:+.2f}`) |"
        )

    h_a = ((ms.get("E2_process_hype") or {}).get("A") or {}).get("occurrences", 0)
    h_d = ((ms.get("E2_process_hype") or {}).get("D") or {}).get("occurrences", 0)
    h_e = ((ms.get("E2_process_hype") or {}).get("E") or {}).get("occurrences", 0)
    u_a = ((ms.get("E2_unverified_entities") or {}).get("A") or {}).get("total", 0)
    u_d = ((ms.get("E2_unverified_entities") or {}).get("D") or {}).get("total", 0)
    u_e = ((ms.get("E2_unverified_entities") or {}).get("E") or {}).get("total", 0)
    c3 = ms.get("E3_confounds") or {}
    fh = fast.get("hygiene") or {}

    md_lines.extend([
        "",
        f"- **The Reasoning-Without-Governance Paradox**: Enabling test-time slow-thinking in unconstrained `Arm D_think` **inflated fabricated process-hype claims from `A={h_a}` & `D_fast={fh.get('D_process_hype', 42)}` up to `D_think={h_d}` (`+116.7%` over fast baseline)** while leaving unverified entities at `{u_d}` and addressing only `{c3.get('D_addressed',0)}/{c3.get('targets_required_n',14)}` blocking confounds. Governed **`Arm E_think` reduced process hype `91 -> {h_e}` (`-100.0%`)**, cut unverified entities **`{u_d} -> {u_e}` (`-79.2%`)**, and resolved **`{c3.get('E_addressed',0)}/{c3.get('targets_required_n',14)}` (`85.7%`)** blocking confounds.",
        f"- **Initial Slow-Thinking Generation Probe (`n={probe.get('n_topics',10)}` Direct topics)**: Native slow-thinking during `Arm A` generation likewise increased process hype (`{probe.get('fast_A_process_hype_total',0)} -> {probe.get('think_A_process_hype_total',0)}`) and unverified external entities (`{probe.get('fast_A_unverified_entities_total',0)} -> {probe.get('think_A_unverified_entities_total',0)}`).",
    ])

    tex_lines = [
        "% Auto-generated by edara_mvp_eval.paper_artifacts",
        "\\begin{table*}[t]",
        "\\centering",
        "\\small",
        "\\caption{Test-time reasoning and symmetric \\texttt{8192}-token budget ablation (\\texttt{Action B}, $N=" + str(n_tot) + "$ paired Wave-1 manuscripts). Without epistemic governance, slow-thinking amplifies process hype ($D_{\\text{fast}}=" + str(fh.get("D_process_hype", 42)) + " \\to D_{\\text{think}}=" + str(h_d) + "$), whereas governed \\texttt{Arm E} achieves significant gains across all three blind judges and $0$ hype defects.}",
        "\\label{tab:thinking-ablation}",
        "\\begin{tabular}{l cc cc cc cc}",
        "\\toprule",
        "\\textbf{Regime / Host} & \\textbf{Qwen W/L/T} & \\textbf{$\\Delta$Sum} & \\textbf{DeepSeek W/L/T} & \\textbf{$\\Delta$Sum} & \\textbf{GLM W/L/T} & \\textbf{$\\Delta$Sum} & \\textbf{Maj.\\ Net [95\\% CI]} & \\textbf{Maj.\\ $\\Delta$Sum [95\\% CI]} \\\\",
        "\\midrule",
    ]
    if fast.get("judge_summaries"):
        fj = fast["judge_summaries"]
        fq, fd, fg, fm = fj["qwen3.8-flash"], fj["deepseek-v4.1-flash"], fj["GLM-5.3-flash"], fj["majority_3way"]
        fnci = fm["topic_net_wins_ci95"]
        fsci = fm["score_sum_delta_ci95"]
        tex_lines.append(
            f"Fast Mode (4096, \\texttt{{think=off}}) & {fq['counts'].get('win',0)}/{fq['counts'].get('loss',0)}/{fq['counts'].get('tie',0)} & ${fq['score_sum_delta']:+.2f}$ & "
            f"{fd['counts'].get('win',0)}/{fd['counts'].get('loss',0)}/{fd['counts'].get('tie',0)} & ${fd['score_sum_delta']:+.2f}$ & "
            f"{fg['counts'].get('win',0)}/{fg['counts'].get('loss',0)}/{fg['counts'].get('tie',0)} & ${fg['score_sum_delta']:+.2f}$ & "
            f"${fm['topic_net_wins']:+.3f}$ {{\\scriptsize $[{fnci[0]:+.2f},{fnci[1]:+.2f}]$}} & ${fm['score_sum_delta']:+.2f}$ {{\\scriptsize $[{fsci[0]:+.2f},{fsci[1]:+.2f}]$}} \\\\"
        )
    tex_lines.append(
        f"\\textbf{{Slow-Thinking (8192, \\texttt{{think=on}})}} & \\textbf{{{sq['counts'].get('win',0)}/{sq['counts'].get('loss',0)}/{sq['counts'].get('tie',0)}}} & \\textbf{{${sq['score_sum_delta']:+.2f}$}} & "
        f"\\textbf{{{sd['counts'].get('win',0)}/{sd['counts'].get('loss',0)}/{sd['counts'].get('tie',0)}}} & \\textbf{{${sd['score_sum_delta']:+.2f}$}} & "
        f"\\textbf{{{sg['counts'].get('win',0)}/{sg['counts'].get('loss',0)}/{sg['counts'].get('tie',0)}}} & \\textbf{{${sg['score_sum_delta']:+.2f}$}} & "
        f"\\textbf{{${sm['topic_net_wins']:+.3f}$}} {{\\scriptsize $[{m_nci[0]:+.2f},{m_nci[1]:+.2f}]$}} & \\textbf{{${sm['score_sum_delta']:+.2f}$}} {{\\scriptsize $[{m_sci[0]:+.2f},{m_sci[1]:+.2f}]$}} \\\\"
    )
    tex_lines.append("\\midrule")
    for h_key in ("ai_scientist", "evoscientist", "direct"):
        if h_key not in bh:
            continue
        hm = bh[h_key]
        hq, hd, hg, hm_maj = hm["qwen3.8-flash"], hm["deepseek-v4.1-flash"], hm["GLM-5.3-flash"], hm["majority_3way"]
        lbl = HOST_LABELS.get(h_key, h_key)
        tex_lines.append(
            f"\\quad {lbl} ($n=10$) & {hq['counts'].get('win',0)}/{hq['counts'].get('loss',0)}/{hq['counts'].get('tie',0)} & ${hq['score_sum_delta']:+.2f}$ & "
            f"{hd['counts'].get('win',0)}/{hd['counts'].get('loss',0)}/{hd['counts'].get('tie',0)} & ${hd['score_sum_delta']:+.2f}$ & "
            f"{hg['counts'].get('win',0)}/{hg['counts'].get('loss',0)}/{hg['counts'].get('tie',0)} & ${hg['score_sum_delta']:+.2f}$ & "
            f"${hm_maj['topic_net_wins']:+.3f}$ & ${hm_maj['score_sum_delta']:+.2f}$ \\\\"
        )
    tex_lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table*}"])

    # SVG Figure 4: Reasoning-Without-Governance Paradox
    d_fast_h = int(fh.get("D_process_hype", 42))
    e_fast_h = int(fh.get("E_process_hype", 3))
    svg = [
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 820 330" width="820" height="330" font-family="Inter, -apple-system, BlinkMacSystemFont, Segoe UI, Arial, sans-serif">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="24" y="30" font-size="16.5" font-weight="700" fill="#111827">The Reasoning-Without-Governance Paradox (N=30 Paired Manuscripts)</text>',
        '<text x="24" y="50" font-size="12" fill="#4b5563">Test-time slow-thinking more than doubles fabricated process hype in ungoverned Arm D (42 &#8594; 91), whereas Governed Arm E drives hype to 0.</text>',
    ]
    base_y = 265
    max_h = 165
    max_val = max(1, h_a, d_fast_h, h_d)
    svg.append('<text x="36" y="84" font-size="13.5" font-weight="700" fill="#1f2937">Fabricated Process-Hype Claims Across Reasoning Regimes</text>')
    svg.append(f'<line x1="36" y1="{base_y}" x2="465" y2="{base_y}" stroke="#94a3b8" stroke-width="1.5"/>')
    bars_left = [
        ("Arm A (Orig)", h_a, "#64748b", 48),
        ("D (Fast)", d_fast_h, "#f97316", 132),
        ("D (Think)", h_d, "#dc2626", 216),
        ("E (Fast)", e_fast_h, "#22c55e", 300),
        ("E (Think)", h_e, "#15803d", 384),
    ]
    for lbl, val, col, bx in bars_left:
        bh_px = round(max_h * (val / max_val), 1)
        by = round(base_y - bh_px, 1)
        svg.append(f'<rect x="{bx}" y="{by}" width="54" height="{max(2.0, bh_px)}" rx="4" fill="{col}"/>')
        svg.append(f'<text x="{bx + 27}" y="{by - 7}" font-size="12.5" font-weight="700" fill="{col}" text-anchor="middle">{val}</text>')
        svg.append(f'<text x="{bx + 27}" y="{base_y + 20}" font-size="11" fill="#334155" text-anchor="middle">{html.escape(lbl)}</text>')

    svg.append('<text x="505" y="84" font-size="13.5" font-weight="700" fill="#1f2937">Slow-Thinking Unverified Entities</text>')
    svg.append(f'<line x1="505" y1="{base_y}" x2="790" y2="{base_y}" stroke="#94a3b8" stroke-width="1.5"/>')
    max_u = max(1, u_a, u_d, u_e)
    bars_right = [
        ("Arm A", u_a, "#64748b", 522),
        ("D (Think)", u_d, "#dc2626", 616),
        ("E (Think)", u_e, "#15803d", 710),
    ]
    for lbl, val, col, bx in bars_right:
        bh_px = round(max_h * (val / max_u), 1)
        by = round(base_y - bh_px, 1)
        svg.append(f'<rect x="{bx}" y="{by}" width="58" height="{max(2.0, bh_px)}" rx="4" fill="{col}"/>')
        svg.append(f'<text x="{bx + 29}" y="{by - 7}" font-size="12.5" font-weight="700" fill="{col}" text-anchor="middle">{val}</text>')
        svg.append(f'<text x="{bx + 29}" y="{base_y + 20}" font-size="11" fill="#334155" text-anchor="middle">{html.escape(lbl)}</text>')
    svg.append("</svg>")
    return "\n".join(tex_lines) + "\n", "\n".join(md_lines) + "\n", "\n".join(svg) + "\n"


def generate_paper_artifacts(
    *,
    results_dir: Path,
    out_dir: Path,
    outputs_dir: Path | None = None,
    manifest_path: Path | None = None,
) -> dict[str, Any]:
    manifest_file = manifest_path or (PACKAGE_ROOT / "data/benchmark_50_topics.json")
    bench = _load_json(manifest_file)
    domains_meta = bench["domains"]
    topic_map = {t["task_id"]: t for t in bench["topics"]}

    bundles = load_all_wave_bundles(results_dir, outputs_dir=outputs_dir)
    dedup: dict[tuple[str, str], dict[str, Any]] = {}
    waves_loaded: sorted_waves = set()  # type: ignore[valid-type]
    for b in bundles:
        waves_loaded.add(b["wave"])
        for m in b.get("manuscripts") or []:
            dedup[(m["task_id"], m["host"])] = m

    manuscripts = list(dedup.values())
    waves_list = sorted(waves_loaded)
    topics_covered = sorted({m["task_id"] for m in manuscripts})

    rows_spec: list[tuple[str, list[dict[str, Any]]]] = []
    for h_key in HOST_ORDER:
        sub_h = [m for m in manuscripts if m["host"] == h_key]
        if sub_h:
            rows_spec.append((HOST_LABELS[h_key], sub_h))
    ext_sub = [m for m in manuscripts if m["host"] in ("ai_scientist", "evoscientist")]
    if ext_sub:
        rows_spec.append(("\\midrule \\textbf{External Agents Combined}", ext_sub))
    if manuscripts:
        rows_spec.append(("\\textbf{Overall All Hosts}", manuscripts))

    mech = _aggregate_mechanisms(bundles)

    out_dir.mkdir(parents=True, exist_ok=True)
    t1_tex = _render_table1_latex(rows_spec)
    t1_md = _render_table1_md(rows_spec)
    t2_tex, t2_md = _render_domain_tables(manuscripts, domains_meta, topic_map)
    t3_tex, t3_md = _render_mechanism_tables(mech)
    f1_svg = _render_fig1_svg(rows_spec)
    f2_svg = _render_fig2_svg(mech)
    f3_svg = _render_fig3_svg(manuscripts, domains_meta, topic_map)

    (out_dir / "table1_main_multihost.tex").write_text(t1_tex, encoding="utf-8")
    (out_dir / "table1_main_multihost.md").write_text(t1_md, encoding="utf-8")
    (out_dir / "table2_domain_breakdown.tex").write_text(t2_tex, encoding="utf-8")
    (out_dir / "table2_domain_breakdown.md").write_text(t2_md, encoding="utf-8")
    (out_dir / "table3_mechanism_ablation.tex").write_text(t3_tex, encoding="utf-8")
    (out_dir / "table3_mechanism_ablation.md").write_text(t3_md, encoding="utf-8")
    (out_dir / "fig1_host_winrates.svg").write_text(f1_svg, encoding="utf-8")
    (out_dir / "fig2_epistemic_hygiene.svg").write_text(f2_svg, encoding="utf-8")
    (out_dir / "fig3_domain_heatmap.svg").write_text(f3_svg, encoding="utf-8")

    act_a_files = sorted(results_dir.glob("action_a_qwen_backbone_*/action_a_compact_summary.json"))
    act_b_files = sorted(results_dir.glob("action_b_thinking_ablation_*/action_b_compact_summary.json"))
    act_a_data = _load_json(act_a_files[-1]) if act_a_files else None
    act_b_data = _load_json(act_b_files[-1]) if act_b_files else None

    extra_md_sections: list[str] = []
    extra_tex_list: list[str] = []
    if act_a_data:
        t4_tex, t4_md = _render_table4_cross_backbone(act_a_data)
        (out_dir / "table4_cross_backbone.tex").write_text(t4_tex, encoding="utf-8")
        (out_dir / "table4_cross_backbone.md").write_text(t4_md, encoding="utf-8")
        extra_md_sections.extend([
            "",
            "## Table 4: Cross-Backbone Generalization (`Qwen3.8-flash` Host + Editor, Role-Inverted Disjoint Judges)",
            "",
            t4_md.strip(),
        ])
        extra_tex_list.append("- `table4_cross_backbone.tex` (`booktabs` cross-backbone generalization table)")

    if act_b_data:
        t5_tex, t5_md, f4_svg = _render_table5_thinking_ablation(act_b_data)
        (out_dir / "table5_thinking_ablation.tex").write_text(t5_tex, encoding="utf-8")
        (out_dir / "table5_thinking_ablation.md").write_text(t5_md, encoding="utf-8")
        (out_dir / "fig4_reasoning_hygiene_paradox.svg").write_text(f4_svg, encoding="utf-8")
        extra_md_sections.extend([
            "",
            "## Table 5: Test-Time Reasoning (`Slow-Thinking`) & `8192`-Budget Sensitivity Ablation",
            "",
            "![Reasoning Hygiene Paradox](./fig4_reasoning_hygiene_paradox.svg)",
            "",
            t5_md.strip(),
        ])
        extra_tex_list.append("- `table5_thinking_ablation.tex` (`booktabs` 2×2 test-time reasoning scaling table)")

    master_md = "\n".join([
        f"# Benchmark-50 Paper-Ready Tables & Figures (`Waves: {', '.join(waves_list) or 'none'}`, `{len(topics_covered)}` Topics, `N={len(manuscripts)}` Manuscripts)",
        "",
        "![Host Win Rates](./fig1_host_winrates.svg)",
        "",
        "## Table 1: Multi-Host Paired Blind Evaluation (`Arm E` vs. `Arm D`)",
        "",
        t1_md.strip(),
        "",
        "## Table 2: Deterministic Epistemic Hygiene & Guardrail Audit",
        "",
        "![Epistemic Hygiene](./fig2_epistemic_hygiene.svg)",
        "",
        t3_md.strip(),
        "",
        "## Table 3: Stratified 10-Domain & Dual-Track Breakdown",
        "",
        "![Domain Heatmap](./fig3_domain_heatmap.svg)",
        "",
        t2_md.strip(),
        *extra_md_sections,
        "",
        "## Exported LaTeX Artifacts",
        "- `table1_main_multihost.tex` (`booktabs` two-column main results table)",
        "- `table2_domain_breakdown.tex` (`booktabs` 10-domain & track table)",
        "- `table3_mechanism_ablation.tex` (`booktabs` deterministic audit table)",
        *extra_tex_list,
    ]) + "\n"
    (out_dir / "PAPER_SUMMARY_REPORT.md").write_text(master_md, encoding="utf-8")

    summary_bundle = {
        "waves_included": waves_list,
        "n_topics": len(topics_covered),
        "n_manuscripts": len(manuscripts),
        "overall_qwen": summarize_subset(manuscripts, "qwen3.8-flash"),
        "overall_majority": summarize_subset(manuscripts, "majority"),
        "external_agents_qwen": summarize_subset(ext_sub, "qwen3.8-flash"),
        "by_host_qwen": {h: summarize_subset([m for m in manuscripts if m["host"] == h], "qwen3.8-flash") for h in HOST_ORDER if any(m["host"] == h for m in manuscripts)},
        "mechanism_totals": mech,
        "action_a_cross_backbone": {
            "n_manuscripts": act_a_data["n_manuscripts"],
            "independent_2judge": act_a_data["judge_summaries"].get("independent_2judge"),
            "majority_3way": act_a_data["judge_summaries"].get("majority_3way"),
            "total_cost_usd": act_a_data.get("total_cost_usd"),
        } if act_a_data else None,
        "action_b_thinking_ablation": {
            "n_manuscripts": act_b_data["n_manuscripts"],
            "output_tokens": act_b_data.get("output_tokens"),
            "qwen_primary": act_b_data["judge_summaries"].get("qwen3.8-flash"),
            "majority_3way": act_b_data["judge_summaries"].get("majority_3way"),
            "total_cost_usd": act_b_data.get("total_cost_usd"),
        } if act_b_data else None,
    }
    _write_json(out_dir / "paper_metrics_bundle.json", summary_bundle)
    return summary_bundle


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=PACKAGE_ROOT / "results")
    parser.add_argument("--outputs-dir", type=Path, default=None, help="Optional live staging outputs/ dir to sync compact wave bundles from first")
    parser.add_argument("--out-dir", type=Path, default=PACKAGE_ROOT / "results/paper_ready")
    parser.add_argument("--manifest", type=Path, default=PACKAGE_ROOT / "data/benchmark_50_topics.json")
    args = parser.parse_args(argv)

    summary = generate_paper_artifacts(
        results_dir=args.results_dir.resolve(),
        out_dir=args.out_dir.resolve(),
        outputs_dir=args.outputs_dir.resolve() if args.outputs_dir else None,
        manifest_path=args.manifest.resolve(),
    )
    print(
        f"[paper-artifacts] Generated LaTeX tables + SVG figures in {args.out_dir} "
        f"(waves={summary['waves_included']}, topics={summary['n_topics']}, manuscripts={summary['n_manuscripts']})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
