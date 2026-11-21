"""CLI entry point for the 15-task MVP & 3-topic Pilot evaluation harness and middleware comparison."""
from __future__ import annotations

import hashlib
import json
import re
import statistics
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import combinations
from pathlib import Path
from typing import Any

import typer
from rich.console import Console

from .baselines import run_retrieve_and_revise, run_self_refine
from .epistemic_audit import run_full_audit
from .llm_client import LLMRunner, ModelEndpoint
from .middleware import TrustworthyNoveltyMiddleware, build_pre_gen_brief
from .pairwise import compare_pair_bidirectional
from .schemas import FullAuditBundle, PairwiseComparison, TaskSpec
from .tasks_loader import (
    build_mvp15_tasks,
    build_pilot3_tasks,
    export_pilot3_external_inputs,
)

app = typer.Typer(add_completion=False, no_args_is_help=True)
console = Console()

DEFAULT_OUT_ROOT = Path(__file__).resolve().parents[2] / "outputs"

_GENERATE_SYSTEM = """You are generating a concrete, budget-feasible, falsifiable research proposal prior to code execution.
Respect the supplied Cutoff Date and Budget Envelope strictly. Never invent fake prior citations.
Return valid JSON with keys:
title, literature_gap, hypothesis, method, minimal_test, decisive_metric, supporting_result, falsifying_result, related_work, limitations, resource_plan."""


def _generate_direct(task: TaskSpec, runner: LLMRunner) -> tuple[dict[str, Any], float]:
    before = runner.usage.cost_usd
    user = f"Task ID: {task.task_id}\n\n{task.prompt_context}"
    draft = runner.chat_json(_GENERATE_SYSTEM, user, temperature=0.3, max_tokens=4096, tag="generate_direct")
    return draft, round(runner.usage.cost_usd - before, 6)


def _load_imported_arm(path: Path) -> dict[str, Any]:
    raw = path.read_text(encoding="utf-8")
    if path.suffix == ".jsonl":
        out: dict[str, Any] = {}
        for line in raw.splitlines():
            if line.strip():
                row = json.loads(line)
                tid = str(row.get("task_id") or row.get("context_id") or row.get("case_id") or "")
                if tid:
                    out[tid] = row
        return out
    data = json.loads(raw)
    if isinstance(data, dict):
        return data
    if isinstance(data, list):
        return {str(r.get("task_id") or r.get("case_id")): r for r in data if isinstance(r, dict)}
    return {}


def _summarize_arm(arm: str, bundles: list[FullAuditBundle], costs: list[float]) -> dict[str, Any]:
    n = max(1, len(bundles))
    ret_items = [b.retention for b in bundles if b.retention is not None]
    method_ret = (
        round(statistics.mean(r.method_token_retention for r in ret_items), 3)
        if ret_items
        else 1.0
    )
    return {
        "arm": arm,
        "n_tasks": len(bundles),
        "gate_pass_rate": round(sum(1 for b in bundles if b.gate.gate_pass) / n, 3),
        "strict_gate_pass_rate": round(sum(1 for b in bundles if b.gate.strict_gate_pass) / n, 3),
        "epistemic_overclaim_rate": round(statistics.mean(b.tri_state.epistemic_overclaim_rate for b in bundles), 3),
        "gap_survival_rate": round(statistics.mean(b.gap_audit.survival_rate for b in bundles), 3),
        "gap_unresolved_count": sum(b.gap_audit.unresolved_count for b in bundles),
        "spec_completeness_mean": round(statistics.mean(b.spec_audit.completeness_score for b in bundles), 3),
        "method_retention_mean": method_ret,
        "claimnov_pass_rate": round(sum(1 for b in bundles if not b.claimnov.veto) / n, 3),
        "trusted_novelty_mean": round(statistics.mean(b.gate.trusted_novelty for b in bundles), 3),
        "rigor_mean": round(statistics.mean(b.gate.rigor_score for b in bundles), 3),
        "actionability_mean": round(statistics.mean(b.gate.actionability_score for b in bundles), 3),
        "mean_gen_cost_usd": round(statistics.mean(costs), 6) if costs else 0.0,
    }


def _render_markdown_summary(
    arm_summaries: list[dict[str, Any]],
    pairwise: list[PairwiseComparison],
    dry_run: bool,
    n_tasks: int,
) -> str:
    lines = [
        f"# E-DARA {n_tasks}-Task Evaluation Summary (`dry_run={dry_run}`)",
        "",
        "| Arm | Tasks | Gate Pass | Strict Gate | Overclaim (↓) | Gap Surv | Gap Unresolved | Spec Comp (↑) | Method Ret (↑) | ClaimNov Pass | Trusted Nov | Rigor | Actionability | Mean Cost ($) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for s in arm_summaries:
        lines.append(
            f"| `{s['arm']}` | {s['n_tasks']} | {s['gate_pass_rate']:.1%} | {s['strict_gate_pass_rate']:.1%} | "
            f"{s['epistemic_overclaim_rate']:.3f} | {s['gap_survival_rate']:.1%} | "
            f"{s.get('gap_unresolved_count', 0)} | "
            f"{s['spec_completeness_mean']:.2f} | {s['method_retention_mean']:.2f} | "
            f"{s['claimnov_pass_rate']:.1%} | {s['trusted_novelty_mean']:.3f} | "
            f"{s['rigor_mean']:.2f} | {s['actionability_mean']:.2f} | {s['mean_gen_cost_usd']:.5f} |"
        )
    if pairwise:
        grouped: dict[tuple[str, str], list[PairwiseComparison]] = defaultdict(list)
        for p in pairwise:
            grouped[(p.arm_x, p.arm_y)].append(p)
        lines.extend(["", "## Order-Swapped Blind Pairwise Comparisons", ""])
        for (ax, ay), items in grouped.items():
            x_wins = sum(1 for p in items if p.folded_winner == "X")
            y_wins = sum(1 for p in items if p.folded_winner == "Y")
            ties = sum(1 for p in items if p.folded_winner == "tie")
            order_sens = sum(1 for p in items if p.order_sensitive)
            lines.append(
                f"- `{ax}` vs `{ay}` ({len(items)} tasks): "
                f"**{ax} wins={x_wins}**, **{ay} wins={y_wins}**, **tie={ties}** "
                f"(order-sensitive -> tie: {order_sens}/{len(items)})"
            )
    return "\n".join(lines) + "\n"


@app.command("prepare-tasks")
def prepare_tasks(
    out_root: Path = typer.Option(DEFAULT_OUT_ROOT, "--out-root"),
) -> None:
    """Build tasks_mvp15.json, curated tasks_pilot3.json, and external runner inputs under pilot3_inputs/."""
    out_root.mkdir(parents=True, exist_ok=True)
    mvp15_path = out_root / "tasks_mvp15.json"
    pilot3_path = out_root / "tasks_pilot3.json"
    for p in (mvp15_path, pilot3_path):
        if p.is_file():
            p.unlink()

    tasks15 = build_mvp15_tasks(cache_path=mvp15_path)
    pilot3 = build_pilot3_tasks(cache_path=pilot3_path, mvp15_cache_path=mvp15_path)
    inputs_dir = export_pilot3_external_inputs(pilot3, out_root / "pilot3_inputs")

    console.print(f"Prepared {len(tasks15)} MVP tasks -> {mvp15_path}")
    console.print(f"Prepared {len(pilot3)} Pilot-3 tasks -> {pilot3_path}")
    console.print(f"Exported AI-Scientist-v2 & E-DARA inputs -> {inputs_dir}")
    for t in pilot3:
        console.print(f"  - [pilot3/{t.source_track}] {t.task_id} | cutoff={t.cutoff_date} | prior={len(t.prior_work)}")


def _select_tasks(tasks_cache: Path, task_ids: str | None, limit: int) -> list[TaskSpec]:
    tasks = build_mvp15_tasks(cache_path=tasks_cache)
    if task_ids:
        wanted = [x.strip() for x in task_ids.split(",") if x.strip()]
        by_id = {t.task_id: t for t in tasks}
        missing = [w for w in wanted if w not in by_id]
        if missing:
            raise typer.BadParameter(f"Unknown task_ids: {missing}")
        return [by_id[w] for w in wanted]
    return tasks[: max(1, limit)]


@app.command("run")
def run_eval(
    tasks_cache: Path = typer.Option(DEFAULT_OUT_ROOT / "tasks_pilot3.json", "--tasks-cache"),
    out_dir: Path = typer.Option(DEFAULT_OUT_ROOT / "latest", "--out"),
    limit: int = typer.Option(15, "--limit", help="Max number of tasks to evaluate from tasks_cache."),
    task_ids: str | None = typer.Option(None, "--task-ids", help="Comma-separated task IDs filter."),
    workers: int = typer.Option(3, "--workers", help="Concurrent tasks to execute in parallel."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Run full pipeline offline with mock fixtures."),
    live_search: bool = typer.Option(False, "--live-search", help="Enable live OpenAlex/S2 pre-cutoff queries."),
    enable_embeddings: bool = typer.Option(False, "--enable-embeddings", help="Try OpenAI embeddings before BM25 fallback."),
    skip_direct: bool = typer.Option(False, "--skip-direct", help="Skip generating Direct / Direct+MW arms."),
    direct_inputs: Path | None = typer.Option(None, "--direct-inputs", help="Optional pre-generated Direct JSONL to reuse identical initial drafts across ablations."),
    with_self_refine: bool = typer.Option(False, "--with-self-refine", help="Also run prompt-only Self-Refine baseline on Direct drafts."),
    with_retrieve_revise: bool = typer.Option(False, "--with-retrieve-revise", help="Also run unconstrained Retrieve-and-Revise baseline on Direct drafts."),
    with_pre_brief: bool = typer.Option(False, "--with-pre-brief", help="Optional BVSR+Lit2Test Pre-Gen Hook ablation arm (direct_pre_mw)."),
    mw_on_imported: bool = typer.Option(False, "--mw-on-imported", help="Also run TrustworthyNoveltyMiddleware over imported arms."),
    skip_pairwise: bool = typer.Option(False, "--skip-pairwise", help="Skip bidirectional pairwise judge."),
    resume: bool = typer.Option(True, "--resume/--no-resume", help="Save per-task checkpoints under <out>/checkpoints and resume completed tasks."),
    gen_profile: Path | None = typer.Option(None, "--gen-profile", help="Optional E-DARA ModelProfile JSON for generation."),
    judge_profile: Path | None = typer.Option(None, "--judge-profile", help="Optional E-DARA ModelProfile JSON for auditing/judging."),
    import_arm: list[str] = typer.Option([], "--import-arm", help="Import external arm outputs as name=/path/to/file.json[l]"),
) -> None:
    """Evaluate Direct, Direct+Middleware, ablation baselines (Self-Refine / Retrieve-and-Revise), and external arms."""
    tasks = _select_tasks(tasks_cache, task_ids, limit)
    gen_ep = ModelEndpoint.from_env_or_profile(profile_path=gen_profile, prefix="EDARA_GEN")
    judge_ep = ModelEndpoint.from_env_or_profile(profile_path=judge_profile, prefix="EDARA_JUDGE")
    gen_runner = LLMRunner(gen_ep, dry_run=dry_run)
    judge_runner = LLMRunner(judge_ep, dry_run=dry_run)
    mw = TrustworthyNoveltyMiddleware(live_search=live_search, enable_embeddings=enable_embeddings)
    direct_map = _load_imported_arm(direct_inputs) if direct_inputs else {}

    imported_maps: dict[str, dict[str, Any]] = {}
    for spec in import_arm:
        if "=" not in spec:
            raise typer.BadParameter(f"Expected arm=path, got {spec!r}")
        name, pstr = spec.split("=", 1)
        imported_maps[name.strip()] = _load_imported_arm(Path(pstr.strip()))

    run_sig = hashlib.sha1(
        json.dumps(
            {
                "dry_run": dry_run,
                "audit_contract": "evidence-bound-governance/independent-spec-v4",
                "live_search": live_search,
                "enable_embeddings": enable_embeddings,
                "skip_direct": skip_direct,
                "has_direct_inputs": bool(direct_inputs),
                "with_self_refine": with_self_refine,
                "with_retrieve_revise": with_retrieve_revise,
                "with_pre_brief": with_pre_brief,
                "mw_on_imported": mw_on_imported,
                "skip_pairwise": skip_pairwise,
                "imported_arms": sorted(imported_maps.keys()),
                "gen_model": gen_ep.model,
                "judge_model": judge_ep.model,
            },
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()[:16]
    ckpt_dir = out_dir / "checkpoints"
    if resume:
        ckpt_dir.mkdir(parents=True, exist_ok=True)

    def _ckpt_path(task_id: str) -> Path:
        slug = re.sub(r"[^A-Za-z0-9._-]+", "_", task_id)[:80]
        digest = hashlib.sha1(task_id.encode("utf-8")).hexdigest()[:8]
        return ckpt_dir / f"{slug}__{digest}.json"

    def _load_ckpt(task_id: str) -> tuple[dict[str, tuple[FullAuditBundle, float]], list[PairwiseComparison], bool] | None:
        if not resume:
            return None
        p = _ckpt_path(task_id)
        if not p.is_file():
            return None
        try:
            raw = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if raw.get("signature") != run_sig:
            return None
        per_arm: dict[str, tuple[FullAuditBundle, float]] = {
            arm_name: (FullAuditBundle.model_validate(entry["bundle"]), float(entry.get("cost", 0.0)))
            for arm_name, entry in (raw.get("per_arm") or {}).items()
        }
        pw_list = [PairwiseComparison.model_validate(item) for item in (raw.get("pairwise") or [])]
        complete = bool(raw.get("complete", True))
        return per_arm, pw_list, complete

    def _save_ckpt(
        task_id: str,
        per_arm: dict[str, tuple[FullAuditBundle, float]],
        pw_list: list[PairwiseComparison],
        *,
        complete: bool,
    ) -> None:
        if not resume:
            return
        p = _ckpt_path(task_id)
        tmp = p.with_suffix(f".{threading.get_ident()}.tmp")
        payload = {
            "task_id": task_id,
            "signature": run_sig,
            "complete": complete,
            "per_arm": {
                arm_name: {"bundle": bundle.model_dump(), "cost": cost}
                for arm_name, (bundle, cost) in per_arm.items()
            },
            "pairwise": [pw.model_dump() for pw in pw_list],
        }
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(p)

    def _eval_one_task(task: TaskSpec) -> tuple[dict[str, tuple[FullAuditBundle, float]], list[PairwiseComparison]]:
        partial = _load_ckpt(task.task_id)
        per_arm: dict[str, tuple[FullAuditBundle, float]] = dict(partial[0]) if partial else {}
        if partial and per_arm:
            print(f"[eval] {task.task_id}: resuming partial checkpoint ({len(per_arm)} arms cached)", flush=True)

        if not skip_direct:
            if "direct" in per_arm and "direct_mw" in per_arm:
                before_bundle, direct_cost = per_arm["direct"]
                direct_draft = direct_map.get(task.task_id) or before_bundle.proposal.model_dump()
            else:
                print(f"[eval] {task.task_id}: start arm=direct (mw=True)", flush=True)
                if task.task_id in direct_map:
                    direct_draft = direct_map[task.task_id]
                    direct_cost = float(direct_draft.get("gen_cost_usd") or 0.0)
                else:
                    direct_draft, direct_cost = _generate_direct(task, gen_runner)
                _, before_bundle, after_bundle, mw_cost = mw.apply(
                    task,
                    direct_draft,
                    repair_runner=gen_runner,
                    audit_runner=judge_runner,
                    target_arm_name="direct_mw",
                )
                per_arm["direct"] = (before_bundle.model_copy(update={"arm": "direct"}), direct_cost)
                per_arm["direct_mw"] = (after_bundle, round(direct_cost + mw_cost, 6))
                _save_ckpt(task.task_id, per_arm, [], complete=False)
                print(f"[eval] {task.task_id}: done arm=direct_mw", flush=True)

            if with_self_refine and "direct_self_refine" not in per_arm:
                print(f"[eval] {task.task_id}: start arm=direct_self_refine", flush=True)
                _, sr_bundle, sr_cost = run_self_refine(
                    task,
                    direct_draft,
                    before_bundle=before_bundle,
                    repair_runner=gen_runner,
                    audit_runner=judge_runner,
                    target_arm_name="direct_self_refine",
                    live_search=live_search,
                    enable_embeddings=enable_embeddings,
                )
                per_arm["direct_self_refine"] = (sr_bundle, round(direct_cost + sr_cost, 6))
                _save_ckpt(task.task_id, per_arm, [], complete=False)
                print(f"[eval] {task.task_id}: done arm=direct_self_refine", flush=True)

            if with_retrieve_revise and "direct_retrieve_revise" not in per_arm:
                print(f"[eval] {task.task_id}: start arm=direct_retrieve_revise", flush=True)
                _, rr_bundle, rr_cost = run_retrieve_and_revise(
                    task,
                    before_bundle=before_bundle,
                    repair_runner=gen_runner,
                    audit_runner=judge_runner,
                    target_arm_name="direct_retrieve_revise",
                    live_search=live_search,
                    enable_embeddings=enable_embeddings,
                )
                per_arm["direct_retrieve_revise"] = (rr_bundle, round(direct_cost + rr_cost, 6))
                _save_ckpt(task.task_id, per_arm, [], complete=False)
                print(f"[eval] {task.task_id}: done arm=direct_retrieve_revise", flush=True)

            if with_pre_brief and "direct_pre_mw" not in per_arm:
                print(f"[eval] {task.task_id}: start arm=direct_pre_mw", flush=True)
                brief = build_pre_gen_brief(task, live_search=live_search)
                pre_task = task.model_copy(update={"prompt_context": f"{brief}\n\n{task.prompt_context}"})
                pre_draft, pre_cost = _generate_direct(pre_task, gen_runner)
                _, _, pre_after, pre_mw_cost = mw.apply(
                    task,
                    pre_draft,
                    repair_runner=gen_runner,
                    audit_runner=judge_runner,
                    target_arm_name="direct_pre_mw",
                )
                per_arm["direct_pre_mw"] = (pre_after, round(pre_cost + pre_mw_cost, 6))
                _save_ckpt(task.task_id, per_arm, [], complete=False)
                print(f"[eval] {task.task_id}: done arm=direct_pre_mw", flush=True)

        for name, amap in imported_maps.items():
            if task.task_id not in amap:
                continue
            already_done = (name in per_arm) and ((not mw_on_imported) or (f"{name}_mw" in per_arm))
            if already_done:
                continue
            print(f"[eval] {task.task_id}: start arm={name} (mw={mw_on_imported})", flush=True)
            if mw_on_imported:
                _, pre_b, post_b, imp_mw_cost = mw.apply(
                    task,
                    amap[task.task_id],
                    repair_runner=gen_runner,
                    audit_runner=judge_runner,
                    target_arm_name=f"{name}_mw",
                )
                per_arm[name] = (pre_b.model_copy(update={"arm": name}), 0.0)
                per_arm[f"{name}_mw"] = (post_b, imp_mw_cost)
            else:
                b = run_full_audit(
                    task,
                    name,
                    amap[task.task_id],
                    judge_runner,
                    live_search=live_search,
                    enable_embeddings=enable_embeddings,
                )
                per_arm[name] = (b, 0.0)
            _save_ckpt(task.task_id, per_arm, [], complete=False)
            print(f"[eval] {task.task_id}: done arm={name}", flush=True)

        pw_list: list[PairwiseComparison] = []
        if not skip_pairwise and len(per_arm) >= 2:
            arm_names = list(per_arm.keys())
            print(f"[eval] {task.task_id}: start pairwise ({len(arm_names)} arms)", flush=True)
            for ax, ay in combinations(arm_names, 2):
                pw_list.append(compare_pair_bidirectional(task, per_arm[ax][0], per_arm[ay][0], judge_runner))
            print(f"[eval] {task.task_id}: done pairwise ({len(pw_list)} pairs)", flush=True)
        _save_ckpt(task.task_id, per_arm, pw_list, complete=True)
        return per_arm, pw_list

    arm_bundles: dict[str, list[FullAuditBundle]] = defaultdict(list)
    arm_costs: dict[str, list[float]] = defaultdict(list)
    pairwise_results: list[PairwiseComparison] = []

    results_by_tid: dict[str, tuple[dict[str, tuple[FullAuditBundle, float]], list[PairwiseComparison]]] = {}
    pending_tasks: list[TaskSpec] = []
    for t in tasks:
        loaded = _load_ckpt(t.task_id)
        if loaded is not None and loaded[2]:
            print(f"[eval] {t.task_id}: loaded from checkpoint ({len(loaded[0])} arms, {len(loaded[1])} pairs)", flush=True)
            results_by_tid[t.task_id] = (loaded[0], loaded[1])
        else:
            pending_tasks.append(t)

    max_workers = max(1, min(workers, max(1, len(pending_tasks))))
    if max_workers == 1:
        for t in pending_tasks:
            results_by_tid[t.task_id] = _eval_one_task(t)
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futs = {pool.submit(_eval_one_task, t): t for t in pending_tasks}
            for fut in as_completed(futs):
                t = futs[fut]
                results_by_tid[t.task_id] = fut.result()

    task_outputs = [results_by_tid[t.task_id] for t in tasks]

    for per_arm, pw_list in task_outputs:
        for arm_name, (bundle, cost) in per_arm.items():
            arm_bundles[arm_name].append(bundle)
            arm_costs[arm_name].append(cost)
        pairwise_results.extend(pw_list)

    arm_summaries = [_summarize_arm(arm, bs, arm_costs[arm]) for arm, bs in arm_bundles.items() if bs]
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "dry_run": dry_run,
        "n_tasks": len(tasks),
        "task_ids": [t.task_id for t in tasks],
        "arm_summaries": arm_summaries,
        "pairwise": [p.model_dump() for p in pairwise_results],
        "bundles": {arm: [b.model_dump() for b in bs] for arm, bs in arm_bundles.items()},
        "usage": {
            "gen_cost_usd": round(gen_runner.usage.cost_usd, 6),
            "judge_cost_usd": round(judge_runner.usage.cost_usd, 6),
        },
    }
    (out_dir / "results.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    md = _render_markdown_summary(arm_summaries, pairwise_results, dry_run, len(tasks))
    (out_dir / "summary.md").write_text(md, encoding="utf-8")
    console.print(md)


@app.command("generate-direct")
def generate_direct_only(
    tasks_cache: Path = typer.Option(DEFAULT_OUT_ROOT / "tasks_pilot3.json", "--tasks-cache"),
    out_path: Path = typer.Option(DEFAULT_OUT_ROOT / "pilot3_direct" / "direct_outputs.jsonl", "--out"),
    limit: int = typer.Option(15, "--limit", help="Max number of tasks to generate from tasks_cache."),
    task_ids: str | None = typer.Option(None, "--task-ids", help="Comma-separated task IDs filter."),
    workers: int = typer.Option(3, "--workers", help="Concurrent tasks to generate in parallel."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Run offline with mock fixtures."),
    gen_profile: Path | None = typer.Option(None, "--gen-profile", help="Optional E-DARA ModelProfile JSON for generation."),
) -> None:
    """Generate Direct baseline drafts ONLY (no judge/audit yet) and save JSONL for later evaluation."""
    tasks = _select_tasks(tasks_cache, task_ids, limit)
    gen_ep = ModelEndpoint.from_env_or_profile(profile_path=gen_profile, prefix="EDARA_GEN")
    gen_runner = LLMRunner(gen_ep, dry_run=dry_run)

    def _one(task: TaskSpec) -> dict[str, Any]:
        draft, cost = _generate_direct(task, gen_runner)
        return {"task_id": task.task_id, "arm": "direct", "model": gen_ep.model, "gen_cost_usd": cost, **draft}

    max_workers = max(1, min(workers, len(tasks)))
    if max_workers == 1:
        rows = [_one(t) for t in tasks]
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            rows = list(pool.map(_one, tasks))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    console.print(
        f"Generated {len(rows)} Direct proposals (model={gen_ep.model}, dry_run={dry_run}, "
        f"cost=${gen_runner.usage.cost_usd:.6f}) -> {out_path}"
    )


@app.command("govern")
def govern_single_draft(
    draft_path: Path = typer.Option(..., "--draft", help="Path to raw proposal / experiment plan JSON."),
    out_path: Path = typer.Option(..., "--out", help="Output path for governed JSON."),
    refs_path: Path | None = typer.Option(None, "--refs", help="Optional JSON/JSONL of prior-work papers."),
    title: str = typer.Option("Research Proposal", "--title", help="Task title."),
    context: str = typer.Option("", "--context", help="Task prompt / background context."),
    cutoff_date: str = typer.Option("2025-12-31", "--cutoff-date", help="Pre-cutoff date YYYY-MM-DD."),
    editor_profile: Path | None = typer.Option(None, "--editor-profile", help="Editor model profile JSON."),
    audit_profile: Path | None = typer.Option(None, "--audit-profile", help="Auditor model profile JSON."),
    mode: str = typer.Option("full", "--mode", help="Governance mode: 'full' (multi-round) or 'local' (single-round)."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Run offline with deterministic mock responses."),
) -> None:
    """Plug-and-play CLI: govern one research proposal JSON and write back host-compatible JSON."""
    from .plugin import govern_proposal

    raw_draft = json.loads(draft_path.read_text(encoding="utf-8"))
    refs: list[dict[str, Any]] = []
    if refs_path and refs_path.is_file():
        txt = refs_path.read_text(encoding="utf-8").strip()
        if refs_path.suffix == ".jsonl":
            refs = [json.loads(line) for line in txt.splitlines() if line.strip()]
        else:
            parsed = json.loads(txt)
            refs = parsed if isinstance(parsed, list) else list(parsed.get("papers") or parsed.get("references") or [])
    outcome = govern_proposal(
        raw_draft,
        task_title=title,
        task_context=context,
        prior_work=refs,
        cutoff_date=cutoff_date,
        editor_profile=editor_profile,
        audit_profile=audit_profile,
        mode=mode,
        dry_run=dry_run,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(outcome.draft, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    console.print(
        f"[govern] status={outcome.status} changed={outcome.changed} rounds={outcome.rounds} "
        f"retrieved={len(outcome.retrieved_papers)} cost=${outcome.cost_usd:.4f} -> {out_path}"
    )
