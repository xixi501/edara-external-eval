"""Action A: Cross-Backbone Generalization across the Stratified 10-Domain Wave-1 Core (N=30 Manuscripts).

Generates fresh Arm A proposals across all 3 external upstream host paradigms (Direct LLM, AI-Scientist-v2,
and EvoScientist v2.1.2) using a completely disjoint model family (`qwen3.8-flash`), executes symmetric
budget-matched `Arm D` (retrieve-and-revise) and `Arm E` (governed middleware v0.2) loops on `qwen3.8-flash`,
and evaluates `E_vs_D` under a role-inverted 3-judge blind panel where `DeepSeek-v4.1-flash` and `GLM-5.3-flash`
serve as strictly independent primary judges (zero generator/editor family overlap) alongside 3-judge majority vote.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any

import numpy as np

PILOTS_DIR = Path(__file__).resolve().parent
if str(PILOTS_DIR) not in sys.path:
    sys.path.insert(0, str(PILOTS_DIR))

from edara_mvp_eval.artifact_export import substitute
from edara_mvp_eval.cli import _generate_direct
from edara_mvp_eval.epistemic_audit import run_full_audit
from edara_mvp_eval.factual_audit import audit_unverified_entities
from edara_mvp_eval.host_adapter import (
    confound_addressed,
    extract_design_elements,
    preserves_design_elements,
)
from edara_mvp_eval.llm_client import LLMRunner, ModelEndpoint
from edara_mvp_eval.middleware import TrustworthyNoveltyMiddleware
from edara_mvp_eval.paired_inputs import (
    _MARKERS,
    Evidence,
    ai_citations,
    direct_aliases,
    project,
)
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
from edara_mvp_eval.schemas import FullAuditBundle, PriorWorkItem, TaskSpec
from edara_mvp_eval.stage_b_evidence import EvidenceStore, write_json
from edara_mvp_eval.unified_eval import (
    CORE_AIB_DIMENSIONS,
    LIT2TEST_DIMENSIONS,
    eval_lit2test_pair,
    evaluate_unified_pair,
    summarize_unified_rows,
)
from run_benchmark50_wave import _NATIVE_COMMON_SHIM, _count_process_hype
from run_evoscientist_wave import EVO_SKILL_VERSION, generate_evoscientist_proposal

_QWEN_AI_SCIENTIST_SHIM = r'''"""Route AI-Scientist-v2 ideation calls to DashScope target model (e.g. qwen3.8-flash)."""
from __future__ import annotations

import json
import os
import re
import runpy
import sys
from types import SimpleNamespace
import openai
import ai_scientist.llm as ai_llm

TARGET_MODEL = os.environ.get("DASHSCOPE_TARGET_MODEL", "qwen3.8-flash")


def _convert_dsml_to_action_blocks(text: str) -> str:
    if not text or ("DSML" not in text) or ("ACTION:" in text):
        return text
    inv_pat = re.compile(
        r'<[｜|]DSML[｜|]\s*invoke\s+name="([^"]+)"\s*>(.*?)</[｜|]DSML[｜|]\s*invoke\s*>',
        re.DOTALL,
    )
    param_pat = re.compile(
        r'<[｜|]DSML[｜|]\s*parameter\s+name="([^"]+)"(?:\s+string="([^"]+)")?\s*>(.*?)</[｜|]DSML[｜|]\s*parameter\s*>',
        re.DOTALL,
    )
    matches = list(inv_pat.finditer(text))
    if not matches:
        return text
    prefix = re.split(r'<[｜|]DSML[｜|]\s*calls\s*>', text, maxsplit=1)[0].strip()
    rendered_blocks = []
    for m in matches:
        tool_name = m.group(1).strip()
        body = m.group(2)
        params = {}
        for pm in param_pat.finditer(body):
            p_name = pm.group(1).strip()
            is_str = (pm.group(2) or "true").lower() == "true"
            raw_val = pm.group(3).strip()
            if not is_str or raw_val.startswith("{") or raw_val.startswith("["):
                try:
                    params[p_name] = json.loads(raw_val)
                    continue
                except Exception:
                    pass
            params[p_name] = raw_val
        if "arguments" in params and isinstance(params["arguments"], dict) and len(params) == 1:
            args_dict = params["arguments"]
        elif tool_name == "FinalizeIdea" and "idea" not in params:
            args_dict = {"idea": params}
        else:
            args_dict = params
        rendered_blocks.append(f"ACTION:\n{tool_name}\n\nARGUMENTS:\n{json.dumps(args_dict, ensure_ascii=False)}")
    return (prefix + "\n\n" if prefix else "") + "\n\n".join(rendered_blocks)


class _CompletionsProxy:
    def __init__(self, inner):
        self._inner = inner

    def create(self, *args, **kwargs):
        if kwargs.get("model") == "deepseek-flash":
            kwargs["model"] = TARGET_MODEL
            extra = dict(kwargs.get("extra_body") or {})
            if "qwen" in TARGET_MODEL.lower():
                extra["enable_thinking"] = False
            else:
                extra.setdefault("thinking", {"type": "disabled"})
            kwargs["extra_body"] = extra
        resp = self._inner.create(*args, **kwargs)
        try:
            if getattr(resp, "choices", None):
                msg = resp.choices[0].message
                if getattr(msg, "content", None):
                    msg.content = _convert_dsml_to_action_blocks(msg.content)
        except Exception:
            pass
        usage_obj = getattr(resp, "usage", None)
        if usage_obj is not None:
            cd = getattr(usage_obj, "completion_tokens_details", None)
            if cd is not None and getattr(cd, "reasoning_tokens", None) is None:
                cd.reasoning_tokens = 0
            pd = getattr(usage_obj, "prompt_tokens_details", None)
            if pd is None:
                usage_obj.prompt_tokens_details = SimpleNamespace(cached_tokens=0)
            elif getattr(pd, "cached_tokens", None) is None:
                pd.cached_tokens = 0
        return resp


class _ChatProxy:
    def __init__(self, inner):
        self.completions = _CompletionsProxy(inner.completions)


class _DashScopeTargetClient:
    def __init__(self):
        self._client = openai.OpenAI(
            api_key=os.environ["DASHSCOPE_API_KEY"],
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        self.chat = _ChatProxy(self._client.chat)


def _patched_create_client(model):
    if model == "deepseek-flash":
        print(f"Using DashScope API ({TARGET_MODEL}) as AI-Scientist-v2 backbone.")
        return _DashScopeTargetClient(), "deepseek-flash"
    return _orig_create_client(model)


_orig_create_client = ai_llm.create_client
ai_llm.create_client = _patched_create_client

target_script = sys.argv[1]
sys.argv = [target_script, *sys.argv[2:]]
runpy.run_path(target_script, run_name="__main__")
'''


def _configure_endpoint(profile_path: Path, *, role: str, dry_run: bool) -> ModelEndpoint:
    if dry_run:
        return ModelEndpoint(model=f"dry-run-{role}")
    ep = ModelEndpoint.from_env_or_profile(profile_path=profile_path)
    low = ep.model.lower()
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
    dry_run: bool,
):
    if dry_run:
        r = LLMRunner(endpoint, dry_run=True)
        r.index = 0  # type: ignore[attr-defined]
        r.reused = 0  # type: ignore[attr-defined]
        return r
    return TrialRunner(endpoint, log_dir, task_id=job, arm=arm, stage=stage, limit=limit, output_tokens=OUTPUT_TOKENS)


def _run_qwen_ai_scientist_one(
    task: TaskSpec,
    out_dir: Path,
    backbone_model: str,
    *,
    dry_run: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    tid = task.task_id
    runs_root = out_dir / "ai_scientist_runs"
    tdir = runs_root / tid
    tdir.mkdir(parents=True, exist_ok=True)
    md_path = tdir / f"{tid}.md"
    boundary_path = tdir / f"{tid}.boundary.json"
    if not md_path.is_file():
        md_content = (
            f"# Title: {task.title}\n\n"
            f"## Task ID & Source\n- task_id: `{tid}`\n- source_track: `{task.source_track}`\n"
            f"- cutoff_date: `{task.cutoff_date}`\n\n"
            f"## Keywords\n{task.field}\n\n"
            f"## Research Problem & Constraints\n{task.prompt_context}\n"
        )
        md_path.write_text(md_content, encoding="utf-8")
    if not boundary_path.is_file():
        write_json(
            boundary_path,
            {
                "protocol": "literature-boundary/v1",
                "case_id": tid,
                "cutoff_date": task.cutoff_date,
                "forbidden_identity_hashes": task.forbidden_title_hashes,
                "seed_document_sha256s": [],
            },
        )

    if dry_run:
        payload = {
            "Name": f"qwen_dry_{tid[:20]}",
            "Title": f"Qwen AI-Scientist Dry-Run Proposal for {tid}",
            "Short Hypothesis": "Targeted structural regularizer improves OOD performance under fixed budget.",
            "Related Work": "Prior methods leave a gap under controlled shift.",
            "Abstract": "We propose a minimal, falsifiable intervention and evaluate against matched baselines.",
            "Experiments": "Compare proposed method vs baseline + matched-compute control across 5 seeds.",
            "Risk Factors and Limitations": "Potential sensitivity to hyperparameter sweep budget.",
            "task_id": tid,
        }
        return payload, []

    out_json = tdir / f"{tid}.json"
    if not (out_json.is_file() and json.loads(out_json.read_text(encoding="utf-8"))):
        shim_dir = out_dir / "_ai_scientist_shim"
        shim_dir.mkdir(parents=True, exist_ok=True)
        (shim_dir / "native_common.py").write_text(_NATIVE_COMMON_SHIM, encoding="utf-8")
        (shim_dir / "run_ideation_qwen.py").write_text(_QWEN_AI_SCIENTIST_SHIM, encoding="utf-8")

        ai_py = Path(os.environ.get("AI_SCIENTIST_PYTHON") or (WORKSPACE / "runtime/ai-scientist-v2-ideation/bin/python"))
        native_root = Path(
            os.environ.get("AI_SCIENTIST_NATIVE_ROOT")
            or (WORKSPACE / "staging/03_active_benchmarks/external-native-eval-20261109")
        )
        source_dir = Path(os.environ.get("AI_SCIENTIST_SOURCE") or (native_root / "sources/ai-scientist-v2"))
        script_path = source_dir / "ai_scientist/perform_ideation_temp_free.py"
        if not ai_py.is_file() or not script_path.is_file():
            raise FileNotFoundError(f"AI-Scientist runtime/source missing: {ai_py} / {script_path}")

        produced_ok = False
        last_err = ""
        for attempt, reflections in ((1, 3), (2, 4), (3, 5)):
            out_json.unlink(missing_ok=True)
            env = os.environ.copy()
            env["DASHSCOPE_TARGET_MODEL"] = backbone_model
            env["PYTHONPATH"] = f"{shim_dir}:{native_root / 'common'}:{source_dir}"
            env["PYTHONDONTWRITEBYTECODE"] = "1"
            env["AI_SCIENTIST_CASE_ID"] = tid
            env["AI_SCIENTIST_BOUNDARY_PATH"] = str(boundary_path)
            env["AI_SCIENTIST_SEARCH_LEDGER"] = str(tdir / f"search_{attempt}.jsonl")
            env["AI_SCIENTIST_TOOL_LEDGER"] = str(tdir / f"tool_{attempt}.jsonl")
            env["AI_SCIENTIST_LLM_LEDGER"] = str(tdir / f"llm_{attempt}.jsonl")
            cmd = [
                str(ai_py),
                str(shim_dir / "run_ideation_qwen.py"),
                str(script_path),
                "--model", "deepseek-flash",
                "--max-num-generations", "1",
                "--num-reflections", str(reflections),
                "--workshop-file", str(md_path),
            ]
            proc = subprocess.run(cmd, env=env, cwd=str(ROOT.parent), capture_output=True, text=True)
            (tdir / f"run_{attempt}.log").write_text(
                (proc.stdout or "") + "\n--- STDERR ---\n" + (proc.stderr or ""),
                encoding="utf-8",
            )
            if proc.returncode == 0 and out_json.is_file():
                try:
                    ideas = json.loads(out_json.read_text(encoding="utf-8"))
                    if isinstance(ideas, list) and ideas and isinstance(ideas[0], dict) and ideas[0].get("Title"):
                        produced_ok = True
                        break
                except Exception as exc:
                    last_err = str(exc)
            last_err = f"rc={proc.returncode} tail={(proc.stderr or '')[-240:]}"
        if not produced_ok:
            raise RuntimeError(f"Qwen AI-Scientist generation failed for {tid}: {last_err}")

    ideas_list = json.loads(out_json.read_text(encoding="utf-8"))
    payload = dict(ideas_list[0])
    payload["task_id"] = tid

    seen_sources: dict[str, dict[str, Any]] = {}
    for s_log in sorted(tdir.glob("search_*.jsonl")):
        for line in s_log.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            for src in rec.get("kept_sources") or []:
                pid = str(src.get("paper_id") or src.get("title") or "")
                if not pid:
                    continue
                item = seen_sources.setdefault(pid, {
                    "paper_id": pid,
                    "title": src.get("title") or "",
                    "quotes": [],
                    "doi": src.get("doi"),
                    "year": src.get("year") or (str(src.get("date") or "")[:4] or None),
                })
                ab = (src.get("abstract") or "").strip()
                if ab and ab not in item["quotes"]:
                    item["quotes"].append(ab)
    return payload, list(seen_sources.values())


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


def _sum_ai_scientist_ledger_cost(out_dir: Path) -> float:
    total = 0.0
    runs_dir = out_dir / "ai_scientist_runs"
    if not runs_dir.is_dir():
        return 0.0
    for ledger in runs_dir.glob("*/llm_*.jsonl"):
        for line in ledger.read_text(encoding="utf-8").splitlines():
            if line.strip():
                total += float(json.loads(line).get("estimated_cost_usd") or 0.0)
    return round(total, 6)


def _compute_mechanism_stats(jobs: list[tuple[str, str, dict[str, Any], dict[str, Any]]], out_dir: Path) -> dict[str, Any]:
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
    for tid, host, a_rec, seed in jobs:
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--wave1-mh-dir", type=Path, default=ROOT / "outputs/benchmark50_wave1_20261122_r2")
    parser.add_argument("--wave1-evosci-dir", type=Path, default=ROOT / "outputs/benchmark50_evosci_wave1_20261122")
    parser.add_argument("--backbone-profile", type=Path, default=WORKSPACE / "data/dashscope-qwen-profile.json")
    parser.add_argument("--audit-profile", type=Path, default=WORKSPACE / "data/p1b-critic-profile.json")
    parser.add_argument("--deepseek-judge-profile", type=Path, default=WORKSPACE / "data/dashscope-deepseek-profile.json")
    parser.add_argument("--glm-judge-profile", type=Path, default=WORKSPACE / "data/p1b-critic-profile.json")
    parser.add_argument("--qwen-judge-profile", type=Path, default=WORKSPACE / "data/dashscope-qwen-profile.json")
    parser.add_argument("--limit-topics", type=int, default=0, help="Optional topic cap for offline dry-run tests")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)

    host_ep = _configure_endpoint(args.backbone_profile, role="host_gen", dry_run=args.dry_run)
    editor_ep = _configure_endpoint(args.backbone_profile, role="editor", dry_run=args.dry_run)
    audit_ep = _configure_endpoint(args.audit_profile, role="audit", dry_run=args.dry_run)
    judges = {
        "deepseek-v4.1-flash": (_configure_endpoint(args.deepseek_judge_profile, role="judge", dry_run=args.dry_run), min(max(args.workers, 3), 4)),
        "GLM-5.3-flash": (_configure_endpoint(args.glm_judge_profile, role="judge", dry_run=args.dry_run), min(args.workers, 2)),
        "qwen3.8-flash": (_configure_endpoint(args.qwen_judge_profile, role="judge", dry_run=args.dry_run), min(args.workers, 3)),
    }

    b50 = load(ROOT / "data/benchmark_50_topics.json")["topics"]
    b50_tasks_map = {t["task_id"]: TaskSpec.model_validate(t) for t in load(ROOT / "data/tasks_benchmark50.json")}
    selected_entries = [e for e in b50 if e["wave"] == "wave1"]
    if args.limit_topics > 0:
        selected_entries = selected_entries[: args.limit_topics]

    w1_inputs = load(args.wave1_mh_dir / "inputs_wave1.json") if (args.wave1_mh_dir / "inputs_wave1.json").is_file() else {}

    write_json(
        out / "manifest.json",
        {
            "protocol_version": PROTOCOL_VERSION,
            "experiment": "action_a_cross_backbone_qwen",
            "wave": "wave1",
            "dry_run": args.dry_run,
            "output_tokens": OUTPUT_TOKENS,
            "code": {p.name: path_hash(p) for p in sorted((ROOT / "src/edara_mvp_eval").glob("*.py"))},
            "models": {
                "host_generator": host_ep.model,
                "editor": editor_ep.model,
                "audit": audit_ep.model,
                "primary_independent_judges": ["deepseek-v4.1-flash", "GLM-5.3-flash"],
                "judges": {k: ep.model for k, (ep, _) in judges.items()},
            },
        },
    )

    # Seed pre-cutoff lookup caches cleanly from Wave 1 (read-only) so searches hit deterministic pre-cutoff snapshots
    for entry in selected_entries:
        tid = entry["task_id"]
        for host, src_root in (
            ("direct", args.wave1_mh_dir),
            ("ai_scientist", args.wave1_mh_dir),
            ("evoscientist", args.wave1_evosci_dir),
        ):
            src_lk = src_root / "lookup" / tid / host
            dst_lk = out / "lookup" / tid / host
            if src_lk.is_dir() and not dst_lk.exists():
                dst_lk.parent.mkdir(parents=True, exist_ok=True)
                shutil.copytree(src_lk, dst_lk)

    # Phase 0: Fresh Qwen3.8-flash Arm A generation across all 3 hosts (Direct, AI-Scientist-v2, EvoScientist)
    print(f"[ActionA:Qwen] Phase 0: Generating 3-host Arm A proposals across {len(selected_entries)} Wave-1 topics...", flush=True)

    def _prepare_topic_all_hosts(entry: dict[str, Any]) -> list[tuple[str, str, dict[str, Any], dict[str, Any]]]:
        tid = entry["task_id"]
        raw_task = b50_tasks_map[tid].model_copy(update={"strict_evidence": True})
        t_gen_dir = out / "host_gen" / tid
        t_gen_dir.mkdir(parents=True, exist_ok=True)

        # Base pre-cutoff seed evidence from Wave 1 topic freeze (or offline task prior_work in dry-run)
        if tid in w1_inputs and not args.dry_run:
            base_task = TaskSpec.model_validate(w1_inputs[tid]["task"])
            base_ev = Evidence.restore(base_task, w1_inputs[tid]["evidence"])
        else:
            base_task = raw_task
            base_ev = Evidence(base_task)
            for p in base_task.prior_work:
                base_ev.add(p)

        # Build prepared_task prompt context with supplied pre-cutoff seed papers for Direct LLM
        exp_task_dir = ROOT / "outputs/repair_v3/expansion/tasks" / tid
        if (exp_task_dir / "prepared.json").is_file() and not args.dry_run:
            prepared_task = TaskSpec.model_validate(load(exp_task_dir / "prepared.json")["task"])
        else:
            prep_file = t_gen_dir / "prepared.json"
            if prep_file.is_file():
                prepared_task = TaskSpec.model_validate(load(prep_file)["task"])
            else:
                store_prep = EvidenceStore(t_gen_dir, base_task, offline=args.dry_run)
                prepared_task = store_prep.prepare() if not args.dry_run else base_task
                if not args.dry_run:
                    write_json(prep_file, {"task": prepared_task.model_dump()})

        # 1) Direct + 2) AI-Scientist-v2 (sharing symmetric topic evidence snapshot, matching main benchmark)
        mh_seed_file = t_gen_dir / "seed_direct_aisci_qwen.json"
        if mh_seed_file.is_file() and not args.dry_run:
            mh_seed = load(mh_seed_file)
        else:
            papers = Evidence.restore(base_task, base_ev.snapshot())
            for p in prepared_task.prior_work:
                papers.add(p)

            d_raw_path = t_gen_dir / "direct_qwen_raw.json"
            if d_raw_path.is_file() and not args.dry_run:
                direct_payload = load(d_raw_path)
            elif args.dry_run:
                direct_payload, _ = _generate_direct(
                    prepared_task,
                    _make_runner(host_ep, t_gen_dir / "direct_gen", f"{tid}/direct", "A", "host_gen", 8, dry_run=True),
                )
            else:
                d_runner = _make_runner(host_ep, t_gen_dir / "direct_gen", f"{tid}/direct", "A", "host_gen", 8, dry_run=False)
                direct_payload, _ = _generate_direct(prepared_task, d_runner)
                write_json(d_raw_path, direct_payload)

            ai_payload, ai_sources = _run_qwen_ai_scientist_one(base_task, out, host_ep.model, dry_run=args.dry_run)
            ai_draft = project(ai_payload)
            ai_alias = ai_citations(
                ROOT,
                tid,
                ai_payload,
                papers,
                ai_sources,
                base_task.prior_work,
                extra_md_dirs=(out / "ai_scientist_runs",),
            )
            ai_unres = sorted(set(tok.lower() for tok in _MARKERS.findall(json.dumps(ai_draft))) - ai_alias.keys())
            ai_draft = substitute(ai_draft, _MARKERS, lambda m: ai_alias.get(m[0].lower(), "[unmapped citation]"))

            d_draft = project(direct_payload)
            d_alias = direct_aliases(direct_payload, papers)
            for old_m, new_m in d_alias.items():
                d_draft = {k: v.replace(old_m, new_m) for k, v in d_draft.items()}

            mh_seed = {
                "task": base_task.model_dump(),
                "development": False,
                "evidence": papers.snapshot(),
                "originals": {
                    "direct": {
                        "status": "ready",
                        "host": "direct",
                        "stage": "proposal",
                        "sha256": digest(d_draft),
                        "draft": d_draft,
                    },
                    "ai_scientist": {
                        "status": "ready",
                        "host": "ai_scientist",
                        "stage": "proposal",
                        "sha256": digest(ai_draft),
                        "draft": ai_draft,
                        "unmapped_citations": ai_unres,
                    },
                },
            }
            if not args.dry_run:
                write_json(mh_seed_file, mh_seed)

        topic_jobs: list[tuple[str, str, dict[str, Any], dict[str, Any]]] = []
        for h_name in ("direct", "ai_scientist"):
            vdir_h = out / "versions" / tid / h_name
            vdir_h.mkdir(parents=True, exist_ok=True)
            rec_h = mh_seed["originals"][h_name]
            write_json(vdir_h / "A.json", rec_h)
            topic_jobs.append((tid, h_name, rec_h, {"task": mh_seed["task"], "evidence": mh_seed["evidence"]}))

        # 3) EvoScientist v2.1.2 on Qwen3.8-flash
        vdir_evo = out / "versions" / tid / "evoscientist"
        vdir_evo.mkdir(parents=True, exist_ok=True)
        a_evo_file = vdir_evo / "A.json"
        ev_evo_file = vdir_evo / "initial_evidence.json"
        if a_evo_file.is_file() and ev_evo_file.is_file() and not args.dry_run:
            evo_rec = load(a_evo_file)
            evo_ev_snap = load(ev_evo_file)
        else:
            evo_runner = _make_runner(host_ep, vdir_evo / "A_gen", f"{tid}/evoscientist", "A", "host_gen", 24, dry_run=args.dry_run)
            evo_rec, evo_ev_snap = generate_evoscientist_proposal(
                base_task,
                mh_seed["evidence"],
                evo_runner,
                out / "lookup" / tid / "evoscientist",
                vdir_evo / "evoscientist_trace",
                dry_run=args.dry_run,
            )
            write_json(a_evo_file, evo_rec)
            write_json(ev_evo_file, evo_ev_snap)
        topic_jobs.append((tid, "evoscientist", evo_rec, {"task": mh_seed["task"], "evidence": evo_ev_snap}))
        print(f"[ActionA:HostGen] {tid} -> direct, ai_scientist, evoscientist ready on {host_ep.model}", flush=True)
        return topic_jobs

    jobs: list[tuple[str, str, dict[str, Any], dict[str, Any]]] = []
    with ThreadPoolExecutor(max_workers=min(args.workers, 3)) as pool:
        for t_list in pool.map(_prepare_topic_all_hosts, selected_entries):
            jobs.extend(t_list)

    # Phase 1: Symmetric Arm D and Arm E on Qwen3.8-flash
    print(f"[ActionA:Qwen] Phase 1: Running symmetric Arm D & Arm E across {len(jobs)} manuscripts...", flush=True)

    def _run_arms_de(item: tuple[str, str, dict[str, Any], dict[str, Any]]) -> None:
        tid, host, a_rec, seed = item
        job = f"{tid}/{host}"
        task = TaskSpec.model_validate(seed["task"])
        vdir = out / "versions" / tid / host
        audit_cache = vdir / "structured_audits"
        audit_cache.mkdir(parents=True, exist_ok=True)
        lookup_dir = out / "lookup" / tid / host
        lim = LIMITS["full"]
        for arm in ("D", "E"):
            auditor = _make_runner(audit_ep, vdir / arm / "audit", job, arm, "audit", lim["audit"], dry_run=args.dry_run)
            editor = _make_runner(editor_ep, vdir / arm / "editor", job, arm, "editor", lim["editor"], dry_run=args.dry_run)
            editor_conf = (
                _make_runner(editor_ep, vdir / arm / "editor_confound", job, arm, "editor", lim["rounds"], dry_run=args.dry_run)
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
                    output_tokens=OUTPUT_TOKENS,
                ),
            )
            if arm == "E":
                acts = [h.get("action") for h in (res.get("history") or [])]
                print(f"[ActionA:ArmE] {job} -> status={res['status']} actions={acts}", flush=True)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(_run_arms_de, jobs))

    # Phase 2: Role-Inverted Independent Primary Judge (DeepSeek-v4.1-flash: Lit2Test + AlphaIdeaBench + ClaimNov)
    ds_ep, ds_workers = judges["deepseek-v4.1-flash"]
    print(f"[ActionA:Qwen] Phase 2: Running Role-Inverted Independent Primary Judge (deepseek-v4.1-flash)...", flush=True)

    def _eval_deepseek_primary(item: tuple[str, str, dict[str, Any], dict[str, Any]]) -> dict[str, Any]:
        tid, host, a_rec, seed = item
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

        rdir = vdir / "review_deepseek" / "E_vs_D"

        def _compute_ds():
            pair_runners = [
                _make_runner(ds_ep, rdir / "lit2test" / o, job, "blind", o, 12, dry_run=args.dry_run)
                for o in ("forward", "reverse")
            ]
            aib_runner = _make_runner(ds_ep, rdir / "alphaideabench", job, "blind", "alphaideabench", 48, dry_run=args.dry_run)
            cn_sub = "claimnov_attempt2" if (rdir / "claimnov" / "0001.json").is_file() else "claimnov"
            cn_runner = _make_runner(ds_ep, rdir / cn_sub, job, "blind", "claimnov", 256, dry_run=args.dry_run)
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

        unified_e_vs_d = cached(vdir / "review_deepseek_E_vs_D.json", _compute_ds)
        l2t = unified_e_vs_d["lit2test"]
        print(f"[ActionA:Judge:DeepSeek] {job} -> {l2t['status']} (sum_delta={l2t.get('score_sum_delta')})", flush=True)
        return {
            "task_id": tid,
            "host": host,
            "source_sha256": a_rec["sha256"],
            "versions": {arm: {k: versions[arm][k] for k in ("status", "sha256", "edit_volume", "queries")} for arm in ("D", "E")},
            "e_actions": [h.get("action") for h in (versions["E"].get("history") or [])],
            "unified_comparisons": {"E_vs_D": unified_e_vs_d},
        }

    with ThreadPoolExecutor(max_workers=ds_workers) as pool:
        primary_rows = list(pool.map(_eval_deepseek_primary, jobs))

    # Phase 3: Cross-Judges (GLM-5.3-flash [Independent] + Qwen3.8-flash [3rd Judge for Majority])
    extra_judges: dict[str, dict[tuple[str, str], dict[str, Any]]] = {}
    for j_name in ("GLM-5.3-flash", "qwen3.8-flash"):
        j_ep, j_workers = judges[j_name]
        slug = "glm" if "glm" in j_name.lower() else "qwen"
        print(f"[ActionA:Qwen] Phase 3 ({j_name}): Running Lit2Test bidirectional judge...", flush=True)

        def _eval_extra(item: tuple[str, str, dict[str, Any], dict[str, Any]], _ep=j_ep, _slug=slug, _jname=j_name):
            tid, host, _, seed = item
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
            print(f"[ActionA:Judge:{_jname}] {job} -> {res['status']} (sum_delta={res.get('score_sum_delta')})", flush=True)
            return (tid, host), res

        with ThreadPoolExecutor(max_workers=j_workers) as pool:
            extra_judges[j_name] = dict(pool.map(_eval_extra, jobs))

    per_ms: list[dict[str, Any]] = []
    for row_p in primary_rows:
        tid, host = row_p["task_id"], row_p["host"]
        d_l2t = row_p["unified_comparisons"]["E_vs_D"]["lit2test"]
        g_l2t = extra_judges["GLM-5.3-flash"][(tid, host)]
        q_l2t = extra_judges["qwen3.8-flash"][(tid, host)]
        maj_status = _majority_vote_status([d_l2t["status"], g_l2t["status"], q_l2t["status"]])
        maj_sum = round(
            float(np.mean([
                float(d_l2t.get("score_sum_delta") or 0.0),
                float(g_l2t.get("score_sum_delta") or 0.0),
                float(q_l2t.get("score_sum_delta") or 0.0),
            ])),
            3,
        )
        maj_dims = {
            dim: round(
                float(np.mean([
                    float((d_l2t.get("dimension_deltas") or {}).get(dim, 0.0)),
                    float((g_l2t.get("dimension_deltas") or {}).get(dim, 0.0)),
                    float((q_l2t.get("dimension_deltas") or {}).get(dim, 0.0)),
                ])),
                3,
            )
            for dim in LIT2TEST_DIMENSIONS
        }
        # Independent 2-Judge Average (DeepSeek + GLM, 100% disjoint from Qwen generator/editor!)
        indep_sum = round(
            0.5 * (float(d_l2t.get("score_sum_delta") or 0.0) + float(g_l2t.get("score_sum_delta") or 0.0)),
            3,
        )
        indep_st = _majority_vote_status([d_l2t["status"], g_l2t["status"]])
        per_ms.append({
            "task_id": tid,
            "host": host,
            "e_status": row_p["versions"]["E"]["status"],
            "e_actions": row_p["e_actions"],
            "judges": {
                "deepseek-v4.1-flash": d_l2t,
                "GLM-5.3-flash": g_l2t,
                "independent_2judge": {
                    "status": indep_st,
                    "order_sensitive": False,
                    "score_sum_delta": indep_sum,
                    "dimension_deltas": {},
                },
                "qwen3.8-flash": q_l2t,
                "majority_3way": {
                    "status": maj_status,
                    "order_sensitive": False,
                    "score_sum_delta": maj_sum,
                    "dimension_deltas": maj_dims,
                },
            },
        })

    judge_keys = ("deepseek-v4.1-flash", "GLM-5.3-flash", "independent_2judge", "qwen3.8-flash", "majority_3way")
    judge_summaries = {jk: _summarize_judge(per_ms, jk) for jk in judge_keys}
    by_host_summaries = {
        h: {jk: _summarize_judge([m for m in per_ms if m["host"] == h], jk) for jk in judge_keys}
        for h in ("ai_scientist", "evoscientist", "direct")
        if any(m["host"] == h for m in per_ms)
    }
    unified_ds_summary = summarize_unified_rows(primary_rows, "E_vs_D")
    mech_stats = _compute_mechanism_stats(jobs, out)

    metered = {k: dict(v) for k, v in usage(out).items()}
    ai_sci_cost = _sum_ai_scientist_ledger_cost(out)
    if ai_sci_cost > 0:
        metered["ai_scientist_ideation"] = {"cost_usd": ai_sci_cost}
    tot_cost = round(sum(float(v.get("cost_usd", 0.0)) for v in metered.values() if isinstance(v, dict)), 4)

    summary_payload = {
        "experiment": "action_a_cross_backbone_qwen",
        "backbone_model": host_ep.model,
        "n_topics": len(selected_entries),
        "n_manuscripts": len(per_ms),
        "judge_summaries": judge_summaries,
        "by_host_summaries": by_host_summaries,
        "deepseek_unified_summary": unified_ds_summary,
        "mechanism_stats": mech_stats,
        "total_cost_usd": tot_cost,
        "usage": metered,
        "manuscripts": per_ms,
    }
    write_json(out / "action_a_cross_backbone_summary.json", summary_payload)

    def _fmt(v: Any) -> str:
        if v is None:
            return "—"
        if isinstance(v, list):
            return f"[{v[0]:+.3f}, {v[1]:+.3f}]"
        return f"{v:+.3f}"

    md = [
        f"# Action A: Cross-Backbone Generalization (`Qwen3.8-flash` Host + Editor, `N={len(per_ms)}` across `{len(selected_entries)}` Domains)",
        "",
        "## 1. Role-Inverted Blind Evaluation (`Arm E` vs. `Arm D`)",
        "",
        "| Judge (Role) | Win / Loss / Tie | Topic Net Wins (95% CI) | Score Sum Delta (95% CI) | Order Sens. |",
        "| --- | :---: | :---: | :---: | ---: |",
    ]
    for label, jk in (
        ("**DeepSeek-v4.1-flash (Independent Primary)**", "deepseek-v4.1-flash"),
        ("**GLM-5.3-flash (Independent Primary)**", "GLM-5.3-flash"),
        ("**Independent 2-Judge Consensus (Disjoint)**", "independent_2judge"),
        ("Qwen3.8-flash (Same-Family Reference)", "qwen3.8-flash"),
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
        "## 2. By Upstream Host (`DeepSeek Independent`, `GLM Independent`, & `3-Judge Majority`)",
        "",
        "| Host | $N$ | DeepSeek W/L/T (`Net`, `ΔSum`) | GLM W/L/T (`Net`, `ΔSum`) | **3-Judge Majority W/L/T (`Net`, `ΔSum`)** |",
        "| --- | :---: | :---: | :---: | :---: |",
    ]
    for h_name in ("ai_scientist", "evoscientist", "direct"):
        if h_name not in by_host_summaries:
            continue
        hm = by_host_summaries[h_name]
        sd, sg, sm = hm["deepseek-v4.1-flash"], hm["GLM-5.3-flash"], hm["majority_3way"]
        cd, cg, cm = sd["counts"], sg["counts"], sm["counts"]
        n_h = sum(cd.values())
        md.append(
            f"| `{h_name}` | {n_h} | `{cd.get('win',0)}/{cd.get('loss',0)}/{cd.get('tie',0)}` (`{_fmt(sd['topic_net_wins'])}`, `{_fmt(sd['score_sum_delta'])}`) | "
            f"`{cg.get('win',0)}/{cg.get('loss',0)}/{cg.get('tie',0)}` (`{_fmt(sg['topic_net_wins'])}`, `{_fmt(sg['score_sum_delta'])}`) | "
            f"**`{cm.get('win',0)}/{cm.get('loss',0)}/{cm.get('tie',0)}`** (`{_fmt(sm['topic_net_wins'])}`, `{_fmt(sm['score_sum_delta'])}`) |"
        )

    md += [
        "",
        "## 3. Deterministic Epistemic Hygiene on `Qwen3.8-flash` Backbone",
        "",
        f"- **Schema Keys Preserved (`E`)**: `{mech_stats['E1_orig_keys_kept_all']} / {len(per_ms)}`",
        f"- **Process Hype (`A -> D -> E`)**: `{mech_stats['E2_process_hype']['A']['occurrences']} -> {mech_stats['E2_process_hype']['D']['occurrences']} -> {mech_stats['E2_process_hype']['E']['occurrences']}`",
        f"- **Unverified External Entities (`A -> D -> E`)**: `{mech_stats['E2_unverified_entities']['A']['total']} -> {mech_stats['E2_unverified_entities']['D']['total']} -> {mech_stats['E2_unverified_entities']['E']['total']}`",
        f"- **Blocking Confounds Addressed (`A -> D -> E`)**: `{mech_stats['E3_confounds']['A_addressed']}/{mech_stats['E3_confounds']['targets_required_n']} -> {mech_stats['E3_confounds']['D_addressed']}/{mech_stats['E3_confounds']['targets_required_n']} -> {mech_stats['E3_confounds']['E_addressed']}/{mech_stats['E3_confounds']['targets_required_n']}`",
        f"- **Trajectory Integrity (`E`)**: `errors={mech_stats['E7_trajectory']['E_error_n']}`, `clean_closed_loop={mech_stats['E7_trajectory']['E_clean_closed_loop_n']}/{len(per_ms)}`",
        f"- **Total Metered Cost (USD)**: `${tot_cost:.4f}`",
    ]
    (out / "action_a_cross_backbone_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"[ActionA:Qwen] Completed! Cost=${tot_cost:.4f} -> {out / 'action_a_cross_backbone_summary.md'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
