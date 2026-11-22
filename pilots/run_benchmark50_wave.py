"""Execute a locked Benchmark-50 wave (wave1/wave2/wave3) across Direct, AI-Scientist-v2, and E-DARA.

Reuses frozen A/B/C/D drafts from v4/ext15 where available, generates missing Direct + AI-Scientist-v2
proposals and symmetric B/C/D arms for brand-new Benchmark-50 topics, runs upgraded Arm E (closed-loop
governance + post-patch consolidation + prerequisite isolation + statistical-power calibration), and
evaluates E_vs_D across all 3 calibrated blind judges (Qwen3.8-flash, DeepSeek-v4.1-flash, GLM-5.3-flash).
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

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

_DASHSCOPE_RUNNER_SHIM = r'''"""Run AI-Scientist-v2 ideation routing deepseek-flash calls to DashScope deepseek-v4.1-flash-1017."""
from __future__ import annotations

import json
import os
import re
import runpy
import sys
from types import SimpleNamespace
import openai
import ai_scientist.llm as ai_llm


def _convert_dsml_to_action_blocks(text: str) -> str:
    """Translate DeepSeek-v4 native <|DSML|> tool-call blocks into AI-Scientist ACTION/ARGUMENTS syntax."""
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
            kwargs["model"] = "deepseek-v4.1-flash-1017"
            extra = dict(kwargs.get("extra_body") or {})
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


class _DashScopeDeepSeekClient:
    def __init__(self):
        self._client = openai.OpenAI(
            api_key=os.environ["DASHSCOPE_API_KEY"],
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        self.chat = _ChatProxy(self._client.chat)


def _patched_create_client(model):
    if model == "deepseek-flash":
        print("Using DashScope DeepSeek API (deepseek-v4.1-flash-1017) as deepseek-flash.")
        return _DashScopeDeepSeekClient(), "deepseek-flash"
    return _orig_create_client(model)


_orig_create_client = ai_llm.create_client
ai_llm.create_client = _patched_create_client

target_script = sys.argv[1]
sys.argv = [target_script, *sys.argv[2:]]
runpy.run_path(target_script, run_name="__main__")
'''

_NATIVE_COMMON_SHIM = '''"""Dynamic per-task boundary + key-free OpenAlex shim for AI-Scientist-v2 Benchmark-50 runs."""
from __future__ import annotations

from datetime import date, datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import unicodedata
import urllib.parse
import urllib.request


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    data = bytes(value) if isinstance(value, (bytes, bytearray)) else (value if isinstance(value, str) else canonical(value)).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\\n", encoding="utf-8")
    os.replace(tmp, p)


def append_receipt(path, receipt):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    line = canonical(receipt) + "\\n"
    with p.open("a", encoding="utf-8") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        try:
            stream.write(line)
            stream.flush()
            os.fsync(stream.fileno())
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _candidate_hashes(title: str, identifier: str = "") -> set[str]:
    out: set[str] = set()
    if title:
        bare = re.sub(r"[^a-z0-9]+", " ", title.lower()).strip()
        if bare:
            out.add(hashlib.sha256(bare.encode("utf-8")).hexdigest())
        uni = " ".join(re.findall(r"\\w+", unicodedata.normalize("NFKC", title).casefold()))
        if uni:
            out.add(hashlib.sha256(f"title:{uni}".encode("utf-8")).hexdigest())
    if identifier:
        text = urllib.parse.unquote(identifier).strip().casefold()
        if text:
            out.add(hashlib.sha256(f"identity:{text}".encode("utf-8")).hexdigest())
            doi = re.search(r"10\\.\\d{4,9}/[^\\s?#]+", text)
            if doi:
                out.add(hashlib.sha256(f"doi:{doi.group(0).rstrip('./')}".encode("utf-8")).hexdigest())
            arxiv = re.search(r"(?:arxiv[:/]|abs/|pdf/|^)(\\d{4}\\.\\d{4,5}|[a-z.-]+/\\d{7})(?:v\\d+)?(?:\\.pdf)?(?:$|[?#/])", text)
            if arxiv:
                out.add(hashlib.sha256(f"arxiv:{arxiv.group(1)}".encode("utf-8")).hexdigest())
    return out


def _is_blocked(title: str, pub_date: str, cutoff_date: str, forbidden: set[str], identifiers=()) -> bool:
    if not title or (_candidate_hashes(title) & forbidden):
        return True
    if any(_candidate_hashes("", ident) & forbidden for ident in identifiers if ident):
        return True
    try:
        stamp = f"{pub_date}-12-31" if len(pub_date) == 4 else pub_date
        return date.fromisoformat(stamp) > date.fromisoformat(cutoff_date)
    except ValueError:
        return True


def _reconstruct_openalex_abstract(inv) -> str:
    if not isinstance(inv, dict) or not inv:
        return ""
    words = []
    for w, positions in inv.items():
        if isinstance(positions, list):
            for pos in positions:
                words.append((int(pos), str(w)))
    words.sort()
    return " ".join(w for _, w in words)[:1800]


def literature_search(*, arm, case_id, query, limit, boundary_path, ledger_path, **_kwargs):
    boundary = read_json(boundary_path)
    cutoff_date = str(boundary.get("cutoff_date") or "2024-12-31")
    forbidden = set(boundary.get("forbidden_identity_hashes") or [])
    papers = []
    provider = "openalex"
    retained = []

    q = urllib.parse.quote(query[:220])
    url = (
        f"https://api.openalex.org/works?search={q}"
        f"&filter=to_publication_date:{cutoff_date}&per-page={max(1, min(int(limit), 20))}"
        f"&select=id,ids,doi,title,publication_date,abstract_inverted_index,cited_by_count,authorships"
    )
    req = urllib.request.Request(url, headers={"User-Agent": "EDARA-Benchmark50/1.0 (research benchmark)"})
    try:
        with urllib.request.urlopen(req, timeout=15.0) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        for item in data.get("results") or []:
            title = str(item.get("title") or "")
            pub = str(item.get("publication_date") or "")
            aliases = [item.get("id"), item.get("doi"), *(item.get("ids") or {}).values()]
            if _is_blocked(title, pub, cutoff_date, forbidden, aliases):
                continue
            abstract = _reconstruct_openalex_abstract(item.get("abstract_inverted_index"))
            doi = item.get("doi") or (item.get("ids") or {}).get("doi")
            retained.append({
                "paper_id": str(item.get("id") or ""),
                "title": title,
                "date": pub,
                "doi": doi,
                "year": pub[:4] if pub else None,
                "abstract": abstract,
                "ids": [x for x in aliases if x],
            })
            authors = [
                {"name": (a.get("author") or {}).get("display_name") or "Unknown"}
                for a in (item.get("authorships") or [])[:5]
                if isinstance(a, dict)
            ]
            papers.append({
                "paperId": str(item.get("id") or ""),
                "title": title,
                "authors": authors,
                "venue": "OpenAlex",
                "year": pub[:4] if pub else "Unknown",
                "abstract": abstract,
                "citationCount": item.get("cited_by_count", "N/A"),
            })
    except Exception:
        papers.clear()
        retained.clear()

    append_receipt(ledger_path, {
        "schema_version": "search-receipt/v1",
        "created_at": now(),
        "arm": arm,
        "case_id": case_id,
        "query": query,
        "provider": provider,
        "date_cutoff": cutoff_date,
        "kept_count": len(papers[: int(limit)]),
        "kept_ids": [str(p.get("paperId") or p.get("title") or "")[:80] for p in papers[: int(limit)]],
        "kept_sources": retained[: int(limit)],
    })
    return papers[: int(limit)], provider


def deepseek_cost_usd(input_tokens, output_tokens):
    return int(input_tokens) * 0.30 / 1_000_000 + int(output_tokens) * 1.20 / 1_000_000
'''


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


def _ensure_ai_scientist_shim(out_dir: Path) -> Path:
    shim_dir = out_dir / "_ai_scientist_shim"
    shim_dir.mkdir(parents=True, exist_ok=True)
    (shim_dir / "native_common.py").write_text(_NATIVE_COMMON_SHIM, encoding="utf-8")
    (shim_dir / "run_ideation_dashscope.py").write_text(_DASHSCOPE_RUNNER_SHIM, encoding="utf-8")
    return shim_dir


def _run_ai_scientist_one(task: TaskSpec, out_dir: Path, *, dry_run: bool) -> tuple[dict[str, Any], list[dict[str, Any]]]:
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
        boundary = {
            "protocol": "literature-boundary/v1",
            "case_id": tid,
            "cutoff_date": task.cutoff_date,
            "forbidden_identity_hashes": task.forbidden_title_hashes,
            "seed_document_sha256s": [],
        }
        write_json(boundary_path, boundary)

    if dry_run:
        payload = {
            "Name": f"dry_run_{tid[:20]}",
            "Title": f"AI-Scientist Dry-Run Proposal for {tid}",
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
        shim_dir = _ensure_ai_scientist_shim(out_dir)
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
            env["PYTHONPATH"] = f"{shim_dir}:{native_root / 'common'}:{source_dir}"
            env["PYTHONDONTWRITEBYTECODE"] = "1"
            env["AI_SCIENTIST_CASE_ID"] = tid
            env["AI_SCIENTIST_BOUNDARY_PATH"] = str(boundary_path)
            env["AI_SCIENTIST_SEARCH_LEDGER"] = str(tdir / f"search_{attempt}.jsonl")
            env["AI_SCIENTIST_TOOL_LEDGER"] = str(tdir / f"tool_{attempt}.jsonl")
            env["AI_SCIENTIST_LLM_LEDGER"] = str(tdir / f"llm_{attempt}.jsonl")
            cmd = [
                str(ai_py),
                str(shim_dir / "run_ideation_dashscope.py"),
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
            raise RuntimeError(f"AI-Scientist generation failed for {tid}: {last_err}")

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


def _prepare_new_topic_seed(
    raw_task: TaskSpec,
    out_dir: Path,
    host_ep: ModelEndpoint,
    *,
    dry_run: bool,
) -> dict[str, Any]:
    tid = raw_task.task_id
    seed_cache = out_dir / "host_gen" / tid / "seed_topic.json"
    if seed_cache.is_file():
        return load(seed_cache)

    task = raw_task.model_copy(update={"strict_evidence": True})
    t_gen_dir = out_dir / "host_gen" / tid
    t_gen_dir.mkdir(parents=True, exist_ok=True)

    exp_task_dir = ROOT / "outputs/repair_v3/expansion/tasks" / tid
    if (exp_task_dir / "lookup").is_dir() and not (t_gen_dir / "lookup").exists():
        shutil.copytree(exp_task_dir / "lookup", t_gen_dir / "lookup")

    if (exp_task_dir / "prepared.json").is_file():
        prepared_task = TaskSpec.model_validate(load(exp_task_dir / "prepared.json")["task"])
    else:
        prep_file = t_gen_dir / "prepared.json"
        if prep_file.is_file():
            prepared_task = TaskSpec.model_validate(load(prep_file)["task"])
        else:
            store = EvidenceStore(t_gen_dir, task, offline=dry_run)
            prepared_task = store.prepare()
            write_json(prep_file, {"task": prepared_task.model_dump(), "evidence_events": store.events})

    papers = Evidence(task)
    for p in task.prior_work:
        papers.add(p)
    for p in prepared_task.prior_work:
        papers.add(p)

    # 1. Direct proposal (reuse repair_v3/expansion/tasks/<tid>/direct.json if already produced)
    direct_payload: dict[str, Any]
    if (exp_task_dir / "direct.json").is_file() and load(exp_task_dir / "direct.json").get("status") == "produced":
        direct_payload = load(exp_task_dir / "direct.json")["candidate"]
    elif dry_run:
        direct_payload = {
            "title": f"Direct Dry-Run Proposal for {tid}",
            "literature_gap": "Existing methods underperform under distribution shift.",
            "hypothesis": "Explicit invariant regularization reduces worst-group error by >=5%.",
            "method": "Introduce lightweight feature-space invariance penalty with matched compute.",
            "minimal_test": "Evaluate on benchmark split vs ERM and compute-matched baseline across 5 seeds.",
            "decisive_metric": "Worst-group accuracy and macro-F1 across 5 random seeds.",
            "supporting_result": "Worst-group accuracy improves by >=5 percentage points.",
            "falsifying_result": "Improvement <1.5 percentage points or worse than matched control.",
            "related_work": "Builds on supplied pre-cutoff literature.",
            "limitations": "Evaluated on medium-scale academic benchmarks.",
            "resource_plan": "Single GPU within stated task budget.",
        }
    else:
        d_raw_path = t_gen_dir / "direct_raw.json"
        if d_raw_path.is_file():
            direct_payload = load(d_raw_path)
        else:
            gen_runner = _make_runner(host_ep, t_gen_dir / "direct_gen", f"{tid}/direct", "A", "host_gen", 8, dry_run=dry_run)
            direct_payload, _ = _generate_direct(prepared_task, gen_runner)
            write_json(d_raw_path, direct_payload)

    # 2. AI-Scientist-v2 proposal
    ai_payload, ai_sources = _run_ai_scientist_one(task, out_dir, dry_run=dry_run)

    # 3. Normalize citations and freeze shared symmetric topic evidence snapshot
    ai_draft = project(ai_payload)
    ai_alias = ai_citations(
        ROOT,
        tid,
        ai_payload,
        papers,
        ai_sources,
        task.prior_work,
        extra_md_dirs=(out_dir / "ai_scientist_runs",),
    )
    ai_unres = sorted(set(tok.lower() for tok in _MARKERS.findall(json.dumps(ai_draft))) - ai_alias.keys())
    ai_draft = substitute(ai_draft, _MARKERS, lambda m: ai_alias.get(m[0].lower(), "[unmapped citation]"))

    d_draft = project(direct_payload)
    d_alias = direct_aliases(direct_payload, papers)
    for old_m, new_m in d_alias.items():
        d_draft = {k: v.replace(old_m, new_m) for k, v in d_draft.items()}

    originals = {
        "direct": {
            "status": "ready",
            "stage": "proposal",
            "version": digest(direct_payload),
            "raw_sha256": digest(direct_payload),
            "sha256": digest(d_draft),
            "draft": d_draft,
            "citation_map": d_alias,
            "unmapped_citations": [],
        },
        "ai_scientist": {
            "status": "ready",
            "stage": "proposal",
            "version": digest(ai_payload),
            "raw_sha256": digest(ai_payload),
            "sha256": digest(ai_draft),
            "draft": ai_draft,
            "citation_map": ai_alias,
            "unmapped_citations": ai_unres,
        },
    }
    seed = {
        "task": task.model_dump(),
        "development": False,
        "evidence": papers.snapshot(),
        "originals": originals,
    }
    if not dry_run:
        write_json(seed_cache, seed)
    print(f"[HostGen:new] {tid} -> direct & ai_scientist ready (evidence={len(papers.rows)} papers)", flush=True)
    return seed


def _prepare_missing_ai_scientist_for_cached_topic(
    tid: str,
    base_seed: dict[str, Any],
    out_dir: Path,
    *,
    dry_run: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Generate missing ai_scientist manuscript for a cached topic (e.g. ws_09) without mutating direct's frozen evidence."""
    ai_seed_cache = out_dir / "host_gen" / tid / "seed_ai_scientist_supplement.json"
    if ai_seed_cache.is_file():
        saved = load(ai_seed_cache)
        return saved["record"], saved["seed"]

    task = TaskSpec.model_validate(base_seed["task"])
    ev = Evidence.restore(task, base_seed["evidence"])
    ai_payload, ai_sources = _run_ai_scientist_one(task, out_dir, dry_run=dry_run)
    ai_draft = project(ai_payload)
    ai_alias = ai_citations(
        ROOT,
        tid,
        ai_payload,
        ev,
        ai_sources,
        task.prior_work,
        extra_md_dirs=(out_dir / "ai_scientist_runs",),
    )
    ai_unres = sorted(set(tok.lower() for tok in _MARKERS.findall(json.dumps(ai_draft))) - ai_alias.keys())
    ai_draft = substitute(ai_draft, _MARKERS, lambda m: ai_alias.get(m[0].lower(), "[unmapped citation]"))
    rec = {
        "status": "ready",
        "stage": "proposal",
        "version": digest(ai_payload),
        "raw_sha256": digest(ai_payload),
        "sha256": digest(ai_draft),
        "draft": ai_draft,
        "citation_map": ai_alias,
        "unmapped_citations": ai_unres,
    }
    ai_seed = {
        "task": base_seed["task"],
        "development": base_seed.get("development", False),
        "evidence": ev.snapshot(),
        "originals": {"ai_scientist": rec},
    }
    if not dry_run:
        write_json(ai_seed_cache, {"record": rec, "seed": ai_seed})
    print(f"[HostGen:supplement] {tid}/ai_scientist -> ready (evidence={len(ev.rows)} papers)", flush=True)
    return rec, ai_seed


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

        init_key = digest([a_draft, ev.packet()])
        sa_path = vdir / "structured_audits" / f"{init_key}.json"
        targets: list[str] = []
        if sa_path.is_file():
            bundle = FullAuditBundle.model_validate(load(sa_path)["bundle"])
            diag = TrustworthyNoveltyMiddleware()._build_diag(bundle)
            targets = confound_targets(diag)
        if targets:
            stats["E3_confounds"]["targets_required_n"] += 1
            for arm_k, field_k in (("A", "A_addressed"), ("D", "D_addressed"), ("E", "E_addressed")):
                if confound_addressed(arm_drafts[arm_k], targets, baseline_draft=a_draft if arm_k != "A" else None) or confound_addressed(arm_drafts[arm_k], targets):
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

    host_ep = _configure_endpoint(args.editor_profile, role="host_gen", dry_run=args.dry_run)
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
            "host_generator": host_ep.model,
            "editor": editor_ep.model,
            "audit": audit_ep.model,
            "judges": {k: ep.model for k, (ep, _) in judges.items()},
        },
        "limits": LIMITS,
        "output_tokens": OUTPUT_TOKENS,
    }
    write_json(out / "manifest.json", manifest_meta)

    b50 = load(ROOT / "data/benchmark_50_topics.json")["topics"]
    b50_tasks_map = {t["task_id"]: TaskSpec.model_validate(t) for t in load(ROOT / "data/tasks_benchmark50.json")}
    allowed_waves = {"wave1", "wave2"} if args.wave == "wave1_2" else {args.wave}
    selected_entries = [e for e in b50 if e["wave"] in allowed_waves]

    v4_inputs = load(args.v4_dir / "inputs_core.json") if (args.v4_dir / "inputs_core.json").is_file() else {}
    ext_inputs = load(args.ext15_dir / "inputs_extension.json") if (args.ext15_dir / "inputs_extension.json").is_file() else {}

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

    # Phase 0: Seed cached v4/ext15 manuscripts and generate missing new25 / supplement originals
    new_entries = [e for e in selected_entries if e["task_id"] not in v4_inputs and e["task_id"] not in ext_inputs]
    new_seeds_by_tid: dict[str, dict[str, Any]] = {}
    if new_entries:
        print(f"[{args.wave}] Phase 0: Generating Direct + AI-Scientist-v2 originals for {len(new_entries)} new Benchmark-50 topics...", flush=True)
        with ThreadPoolExecutor(max_workers=min(args.workers, 3)) as pool:
            futs = {
                e["task_id"]: pool.submit(_prepare_new_topic_seed, b50_tasks_map[e["task_id"]], out, host_ep, dry_run=args.dry_run)
                for e in new_entries
            }
            for tid, fut in futs.items():
                new_seeds_by_tid[tid] = fut.result()

    wave_inputs: dict[str, Any] = {}
    jobs: list[tuple[str, str, dict[str, Any], dict[str, Any], str]] = []
    for entry in selected_entries:
        tid = entry["task_id"]
        if tid in v4_inputs:
            src_name, src_dir, seed = "v4", args.v4_dir, v4_inputs[tid]
            wave_inputs[tid] = seed
            for host, rec in seed["originals"].items():
                if rec["status"] == "ready":
                    _seed_cached_manuscript(src_dir, out, tid, host, rec, seed)
                    jobs.append((tid, host, rec, seed, src_name))
                elif host == "ai_scientist":
                    ai_rec, ai_seed = _prepare_missing_ai_scientist_for_cached_topic(tid, seed, out, dry_run=args.dry_run)
                    vdir_ai = out / "versions" / tid / "ai_scientist"
                    vdir_ai.mkdir(parents=True, exist_ok=True)
                    cached(vdir_ai / "A.json", lambda r=ai_rec: r)
                    jobs.append((tid, "ai_scientist", ai_rec, ai_seed, "v4_supplement"))
        elif tid in ext_inputs:
            src_name, src_dir, seed = "ext15", args.ext15_dir, ext_inputs[tid]
            wave_inputs[tid] = seed
            for host, rec in seed["originals"].items():
                if rec["status"] != "ready":
                    continue
                _seed_cached_manuscript(src_dir, out, tid, host, rec, seed)
                jobs.append((tid, host, rec, seed, src_name))
        else:
            seed = new_seeds_by_tid[tid]
            wave_inputs[tid] = seed
            for host, rec in seed["originals"].items():
                if rec["status"] != "ready":
                    continue
                vdir_h = out / "versions" / tid / host
                vdir_h.mkdir(parents=True, exist_ok=True)
                cached(vdir_h / "A.json", lambda r=rec: r)
                jobs.append((tid, host, rec, seed, "new25"))

    write_json(out / f"inputs_{args.wave}.json", wave_inputs)
    print(f"[{args.wave}] Prepared {len(selected_entries)} topics, {len(jobs)} ready manuscripts.", flush=True)

    # Phase 1: Run Arms B, C, D (cached for v4/ext15; executed for new25/supplement) + upgraded Arm E
    def _run_one_arms(item: tuple[str, str, dict[str, Any], dict[str, Any], str]) -> dict[str, Any]:
        tid, host, rec, seed, src_name = item
        job = f"{tid}/{host}"
        task = TaskSpec.model_validate(seed["task"])
        vdir = out / "versions" / tid / host
        audit_cache = vdir / "structured_audits"
        audit_cache.mkdir(parents=True, exist_ok=True)
        lookup_dir = out / "lookup" / tid / host
        e_row: dict[str, Any] = {}
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
                    rec["draft"],
                    task,
                    seed["evidence"],
                    au,
                    ed,
                    lookup_dir,
                    audit_cache=audit_cache if a in ("C", "E") else None,
                    editor_confound=ec,
                ),
            )
            if arm == "E":
                e_row = res
        acts = [h.get("action") for h in (e_row.get("history") or [])]
        print(f"[Arm E] {job} ({src_name}) -> status={e_row['status']} actions={acts}", flush=True)
        return e_row

    print(f"[{args.wave}] Phase 1: Running trial arms (B/C/D where needed + upgraded E) across {len(jobs)} manuscripts (workers={args.workers})...", flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(_run_one_arms, jobs))

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
    ai_sci_gen_cost = _sum_ai_scientist_ledger_cost(out)
    if ai_sci_gen_cost > 0:
        metered["ai_scientist_ideation"] = {"cost_usd": ai_sci_gen_cost}

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

    def _fmt(v: Any) -> str:
        if v is None:
            return "—"
        if isinstance(v, list):
            return f"[{v[0]:+.3f}, {v[1]:+.3f}]"
        return f"{v:+.3f}"

    md = [
        f"# Benchmark-50 `{args.wave}` Multi-Host Evaluation Report (`N={len(per_ms)}` Manuscripts across `{len(selected_entries)}` Domains)",
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
        f"- **Process Hype Defects**: `A={mech_stats['E2_process_hype']['A']['occurrences']} -> D={mech_stats['E2_process_hype']['D']['occurrences']} -> E={mech_stats['E2_process_hype']['E']['occurrences']}`",
        f"- **Unverified External Entities**: `A={mech_stats['E2_unverified_entities']['A']['total']} -> D={mech_stats['E2_unverified_entities']['D']['total']} -> E={mech_stats['E2_unverified_entities']['E']['total']}`",
        f"- **Blocking Confounds Addressed**: `D={mech_stats['E3_confounds']['D_addressed']}/{mech_stats['E3_confounds']['targets_required_n']}` -> `E={mech_stats['E3_confounds']['E_addressed']}/{mech_stats['E3_confounds']['targets_required_n']}`",
        f"- **Trajectory Integrity (`E`)**: `errors={mech_stats['E7_trajectory']['E_error_n']}`, `clean_closed_loop={mech_stats['E7_trajectory']['E_clean_closed_loop_n']}/{len(per_ms)}`, `consolidated={mech_stats['E7_trajectory']['E_consolidated_n']}`, `orig_keys_kept={mech_stats['E1_orig_keys_kept_all']}/{len(per_ms)}`",
        "",
        "## 3. By Host (`Qwen3.8-flash Primary` & `3-Judge Majority`)",
        "",
        "| Host | n | Qwen W/L/T (`Net`, `Δsum`) | 3-Judge Majority W/L/T (`Net`, `Δsum`) |",
        "| --- | :---: | :---: | :---: |",
    ]
    for h_name, h_map in by_host_summaries.items():
        ms_h = [m for m in per_ms if m["host"] == h_name]
        sq = h_map["qwen3.8-flash"]
        cq = sq["counts"]
        sm = h_map["majority_3way"]
        cm = sm["counts"]
        md.append(
            f"| `{h_name}` | {len(ms_h)} | `{cq.get('win',0)}/{cq.get('loss',0)}/{cq.get('tie',0)}` (`{_fmt(sq['topic_net_wins'])}`, `{_fmt(sq['score_sum_delta'])}`) "
            f"| `{cm.get('win',0)}/{cm.get('loss',0)}/{cm.get('tie',0)}` (`{_fmt(sm['topic_net_wins'])}`, `{_fmt(sm['score_sum_delta'])}`) |"
        )

    md += [
        "",
        "## 4. Manuscript-by-Manuscript Breakdown",
        "",
        "| Topic | Host | Cohort | `E` Trajectory | Qwen | DeepSeek | GLM | **Majority (`Δsum`)** |",
        "| --- | :---: | :---: | --- | :---: | :---: | :---: | :---: |",
    ]
    for m in per_ms:
        traj_str = f"`{m['e_status']}` `{m['e_actions']}`"
        q_st = m["judges"]["qwen3.8-flash"]["status"]
        d_st = m["judges"]["deepseek-v4.1-flash"]["status"]
        g_st = m["judges"]["GLM-5.3-flash"]["status"]
        maj = m["judges"]["majority_3way"]
        md.append(
            f"| `{m['task_id'][:34]}` | `{m['host']}` | `{m['source_cohort']}` | {traj_str} | `{q_st}` | `{d_st}` | `{g_st}` | **`{maj['status']}` (`{_fmt(maj['score_sum_delta'])}`)** |"
        )

    e_cost = float(metered.get("E", {}).get("cost_usd", 0.0))
    j_cost = float(metered.get("blind", {}).get("cost_usd", 0.0))
    other_cost = sum(
        float(v.get("cost_usd", 0.0))
        for k, v in metered.items()
        if k not in ("E", "blind") and isinstance(v, dict)
    )
    tot_cost = e_cost + j_cost + other_cost
    md += [
        "",
        f"- **Metered Cost (USD)**: `New A/B/C/D = ${other_cost:.4f}`, `Arm E = ${e_cost:.4f}`, `3-Judge Blind Eval = ${j_cost:.4f}`, `Total = ${tot_cost:.4f}`",
    ]
    (out / f"{args.wave}_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"[{args.wave}] Completed! Summary written to {out / f'{args.wave}_summary.md'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
