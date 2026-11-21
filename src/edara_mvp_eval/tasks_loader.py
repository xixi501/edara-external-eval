"""Build and load the deterministic 15-task Tri-Track MVP benchmark."""
from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

import yaml
from pypdf import PdfReader

from .schemas import PriorWorkItem, TaskSpec

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
REFS_ROOT = Path(os.environ.get("EDARA_REFS_ROOT") or PACKAGE_ROOT.parent)
WORKSPACE_ROOT = Path(
    os.environ.get("EDARA_WORKSPACE")
    or (PACKAGE_ROOT.parents[2] if len(PACKAGE_ROOT.parents) >= 3 else PACKAGE_ROOT.parent)
)
DATA_ROOT = Path(os.environ.get("EDARA_DATA_ROOT") or (WORKSPACE_ROOT / "data"))
AIIB_ROOT = Path(os.environ.get("AIIB_ROOT") or (DATA_ROOT / "aiib-pilot-20261105/prepared"))
AIB_ROOT = Path(os.environ.get("ALPHAIDEABENCH_ROOT") or (REFS_ROOT / "AlphaIdeaBench"))
LIT2TEST_ROOT = Path(os.environ.get("LIT2TEST_ROOT") or (REFS_ROOT / "lit2test-benchmark"))
LIT2TEST_DATA = LIT2TEST_ROOT / "data/lit2test_v02_expansion40_smallari_contexts.jsonl"
LIT2TEST_FIFTH40_DATA = LIT2TEST_ROOT / "data/lit2test_v02_fifth40_smallari_contexts.jsonl"
BENCHMARK_50_MANIFEST = PACKAGE_ROOT / "data/benchmark_50_topics.json"
FROZEN_MVP15_PATH = PACKAGE_ROOT / "data/tasks_mvp15.json"
FROZEN_BENCHMARK50_PATH = PACKAGE_ROOT / "data/tasks_benchmark50.json"

DEFAULT_AIB_IDS = (
    "ws_02_agentic_ai",
    "survey_06_llm_safety_jailbreak",
    "survey_08_kv_cache_compression",
    "ws_18_scientific_foundation_models",
    "ws_09_world_models_simulation",
)

DEFAULT_PILOT3_IDS = (
    "aiib-82",
    "smallari_expansion40_001_cache_compression_llms_long_context__v02_expansion40",
    "ws_02_agentic_ai",
)

# Diverse 5-field substring anchors inside Lit2Test expansion40 contexts.
DEFAULT_LIT2TEST_ANCHORS = (
    "cache, compression, llms",
    "unlearning, machine, language",
    "concept, concepts, information",
    "actor-critic, actor, continuous",
    "test-time, adaptation, active",
)


def normalize_title_hash(title: str) -> str:
    norm = re.sub(r"[^a-z0-9]+", " ", (title or "").lower()).strip()
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()


def _extract_pdf_snippet(pdf_path: Path, *, max_pages: int = 2, max_chars: int = 3500) -> str:
    if not pdf_path.is_file():
        return ""
    try:
        reader = PdfReader(str(pdf_path))
        chunks: list[str] = []
        for page in reader.pages[:max_pages]:
            txt = (page.extract_text() or "").strip()
            if txt:
                chunks.append(txt)
        merged = re.sub(r"\s+", " ", "\n".join(chunks)).strip()
        return merged[:max_chars]
    except Exception:
        return ""


def _abstract_from_pdf_snippet(snippet: str) -> str:
    if not snippet:
        return ""
    m = re.search(r"\bAbstract\b[:.\s]*(.+?)(?:\b1\.?\s+Introduction\b|\bKeywords\b|$)", snippet, re.I)
    if m:
        return m.group(1).strip()[:1600]
    return snippet[:1200]


def load_aiib_tasks(aiib_root: Path = AIIB_ROOT) -> list[TaskSpec]:
    public_dir = aiib_root / "public"
    inputs_dir = aiib_root / "inputs"
    tasks: list[TaskSpec] = []
    for path in sorted(public_dir.glob("*.json")):
        raw = json.loads(path.read_text(encoding="utf-8"))
        case_id = str(raw["case_id"])
        boundary = raw.get("source_boundary") or {}
        cutoff = str(boundary.get("cutoff_date") or "2024-06-30")
        forbidden = list(boundary.get("forbidden_identity_hashes") or [])
        prior_work: list[PriorWorkItem] = []
        ref_lines: list[str] = []
        for ref in raw.get("references") or []:
            rid = str(ref.get("reference_id") or "")
            title = str(ref.get("title") or "").strip()
            pdf_path = inputs_dir / case_id / f"{rid}.pdf"
            snippet = _extract_pdf_snippet(pdf_path)
            abstract = _abstract_from_pdf_snippet(snippet)
            prior_work.append(
                PriorWorkItem(
                    id=rid,
                    title=title,
                    abstract=abstract,
                    fulltext_snippet=snippet,
                    url=ref.get("pdf_url"),
                    source="local_pdf",
                )
            )
            ref_lines.append(f"- [{rid}] {title}: {abstract[:450]}")

        prompt_context = (
            f"Topic: {raw.get('topic', '')}\n"
            f"Setting: {raw.get('setting', '')}\n"
            f"Cutoff Date: {cutoff}\n\n"
            f"Supplied Seed References (PDF-extracted):\n" + "\n".join(ref_lines)
        )
        tasks.append(
            TaskSpec(
                task_id=case_id,
                source_track="aiib",
                field="aiib_seed_pdf",
                title=str(raw.get("topic") or case_id),
                prompt_context=prompt_context,
                cutoff_date=cutoff,
                forbidden_title_hashes=forbidden,
                budget={"gpu": "academic_lab_default", "note": "Keep minimal falsifiable experiment feasible on public datasets."},
                prior_work=prior_work,
            )
        )
    return tasks


def load_lit2test_tasks(
    jsonl_path: Path = LIT2TEST_DATA,
    anchors: tuple[str, ...] = DEFAULT_LIT2TEST_ANCHORS,
) -> list[TaskSpec]:
    rows = [json.loads(line) for line in jsonl_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    picked: list[dict[str, Any]] = []
    used_ids: set[str] = set()
    for anchor in anchors:
        for r in rows:
            cid = str(r.get("context_id") or "")
            if cid not in used_ids and anchor in (r.get("field") or "").lower():
                picked.append(r)
                used_ids.add(cid)
                break
    for r in rows:
        if len(picked) >= len(anchors):
            break
        cid = str(r.get("context_id") or "")
        if cid not in used_ids:
            picked.append(r)
            used_ids.add(cid)

    tasks: list[TaskSpec] = []
    for r in picked[: len(anchors)]:
        prior_work: list[PriorWorkItem] = []
        paper_blocks: list[str] = []
        years: list[int] = []
        for idx, p in enumerate(r.get("papers") or [], 1):
            pid = str(p.get("paper_id") or f"p{idx}")
            yr = str(p.get("year") or "2025")
            if yr.isdigit():
                years.append(int(yr))
            prior_work.append(
                PriorWorkItem(
                    id=pid,
                    title=str(p.get("title") or ""),
                    abstract=str(p.get("abstract") or ""),
                    key_contribution=str(p.get("key_contribution") or ""),
                    limitation=str(p.get("limitation") or ""),
                    important_result=str(p.get("important_result") or ""),
                    year=yr,
                    url=p.get("url"),
                    source="lit2test_paper",
                )
            )
            paper_blocks.append(
                f"- [{pid}] ({yr}) {p.get('title', '')}\n"
                f"  Contribution: {p.get('key_contribution', '')}\n"
                f"  Important Result: {p.get('important_result', '')}\n"
                f"  Reviewer Limitation: {p.get('limitation', '')}\n"
                f"  Abstract: {str(p.get('abstract') or '')[:450]}"
            )
        max_year = max(years) if years else 2025
        cutoff = f"{max_year}-12-31"
        prompt_context = (
            f"Research Context: {r.get('research_context', '')}\n"
            f"Open Problem: {r.get('open_problem', '')}\n"
            f"Cutoff Date: {cutoff}\n"
            f"Resource Constraint: {r.get('resource_constraint', '')}\n"
            f"Task Instruction: {r.get('task_instruction', '')}\n\n"
            f"Literature Neighborhood:\n" + "\n".join(paper_blocks)
        )
        tasks.append(
            TaskSpec(
                task_id=str(r["context_id"]),
                source_track="lit2test",
                field=str(r.get("field") or "lit2test"),
                title=str(r.get("open_problem") or r["context_id"])[:140],
                prompt_context=prompt_context,
                cutoff_date=cutoff,
                forbidden_title_hashes=[],
                budget={"constraint_text": str(r.get("resource_constraint") or "")},
                prior_work=prior_work,
            )
        )
    return tasks


def _parse_frontmatter_md(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_text(encoding="utf-8")
    if raw.startswith("---"):
        parts = raw.split("---", 2)
        if len(parts) == 3:
            meta = yaml.safe_load(parts[1]) or {}
            body = parts[2].strip()
            return meta, body
    return {}, raw.strip()


def load_alphaideabench_tasks(
    aib_root: Path = AIB_ROOT,
    task_ids: tuple[str, ...] = DEFAULT_AIB_IDS,
) -> list[TaskSpec]:
    index = {item["id"]: item for item in json.loads((aib_root / "tasks/index.json").read_text(encoding="utf-8"))}
    tasks: list[TaskSpec] = []
    for tid in task_ids:
        entry = index[tid]
        md_path = aib_root / entry["path"]
        meta, body = _parse_frontmatter_md(md_path)
        budget = meta.get("budget") or {}
        # Represent the topic document itself as a local baseline context item for grounding checks.
        prior_work = [
            PriorWorkItem(
                id=f"{tid}_topic_doc",
                title=str(meta.get("title") or entry["title"]),
                abstract=body[:2000],
                fulltext_snippet=body[:4000],
                source="topic_doc",
            )
        ]
        prompt_context = (
            f"Title: {meta.get('title') or entry['title']}\n"
            f"Field: {meta.get('field') or entry['field']}\n"
            f"Cutoff Date: 2025-12-31\n"
            f"Budget Envelope: {json.dumps(budget, ensure_ascii=False)}\n\n"
            f"Research Background & Open Challenges:\n{body[:4500]}"
        )
        tasks.append(
            TaskSpec(
                task_id=tid,
                source_track="alphaideabench",
                field=str(meta.get("field") or entry["field"]),
                title=str(meta.get("title") or entry["title"]),
                prompt_context=prompt_context,
                cutoff_date="2025-12-31",
                forbidden_title_hashes=[],
                budget=budget,
                prior_work=prior_work,
            )
        )
    return tasks


def load_lit2test_by_ids(
    task_ids: tuple[str, ...] | list[str],
    jsonl_path: Path = LIT2TEST_DATA,
) -> list[TaskSpec]:
    rows_by_id = {
        str(row["context_id"]): row
        for line in jsonl_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
        for row in (json.loads(line),)
    }
    tasks: list[TaskSpec] = []
    for tid in task_ids:
        if tid not in rows_by_id:
            raise KeyError(f"Lit2Test context_id not found: {tid}")
        r = rows_by_id[tid]
        prior_work: list[PriorWorkItem] = []
        paper_blocks: list[str] = []
        years: list[int] = []
        for idx, p in enumerate(r.get("papers") or [], 1):
            pid = str(p.get("paper_id") or f"p{idx}")
            yr = str(p.get("year") or "2025")
            if yr.isdigit():
                years.append(int(yr))
            prior_work.append(
                PriorWorkItem(
                    id=pid,
                    title=str(p.get("title") or ""),
                    abstract=str(p.get("abstract") or ""),
                    key_contribution=str(p.get("key_contribution") or ""),
                    limitation=str(p.get("limitation") or ""),
                    important_result=str(p.get("important_result") or ""),
                    year=yr,
                    url=p.get("url"),
                    source="lit2test_paper",
                )
            )
            paper_blocks.append(
                f"- [{pid}] ({yr}) {p.get('title', '')}\n"
                f"  Contribution: {p.get('key_contribution', '')}\n"
                f"  Important Result: {p.get('important_result', '')}\n"
                f"  Reviewer Limitation: {p.get('limitation', '')}\n"
                f"  Abstract: {str(p.get('abstract') or '')[:450]}"
            )
        max_year = max(years) if years else 2025
        cutoff = f"{max_year}-12-31"
        prompt_context = (
            f"Research Context: {r.get('research_context', '')}\n"
            f"Open Problem: {r.get('open_problem', '')}\n"
            f"Cutoff Date: {cutoff}\n"
            f"Resource Constraint: {r.get('resource_constraint', '')}\n"
            f"Task Instruction: {r.get('task_instruction', '')}\n\n"
            f"Literature Neighborhood:\n" + "\n".join(paper_blocks)
        )
        tasks.append(
            TaskSpec(
                task_id=tid,
                source_track="lit2test",
                field=str(r.get("field") or "lit2test"),
                title=str(r.get("open_problem") or tid)[:140],
                prompt_context=prompt_context,
                cutoff_date=cutoff,
                forbidden_title_hashes=[],
                budget={"constraint_text": str(r.get("resource_constraint") or "")},
                prior_work=prior_work,
            )
        )
    return tasks


def load_benchmark50_tasks(
    *,
    wave: str | None = None,
    manifest_path: Path = BENCHMARK_50_MANIFEST,
    lit2test_jsonl: Path = LIT2TEST_DATA,
    fifth40_jsonl: Path = LIT2TEST_FIFTH40_DATA,
    aib_root: Path = AIB_ROOT,
) -> list[TaskSpec]:
    """Load the locked 50-topic (10 domains x 5 topics, 25 Lit2Test + 25 AlphaIdeaBench) benchmark or a specific wave."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = manifest["topics"]
    if fifth40_jsonl.is_file():
        fifth_ids = {
            str(json.loads(line)["context_id"])
            for line in fifth40_jsonl.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
        overlap = {e["task_id"] for e in entries} & fifth_ids
        if overlap:
            raise ValueError(f"Benchmark-50 overlaps with fifth40 calibration suite: {sorted(overlap)}")
    if wave is not None:
        if wave not in {"wave1", "wave2", "wave3", "wave1_2"}:
            raise ValueError(f"Unsupported wave {wave!r}; choose wave1, wave2, wave3, wave1_2, or None")
        allowed_waves = {"wave1", "wave2"} if wave == "wave1_2" else {wave}
        entries = [e for e in entries if e["wave"] in allowed_waves]

    lit_ids = [e["task_id"] for e in entries if e["source_track"] == "lit2test"]
    aib_ids = [e["task_id"] for e in entries if e["source_track"] == "alphaideabench"]
    by_id: dict[str, TaskSpec] = {}
    raw_available = (not lit_ids or lit2test_jsonl.is_file()) and (not aib_ids or (aib_root / "tasks/index.json").is_file())
    if raw_available:
        if lit_ids:
            for t in load_lit2test_by_ids(lit_ids, jsonl_path=lit2test_jsonl):
                by_id[t.task_id] = t
        if aib_ids:
            for t in load_alphaideabench_tasks(aib_root=aib_root, task_ids=tuple(aib_ids)):
                by_id[t.task_id] = t
    elif FROZEN_BENCHMARK50_PATH.is_file():
        frozen_items = json.loads(FROZEN_BENCHMARK50_PATH.read_text(encoding="utf-8"))
        for item in frozen_items:
            t = TaskSpec.model_validate(item)
            by_id[t.task_id] = t
    else:
        raise FileNotFoundError(
            f"Raw benchmark datasets ({lit2test_jsonl}, {aib_root}) and frozen snapshot ({FROZEN_BENCHMARK50_PATH}) not found."
        )
    ordered: list[TaskSpec] = []
    for e in entries:
        spec = by_id[e["task_id"]]
        ordered.append(spec.model_copy(update={"field": f"{e['domain_id']}:{e['domain_en']}"}))
    return ordered


def build_mvp15_tasks(cache_path: Path | None = None) -> list[TaskSpec]:
    if cache_path and cache_path.is_file():
        data = json.loads(cache_path.read_text(encoding="utf-8"))
        return [TaskSpec.model_validate(item) for item in data]

    raw_available = (
        (AIIB_ROOT / "public").is_dir()
        and LIT2TEST_DATA.is_file()
        and (AIB_ROOT / "tasks/index.json").is_file()
    )
    if raw_available:
        tasks = load_aiib_tasks() + load_lit2test_tasks() + load_alphaideabench_tasks()
    elif FROZEN_MVP15_PATH.is_file():
        data = json.loads(FROZEN_MVP15_PATH.read_text(encoding="utf-8"))
        tasks = [TaskSpec.model_validate(item) for item in data]
    else:
        tasks = load_aiib_tasks() + load_lit2test_tasks() + load_alphaideabench_tasks()
    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(
            json.dumps([t.model_dump() for t in tasks], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    return tasks


def build_pilot3_tasks(
    cache_path: Path | None = None,
    *,
    mvp15_cache_path: Path | None = None,
) -> list[TaskSpec]:
    if cache_path and cache_path.is_file():
        data = json.loads(cache_path.read_text(encoding="utf-8"))
        return [TaskSpec.model_validate(item) for item in data]

    all_tasks = {t.task_id: t for t in build_mvp15_tasks(cache_path=mvp15_cache_path)}
    pilot3 = [all_tasks[tid] for tid in DEFAULT_PILOT3_IDS if tid in all_tasks]
    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(
            json.dumps([t.model_dump() for t in pilot3], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    return pilot3


def export_pilot3_external_inputs(
    pilot3_tasks: list[TaskSpec],
    out_dir: Path,
) -> Path:
    """Export per-topic markdown + literature-boundary JSON for AI-Scientist-v2 and reference bundles for E-DARA."""
    ai_dir = out_dir / "ai_scientist_topics"
    edara_dir = out_dir / "edara_cases"
    ai_dir.mkdir(parents=True, exist_ok=True)
    edara_dir.mkdir(parents=True, exist_ok=True)

    for task in pilot3_tasks:
        md_content = (
            f"# Title: {task.title}\n\n"
            f"## Task ID & Source\n- task_id: `{task.task_id}`\n- source_track: `{task.source_track}`\n"
            f"- cutoff_date: `{task.cutoff_date}`\n\n"
            f"## Keywords\n{task.field}\n\n"
            f"## Research Problem & Constraints\n{task.prompt_context}\n"
        )
        (ai_dir / f"{task.task_id}.md").write_text(md_content, encoding="utf-8")

        boundary = {
            "protocol": "literature-boundary/v1",
            "case_id": task.task_id,
            "cutoff_date": task.cutoff_date,
            "forbidden_identity_hashes": task.forbidden_title_hashes,
            "seed_document_sha256s": [],
        }
        (ai_dir / f"{task.task_id}.boundary.json").write_text(
            json.dumps(boundary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        # Export IdeaBench-compatible case envelope for non-AIIB tracks (and symlink/reference for aiib-82).
        edara_case = {
            "schema_version": "edara.mvp-pilot-case/v1",
            "case_id": task.task_id,
            "source_track": task.source_track,
            "topic": task.title,
            "cutoff_date": task.cutoff_date,
            "budget": task.budget,
            "source_boundary": boundary,
            "native_aiib_public_json": (
                str(AIIB_ROOT / "public" / f"{task.task_id}.json") if task.source_track == "aiib" else None
            ),
            "references": [
                {
                    "reference_id": p.id,
                    "title": p.title,
                    "abstract": p.abstract or p.key_contribution or p.fulltext_snippet[:1500],
                    "year": p.year,
                    "url": p.url,
                }
                for p in task.prior_work
            ],
        }
        (edara_dir / f"{task.task_id}.json").write_text(
            json.dumps(edara_case, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        # Also materialize per-topic E-DARA bootstrap Markdown files + exact 4-field literature-boundary.json.
        t_dir = edara_dir / task.task_id
        in_dir = t_dir / "inputs"
        in_dir.mkdir(parents=True, exist_ok=True)
        seed_sha256s: list[str] = []
        if task.source_track != "aiib":
            for idx, p in enumerate(task.prior_work, 1):
                md_text = (
                    f"# {p.title}\n\n"
                    f"- Source ID: `{p.id}` ({p.year or 'n/a'})\n\n"
                    f"## Abstract / Context\n{p.abstract or p.fulltext_snippet}\n"
                )
                if p.key_contribution:
                    md_text += f"\n## Key Contribution\n{p.key_contribution}\n"
                if p.important_result:
                    md_text += f"\n## Important Result\n{p.important_result}\n"
                if p.limitation:
                    md_text += f"\n## Limitation\n{p.limitation}\n"
                ref_path = in_dir / f"ref_{idx:02d}.md"
                ref_path.write_text(md_text, encoding="utf-8")
                seed_sha256s.append(hashlib.sha256(ref_path.read_bytes()).hexdigest())
        if task.source_track == "aiib":
            aiib_pub = AIIB_ROOT / "public" / f"{task.task_id}.json"
            raw_aiib = json.loads(aiib_pub.read_text(encoding="utf-8")) if aiib_pub.is_file() else {}
            edara_boundary = raw_aiib.get("source_boundary") or {
                "protocol": "literature-boundary/v1",
                "cutoff_date": task.cutoff_date,
                "forbidden_identity_hashes": task.forbidden_title_hashes,
                "seed_document_sha256s": [],
            }
        else:
            # E-DARA boundary.validate requires non-empty unique 64-hex SHA-256 lists for both fields.
            forbidden_hashes = list(task.forbidden_title_hashes) or [
                hashlib.sha256(f"edara-sentinel-no-heldout-target:{task.task_id}".encode("utf-8")).hexdigest()
            ]
            edara_boundary = {
                "protocol": "literature-boundary/v1",
                "cutoff_date": task.cutoff_date,
                "forbidden_identity_hashes": forbidden_hashes,
                "seed_document_sha256s": seed_sha256s,
            }
        (t_dir / "literature-boundary.json").write_text(
            json.dumps(edara_boundary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    def _prof(name: str) -> str:
        cand = DATA_ROOT / name
        if not cand.is_file() and (PACKAGE_ROOT / "configs" / name).is_file():
            cand = PACKAGE_ROOT / "configs" / name
        return str(cand)

    edara_live_config = {
        "budget_microusd": 1_500_000,
        "policy_model_config": _prof("p1a-live-profile.json"),
        "topic_model_config": _prof("p1a-live-profile.json"),
        "audit_model_config": _prof("p1b-critic-profile.json"),
        "coverage_model_config": _prof("p1b-critic-profile.json"),
        "p2_model_config": _prof("p1b-idea-profile.json"),
        "p2_critic_model_config": _prof("p1b-critic-profile.json"),
        "p3_editor_model_config": _prof("p1b-idea-profile.json"),
        "p3_reviewer_model_config": _prof("p1b-critic-profile.json"),
        "p4b_generator_model_config": _prof("p2-deep-profile.json"),
        "p4b_judge_model_config": _prof("p1b-critic-profile.json"),
        "review_model_configs": [
            _prof("p1b-critic-profile.json"),
            _prof("p1b-critic-profile.json"),
            _prof("p1b-critic-profile.json"),
        ],
        "review_arbiter_model_config": _prof("p1b-critic-profile.json"),
        "supplement_source": "openalex",
    }
    (edara_dir / "edara_live_config.json").write_text(
        json.dumps(edara_live_config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    shim_dir = out_dir / "ai_scientist_shim"
    shim_dir.mkdir(parents=True, exist_ok=True)
    (shim_dir / "native_common.py").write_text(_AI_SCIENTIST_NATIVE_COMMON_SHIM, encoding="utf-8")
    return out_dir


_AI_SCIENTIST_NATIVE_COMMON_SHIM = '''"""Dynamic per-task boundary + key-free OpenAlex shim for AI-Scientist-v2 Pilot-3 runs."""
from __future__ import annotations

from datetime import datetime, timezone
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


def _is_blocked(title: str, pub_date: str, cutoff_date: str, forbidden: set[str], identifier: str = "") -> bool:
    if not title or (_candidate_hashes(title, identifier) & forbidden):
        return True
    if pub_date and len(pub_date) >= 4:
        norm_d = f"{pub_date[:4]}-12-31" if len(pub_date) == 4 else pub_date[:10]
        if norm_d > cutoff_date:
            return True
    return False


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

    s2_key = os.environ.get("S2_API_KEY")
    if s2_key:
        try:
            q = urllib.parse.quote(query[:220])
            url = (
                f"https://api.semanticscholar.org/graph/v1/paper/search?query={q}"
                f"&publicationDateOrYear=:{cutoff_date}&limit={max(1, min(int(limit), 20))}"
                f"&fields=paperId,title,authors,year,venue,abstract,citationCount,publicationDate"
            )
            req = urllib.request.Request(url, headers={"x-api-key": s2_key, "User-Agent": "EDARA-Pilot3/0.1"})
            with urllib.request.urlopen(req, timeout=15.0) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            for item in data.get("data") or []:
                title = str(item.get("title") or "")
                pub = str(item.get("publicationDate") or item.get("year") or "")
                ident = str(item.get("paperId") or item.get("url") or "")
                if not _is_blocked(title, pub, cutoff_date, forbidden, ident):
                    papers.append(item)
            if papers:
                provider = "semantic_scholar"
        except Exception:
            papers = []

    if not papers:
        q = urllib.parse.quote(query[:220])
        url = (
            f"https://api.openalex.org/works?search={q}"
            f"&filter=to_publication_date:{cutoff_date}&per-page={max(1, min(int(limit), 20))}"
            f"&select=id,title,publication_date,abstract_inverted_index,cited_by_count,authorships"
        )
        req = urllib.request.Request(url, headers={"User-Agent": "EDARA-Pilot3/0.1 (research benchmark)"})
        try:
            with urllib.request.urlopen(req, timeout=15.0) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            for item in data.get("results") or []:
                title = str(item.get("title") or "")
                pub = str(item.get("publication_date") or "")
                ident = str(item.get("id") or "")
                if _is_blocked(title, pub, cutoff_date, forbidden, ident):
                    continue
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
                    "abstract": _reconstruct_openalex_abstract(item.get("abstract_inverted_index")),
                    "citationCount": item.get("cited_by_count", "N/A"),
                })
        except Exception:
            papers = []

    append_receipt(ledger_path, {
        "schema_version": "search-receipt/v1",
        "created_at": now(),
        "arm": arm,
        "case_id": case_id,
        "query": query,
        "provider": provider,
        "date_cutoff": cutoff_date,
        "kept_count": len(papers),
        "kept_ids": [str(p.get("paperId") or p.get("title") or "")[:80] for p in papers],
    })
    return papers[: int(limit)], provider


def deepseek_cost_usd(input_tokens, output_tokens):
    return int(input_tokens) * 0.30 / 1_000_000 + int(output_tokens) * 1.20 / 1_000_000
'''
