"""Frozen manuscripts and symmetric evidence packets for the v1 trial."""

from __future__ import annotations

import json
import re
from copy import deepcopy
from contextlib import closing
from datetime import date
from pathlib import Path
from typing import Any

from .artifact_export import EVIDENCE_ID, _read_only, substitute
from .governance import admissible
from .schemas import PriorWorkItem, TaskSpec
from .stage_b_evidence import digest

HOSTS = ("direct", "ai_scientist", "edara")
Draft = dict[str, str]
_MARKERS = re.compile(r"\[r\d+\]", re.IGNORECASE)
_METADATA = {"task_id", "_edara", "Name", "mandate_version", "proposal_version_id", "rq_ids",
             "exposed_result_ids", "result_informed", "rq_coverage"}
_AI_FIELDS = {"Title": "title", "Short Hypothesis": "hypothesis", "Abstract": "approach",
              "Related Work": "related_work", "Experiments": "validation",
              "Risk Factors and Limitations": "limitations"}


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def project(payload: dict) -> Draft:
    """Expose native text leaf fields; retain nested scientific details and empty fields."""
    result = {}

    def visit(key, value):
        if isinstance(value, dict):
            for child, item in value.items():
                visit(f"{key}.{child}" if key else str(child), item)
        elif isinstance(value, list):
            for index, item in enumerate(value, 1):
                visit(f"{key}[{index}]", item)
        elif isinstance(value, str) or value is None:
            result[key] = value or ""
        else:
            result[key] = json.dumps(value, ensure_ascii=False)

    for key, value in payload.items():
        if key not in _METADATA:
            visit(_AI_FIELDS.get(key, key), value)
    return result


def dated(value: str | None, cutoff: str) -> bool:
    try:
        if value and len(value) == 4:
            return int(value) < date.fromisoformat(cutoff).year
        return bool(value and date.fromisoformat(value) < date.fromisoformat(cutoff))
    except ValueError:
        return False


def identities(paper: PriorWorkItem) -> set[str]:
    keys = {f"title:{re.sub(r'[^a-z0-9]+', '', paper.title.casefold())}"}
    for value in (paper.id, paper.url or ""):
        match = re.search(r"10\.\d{4,}/[^\s]+", value.casefold())
        if match:
            keys.add(match.group())
    return keys


class Evidence:
    def __init__(self, task: TaskSpec):
        self.task = task
        self.rows: dict[str, dict] = {}
        self.keys: dict[str, str] = {}
        self.audit_items: dict[str, PriorWorkItem] = {}

    def add(self, paper: PriorWorkItem) -> str | None:
        passages = [text.strip() for text in (paper.fulltext_snippet, paper.abstract) if text.strip()]
        if not passages or paper.source == "topic_doc":
            return None
        keys = identities(paper)
        marker = next((self.keys[key] for key in sorted(keys) if key in self.keys), None)
        marker = marker or "S" + digest(sorted(keys)[0])[:12]
        published = paper.publication_date or paper.year
        verified_date = bool(paper.version and paper.date_source and paper.date_basis == "publication")
        prior = verified_date and admissible(self.task, paper)
        row = self.rows.setdefault(marker, {"paper_id": marker, "title": paper.title, "sources": [],
                                           "attribution_passages": [], "novelty_passages": [], "dates": []})
        row["attribution_passages"] = list(dict.fromkeys(row["attribution_passages"] + passages))
        if prior:
            row["novelty_passages"] = list(dict.fromkeys(row["novelty_passages"] + passages))
        if published:
            row["dates"] = list(dict.fromkeys(row["dates"] + [published]))
        row["sources"] = list(dict.fromkeys(row["sources"] + [paper.id]))
        self.keys.update(dict.fromkeys(keys, marker))
        if row["novelty_passages"]:
            current = self.audit_items.get(marker)
            basis = paper if prior else current
            self.audit_items[marker] = basis.model_copy(update={
                "id": marker, "abstract": "", "fulltext_snippet": "\n".join(row["novelty_passages"]),
            })
        else:
            self.audit_items[marker] = paper.model_copy(update={
                "id": marker, "abstract": "", "fulltext_snippet": "\n".join(row["attribution_passages"]),
                "source": "native_export", "date_basis": "participant_provenance",
                "version": paper.version or "saved-source",
                "date_source": paper.date_source or "saved-source",
            })
        return marker

    def packet(self) -> list[dict]:
        return [{key: row[key] for key in (
            "paper_id", "title", "attribution_passages", "novelty_passages", "dates"
        )} for row in self.rows.values()]

    def audit_task(self) -> TaskSpec:
        return self.task.model_copy(update={"prior_work": list(self.audit_items.values())})

    def snapshot(self) -> dict:
        return {"rows": self.rows, "keys": self.keys,
                "audit_items": {key: value.model_dump() for key, value in self.audit_items.items()}}

    @classmethod
    def restore(cls, task: TaskSpec, snapshot: dict) -> Evidence:
        obj = cls(task)
        obj.rows = deepcopy(snapshot["rows"])
        obj.keys = dict(snapshot["keys"])
        obj.audit_items = {key: PriorWorkItem.model_validate(value)
                           for key, value in snapshot["audit_items"].items()}
        return obj


def native_citations(root: Path, row: dict, evidence: Evidence) -> dict[str, str]:
    db = Path(row["_edara"]["db"]).resolve()
    if not db.is_relative_to(root.parent.parent.resolve()):
        raise ValueError("Native artifact DB path is outside staging")
    identifiers = sorted(set(EVIDENCE_ID.findall(json.dumps(row, ensure_ascii=False))))
    if not identifiers:
        return {}
    with closing(_read_only(db)) as conn:
        refs = {}
        for identifier in identifiers:
            record = conn.execute(
                """select e.paper_id, e.quote, p.title, s.year, s.url, s.abstract
                   from evidence e left join papers p on p.paper_id = e.paper_id
                   left join scout_hits s on s.admitted_paper_id = p.paper_id
                   where e.evidence_id = ?
                   order by case when s.year is null then 1 else 0 end, s.hit_id
                   limit 1""", (identifier,),
            ).fetchone()
            if record is None or not record["quote"]:
                continue
            year = str(record["year"] or "") or None
            marker = evidence.add(PriorWorkItem(
                id=record["paper_id"], title=record["title"] or record["paper_id"],
                year=year, url=record["url"], source="native_export",
                fulltext_snippet=record["quote"],
                version="dated-scout-abstract" if year and record["abstract"] else None,
                date_source="native-scout-metadata" if year and record["abstract"] else None,
                abstract=record["abstract"] or "",
                date_basis="publication" if record["abstract"] and record["abstract"] == record["quote"]
                else "participant_provenance",
            ))
            if year and record["abstract"] and record["abstract"] != record["quote"]:
                # The stored abstract has separate date provenance. Do not attach that date to the quote.
                evidence.add(PriorWorkItem(
                    id=record["paper_id"], title=record["title"] or record["paper_id"],
                    year=year, url=record["url"], source="native_export", abstract=record["abstract"],
                    version="dated-scout-abstract", date_source="native-scout-metadata",
                ))
            if marker:
                refs[identifier] = f"[{marker}]"
        return refs


def source_text(value: str) -> str:
    return re.sub(r"[\W_]+", " ", value.casefold()).strip()


def ai_citations(root: Path, task_id: str, draft: dict, evidence: Evidence, sources: list[dict],
                 originals: list[PriorWorkItem]) -> dict[str, str]:
    body = json.dumps(draft, ensure_ascii=False)
    normalized = source_text(body)
    paths = (root.parent / "pilot3_runs/ai_scientist_v2" / f"{task_id}.md",
             root / "outputs/hosts_v1/ai_scientist_topics" / f"{task_id}.md")
    path = next((file for file in paths if file.exists()), None)
    aliases = {}
    if path:
        for line in path.read_text(encoding="utf-8").splitlines():
            match = re.match(r"^\s*-\s*(\[r\d+\])\s*(.*)$", line, flags=re.IGNORECASE)
            if not match or match[1] not in body:
                continue
            citation = source_text(match[2])
            matches = [paper for paper in originals if citation.startswith(source_text(paper.title) + " ")
                       or citation == source_text(paper.title)]
            if matches:
                paper = max(matches, key=lambda item: len(item.title))
                marker = evidence.add(paper)
                if marker:
                    aliases[match[1].lower()] = f"[{marker}]"
    # Exact title/DOI matches add a source even if it was absent from the old rank-limited pool.
    for row in sources:
        title = source_text(row["title"])
        doi = (row.get("doi") or "").removeprefix("https://doi.org/")
        if not (title and f" {title} " in f" {normalized} " or doi and doi.casefold() in body.casefold()):
            continue
        passage = "\n".join(text for text in row.get("quotes", []) if text)
        year = str(row.get("year") or "") or None
        evidence.add(PriorWorkItem(id=row["paper_id"], title=row["title"], year=year,
                                   url=row.get("doi"), source="openalex", abstract=passage,
                                   date_source="saved-openalex-metadata" if year else None,
                                   version="indexed-abstract" if year else None))
    return aliases


def direct_aliases(payload: dict, evidence: Evidence) -> dict[str, str]:
    body = json.dumps(payload, ensure_ascii=False)
    return {f"[{old}]": f"[{marker}]" for marker, row in evidence.rows.items()
            for old in row["sources"] if f"[{old}]" in body}


def prepare_inputs(root: Path, cohort: str) -> dict:
    outputs = root / "outputs"
    hosts = outputs / "hosts_v1"
    run = outputs / ("repair_v3/expansion" if cohort == "extension" else "mvp15_ablation_v2")
    development = {task["task_id"] for task in load(outputs / "tasks_pilot3.json")}
    ai = {row["task_id"]: row for row in read_rows(hosts / "ai_scientist_v1/ai_scientist_15.jsonl")}
    native = {row["task_id"]: row for row in read_rows(hosts / "edara_15.jsonl")}
    ai_sources = load(hosts / "ai_scientist_15_evidence.json")
    topics = {}
    for raw in load(run / "tasks_frozen.json"):
        task = TaskSpec.model_validate(raw).model_copy(update={"strict_evidence": True})
        key = task.task_id
        if cohort == "pilot" and key not in development:
            continue
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", key) or key in {".", ".."} or key in topics:
            raise ValueError("Invalid or duplicate task ID")
        papers = Evidence(task)
        for paper in task.prior_work:
            papers.add(paper)
        saved_pool = run / "tasks" / key / "evaluation_evidence.json"
        if saved_pool.is_file():
            inventory = load(saved_pool)
            ids = {row["paper_id"] for row in inventory["evidence"]}
            for row in inventory["all_admitted"]:
                if row["id"] in ids:
                    papers.add(PriorWorkItem.model_validate(row))
        direct = load(run / "tasks" / key / "direct.json")
        originals = {"direct": direct.get("candidate") if direct["status"] == "produced" else None}
        if cohort != "extension":
            originals.update({"ai_scientist": ai.get(key), "edara": native.get(key)})
        versions = {}
        for host, payload in originals.items():
            if not payload:
                versions[host] = {"status": "no_artifact"}
                continue
            draft = project(payload)
            aliases = direct_aliases(payload, papers) if host == "direct" else {}
            unresolved = []
            if host == "edara":
                aliases = native_citations(root, payload, papers)
                unresolved = sorted(set(EVIDENCE_ID.findall(json.dumps(draft))) - aliases.keys())
                draft = substitute(draft, EVIDENCE_ID, lambda match: aliases.get(match[0], "[unmapped citation]"))
            if host == "ai_scientist":
                aliases = ai_citations(root, key, payload, papers, ai_sources.get(key, []), task.prior_work)
                unresolved = sorted(set(token.lower() for token in _MARKERS.findall(json.dumps(draft))) - aliases.keys())
                draft = substitute(draft, _MARKERS, lambda match: aliases.get(match[0].lower(), "[unmapped citation]"))
            if host == "direct":
                for old, new in aliases.items():
                    draft = {field: value.replace(old, new) for field, value in draft.items()}
            versions[host] = {"status": "ready", "stage": payload.get("_edara", {}).get("stage", "proposal"),
                              "version": payload.get("_edara", {}).get("version_id") or digest(payload),
                              "raw_sha256": digest(payload), "sha256": digest(draft), "draft": draft,
                              "citation_map": aliases, "unmapped_citations": unresolved}
        topics[key] = {"task": task.model_dump(), "development": key in development,
                       "evidence": papers.snapshot(), "originals": versions}
    return topics
