"""Export native research artifacts with resolvable anonymous citation markers.

The historical export replaced every ``ev-*`` identifier with ``[REF]``, which
destroyed the link between a claim and the exact source passage. This module
rebuilds that link without changing a single other character: it reconciles a
saved export against the native ``research.db`` payload field by field, then
rewrites only the citation tokens.
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
from pathlib import Path
from typing import Any, Iterable

EXPORT_SCHEMA = "edara-artifact-export/v1"
REF_TOKEN = "[REF]"
EVIDENCE_ID = re.compile(r"ev-[0-9a-f]{24}")
SENTINEL = "\x00EV\x00"
MARKER = re.compile(r"\[S(\d+)\]")
DROPPED_FIELDS = ("rq_coverage",)
INJECTED_FIELDS = ("task_id",)


class ExportError(RuntimeError):
    """Raised when an export cannot be reconciled with its native payload."""


def walk_strings(value: Any) -> Iterable[str]:
    if isinstance(value, dict):
        for key in value:
            yield from walk_strings(value[key])
    elif isinstance(value, list):
        for item in value:
            yield from walk_strings(item)
    elif isinstance(value, str):
        yield value


def count_tokens(pattern: re.Pattern[str], value: Any) -> int:
    return sum(len(pattern.findall(text)) for text in walk_strings(value))


def substitute(value: Any, pattern: re.Pattern[str], replacement: str | Any) -> Any:
    if isinstance(value, dict):
        return {key: substitute(item, pattern, replacement) for key, item in value.items()}
    if isinstance(value, list):
        return [substitute(item, pattern, replacement) for item in value]
    if isinstance(value, str):
        if callable(replacement):
            return pattern.sub(replacement, value)
        return pattern.sub(replacement, value)
    return value


def masked_json(value: Any, pattern: re.Pattern[str]) -> str:
    return json.dumps(substitute(value, pattern, SENTINEL), ensure_ascii=False)


def align_fields(exported: dict[str, Any], native: dict[str, Any]) -> list[tuple[str, int, str]]:
    """Return (field, index_in_field, evidence_id) triples in exported text order."""
    missing = sorted(set(exported) - set(native))
    unexpected = [field for field in missing if field not in INJECTED_FIELDS]
    if unexpected:
        raise ExportError(f"fields present only in the export: {unexpected}")
    for field in missing:
        if count_tokens(re.compile(re.escape(REF_TOKEN)), exported[field]):
            raise ExportError(f"injected field {field!r} unexpectedly carries references")
    extra = sorted(set(native) - set(exported))
    if extra:
        raise ExportError(f"fields present only in the native payload: {extra}")
    pairs: list[tuple[str, int, str]] = []
    for field in exported:
        if field in missing:
            continue
        exported_refs = count_tokens(re.compile(re.escape(REF_TOKEN)), exported[field])
        native_ids = EVIDENCE_ID.findall(json.dumps(native[field], ensure_ascii=False))
        if exported_refs != len(native_ids):
            raise ExportError(
                f"field {field!r}: {exported_refs} exported refs but {len(native_ids)} native ids"
            )
        if masked_json(exported[field], re.compile(re.escape(REF_TOKEN))) != masked_json(
            native[field], EVIDENCE_ID
        ):
            raise ExportError(f"field {field!r}: text differs beyond citation tokens")
        pairs.extend((field, index, evidence_id) for index, evidence_id in enumerate(native_ids))
    return pairs


def build_citation_map(
    pairs: list[tuple[str, int, str]],
    evidence_rows: dict[str, dict[str, Any]],
    paper_rows: dict[str, dict[str, Any]],
    years: dict[str, dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    """Group evidence ids by paper and number the papers by first appearance."""
    order: list[str] = []
    per_paper: dict[str, list[str]] = {}
    for _, _, evidence_id in pairs:
        row = evidence_rows.get(evidence_id)
        if row is None:
            raise ExportError(f"evidence id {evidence_id} is absent from the native store")
        paper_id = row["paper_id"]
        if paper_id not in per_paper:
            per_paper[paper_id] = []
            order.append(paper_id)
        if evidence_id not in per_paper[paper_id]:
            per_paper[paper_id].append(evidence_id)
    citation_map: dict[str, dict[str, Any]] = {}
    for index, paper_id in enumerate(order, start=1):
        paper = paper_rows.get(paper_id, {})
        hit = years.get(paper_id, {})
        citation_map[f"S{index}"] = {
            "paper_id": paper_id,
            "title": paper.get("title", ""),
            "year": hit.get("year") or "",
            "publication_date": None,
            "source": hit.get("source") or paper.get("source") or "native",
            "external_id": hit.get("external_id") or "",
            "evidence_ids": per_paper[paper_id],
            "quotes": {eid: evidence_rows[eid]["quote"] for eid in per_paper[paper_id]},
            "section_ids": {eid: evidence_rows[eid].get("section_id", "")
                            for eid in per_paper[paper_id]},
            "section_id": evidence_rows[per_paper[paper_id][0]].get("section_id", ""),
        }
    return citation_map, order


def mark_exported(exported: dict[str, Any], pairs: list[tuple[str, int, str]],
                  evidence_rows: dict[str, dict[str, Any]], citation_map: dict[str, dict[str, Any]]) -> tuple[dict[str, Any], str]:
    """Replace the k-th ``[REF]`` of each field with the marker of its native source."""
    paper_to_marker = {row["paper_id"]: marker for marker, row in citation_map.items()}
    by_field: dict[str, list[str]] = {}
    for field, _, evidence_id in pairs:
        by_field.setdefault(field, []).append(paper_to_marker[evidence_rows[evidence_id]["paper_id"]])
    marked: dict[str, Any] = {}
    for field in exported:
        marker_list = by_field.get(field, [])
        position = {"i": 0}

        def replace(_match: re.Match[str]) -> str:
            index = position["i"]
            position["i"] += 1
            return f"[{marker_list[index]}]"

        marked[field] = substitute(exported[field], re.compile(re.escape(REF_TOKEN)), replace)
        if position["i"] != len(marker_list):
            raise ExportError(f"field {field!r}: marker count mismatch")
    return marked, json.dumps(marked, ensure_ascii=False)


def reconcile(
    exported: dict[str, Any],
    original_line: str,
    native_payload: dict[str, Any],
    evidence_rows: dict[str, dict[str, Any]],
    paper_rows: dict[str, dict[str, Any]],
    years: dict[str, dict[str, Any]],
    *,
    task_id: str,
    stage: str,
    version_id: str,
    source_db: str,
) -> dict[str, Any]:
    if json.dumps(exported, ensure_ascii=False) != original_line:
        raise ExportError("exported object does not round-trip to the saved line")
    native = {key: value for key, value in native_payload.items() if key not in DROPPED_FIELDS}
    pairs = align_fields(exported, native)
    citation_map, paper_order = build_citation_map(pairs, evidence_rows, paper_rows, years)
    _, text = mark_exported(exported, pairs, evidence_rows, citation_map)
    inverse = MARKER.sub(REF_TOKEN, text)
    if inverse != original_line:
        raise ExportError("marker substitution is not textually reversible")
    if inverse.count(REF_TOKEN) != text.count("[S"):
        raise ExportError("marker count does not match the original reference count")
    return {
        "schema": EXPORT_SCHEMA,
        "task_id": task_id,
        "stage": stage,
        "native_version_id": version_id,
        "source_db": source_db,
        "text": text,
        "citation_map": citation_map,
        "proof": {
            "round_trip_ok": True,
            "inverse_ok": True,
            "masked_equal": True,
            "refs": sum(1 for _ in pairs),
            "sources": len(citation_map),
            "papers": paper_order,
        },
    }


def _read_only(db: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA query_only=ON")
    return connection


def load_native(db: Path, *, stage: str = "proposal") -> dict[str, Any]:
    connection = _read_only(db)
    try:
        row = connection.execute(
            "select version_id, payload_json from research_versions"
            " where kind = ? order by created_at desc limit 1",
            (stage,),
        ).fetchone()
        if row is None:
            raise ExportError(f"no {stage} version in {db}")
        evidence_rows = {
            record["evidence_id"]: dict(record)
            for record in connection.execute(
                "select evidence_id, paper_id, quote, section_id from evidence"
            )
        }
        paper_rows = {
            record["paper_id"]: dict(record)
            for record in connection.execute("select paper_id, title from papers")
        }
        years = {
            record["admitted_paper_id"]: dict(record)
            for record in connection.execute(
                "select admitted_paper_id, year, source, external_id from scout_hits"
                " where admitted_paper_id is not null and admitted_paper_id != 'None'"
            )
        }
    finally:
        connection.close()
    return {
        "version_id": row["version_id"],
        "payload": json.loads(row["payload_json"]),
        "evidence": evidence_rows,
        "papers": paper_rows,
        "years": years,
    }


def export_one(db: Path, exported_lines: Path, task_id: str, out: Path, *, stage: str = "proposal") -> dict[str, Any]:
    lines = [line for line in exported_lines.read_text(encoding="utf-8").splitlines() if line.strip()]
    match = None
    for line in lines:
        parsed = json.loads(line)
        if str(parsed.get("task_id")) == task_id:
            match = (parsed, line)
            break
    if match is None:
        raise ExportError(f"task {task_id} is absent from {exported_lines}")
    exported, original_line = match
    native = load_native(db, stage=stage)
    result = reconcile(
        exported, original_line, native["payload"], native["evidence"], native["papers"], native["years"],
        task_id=task_id, stage=stage, version_id=native["version_id"], source_db=str(db),
    )
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{task_id}.artifact.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    (out / f"{task_id}.citation_map.json").write_text(
        json.dumps(result["citation_map"], ensure_ascii=False, indent=2) + "\n"
    )
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--exported", type=Path, required=True)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--stage", default="proposal")
    args = parser.parse_args(argv)
    result = export_one(args.db, args.exported, args.task_id, args.out, stage=args.stage)
    proof = result["proof"]
    print(f"[export] {args.task_id}: refs={proof['refs']} sources={proof['sources']} "
          f"round_trip={proof['round_trip_ok']} inverse={proof['inverse_ok']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
