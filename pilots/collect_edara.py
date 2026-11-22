"""Collect native E-DARA artifacts and their cited sources into host draft files."""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
from pathlib import Path

WORK = Path(os.environ.get("EDARA_EVAL_ROOT") or Path(__file__).resolve().parents[1])
REFS = Path(os.environ.get("EDARA_REFS_ROOT") or WORK.parent)
NATIVE15 = WORK / "outputs/hosts_v1/native15"

LEGACY = {
    "aiib-82": REFS / "pilot3_runs/edara/aiib82_arxiv_run/cases/aiib-82/topics/the-topic-of-this-paper-is-open-vocabula-b3161cec/research.db",
    "smallari_expansion40_001_cache_compression_llms_long_context__v02_expansion40": (
        REFS / "pilot3_runs/edara/smallari_expansion40_001_cache_compression_llms_long_context__v02_expansion40/topics/propose-one-concrete-next-step-research--5a8e02ac/research.db"
    ),
    "ws_02_agentic_ai": REFS / "pilot3_runs/edara/ws_02_agentic_ai/topics/towards-agentic-ai-for-science-hypothesi-18263a5e/research.db",
}
RANK = {"plan_reviewed": 3, "proposal_reviewed": 2, "inconclusive": 1}


def open_db(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA query_only=ON")
    return connection


def attempt_dirs(task: str) -> list[Path]:
    root = NATIVE15 / task
    return sorted(path for path in root.glob("attempt*") if path.is_dir()) if root.is_dir() else []


def workspace_db(attempt: Path) -> Path | None:
    bootstrap = attempt / "bootstrap-result.json"
    if not bootstrap.exists():
        return None
    workspace = json.loads(bootstrap.read_text()).get("workspace")
    if not workspace:
        return None
    db = Path(workspace) / "research.db"
    return db if db.exists() else None


def best(db_paths: list[Path]) -> tuple[Path, dict] | None:
    best_pick = None
    for path in db_paths:
        connection = open_db(path)
        statuses = [row["status"] for row in connection.execute("select status from policy_runs")]
        version = connection.execute(
            "select version_id, kind, payload_json, created_at from research_versions"
            " where kind in ('experiment_plan','proposal') order by"
            " case kind when 'experiment_plan' then 0 else 1 end, created_at desc limit 1").fetchone()
        connection.close()
        if version is None:
            continue
        score = max((RANK.get(status, 0) for status in statuses), default=0)
        payload = json.loads(version["payload_json"])
        pick = (score, path, {"version_id": version["version_id"], "stage": version["kind"],
                              "payload": payload, "policy_status": statuses[-1] if statuses else None,
                              "statuses": statuses})
        if best_pick is None or score > best_pick[0]:
            best_pick = pick
    return None if best_pick is None else (best_pick[1], best_pick[2])


def sources(db: Path) -> list[dict]:
    connection = open_db(db)
    rows = connection.execute(
        "select e.paper_id, p.title, e.evidence_id, e.quote from evidence e"
        " left join papers p on p.paper_id = e.paper_id").fetchall()
    connection.close()
    grouped: dict[str, dict] = {}
    for row in rows:
        entry = grouped.setdefault(row["paper_id"], {"paper_id": row["paper_id"],
                                                     "title": row["title"] or row["paper_id"],
                                                     "quotes": []})
        if row["quote"]:
            entry["quotes"].append(row["quote"])
    return list(grouped.values())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks-cache", type=Path, default=WORK / "outputs/tasks_mvp15.json")
    parser.add_argument("--out-dir", type=Path, default=WORK / "outputs/hosts_v1")
    args = parser.parse_args(argv)
    tasks = [row["task_id"] for row in json.loads(args.tasks_cache.read_text())]
    rows, evidence = [], {}
    for task in tasks:
        candidates = [LEGACY[task]] if task in LEGACY else []
        for attempt in attempt_dirs(task):
            db = workspace_db(attempt)
            if db:
                candidates.append(db)
        picked = best(candidates)
        if picked is None:
            print(f"[collect] MISSING {task[:50]}")
            continue
        db, info = picked
        rows.append({"task_id": task, **info["payload"], "_edara": {
            "stage": info["stage"], "version_id": info["version_id"],
            "policy_status": info["policy_status"], "statuses": info["statuses"], "db": str(db)}})
        evidence[task] = sources(db)
        print(f"[collect] {task[:50]:50s} stage={info['stage']:16s} status={info['policy_status']} "
              f"sources={len(evidence[task])}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "edara_15.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n")
    (args.out_dir / "edara_15_evidence.json").write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + "\n")
    print(f"[collect] rows={len(rows)} with_evidence={len(evidence)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
