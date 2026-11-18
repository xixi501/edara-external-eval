"""Audit every host artifact against one shared per-task pool.

The audit produces the structured claims the structural metrics need: prior-fact
status, overclaims, specification defects and gap survival. The same pool is used
for every host of a task, so the comparison is not confounded by evidence coverage.
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from edara_mvp_eval.canonical_render import render
from edara_mvp_eval.epistemic_audit import run_full_audit
from edara_mvp_eval.llm_client import ModelEndpoint
from edara_mvp_eval.schemas import PriorWorkItem, TaskSpec
from edara_mvp_eval.stage_b_evidence import write_json
from edara_mvp_eval.stage_b_meter import RecordedRunner

from run_host_eval import build_pool, citation_key, jsonl, load

HOSTS = ("direct", "ai_scientist", "edara")


def pool_prior_work(pool: list[dict]) -> list[PriorWorkItem]:
    items = []
    for row in pool:
        passages = row.get("passages") or ""
        if not passages.strip():
            continue
        items.append(PriorWorkItem(
            id=row["paper_id"], title=row.get("title") or row["paper_id"],
            abstract=passages, version="host-pool/v1", date_source="shared-evaluation-pool",
            source="native_export", date_basis="participant_provenance"))
    return items


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--hosts-dir", type=Path, required=True)
    parser.add_argument("--judge-profile", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args(argv)

    tasks = [TaskSpec.model_validate(row) for row in load(args.run / "tasks_frozen.json")]
    artifacts = {
        "direct": None,
        "ai_scientist": jsonl(args.hosts_dir / "ai_scientist_v1/ai_scientist_15.jsonl"),
        "edara": jsonl(args.hosts_dir / "edara_15.jsonl"),
    }
    evidence = {}
    for host, name in (("edara", "edara_15_evidence.json"),
                       ("ai_scientist", "ai_scientist_15_evidence.json")):
        path = args.hosts_dir / name
        evidence[host] = load(path) if path.exists() else {}
    endpoint = ModelEndpoint.from_env_or_profile(profile_path=args.judge_profile, prefix="EDARA_JUDGE")

    def audit_task(task: TaskSpec) -> dict:
        drafts = {host: (load(args.run / "tasks" / task.task_id / "direct.json").get("candidate")
                         if host == "direct" else artifacts[host].get(task.task_id))
                  for host in HOSTS}
        focus = {host: render(host, drafts[host]) for host in HOSTS if drafts.get(host)}
        pool = build_pool(task.task_id, args.run, evidence, focus=focus)
        scoped = task.model_copy(update={"prior_work": pool_prior_work(pool), "strict_evidence": True})
        written = []
        for host in HOSTS:
            if not drafts.get(host):
                continue
            key = citation_key(task, pool) if host == "ai_scientist" else None
            text = render(host, drafts[host], citation_key=key)
            runner = RecordedRunner(endpoint, args.out / "requests" / task.task_id / host,
                                    task_id=task.task_id, arm=host, stage="audit")
            try:
                bundle = run_full_audit(scoped, host, text, runner, live_search=False)
            except Exception as exc:  # noqa: BLE001 - record and continue
                print(f"[audit] FAILED {task.task_id[:32]}/{host}: {type(exc).__name__}", flush=True)
                write_json(args.out / "bundles" / task.task_id / f"{host}.json",
                           {"status": "failed", "error": type(exc).__name__})
                continue
            payload = bundle.model_dump()
            payload["_structural"] = {
                "pool_size": len(pool),
                "chars": len(text),
            }
            write_json(args.out / "bundles" / task.task_id / f"{host}.json", payload)
            written.append(host)
        print(f"[audit] {task.task_id[:40]:40s} pool={len(pool):3d} audited={written}", flush=True)
        return {"task_id": task.task_id, "pool": len(pool), "audited": written}

    args.out.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool_exec:
        records = list(pool_exec.map(audit_task, tasks))
    write_json(args.out / "index.json", records)
    total = sum(1 for task_dir in (args.out / "bundles").glob("*") for _ in task_dir.glob("*.json"))
    print(f"[audit] bundles={total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
