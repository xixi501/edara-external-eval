"""Score every host artifact on the core task set under one frozen pool and rubric."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from edara_mvp_eval.canonical_render import render
from edara_mvp_eval.llm_client import ModelEndpoint
from edara_mvp_eval.schemas import TaskSpec
from edara_mvp_eval.stage_b_evidence import digest, write_json
from edara_mvp_eval.stage_b_meter import RecordedRunner
from edara_mvp_eval.stage_b_review import AXES, review_candidate, review_packet

HOSTS = ("direct", "ai_scientist", "edara")


def citation_key(task, pool: list[dict]) -> list[str]:
    """Map the numbered seed markers used by imported generators onto pool ids."""
    import re
    by_title = {re.sub(r"\W+", "", str(row["title"]).lower()): row for row in pool}
    lines = []
    for index, item in enumerate(task.prior_work, 1):
        match = by_title.get(re.sub(r"\W+", "", str(item.title).lower()))
        if match:
            lines.append(f"[r{index:02d}] = {match['paper_id']} | {match['title']}")
    return lines


def load(path: Path):
    return json.loads(path.read_text())


def jsonl(path: Path) -> dict[str, dict]:
    return {row["task_id"]: row for row in
            (json.loads(line) for line in path.read_text().splitlines() if line.strip())}


HOST_SOURCE_LIMIT = 12


def rank_sources(rows: list[dict], focus: str, limit: int) -> list[dict]:
    """Keep the host sources whose title words appear in that host's own text."""
    import re
    words = {word for word in re.findall(r"[a-z][a-z0-9-]{3,}", (focus or "").lower())}
    scored = []
    for index, row in enumerate(rows):
        title_words = set(re.findall(r"[a-z][a-z0-9-]{3,}", str(row.get("title", "")).lower()))
        scored.append((len(words & title_words), -index, row))
    scored.sort(reverse=True, key=lambda item: (item[0], item[1]))
    return [row for score, _, row in scored[:limit]]


def build_pool(task_id: str, run: Path, host_evidence: dict[str, dict],
               focus: dict[str, str] | None = None) -> list[dict]:
    """Union the frozen pool with every host's own retrieved sources."""
    frozen = load(run / "tasks" / task_id / "evaluation_evidence.json")
    pool = list(frozen["evidence"])
    known = {row["paper_id"] for row in pool}
    for host, table in host_evidence.items():
        rows = table.get(task_id, [])
        if focus and host in focus:
            rows = rank_sources(rows, focus[host], HOST_SOURCE_LIMIT)
        for source in rows:
            paper_id = f"{host}:{source['paper_id']}"
            if paper_id in known or not source.get("quotes"):
                continue
            pool.append({"paper_id": paper_id, "title": source["title"], "date": "",
                         "passages": "\n".join(q for q in source["quotes"] if q)})
            known.add(paper_id)
    return pool


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True, help="Frozen core run with candidates and pools")
    parser.add_argument("--hosts-dir", type=Path, required=True)
    parser.add_argument("--judge-profile", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--rubric", default="v3")
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--only-hosts", default=",".join(HOSTS))
    args = parser.parse_args(argv)

    tasks = [TaskSpec.model_validate(row) for row in load(args.run / "tasks_frozen.json")]
    hosts = tuple(h.strip() for h in args.only_hosts.split(",") if h.strip())
    artifacts = {
        "direct": None,
        "ai_scientist": jsonl(args.hosts_dir / "ai_scientist_v1/ai_scientist_15.jsonl")
        if (args.hosts_dir / "ai_scientist_v1/ai_scientist_15.jsonl").exists() else {},
        "edara": jsonl(args.hosts_dir / "edara_15.jsonl") if (args.hosts_dir / "edara_15.jsonl").exists() else {},
    }
    host_evidence = {}
    for host, name in (("edara", "edara_15_evidence.json"),
                       ("ai_scientist", "ai_scientist_15_evidence.json")):
        path = args.hosts_dir / name
        host_evidence[host] = load(path) if path.exists() else {}
    judge = ModelEndpoint.from_env_or_profile(profile_path=args.judge_profile, prefix="EDARA_JUDGE")

    def review_task(task: TaskSpec) -> dict:
        drafts = {}
        for host in hosts:
            if host == "direct":
                drafts[host] = load(args.run / "tasks" / task.task_id / "direct.json").get("candidate")
            else:
                drafts[host] = artifacts.get(host, {}).get(task.task_id)
        focus = {host: render(host, drafts[host]) for host in hosts if drafts.get(host)}
        pool = build_pool(task.task_id, args.run, host_evidence, focus=focus)
        packet = review_packet(task, pool)
        rows = {}
        for host in hosts:
            draft = drafts.get(host)
            if not draft:
                rows[host] = {"status": "no_artifact"}
                continue
            key = citation_key(task, pool) if host == "ai_scientist" else None
            text = render(host, draft, citation_key=key)
            draws = []
            for index in range(args.repeat):
                runner = RecordedRunner(judge, args.out / "requests" / task.task_id / host / f"r{index}",
                                        task_id=task.task_id, arm=host, stage=f"final_verdict_r{index}")
                draws.append(review_candidate(packet, text, runner, retry=True, rubric=args.rubric))
            rows[host] = {"status": "assessed", "text_sha256": digest(text),
                          "chars": len(text), "primary": draws[0], "repeats": draws}
        print(f"[hosts] {task.task_id[:48]:48s} " +
              " ".join(f"{host}={rows[host].get('status')}" for host in hosts), flush=True)
        return {"task_id": task.task_id, "pool": len(pool), "hosts": rows}

    args.out.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        results = list(pool.map(review_task, tasks))

    stability = {"joint_flips": 0, "axis_flips": 0, "dims": 0, "assessed": 0}
    for result in results:
        for host in hosts:
            entry = result["hosts"][host]
            if entry.get("status") != "assessed":
                continue
            stability["assessed"] += 1
            draws = entry["repeats"]
            if len({draw.get("joint_pass") for draw in draws}) > 1:
                stability["joint_flips"] += 1
            for axis in AXES:
                stability["dims"] += 1
                if len({draw.get("axes", {}).get(axis, {}).get("status") for draw in draws}) > 1:
                    stability["axis_flips"] += 1

    payload = {"rubric": args.rubric, "repeat": args.repeat, "hosts": list(hosts),
               "results": results, "stability": stability}
    write_json(args.out / "results.json", payload)
    usage = [load(path) for path in sorted((args.out / "requests").glob("**/*.json"))]
    write_json(args.out / "usage.json", [{k: row.get(k) for k in
                                          ("task_id", "arm", "stage", "tag", "status", "usage")}
                                         for row in usage])
    total = sum((row.get("usage") or {}).get("cost_usd", 0) for row in usage)
    (args.out / "summary.md").write_text(summarize(payload) + f"\nTotal metered API cost: ${total:.6f}.\n")
    print(f"[hosts] stability={stability} cost=${total:.4f}", flush=True)
    import sys as _sys
    _sys.stdout.flush()
    _sys.stderr.flush()
    import os as _os
    _os._exit(0)


def summarize(payload: dict) -> str:
    results = payload["results"]
    hosts = payload["hosts"]
    lines = ["# Cross-host evaluation on the core tasks", "",
             f"Rubric: {payload['rubric']}. Repeats: {payload['repeat']}. "
             "All hosts share the frozen pool of their task.", "",
             "| Host | Joint (draws) | Honesty P/F/U | Credibility P/F/U | Novelty P/F/U | Artifacts |",
             "|---|---:|---|---|---|---:|"]
    for host in hosts:
        entries = [row["hosts"][host] for row in results if row["hosts"][host].get("status") == "assessed"]
        per = [sum(1 for entry in entries if entry["repeats"][i].get("joint_pass"))
               for i in range(payload["repeat"])]
        cells = []
        for axis in AXES:
            counts = {status: 0 for status in ("pass", "fail", "unresolved")}
            for entry in entries:
                counts[entry["primary"]["axes"][axis]["status"]] += 1
            cells.append(" / ".join(str(counts[s]) for s in ("pass", "fail", "unresolved")))
        lines.append(f"| {host} | {' / '.join(map(str, per))} | " + " | ".join(cells) + f" | {len(entries)} |")
    stability = payload["stability"]
    lines.extend(["", "## Stability", "",
                  f"- assessed verdicts: {stability['assessed']}",
                  f"- joint verdicts that changed across repeats: {stability['joint_flips']}",
                  f"- axis labels that changed across repeats: {stability['axis_flips']}/{stability['dims']}"])
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
