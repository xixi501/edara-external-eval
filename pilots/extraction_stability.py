"""Measure how stable the claim extraction is: renormalise a sample and compare claims."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from edara_mvp_eval.canonical_render import render
from edara_mvp_eval.epistemic_audit import normalize_artifact
from edara_mvp_eval.llm_client import ModelEndpoint
from edara_mvp_eval.schemas import TaskSpec
from edara_mvp_eval.stage_b_evidence import write_json
from edara_mvp_eval.stage_b_meter import RecordedRunner

from audit_all_hosts import pool_prior_work
from run_host_eval import build_pool, citation_key, jsonl, load

HOSTS = ("direct", "ai_scientist", "edara")


def tokens(text: str) -> set[str]:
    return {word for word in re.findall(r"[a-z][a-z0-9-]{3,}", (text or "").lower())}


def jaccard(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 1.0
    return len(left & right) / max(1, len(left | right))


def overlap(left: list[str], right: list[str]) -> float:
    """Greedy one-to-one match on token Jaccard >= 0.5."""
    if not left or not right:
        return 0.0 if left or right else 1.0
    used, hits = set(), 0
    for claim in left:
        best, best_score = None, 0.0
        for index, other in enumerate(right):
            if index in used:
                continue
            score = jaccard(tokens(claim), tokens(other))
            if score > best_score:
                best, best_score = index, score
        if best is not None and best_score >= 0.5:
            used.add(best)
            hits += 1
    return hits / max(len(left), len(right))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--hosts-dir", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--judge-profile", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=9)
    args = parser.parse_args(argv)

    tasks = {row["task_id"]: TaskSpec.model_validate(row) for row in load(args.run / "tasks_frozen.json")}
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

    bundles = sorted((args.audit / "bundles").glob("*/*.json"))
    chosen = []
    seen_host = {}
    for path in bundles:
        host = path.stem
        if host not in HOSTS:
            continue
        if seen_host.get(host, 0) >= max(1, args.limit // len(HOSTS)):
            continue
        payload = json.loads(path.read_text())
        if "proposal" not in payload:
            continue
        chosen.append((path, payload))
        seen_host[host] = seen_host.get(host, 0) + 1

    rows = []
    for path, payload in chosen:
        task_id, host = path.parent.name, path.stem
        task = tasks[task_id]
        drafts = {h: (load(args.run / "tasks" / task_id / "direct.json").get("candidate")
                      if h == "direct" else artifacts[h].get(task_id)) for h in HOSTS}
        focus = {h: render(h, drafts[h]) for h in HOSTS if drafts.get(h)}
        pool = build_pool(task_id, args.run, evidence, focus=focus)
        scoped = task.model_copy(update={"prior_work": pool_prior_work(pool), "strict_evidence": True})
        key = citation_key(task, pool) if host == "ai_scientist" else None
        text = render(host, drafts[host], citation_key=key)
        runner = RecordedRunner(endpoint, args.out / "requests" / task_id / host,
                                task_id=task_id, arm=host, stage="renormalize")
        try:
            second = normalize_artifact(scoped, text, runner)
        except Exception as exc:  # noqa: BLE001
            print(f"[stability] FAILED {task_id[:30]}/{host}: {type(exc).__name__}")
            continue
        first_claims = [c["claim"] for c in payload["proposal"].get("prior_facts", [])]
        second_claims = [c.claim for c in second.prior_facts]
        score = overlap(first_claims, second_claims)
        rows.append({"task_id": task_id, "host": host, "first": len(first_claims),
                     "second": len(second_claims), "agreement": round(score, 3)})
        print(f"[stability] {task_id[:34]:34s} {host:13s} first={len(first_claims):2d} "
              f"second={len(second_claims):2d} agreement={score:.2f}", flush=True)

    mean = sum(row["agreement"] for row in rows) / len(rows) if rows else 0.0
    write_json(args.out / "stability.json", {"rows": rows, "mean_agreement": round(mean, 3)})
    print(f"[stability] mean agreement={mean:.3f} over {len(rows)} artifacts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
