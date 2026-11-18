"""Judge the frozen expansion candidates once each, under the shared evaluation rubric."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from edara_mvp_eval.governance import evidence_payload
from edara_mvp_eval.llm_client import ModelEndpoint
from edara_mvp_eval.schemas import PriorWorkItem, TaskSpec
from edara_mvp_eval.stage_b_evidence import write_json
from edara_mvp_eval.stage_b_meter import RecordedRunner
from edara_mvp_eval.stage_b_review import AXES, review_candidate, review_packet

ARMS = ("direct", "direct_self_refine", "direct_retrieve_revise", "direct_mw", "direct_feedback_mw")


def load(path: Path):
    return json.loads(path.read_text())


def build_pool(task: TaskSpec, rows: dict[str, dict]) -> list[dict]:
    items = list(task.prior_work)
    known = {item.id for item in items}
    for row in rows.values():
        for raw in row.get("evidence") or []:
            try:
                item = PriorWorkItem.model_validate(raw)
            except Exception:  # noqa: BLE001
                continue
            if item.id not in known:
                items.append(item)
                known.add(item.id)
    return evidence_payload(task, items)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--judge-profile", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--rubric", default="v3")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args(argv)

    tasks = [TaskSpec.model_validate(row) for row in load(args.run / "tasks_frozen.json")]
    endpoint = ModelEndpoint.from_env_or_profile(profile_path=args.judge_profile, prefix="EDARA_JUDGE")

    def judge_task(task: TaskSpec) -> dict:
        root = args.run / "tasks" / task.task_id
        rows = {}
        for arm in ARMS:
            path = root / f"{arm}.json"
            rows[arm] = load(path) if path.exists() else {"status": "missing"}
        pool = build_pool(task, {arm: row for arm, row in rows.items() if row.get("status") == "produced"})
        packet = review_packet(task, pool)
        verdicts = {}
        for arm in ARMS:
            row = rows[arm]
            if row.get("status") != "produced":
                verdicts[arm] = {"status": "no_candidate"}
                continue
            draws = []
            for index in range(args.repeat):
                runner = RecordedRunner(endpoint, args.out / "requests" / task.task_id / arm / f"r{index}",
                                        task_id=task.task_id, arm=arm, stage=f"final_verdict_r{index}")
                draws.append(review_candidate(packet, row["candidate"], runner, retry=True, rubric=args.rubric))
            verdicts[arm] = {"status": "assessed", "primary": draws[0], "repeats": draws}
        print(f"[expansion] {task.task_id[:44]:44s} pool={len(pool)}", flush=True)
        return {"task_id": task.task_id, "source": task.source_track, "pool": len(pool), "arms": verdicts}

    args.out.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool_exec:
        results = list(pool_exec.map(judge_task, tasks))
    write_json(args.out / "results.json", {"rubric": args.rubric, "repeat": args.repeat, "results": results})

    lines = ["# 扩题集判定（25 题 × 5 臂）", "",
             "| 臂 | 联合（抽样均值） | 诚实 P/F/U | 可信 P/F/U | 新颖 P/F/U |",
             "|---|---:|---|---|---|"]
    for arm in ARMS:
        entries = [row["arms"][arm] for row in results if row["arms"][arm].get("status") == "assessed"]
        joint = sum(1 for entry in entries
                    if any(draw.get("joint_pass") for draw in entry["repeats"]))
        cells = []
        for axis in AXES:
            counts = {"pass": 0, "fail": 0, "unresolved": 0}
            for entry in entries:
                status = entry["primary"].get("axes", {}).get(axis, {}).get("status")
                if status in counts:
                    counts[status] += 1
            cells.append(" / ".join(str(counts[s]) for s in ("pass", "fail", "unresolved")))
        lines.append(f"| {arm} | {joint}/{len(entries)} | " + " | ".join(cells) + " |")
    usage = [load(path) for path in sorted((args.out / "requests").glob("**/*.json"))]
    total = sum((row.get("usage") or {}).get("cost_usd", 0) for row in usage)
    lines.append(f"\n成本：${total:.4f}")
    (args.out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
