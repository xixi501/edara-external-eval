"""Build the expansion task cache: unused Lit2Test rows plus unused AlphaIdeaBench tasks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from edara_mvp_eval.schemas import TaskSpec
from edara_mvp_eval.tasks_loader import (
    AIB_ROOT,
    LIT2TEST_DATA,
    build_mvp15_tasks,
    load_alphaideabench_tasks,
    load_lit2test_tasks,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--lit2test", type=int, default=20)
    parser.add_argument("--alphaideabench", type=int, default=5)
    args = parser.parse_args(argv)

    core = build_mvp15_tasks()
    used = {task.task_id for task in core}

    rows = [json.loads(line) for line in LIT2TEST_DATA.read_text(encoding="utf-8").splitlines() if line.strip()]
    free_rows = sorted((row for row in rows if str(row.get("context_id")) not in used),
                       key=lambda row: str(row.get("context_id")))
    chosen_rows = free_rows[: args.lit2test]
    anchors = tuple(str(row.get("field") or "").lower() for row in chosen_rows)
    lit_tasks = load_lit2test_tasks(anchors=anchors)
    wanted = {str(row["context_id"]) for row in chosen_rows}
    if {task.task_id for task in lit_tasks} != wanted:
        raise SystemExit("Lit2Test anchor selection did not reproduce the chosen rows")

    index = json.loads((AIB_ROOT / "tasks/index.json").read_text(encoding="utf-8"))
    free_ids = sorted(item["id"] for item in index if item["id"] not in used)
    aib_ids = tuple(free_ids[: args.alphaideabench])
    aib_tasks = load_alphaideabench_tasks(task_ids=aib_ids)

    tasks: list[TaskSpec] = [*lit_tasks, *aib_tasks]
    ids = [task.task_id for task in tasks]
    if len(set(ids)) != len(ids) or set(ids) & used:
        raise SystemExit("expansion task ids are not unique or overlap the core set")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps([task.model_dump() for task in tasks], ensure_ascii=False, indent=2) + "\n")
    print(f"[expand] wrote {len(tasks)} tasks -> {args.out}")
    for task in tasks:
        print(f"  {task.source_track:14s} {task.task_id[:58]:58s} cutoff={task.cutoff_date} "
              f"prior={len(task.prior_work)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
