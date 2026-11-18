"""Export AI-Scientist-v2 and E-DARA input bundles for a frozen task cache."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from edara_mvp_eval.schemas import TaskSpec
from edara_mvp_eval.tasks_loader import export_pilot3_external_inputs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks-cache", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    tasks = [TaskSpec.model_validate(row) for row in json.loads(args.tasks_cache.read_text())]
    export_pilot3_external_inputs(tasks, args.out)
    ai = args.out / "ai_scientist_topics"
    edara = args.out / "edara_cases"
    print(f"[export] tasks={len(tasks)} ai_scientist={len(list(ai.glob('*.md')))} "
          f"edara_cases={len(list(edara.glob('*.json')))}")
    for task in tasks:
        refs = list((edara / task.task_id / "inputs").glob("ref_*.md"))
        print(f"  {task.source_track:14s} {task.task_id[:56]:56s} refs={len(refs)} cutoff={task.cutoff_date}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
