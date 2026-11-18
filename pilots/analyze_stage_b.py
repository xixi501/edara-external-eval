"""Summarize paired axis transitions, effort and failures from a finished Stage B run."""

import argparse
import json
from collections import Counter
from pathlib import Path

from edara_mvp_eval.stage_b import ARMS
from edara_mvp_eval.stage_b_review import AXES


def load(path):
    return json.loads(path.read_text())


def status(row, arm, axis):
    verdict = row["verdicts"][arm]
    return verdict.get("axes", {}).get(axis, {}).get("status", verdict["status"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    rows = load(args.run / "results.json")
    usage = load(args.run / "usage.json")
    treated = ARMS[-1]
    output = {
        "n_tasks": len(rows),
        "completed_reviews": sum(v["status"] == "assessed" for r in rows for v in r["verdicts"].values()),
        "cohorts": {},
        "feedback_route_counts": {},
        "analysis_scope": "Single model reviewer; novelty relative to supplied evidence.",
        "exposure": "Expansion cases were used in the historical dry-run; no matching live rows "
                    "in the inspected historical pilot comparison files.",
    }

    for name, selection in [
        ("all", rows),
        ("development", [r for r in rows if r["development"]]),
        ("expansion", [r for r in rows if not r["development"]]),
    ]:
        output["cohorts"][name] = {"n": len(selection), "comparisons": {}}
        for base in ARMS[:-1]:
            result = {}
            for axis in AXES:
                changes = Counter(f"{status(r, base, axis)} -> {status(r, treated, axis)}" for r in selection)
                result[axis] = dict(changes)
            output["cohorts"][name]["comparisons"][base] = result

    output["feedback_route_counts"] = dict(Counter(
        step["action"] for r in rows for step in r["arms"][treated].get("history", [])
    ))
    output["tasks_with_failed_or_unresolved_feedback"] = [
        {"task_id": row["task_id"], "verdict": row["verdicts"][treated],
         "model_actions": [entry["action"] for entry in row["arms"][treated].get("history", [])]}
        for row in rows if not row["verdicts"][treated].get("joint_pass", False)
    ]
    production_stages = {"revision", "governance_audit"}
    output["production_cost"] = {
        arm: {
            "requests": sum(row["arm"] == arm and row["stage"] in production_stages for row in usage),
            "metered_usd": round(sum((row["usage"] or {}).get("cost_usd", 0)
                                     for row in usage
                                     if row["arm"] == arm and row["stage"] in production_stages), 6),
            "unknown_usage": sum(row["arm"] == arm and row["stage"] in production_stages
                                 and row["usage"] is None for row in usage),
        }
        for arm in ARMS
    }
    output["production_cost_notes"] = (
        "Direct governance_audit is the shared initial diagnosis. "
        "Initial generation and final review are excluded from this table; see usage.json. "
        "SDK retries inside one request are not individually metered."
    )
    output["search"] = {}
    for arm in ARMS:
        events = [event for r in rows for event in r["arms"][arm].get("search_events", []) if "url" in event]
        output["search"][arm] = dict(Counter(event["status"] for event in events))
    (args.run / "analysis.json").write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({k: output[k] for k in ("n_tasks", "completed_reviews", "feedback_route_counts",
                                             "production_cost")}, indent=2))


if __name__ == "__main__":
    main()
