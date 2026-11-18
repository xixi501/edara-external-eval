"""Verify frozen inputs and completed comparisons in a v1 paired trial."""

import argparse
import json
from collections import Counter
from pathlib import Path

from edara_mvp_eval.paired_trial import ARMS, COMPARISONS, LIMITS
from edara_mvp_eval.stage_b_evidence import digest

EXPECTED = {"pilot": (3, 9, 0), "core": (15, 38, 7), "extension": (25, 25, 0)}


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def verify(run: Path, cohort: str) -> dict:
    frozen = read(run / f"inputs_{cohort}.json")
    data = read(run / f"{cohort}_results.json")
    expected = EXPECTED[cohort]
    if (len(frozen), len(data["results"]), len(data["missing"])) != expected:
        raise ValueError(f"Unexpected frozen/assessed/missing counts for {cohort}")
    seen = set()
    states = {cmp: {axis: Counter() for axis in ("honesty", "credibility", "novelty")} for cmp in COMPARISONS}
    metering = {arm: {"requests": 0, "shared_audit_requests": 0, "overlap_records": 0,
                      "recorded_cost_usd": 0.0} for arm in ARMS[1:]}
    for row in data["results"]:
        task, host = row["task_id"], row["host"]
        if (task, host) in seen:
            raise ValueError(f"Duplicate result for {task}/{host}")
        seen.add((task, host))
        case = run / "versions" / task / host
        versions = {arm: read(case / f"{arm}.json") for arm in ARMS}
        original = frozen[task]["originals"][host]
        if versions["A"] != original:
            raise ValueError(f"Original changed at {task}/{host}")
        if len({versions[arm]["initial_evidence_sha256"] for arm in ARMS[1:]}) != 1:
            raise ValueError(f"Starting evidence mismatch at {task}/{host}")
        for arm in ARMS[1:]:
            result = versions[arm]
            limits = LIMITS["local" if arm in ("B", "C") else "full"]
            if digest(result["draft"]) != result["sha256"] or digest(result["draft"]) != row["versions"][arm]["sha256"]:
                raise ValueError(f"Draft hash mismatch for {task}/{host}/{arm}")
            if len(result["history"]) > limits["rounds"] or len(result["queries"]) > limits["queries"]:
                raise ValueError(f"Action budget exceeded for {task}/{host}/{arm}")
            if arm in ("B", "C") and sum(len(step.get("edits", [])) for step in result["history"]) > 4:
                raise ValueError(f"Edit budget exceeded for {task}/{host}/{arm}")
            editor_requests = list((case / arm / "editor").glob("[0-9][0-9][0-9][0-9].json"))
            audit_requests = list((case / arm / "audit").glob("[0-9][0-9][0-9][0-9].json"))
            shared_count = 0
            if arm == "E" and any(step.get("audit_owner") == "C" for step in result["history"]):
                shared_count = len(list((case / "C" / "audit").glob("[0-9][0-9][0-9][0-9].json")))
            if len(editor_requests) > limits["editor"]:
                raise ValueError(f"Editor call budget exceeded for {task}/{host}/{arm}")
            if len(audit_requests) > limits["audit"]:
                raise ValueError(f"Auditor call budget exceeded for {task}/{host}/{arm}")
            metering[arm]["shared_audit_requests"] += shared_count
            for request in editor_requests + audit_requests:
                metering[arm]["requests"] += 1
                usage = read(request).get("usage") or {}
                metering[arm]["overlap_records"] += usage.get("calls", 0) > 1
                metering[arm]["recorded_cost_usd"] += usage.get("cost_usd", 0.0)
        record = read(case / "review.json")
        for label in COMPARISONS:
            result = record[label]
            if digest(result["evidence"]) != result["evidence_sha256"]:
                raise ValueError(f"Review evidence hash mismatch for {task}/{host}/{label}")
            if result["arms"] != row["comparisons"][label]:
                raise ValueError(f"Review summary diverged for {task}/{host}/{label}")
            for axis, value in result["arms"].items():
                expected_delta = {"win": lambda delta: delta > 0, "loss": lambda delta: delta < 0,
                                  "tie": lambda delta: delta == 0}
                if value["status"] in expected_delta and not expected_delta[value["status"]](value["delta"]):
                    raise ValueError(f"Wrong sign for {task}/{host}/{label}/{axis}")
                states[label][axis][value["status"]] += 1
    return {"cohort": cohort, "topics": expected[0], "originals": expected[1], "missing": expected[2],
            "comparisons": states, "metering": metering,
            "metering_note": "Counts use unique request records. A nonzero overlap_records means recorded cost "
                             "double-counts overlapping usage and cannot serve as an exact cost estimate."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--cohort", choices=EXPECTED, action="append")
    args = parser.parse_args()
    for cohort in args.cohort or EXPECTED:
        print(json.dumps(verify(args.run, cohort), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
