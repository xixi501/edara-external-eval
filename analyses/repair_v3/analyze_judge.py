"""Summarise a re-evaluation run: pass rates, stability, and review-failure mix."""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path


def load(path: Path):
    return json.loads(path.read_text())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--label", default="v3")
    args = parser.parse_args(argv)
    payload = load(args.run / "results.json")
    arms = [name for name in payload["arms"]]
    repeat = payload["repeat"]
    rows = payload["results"]

    table = {}
    for arm in arms:
        per = [sum(1 for row in rows if row["arms"][arm].get("status") == "assessed"
                   and row["arms"][arm]["repeats"][i].get("joint_pass")) for i in range(repeat)]
        entries = [row["arms"][arm] for row in rows if row["arms"][arm].get("status") == "assessed"]
        stable = sum(1 for entry in entries if all(d.get("joint_pass") for d in entry["repeats"]))
        never = sum(1 for entry in entries if not any(d.get("joint_pass") for d in entry["repeats"]))
        table[arm] = {"per_draw": per, "mean": round(sum(per) / repeat, 2),
                      "stable_pass": stable, "stable_fail": never, "assessed": len(entries)}

    axis = collections.Counter()
    validations = collections.Counter()
    aspects = collections.Counter()
    for row in rows:
        for arm in arms:
            entry = row["arms"][arm]
            if entry.get("status") != "assessed":
                continue
            for draw in entry["repeats"]:
                for name, value in (draw.get("axes") or {}).items():
                    axis[(name, value.get("status"))] += 1
                    if value.get("validation"):
                        validations[value["validation"]] += 1
                    if name == "novelty" and value.get("status") == "pass":
                        aspects[value.get("aspect_normalized") or "(none)"] += 1

    total_dims = sum(axis.values())
    stability = payload["stability"]
    summary = {
        "label": args.label, "rubric": payload.get("rubric"), "repeat": repeat,
        "arms": table,
        "axis_pass_rate": {name: round(axis[(name, "pass")] / max(1, sum(v for (n, _), v in axis.items() if n == name)), 3)
                           for name in ("honesty", "credibility", "novelty")},
        "axis_counts": {f"{name}:{status}": value for (name, status), value in axis.items()},
        "dimensions": total_dims,
        "review_failures": dict(validations),
        "joint_flip_rate": round(stability["joint_flips"] / max(1, stability["assessed"]), 3),
        "axis_flip_rate": round(stability["axis_flips"] / max(1, stability["axis_dims"]), 3),
        "aspects": dict(aspects.most_common(8)),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / f"{args.label}_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k not in ("axis_counts",)}, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
