"""Export a compact per-manuscript verdict table from a paired-trial run.

Usage: python analyses/export_results.py <run_dir> <out_csv>
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

AXES = ("honesty", "credibility", "novelty")
COMPARISONS = ("C_vs_B", "E_vs_D")


def main(run: Path, out: Path) -> None:
    report = json.loads((run / "core_results.json").read_text(encoding="utf-8"))
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["task_id", "host", "stage", "activation", "comparison", "axis",
                         "status", "detail", "delta", "band"])
        for row in report["results"]:
            for comparison in COMPARISONS:
                for axis in AXES:
                    case = row["comparisons"][comparison][axis]
                    writer.writerow([
                        row["task_id"], row["host"], row.get("stage") or "", row.get("activation") or "",
                        comparison, axis, case.get("status"), case.get("detail") or "",
                        case.get("delta", ""), case.get("unknown_band", ""),
                    ])
    print(f"wrote {out}")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
