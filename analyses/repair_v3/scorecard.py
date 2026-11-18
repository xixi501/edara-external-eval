"""Structural scorecard from the shared-pool audit bundles."""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

ELEMENTS = {"baseline_alignment", "metric_or_decision_rule", "failure_regime", "dataset_and_split"}
HOSTS = ("direct", "ai_scientist", "edara")


def load(path: Path):
    return json.loads(path.read_text())


def metrics(bundle: dict) -> dict:
    tri = bundle.get("tri_state") or {}
    audits = tri.get("prior_fact_audits") or []
    total = len(audits) or 0
    supported = sum(1 for row in audits if row.get("citation_status") in
                    ("verified_local", "verified_pre_cutoff"))
    unverified = sum(1 for row in audits if row.get("citation_status") == "unverified")
    fabricated = sum(1 for row in audits if row.get("citation_status") == "fabricated_or_missing")
    spec = bundle.get("spec_audit") or {}
    blocked = {row.get("category") for row in (spec.get("defects") or [])
               if row.get("severity") == "blocking"}
    covered = len(ELEMENTS - blocked)
    gate = bundle.get("gate") or {}
    compliance = bool(gate.get("gate_pass")) and fabricated == 0 and covered >= 3
    return {
        "claims": total,
        "support_rate": (supported / total) if total else None,
        "unverified_rate": (unverified / total) if total else None,
        "fabricated_rate": (fabricated / total) if total else None,
        "overclaims": int(tri.get("epistemic_overclaim_count") or 0),
        "max_smoothing": int(tri.get("max_smoothing_score") or 1),
        "spec_completeness": float(spec.get("completeness_score") or 0.0),
        "elements_covered": covered,
        "blocking": int(spec.get("blocking_count") or 0),
        "gap_survival": float((bundle.get("gap_audit") or {}).get("survival_rate") or 0.0),
        "gate_pass": bool(gate.get("gate_pass")),
        "compliant": compliance,
    }


def bootstrap(values: list[float], draws: int = 4000, seed: int = 20261117) -> tuple[float, float]:
    if not values:
        return (0.0, 0.0)
    rng = random.Random(seed)
    means = []
    n = len(values)
    for _ in range(draws):
        means.append(sum(rng.choice(values) for _ in range(n)) / n)
    means.sort()
    return (means[int(0.025 * draws)], means[int(0.975 * draws)])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    per_host: dict[str, list[dict]] = defaultdict(list)
    per_task: dict[str, dict[str, dict]] = defaultdict(dict)
    for path in sorted((args.audit / "bundles").glob("*/*.json")):
        payload = load(path)
        if "tri_state" not in payload:
            continue
        host, task = path.stem, path.parent.name
        row = metrics(payload)
        row.update({"task_id": task, "host": host})
        per_host[host].append(row)
        per_task[task][host] = row

    lines = ["# 结构指标记分卡（同一证据池，15 题）", ""]
    lines += ["| 宿主 | 产物 | 断言支持率 | 未核实率 | 查无此文率 | 四要素覆盖 | 合规产物 |",
              "|---|---:|---|---|---|---:|---:|"]
    summary = {}
    for host in HOSTS:
        rows = per_host.get(host, [])
        if not rows:
            continue
        supports = [row["support_rate"] for row in rows if row["support_rate"] is not None]
        mean = sum(supports) / len(supports) if supports else 0.0
        low, high = bootstrap(supports)
        unver = sum(row["unverified_rate"] or 0 for row in rows) / len(rows)
        fab = sum(row["fabricated_rate"] or 0 for row in rows) / len(rows)
        elements = sum(row["elements_covered"] for row in rows) / len(rows)
        compliant = sum(1 for row in rows if row["compliant"])
        summary[host] = {"n": len(rows), "support": mean, "ci": (low, high),
                         "unverified": unver, "fabricated": fab, "elements": elements,
                         "compliant": compliant}
        lines.append(f"| {host} | {len(rows)} | {mean:.3f} [{low:.3f}, {high:.3f}] | {unver:.3f} | "
                     f"{fab:.3f} | {elements:.2f}/4 | {compliant}/{len(rows)} |")

    lines += ["", "## 与 Direct 的配对差", "", "| 对比 | 断言支持率差 | 四要素差 | 断言数 |", "|---|---|---|---:|"]
    base = {row["task_id"]: row for row in per_host.get("direct", [])}
    for host in HOSTS[1:]:
        diffs_support, diffs_elements, pairs = [], [], 0
        for row in per_host.get(host, []):
            other = base.get(row["task_id"])
            if not other or row["support_rate"] is None or other["support_rate"] is None:
                continue
            diffs_support.append(row["support_rate"] - other["support_rate"])
            diffs_elements.append((row["elements_covered"] - other["elements_covered"]) / 4)
            pairs += 1
        if not diffs_support:
            continue
        s_mean = sum(diffs_support) / len(diffs_support)
        s_low, s_high = bootstrap(diffs_support)
        e_mean = sum(diffs_elements) / len(diffs_elements)
        e_low, e_high = bootstrap(diffs_elements)
        lines.append(f"| {host} − direct | {s_mean:+.3f} [{s_low:+.3f}, {s_high:+.3f}] | "
                     f"{e_mean:+.2f} [{e_low:+.2f}, {e_high:+.2f}] | {pairs} |")

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "scorecard.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    (args.out / "SCORECARD_ZH.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
