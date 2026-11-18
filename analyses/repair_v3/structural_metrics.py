"""Compute structural metrics from the audit bundles (no model calls)."""

from __future__ import annotations

import argparse
import collections
import json
import random
from pathlib import Path

CATEGORIES = ("baseline_alignment", "metric_or_decision_rule", "failure_regime", "dataset_and_split")
HOSTS = ("direct", "ai_scientist", "edara")


def load(path: Path):
    return json.loads(path.read_text())


def metrics(bundle: dict) -> dict:
    tri = bundle.get("tri_state") or {}
    audits = tri.get("prior_fact_audits") or []
    total = len(audits)
    supported = sum(1 for row in audits if row.get("citation_status") in ("verified_local", "verified_pre_cutoff"))
    unverified = sum(1 for row in audits if row.get("citation_status") == "unverified")
    fabricated = sum(1 for row in audits if row.get("citation_status") == "fabricated_or_missing")
    spec = bundle.get("spec_audit") or {}
    blocking = [defect for defect in (spec.get("defects") or [])
                if defect.get("severity") == "blocking"]
    blocked_categories = {defect.get("category") for defect in blocking}
    coverage = sum(1 for category in CATEGORIES if category not in blocked_categories)
    gap = bundle.get("gap_audit") or {}
    gate = bundle.get("gate") or {}
    joint = bool(gate.get("gate_pass")) and fabricated == 0 and coverage >= 3 and len(blocking) == 0
    return {
        "claims": total,
        "support_rate": supported / total if total else None,
        "unverified_rate": unverified / total if total else None,
        "fabricated_rate": fabricated / total if total else None,
        "overclaim": tri.get("epistemic_overclaim_count", 0),
        "max_smoothing": tri.get("max_smoothing_score", 1),
        "spec_completeness": spec.get("completeness_score"),
        "coverage": coverage,
        "blocking": len(blocking),
        "gap_survival": gap.get("survival_rate"),
        "gate_pass": bool(gate.get("gate_pass")),
        "joint_compliant": joint,
    }


def mean(values):
    values = [value for value in values if value is not None]
    return sum(values) / len(values) if values else None


def bootstrap(values, *, draws=2000, seed=20261117):
    values = [value for value in values if value is not None]
    if len(values) < 2:
        return None
    rng = random.Random(seed)
    means = sorted(sum(rng.choices(values, k=len(values))) / len(values) for _ in range(draws))
    return means[int(0.025 * draws)], means[int(0.975 * draws)]


def summarise(rows: list[dict]) -> dict:
    out = {"n": len(rows)}
    for key in ("support_rate", "unverified_rate", "fabricated_rate", "coverage",
                "spec_completeness", "gap_survival", "overclaim", "max_smoothing", "blocking"):
        values = [row[key] for row in rows]
        out[key] = mean(values)
        interval = bootstrap(values)
        out[f"{key}_ci"] = interval
    out["joint_rate"] = mean([1.0 if row["joint_compliant"] else 0.0 for row in rows])
    out["claims_total"] = sum(row["claims"] for row in rows)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    rows: dict[str, list[dict]] = collections.defaultdict(list)
    for path in sorted((args.audit / "bundles").glob("*/*.json")):
        bundle = load(path)
        if bundle.get("status") == "failed":
            continue
        host = path.stem
        if host not in HOSTS:
            continue
        rows[host].append({"task": path.parent.name, **metrics(bundle)})

    summary = {host: summarise(rows[host]) for host in HOSTS if rows[host]}
    (args.out).parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"summary": summary,
                                    "documents": {h: rows[h] for h in rows}}, ensure_ascii=False, indent=2) + "\n")

    def fmt(value, digits=2):
        return "—" if value is None else f"{value:.{digits}f}"

    def ci(host, key):
        interval = summary[host].get(f"{key}_ci")
        return "—" if not interval else f"{interval[0]:.2f}–{interval[1]:.2f}"

    lines = ["# 结构指标（审计派生，零额外模型调用）", "",
             "| 宿主 | 文档数 | 断言数 | 断言支持率 | 未核实率 | 查无此文率 | 四要素覆盖(0–4) | 规范完整度 | 联合合规率 |",
             "|---|---:|---:|---|---|---|---|---|---:|"]
    for host in HOSTS:
        if host not in summary:
            continue
        row = summary[host]
        lines.append(
            f"| {host} | {row['n']} | {row['claims_total']} | {fmt(row['support_rate'])} "
            f"[{ci(host, 'support_rate')}] | {fmt(row['unverified_rate'])} | {fmt(row['fabricated_rate'])} | "
            f"{fmt(row['coverage'])} | {fmt(row['spec_completeness'])} | {fmt(row['joint_rate'])} |")
    lines += ["", "| 宿主 | 过度断言(均) | 最大平滑分 | 缺口存活率 | 阻塞缺陷(均) |",
              "|---|---:|---:|---:|---:|"]
    for host in HOSTS:
        if host not in summary:
            continue
        row = summary[host]
        lines.append(f"| {host} | {fmt(row['overclaim'])} | {fmt(row['max_smoothing'], 1)} | "
                     f"{fmt(row['gap_survival'])} | {fmt(row['blocking'], 1)} |")
    (args.out.parent / "STRUCTURAL_ZH.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
