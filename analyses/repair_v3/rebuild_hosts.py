"""Rebuild the cross-host table from saved judge responses (no model calls)."""

from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

ROOT = Path("/workspace/edara/staging/04_external_eval_refs/edara_mvp_eval")
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "pilots"))

from edara_mvp_eval.canonical_render import render  # noqa: E402
from edara_mvp_eval.schemas import TaskSpec  # noqa: E402
from edara_mvp_eval.stage_b_review import AXES, Verdict, _assess, review_packet  # noqa: E402
from run_host_eval import build_pool, citation_key, jsonl, load  # noqa: E402

RUN = ROOT / "outputs/mvp15_ablation_v2"
HOSTS_DIR = ROOT / "outputs/hosts_v1"
OUT = ROOT / "outputs/repair_v3/hosts"
HOSTS = ("direct", "ai_scientist", "edara")


def assess(task_id: str, host: str, index: int, packet: dict, text: str):
    base = OUT / "requests" / task_id / host / f"r{index}"
    first = base / "0001.json"
    if not first.exists():
        return None
    try:
        verdict = Verdict.model_validate(json.loads(json.loads(first.read_text())["response"]))
    except (json.JSONDecodeError, ValueError):
        return None
    rows = _assess(verdict, packet, text, "v3")
    second = base / "0002.json"
    if second.exists():
        try:
            retry = Verdict.model_validate(json.loads(json.loads(second.read_text())["response"]))
        except (json.JSONDecodeError, ValueError):
            retry = None
        retry_rows = _assess(retry, packet, text, "v3") if retry else None
        for axis in AXES:
            if retry_rows and rows[axis].get("validation") and not retry_rows[axis].get("validation"):
                rows[axis] = retry_rows[axis]
    return {"axes": rows, "joint_pass": all(row["status"] == "pass" for row in rows.values())}


def main() -> int:
    tasks = [TaskSpec.model_validate(row) for row in load(RUN / "tasks_frozen.json")]
    artifacts = {
        "direct": None,
        "ai_scientist": jsonl(HOSTS_DIR / "ai_scientist_v1/ai_scientist_15.jsonl"),
        "edara": jsonl(HOSTS_DIR / "edara_15.jsonl"),
    }
    evidence = {}
    for host, name in (("edara", "edara_15_evidence.json"),
                       ("ai_scientist", "ai_scientist_15_evidence.json")):
        path = HOSTS_DIR / name
        evidence[host] = load(path) if path.exists() else {}
    results = []
    for task in tasks:
        drafts = {}
        for host in HOSTS:
            drafts[host] = (load(RUN / "tasks" / task.task_id / "direct.json").get("candidate")
                            if host == "direct" else artifacts[host].get(task.task_id))
        focus = {host: render(host, drafts[host]) for host in HOSTS if drafts.get(host)}
        pool = build_pool(task.task_id, RUN, evidence, focus=focus)
        packet = review_packet(task, pool)
        row = {"task_id": task.task_id, "hosts": {}}
        for host in HOSTS:
            draft = drafts.get(host)
            if not draft:
                row["hosts"][host] = {"status": "no_artifact"}
                continue
            key = citation_key(task, pool) if host == "ai_scientist" else None
            text = render(host, draft, citation_key=key)
            draws = [assess(task.task_id, host, index, packet, text) for index in range(3)]
            if any(draw is None for draw in draws):
                row["hosts"][host] = {"status": "invalid_review"}
                continue
            row["hosts"][host] = {"status": "assessed", "chars": len(text),
                                 "primary": draws[0], "repeats": draws}
        results.append(row)

    stability = {"assessed": 0, "joint_flips": 0, "axis_flips": 0, "dims": 0}
    for row in results:
        for host in HOSTS:
            entry = row["hosts"][host]
            if entry.get("status") != "assessed":
                continue
            stability["assessed"] += 1
            draws = entry["repeats"]
            if len({draw["joint_pass"] for draw in draws}) > 1:
                stability["joint_flips"] += 1
            for axis in AXES:
                stability["dims"] += 1
                if len({draw["axes"][axis]["status"] for draw in draws}) > 1:
                    stability["axis_flips"] += 1

    payload = {"rubric": "v3", "repeat": 3, "hosts": list(HOSTS), "results": results,
               "stability": stability, "rebuilt_from_saved_responses": True}
    (OUT / "results.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")

    lines = ["# 跨宿主对照（15 题，判定 v3，三次抽样）", "",
             "同一题所有宿主共用同一份冻结证据池。E-DARA 缺失记为 no_artifact，不补造。", "",
             "| 宿主 | 联合（三次抽样） | 诚实 P/F/U | 可信 P/F/U | 新颖 P/F/U | 有产物 |",
             "|---|---:|---|---|---|---:|"]
    for host in HOSTS:
        entries = [row["hosts"][host] for row in results if row["hosts"][host].get("status") == "assessed"]
        per = [sum(1 for entry in entries if entry["repeats"][i]["joint_pass"]) for i in range(3)]
        cells = []
        for axis in AXES:
            counts = collections.Counter(entry["primary"]["axes"][axis]["status"] for entry in entries)
            cells.append(" / ".join(str(counts[s]) for s in ("pass", "fail", "unresolved")))
        lines.append(f"| {host} | {' / '.join(map(str, per))} | " + " | ".join(cells) + f" | {len(entries)} |")
    lines += ["", f"- 三次抽样联合翻转：{stability['joint_flips']}/{stability['assessed']}",
              f"- 轴级翻动：{stability['axis_flips']}/{stability['dims']}", ""]
    (OUT / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
