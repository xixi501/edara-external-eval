"""Offline acceptance replay for the Stage B measurement repair.

Runs against saved records only: no network, no model calls, no writes outside the
replay output directory. Produces acceptance.json and REPORT_ZH.md.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sqlite3
import sys
import urllib.parse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from edara_mvp_eval.feedback import PolicyState, eligible_actions  # noqa: E402
from edara_mvp_eval.schemas import FullAuditBundle, TaskSpec  # noqa: E402
from edara_mvp_eval.stage_b_evidence import _relevant, digest  # noqa: E402
from edara_mvp_eval.stage_b_review import (  # noqa: E402
    candidate_text,
    locate_span,
    normalize,
    quote_in_passage,
    review_packet,
)

RUN = ROOT / "outputs/mvp15_ablation_v2"
EXPORTS = ROOT / "outputs/repair_v2/exports"
ATTRIBUTION = ROOT / "analyses/failure_attribution_20261117/evidence.json"
BASELINE = Path("/workspace/edara/staging/_repair_v2_backup_20261117/baseline.sha256")


def load(path: Path):
    return json.loads(path.read_text())


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def check_e1() -> dict:
    results = []
    unique_evidence = set()
    for artifact_path in sorted(EXPORTS.glob("*.artifact.json")):
        artifact = load(artifact_path)
        text = artifact["text"]
        inverse = re.sub(r"\[S\d+\]", "[REF]", text)
        db = Path(artifact["source_db"])
        sections = {}
        if db.exists():
            connection = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
            connection.execute("PRAGMA query_only=ON")
            sections = {row[0]: row[1] for row in
                        connection.execute("select section_id, text from sections")}
            connection.close()
        missing_quotes = 0
        quotes = 0
        for marker, source in artifact["citation_map"].items():
            unique_evidence.update(source["evidence_ids"])
            for evidence_id, quote in source["quotes"].items():
                quotes += 1
                section_id = (source.get("section_ids") or {}).get(evidence_id, source.get("section_id", ""))
                section = sections.get(section_id, "")
                if not display(quote, section):
                    missing_quotes += 1
        results.append({
            "task_id": artifact["task_id"], "stage": artifact["stage"],
            "refs": artifact["proof"]["refs"], "sources": artifact["proof"]["sources"],
            "inverse_matches_original": inverse == original_line(artifact["task_id"]),
            "quotes": quotes, "quotes_not_located": missing_quotes,
            "markers": len(re.findall(r"\[S\d+\]", text)),
            "ref_placeholders_left": text.count("[REF]"),
        })
    return {"tasks": results, "unique_evidence_ids": len(unique_evidence),
            "unique_sources": sum(row["sources"] for row in results),
            "all_inverse": all(row["inverse_matches_original"] for row in results),
            "all_quotes_located": all(row["quotes_not_located"] == 0 for row in results)}


def display(quote: str, section: str) -> bool:
    if not section:
        return False
    return quote_in_passage(quote, section) or normalize(quote) in normalize(section)


def original_line(task_id: str) -> str:
    path = ROOT.parent / "pilot3_runs/edara/edara_outputs.jsonl"
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip() and json.loads(line).get("task_id") == task_id:
            return line
    raise KeyError(task_id)


def candidate_for(task_id: str, arm: str):
    row = load(RUN / "tasks" / task_id / f"{arm}.json")
    return row.get("candidate")


def check_e2() -> dict:
    issues = load(ATTRIBUTION)["stage_b"]["anchor_issues"]
    rows = []
    for issue in issues:
        candidate = candidate_for(issue["task_id"], issue["arm"])
        if candidate is None:
            continue
        found, mode = locate_span(candidate_text(candidate), issue["candidate_span"])
        rows.append({"edge_fixable": issue["invalid_span"] and issue["span_matches_after_edge_quotes_removed"],
                     "invalid_span": issue["invalid_span"],
                     "quote_issue": bool(issue["invalid_evidence_quotes"]),
                     "found": found, "mode": mode})
    edge = [row for row in rows if row["edge_fixable"]]
    pure = [row for row in edge if not row["quote_issue"]]
    quote_rows = [row for row in rows if row["quote_issue"]]
    quote_total = quote_recovered = 0
    for issue in issues:
        if not issue["invalid_evidence_quotes"]:
            continue
        frozen_path = RUN / "tasks" / issue["task_id"] / "evaluation_evidence.json"
        if not frozen_path.exists():
            continue
        passages = {row["paper_id"]: row["passages"] for row in load(frozen_path)["evidence"]}
        for reference in issue["invalid_evidence_quotes"]:
            quote_total += 1
            if quote_in_passage(reference["quote"], passages.get(reference["paper_id"], "")):
                quote_recovered += 1
    negative = []
    for issue in issues[:20]:
        candidate = candidate_for(issue["task_id"], issue["arm"])
        span = issue["candidate_span"]
        if candidate is None or not span:
            continue
        mutated = span[: max(1, len(span) // 2)] + " zzq " + span[max(1, len(span) // 2):]
        found, _ = locate_span(candidate_text(candidate), mutated)
        negative.append(found)
    return {
        "issues": len(rows),
        "edge_fixable": len(edge),
        "edge_fixable_located": sum(1 for row in edge if row["found"]),
        "pure_edge_cases": len(pure),
        "pure_edge_located": sum(1 for row in pure if row["found"]),
        "quote_issue_rows": len(quote_rows),
        "quote_rows": len(quote_rows),
        "quote_total": quote_total,
        "quote_recovered": quote_recovered,
        "negative_control_false_accepts": sum(negative),
        "negative_control_tested": len(negative),
        "modes": sorted({row["mode"] for row in rows}),
    }


def check_e3() -> dict:
    tasks = [TaskSpec.model_validate(row) for row in load(RUN / "tasks_frozen.json")]
    task = tasks[0]
    frozen = load(RUN / "tasks" / task.task_id / "evaluation_evidence.json")
    packet_a = json.dumps(review_packet(task, frozen["evidence"]), ensure_ascii=False, sort_keys=True)
    packet_b = json.dumps(review_packet(task, list(reversed(frozen["evidence"]))), ensure_ascii=False,
                          sort_keys=True)
    candidate = candidate_for(task.task_id, "direct")
    shuffled = dict(reversed(list(candidate.items())))
    return {
        "packet_stable_under_evidence_order": sha256_text(packet_a) == sha256_text(packet_b),
        "candidate_text_stable_under_key_order": candidate_text(candidate) == candidate_text(shuffled),
        "prompt_sha256": sha256_text(packet_a),
        "evidence_rows": len(frozen["evidence"]),
    }


def check_e4() -> dict:
    rows = []
    for path in sorted((RUN / "tasks").glob("*/direct.json")):
        row = load(path)
        if not row.get("audit"):
            continue
        bundle = FullAuditBundle.model_validate(row["audit"])
        fresh = [name for name, _ in eligible_actions(bundle)]
        after_search = [name for name, _ in eligible_actions(bundle, PolicyState(searched=True))]
        history = load(path.parent / "direct_feedback_mw.json").get("history", []) if (
            path.parent / "direct_feedback_mw.json").exists() else []
        historical = sorted({entry.get("action") for entry in history if entry.get("action")})
        rows.append({"task_id": path.parent.name,
                     "blocking_spec": bundle.spec_audit.blocking_count,
                     "initial_action": bundle.governance.action if bundle.governance else None,
                     "fresh_eligible": fresh,
                     "after_search_eligible": after_search,
                     "historical_actions": historical})
    with_blocking = [row for row in rows if row["blocking_spec"]]
    return {
        "tasks": len(rows),
        "tasks_with_blocking_spec": len(with_blocking),
        "clarify_eligible_now": sum(1 for row in with_blocking if "clarify" in row["fresh_eligible"]),
        "clarify_eligible_after_search": sum(1 for row in with_blocking
                                             if "clarify" in row["after_search_eligible"]),
        "historical_non_search_actions": sum(1 for row in rows
                                             if set(row["historical_actions"]) & {"clarify", "local_patch",
                                                                                  "regenerate"}),
        "detail": rows,
    }


GREP_PATTERNS = {
    "datacite": re.compile(r"query=titles\.title:\("),
    "crossref": re.compile(r"query\.(?:bibliographic|title)="),
}


def parse_query(url: str) -> str:
    parsed = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)
    for key in ("query", "query.bibliographic", "query.title"):
        value = parsed.get(key)
        if value:
            text = value[0]
            text = re.sub(r"AND prefix:.*$", "", text)
            text = text.replace("titles.title:(", "").strip('()" ')
            return text
    return ""


def cached_items(record: dict) -> list[tuple[str, str]]:
    data = record.get("data") or {}
    rows = data.get("data") if isinstance(data.get("data"), list) else data.get("message", {}).get("items", [])
    items = []
    for row in rows or []:
        attributes = row.get("attributes", row)
        titles = attributes.get("titles") or []
        title = titles[0].get("title", "") if titles else (attributes.get("title") or [""])[0] if isinstance(
            attributes.get("title"), list) else attributes.get("title", "")
        abstract = ""
        for block in attributes.get("descriptions", []) or []:
            if block.get("descriptionType") == "Abstract":
                abstract = block.get("description", "")
                break
        if not abstract:
            abstract = attributes.get("abstract") or ""
        if title:
            items.append((title, re.sub(r"<[^>]*>", " ", abstract)))
    return items


def check_e5() -> dict:
    known_bad = [
        ("The Eviction Economy", "A review of housing eviction and reform"),
        ("Generative Adversarial Network Based on Self-Attention Mechanism for Automatic Page Layout "
         "Generation", "We generate page layouts with a conditional GAN"),
        ("A Proposal for Legalizing Small-Scale Physical Copyright Piracy", "Book publishing and films"),
    ]
    query = "KV cache compression token eviction long-context inference"
    rejected_known = sum(1 for title, abstract in known_bad if not _relevant(query, title, abstract))
    admitted = rejected = 0
    for task_dir in sorted((RUN / "tasks").glob("*")):
        frozen_path = task_dir / "evaluation_evidence.json"
        if not frozen_path.exists():
            continue
        titles = {normalize(row["title"]).lower() for row in load(frozen_path)["evidence"]}
        for arm_path in sorted(task_dir.glob("*.json")):
            if arm_path.name == "evaluation_evidence.json":
                continue
            for event in load(arm_path).get("search_events", []) or []:
                url = event.get("url", "")
                query_text = parse_query(url)
                if not query_text:
                    continue
                cached = task_dir / "lookup" / f"{digest(url)}.json"
                if not cached.exists():
                    continue
                for title, abstract in cached_items(load(cached)):
                    if normalize(title).lower() not in titles:
                        continue
                    admitted += 1
                    if not _relevant(query_text, title, abstract):
                        rejected += 1
    return {"known_false_positives": len(known_bad), "known_false_positives_rejected": rejected_known,
            "pool_items_replayed": admitted, "pool_items_now_rejected": rejected,
            "reject_ratio": (rejected / admitted) if admitted else 0.0}


def check_e7() -> dict:
    if not BASELINE.exists():
        return {"available": False}
    entries = {}
    for line in BASELINE.read_text().splitlines():
        parts = line.split()
        if len(parts) == 2 and len(parts[0]) == 64:
            entries[parts[1]] = parts[0]
    checks = {}
    for name in ("../pilot3_runs/edara/edara_outputs.jsonl", "outputs/mvp15_ablation_v2/REPORT_ZH.md",
                 "outputs/mvp15_ablation_v2/results.json", "outputs/pilot3_full_compare_v4/summary.md"):
        path = ROOT / name
        actual = hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None
        checks[name] = {"recorded": entries.get(name), "actual": actual,
                        "unchanged": entries.get(name) == actual}
    return {"available": True, "files": checks, "all_unchanged": all(v["unchanged"] for v in checks.values())}


def verdicts(report: dict) -> dict:
    e1, e2, e3, e4, e5, e7 = report["e1"], report["e2"], report["e3"], report["e4"], report["e5"], report["e7"]
    return {
        "E1": bool(e1["all_inverse"] and e1["all_quotes_located"] and e1["unique_evidence_ids"] == 23),
        "E2": bool(e2["edge_fixable_located"] == e2["edge_fixable"] == 22
                   and e2["pure_edge_located"] == e2["pure_edge_cases"] == 19
                   and e2["negative_control_false_accepts"] == 0),
        "E3": bool(e3["packet_stable_under_evidence_order"] and e3["candidate_text_stable_under_key_order"]),
        "E4": bool(e4["tasks_with_blocking_spec"] and
                   e4["clarify_eligible_after_search"] == e4["tasks_with_blocking_spec"]),
        "E5": bool(e5["known_false_positives_rejected"] == 3 and e5["reject_ratio"] <= 0.10),
        "E6": bool(report["e6"]["passed"] >= 55),
        "E7": bool(e7.get("all_unchanged", False)),
    }


def render(report: dict) -> str:
    e1, e2, e4, e5, e7 = report["e1"], report["e2"], report["e4"], report["e5"], report["e7"]
    lines = ["# 测量与机制修复：离线验收回放", "",
             "日期：2026-11-17。全部检查只读保存记录，不使用网络和模型调用。", "",
             "## 验收结论", "",
             "| 项 | 判据 | 结果 |", "|---|---|---|"]
    rules = {
        "E1": "三份文本逆变换等于原行；23 个唯一 evidence ID；引文全部可在原生 section 定位",
        "E2": "22/22 首尾引号 span 可定位；19/19 纯格式案例恢复原判；0 个负样本误放行",
        "E3": "证据换序与字段换序不改变评价输入",
        "E4": "带 blocking spec 的题目在搜证耗尽后仍可选到非搜证动作",
        "E5": "3/3 已知误匹配被拒；入池召回损失 ≤ 10%",
        "E6": "pytest ≥ 55 passed",
        "E7": "历史产物 sha256 不变",
    }
    for key, rule in rules.items():
        lines.append(f"| {key} | {rule} | {'通过' if report['checks'][key] else '未通过'} |")
    lines += ["",
             "## E1 导出可逆与证据可定位", "",
             "| 题目 | 阶段 | 引用数 | 来源数 | 逆变换等于原行 | 引文未定位 |",
             "|---|---|---:|---:|---|---:|"]
    for row in e1["tasks"]:
        lines.append(f"| {row['task_id'][:30]} | {row['stage']} | {row['refs']} | {row['sources']} | "
                     f"{'是' if row['inverse_matches_original'] else '否'} | {row['quotes_not_located']} |")
    lines += ["", f"- 唯一原生 evidence ID：**{e1['unique_evidence_ids']}**；全部逆变换一致：{e1['all_inverse']}；"
                  f"全部引文可定位：{e1['all_quotes_located']}", "",
              "## E2 锚点回归（69 个历史失败）", "",
              f"- 首尾引号可修复的 span：{e2['edge_fixable']}，其中新定位成功：{e2['edge_fixable_located']}",
              f"- 纯格式案例（无引文问题）：{e2['pure_edge_cases']}，新定位成功：{e2['pure_edge_located']}",
              f"- 引文重新匹配：{e2['quote_recovered']}/{e2['quote_total']}（涉及 {e2['quote_rows']} 行）",
              f"- 负样本（篡改 span）误放行：{e2['negative_control_false_accepts']}/{e2['negative_control_tested']}",
              f"- 判定模式分布：{e2['modes']}", "",
              "## E3 评价输入确定性", ""]
    e3 = report["e3"]
    lines += [f"- 证据换序后 packet 不变：{e3['packet_stable_under_evidence_order']}",
              f"- 候选字段换序后文本不变：{e3['candidate_text_stable_under_key_order']}",
              f"- packet sha256：`{e3['prompt_sha256'][:16]}…`（{e3['evidence_rows']} 条证据）", "",
              "## E4 调度可执行性", "",
              f"- 题目数：{e4['tasks']}；带 blocking spec：{e4['tasks_with_blocking_spec']}",
              f"- 新策略下 clarify 立即可选：{e4['clarify_eligible_now']}",
              f"- 搜证耗尽后 clarify 仍可选：{e4['clarify_eligible_after_search']}",
              f"- 历史实际发生非搜证动作的题目：{e4['historical_non_search_actions']}", "",
              "| 题目 | blocking | 初始动作 | 新策略可选 | 历史动作 |",
              "|---|---:|---|---|---|"]
    for row in e4["detail"]:
        lines.append(f"| {row['task_id'][:28]} | {row['blocking_spec']} | {row['initial_action']} | "
                     f"{', '.join(row['fresh_eligible'])} | {', '.join(row['historical_actions']) or '—'} |")
    lines += ["", "## E5 检索相关性", "",
              f"- 三个已知误匹配被拒：{e5['known_false_positives_rejected']}/{e5['known_false_positives']}",
              f"- 已入池来源重放：{e5['pool_items_replayed']}，新门驳回：{e5['pool_items_now_rejected']}"
              f"（{e5['reject_ratio']:.1%}）", "",
              "## E7 历史产物不变", ""]
    if not e7.get("available"):
        lines.append("- 未找到基线哈希文件，跳过。")
    else:
        for name, row in e7["files"].items():
            lines.append(f"- `{name}`：{'未变' if row['unchanged'] else '已变'}")
        lines.append(f"- 全部未变：{e7['all_unchanged']}")
    lines += ["", "## E6 测试", "", f"- pytest：**{report['e6']['passed']} passed**", "",
              "## 结论与边界", "",
              "- 这七项只证明**测量与调度的失效模式已被消除**，不证明外挂的质量增益；后者需要 Phase 1 的付费重评。",
              "- E2 中恢复的是模型的原判，不是证明原判正确。19 个纯格式案例只是重新进入科学判定，仍可能是错的。",
              f"- {e2['quote_total']} 条引文里仍有 {e2['quote_total'] - e2['quote_recovered']} 条无法在给定段落中找到，"
              "它们继续作为未决，属于引用本身不可核对，不是格式问题。",
              f"- E5 的召回代价为 {e5['reject_ratio']:.1%}，已接近 10% 上限；Phase 1 若发现关键来源被误删，应改为按机制词加权而不是继续收紧。",
              "- 本轮没有发起任何付费调用；smoke 目录里的金额是 dry-run 的模拟值，不是真实计费。",
              ""]
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path(__file__).parent)
    parser.add_argument("--tests-passed", type=int, default=0)
    args = parser.parse_args(argv)
    report = {"e1": check_e1(), "e2": check_e2(), "e3": check_e3(), "e4": check_e4(),
              "e5": check_e5(), "e7": check_e7()}
    report["e6"] = {"passed": args.tests_passed}
    report["checks"] = verdicts(report)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "acceptance.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    (args.out / "REPORT_ZH.md").write_text(render(report) + "\n")
    print(json.dumps({key: {k: v for k, v in value.items() if k != "detail"} if isinstance(value, dict) else value
                      for key, value in report.items()}, ensure_ascii=False, indent=1)[:2000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
