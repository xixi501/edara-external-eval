"""Read-only attribution of saved native outputs and Stage B decisions; no model calls.

Run with Python's standard library. Writes only new analysis files in --out.
Existing result files and all native databases remain read-only.
"""
import argparse
from collections import Counter
from contextlib import closing
import hashlib
import json
from pathlib import Path
import re
import sqlite3


ROOT = Path(__file__).resolve().parents[2]
NATIVE = ROOT.parent / "pilot3_runs/edara"
STAGE_B = ROOT / "outputs/mvp15_ablation_v2"
INPUT_HASHES = {}


def file_hash(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def tracked(path):
    path = Path(path)
    INPUT_HASHES.setdefault(str(path), file_hash(path))
    return path


def read(path):
    return json.loads(tracked(path).read_text())


def norm(value):
    return " ".join(value.split())


def text(candidate):
    if isinstance(candidate, str):
        return candidate
    return json.dumps({k: v for k, v in candidate.items()
                       if k not in {"task_id", "arm", "model", "gen_cost_usd"}},
                      ensure_ascii=False, indent=2)


def content_hash(candidate):
    return hashlib.sha256(text(candidate).encode()).hexdigest()


def parse_response(response):
    if isinstance(response, dict):
        return response
    try:
        return json.loads(response)
    except ValueError:
        start, end = response.find("{"), response.rfind("}")
        return json.loads(response[start:end + 1])


def native_analysis():
    exports = [json.loads(line) for line in tracked(NATIVE / "edara_outputs.jsonl").read_text().splitlines() if line]
    external = read(ROOT / "outputs/pilot3_full_compare_v4/results.json")
    matches, database_inventory = [], []
    for db in sorted(NATIVE.rglob("research.db")):
        tracked(db)
        with closing(sqlite3.connect(db.as_uri() + "?mode=ro", uri=True)) as conn:
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA query_only=ON")
            tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            if "research_versions" not in tables:
                continue
            versions = conn.execute("SELECT * FROM research_versions").fetchall()
            inventory = {"database": str(db), "version_counts": dict(Counter(v["kind"] for v in versions)),
                         "mechanism_record_counts": {}}
            for table in ("research_candidate_comparisons", "research_idea_portfolios"):
                inventory["mechanism_record_counts"][table] = conn.execute(f'SELECT count(*) FROM "{table}"').fetchone()[0] if table in tables else 0
            database_inventory.append(inventory)
            for version in versions:
                payload = json.loads(version["payload_json"])
                for exported in exports:
                    if payload.get("title") != exported["title"]:
                        continue
                    masked = json.loads(re.sub(r"ev-[a-f0-9]+", "[REF]", json.dumps(payload, ensure_ascii=False)))
                    differences = [k for k in exported if k != "task_id" and exported[k] != masked.get(k)]
                    reviews = []
                    for bundle in conn.execute("SELECT * FROM research_review_bundles WHERE json_extract(binding_json,'$.version_id')=?", (version["version_id"],)):
                        policy = json.loads(bundle["policy_json"])
                        attempts = []
                        for attempt in conn.execute("SELECT * FROM research_review_attempts WHERE bundle_id=?", (bundle["bundle_id"],)):
                            result = json.loads(attempt["result_json"] or "{}")
                            attempts.append({"attempt_id": attempt["attempt_id"], "role": attempt["role"],
                                             "status": attempt["status"], "model_family": attempt["model_family"],
                                             "result": result})
                        reviews.append({"bundle_id": bundle["bundle_id"], "binding": json.loads(bundle["binding_json"]),
                                        "policy_version": policy.get("policy_version"), "rubric_protocol": policy.get("rubric_protocol"),
                                        "required_reviewers": policy.get("required_reviewers"), "attempts": attempts})
                    claims = []
                    for claim in payload.get("claims", []):
                        sources = []
                        for eid in claim.get("evidence_ids", []):
                            ev = conn.execute("SELECT e.*,p.title AS paper_title,s.text AS section_text FROM evidence e JOIN papers p ON p.paper_id=e.paper_id JOIN sections s ON s.section_id=e.section_id WHERE e.evidence_id=?", (eid,)).fetchone()
                            if ev:
                                sources.append({"evidence_id": eid, "paper_id": ev["paper_id"], "paper_title": ev["paper_title"],
                                                "status": ev["status"], "review_status": ev["review_status"], "quote": ev["quote"],
                                                "quote_in_saved_section": norm(ev["quote"]) in norm(ev["section_text"])})
                            else:
                                sources.append({"evidence_id": eid, "missing": True})
                        claims.append({"claim": claim, "sources": sources})
                    plans = []
                    for plan in versions:
                        if plan["kind"] != "experiment_plan":
                            continue
                        plans.append({"version_id": plan["version_id"], "dependencies": json.loads(plan["dependencies_json"]),
                                      "payload": json.loads(plan["payload_json"]),
                                      "reviews": [dict(r) for r in conn.execute("SELECT b.bundle_id,a.status,a.role,a.result_json FROM research_review_bundles b LEFT JOIN research_review_attempts a ON a.bundle_id=b.bundle_id WHERE json_extract(b.binding_json,'$.version_id')=?", (plan["version_id"],))]})
                    audited = next(b for b in external["bundles"]["edara"] if b["task_id"] == exported["task_id"])
                    matches.append({"task_id": exported["task_id"], "database": str(db), "version_id": version["version_id"],
                                    "kind": version["kind"], "created_at": version["created_at"], "content_hash": version["content_hash"],
                                    "export_matches_after_reference_mask": not differences,
                                    "different_fields": differences, "omitted_fields": sorted(set(payload) - set(exported)),
                                    "native_payload": payload, "evidence_claims": claims, "native_reviews": reviews,
                                    "saved_plans": plans, "external_normalized_proposal": audited["proposal"],
                                    "external_fact_audits": audited["tri_state"]["prior_fact_audits"],
                                    "external_spec_defects": audited["spec_audit"]["defects"]})
    assert len(matches) == len(exports) == 3
    assert all(m["export_matches_after_reference_mask"] for m in matches)
    return {"database_inventory": database_inventory, "matches": matches}


def stage_b_analysis():
    results = read(STAGE_B / "results.json")
    tasks, anchor_issues = [], []
    for result in results:
        tid = result["task_id"]
        root = STAGE_B / "tasks" / tid
        arms = {arm: read(root / f"{arm}.json") for arm in result["verdicts"]}
        initial, feedback = arms["direct"], arms["direct_feedback_mw"]
        changed = text(initial["candidate"]) != text(feedback["candidate"])
        evidence = read(root / "evaluation_evidence.json")
        passages = {p["paper_id"]: norm(p["passages"]) for p in evidence["evidence"]}
        for arm, row in arms.items():
            paths = sorted((root / "requests" / arm / "final_verdict").glob("*.json"))
            request = next(read(p) for p in reversed(paths) if read(p).get("tag") == "stage_b_verdict")
            raw = parse_response(request["response"])
            for axis, saved in result["verdicts"][arm]["axes"].items():
                original = raw[axis]
                span = norm(original.get("candidate_span", ""))
                valid_span = bool(span) and span in norm(text(row["candidate"]))
                quote_errors = [ref for ref in original.get("evidence", []) if norm(ref["quote"]) not in passages.get(ref["paper_id"], "")]
                if saved.get("validation"):
                    anchor_issues.append({"task_id": tid, "arm": arm, "axis": axis,
                                          "raw_status": original["status"], "saved_status": saved["status"],
                                          "invalid_span": not valid_span,
                                          "span_matches_after_edge_quotes_removed": bool(span.strip('"“”')) and span.strip('"“”') in norm(text(row["candidate"])),
                                          "invalid_evidence_quotes": quote_errors,
                                          "missing_novelty_evidence": axis == "novelty" and original["status"] == "pass" and not original.get("evidence"),
                                          "candidate_span": original.get("candidate_span"), "reason": original["reason"]})
        requests = Counter()
        for path in sorted((root / "requests/direct_self_refine/revision").glob("*.json")):
            record = read(path)
            requests[record.get("tag")] += 1
        tasks.append({"task_id": tid, "development": result["development"], "changed_by_feedback": changed,
                      "candidate_hashes": {a: content_hash(r["candidate"]) for a, r in arms.items()},
                      "initial_governance": initial["audit"]["governance"],
                      "initial_overclaim_count": initial["audit"]["tri_state"]["epistemic_overclaim_count"],
                      "initial_blocking_spec_count": initial["audit"]["spec_audit"]["blocking_count"],
                      "initial_spec_defects": initial["audit"]["spec_audit"]["defects"],
                      "feedback_history": feedback["history"], "feedback_final_state": feedback["governance_state"],
                      "feedback_final_governance": feedback["audit"]["governance"],
                      "self_refine_revision_requests": dict(requests),
                      "same_text_joint_flip": not changed and result["verdicts"]["direct"]["joint_pass"] != result["verdicts"]["direct_feedback_mw"]["joint_pass"],
                      "same_text_axis_flips": [axis for axis in ("honesty", "credibility", "novelty") if not changed and result["verdicts"]["direct"]["axes"][axis]["status"] != result["verdicts"]["direct_feedback_mw"]["axes"][axis]["status"]],
                      "verdicts": result["verdicts"],
                      "evidence_titles": [p["title"] for p in evidence["evidence"]]})
    summary = {"tasks": len(tasks), "feedback_changed": sum(t["changed_by_feedback"] for t in tasks),
               "same_text_joint_flips": sum(t["same_text_joint_flip"] for t in tasks),
               "same_text_axis_flips": sum(len(t["same_text_axis_flips"]) for t in tasks),
               "initial_governance_actions": dict(Counter(t["initial_governance"]["action"] for t in tasks)),
               "initial_overclaim_counts": dict(Counter(t["initial_overclaim_count"] for t in tasks)),
               "initial_blocking_spec_total": sum(t["initial_blocking_spec_count"] for t in tasks),
               "unchanged_with_blocking_spec": sum(not t["changed_by_feedback"] and t["initial_blocking_spec_count"] > 0 for t in tasks),
               "feedback_actions": dict(Counter(h["action"] for t in tasks for h in t["feedback_history"])),
               "feedback_final_reasons": dict(Counter(t["feedback_final_governance"].get("unresolved_reason") for t in tasks)),
               "joint_pass": {a: sum(t["verdicts"][a]["joint_pass"] for t in tasks) for a in tasks[0]["verdicts"]},
               "anchor_downgrades": len(anchor_issues),
               "anchor_raw_statuses": dict(Counter(a["raw_status"] for a in anchor_issues)),
               "anchor_invalid_span": sum(a["invalid_span"] for a in anchor_issues),
               "anchor_invalid_quotes": sum(bool(a["invalid_evidence_quotes"]) for a in anchor_issues),
               "anchor_span_fixed_by_edge_quotes_only": sum(a["invalid_span"] and a["span_matches_after_edge_quotes_removed"] for a in anchor_issues)}
    assert summary["tasks"] == 15
    assert summary["feedback_changed"] == 2
    return {"summary": summary, "tasks": tasks, "anchor_issues": anchor_issues}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    output = {"native": native_analysis(), "stage_b": stage_b_analysis()}
    changed = [p for p, sha in INPUT_HASHES.items() if file_hash(Path(p)) != sha]
    assert not changed, changed
    output["input_hashes"] = INPUT_HASHES
    output["verification"] = {"inputs_unchanged": True, "tracked_files": len(INPUT_HASHES), "model_calls": 0}
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "evidence.json").open("x") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(json.dumps(output["stage_b"]["summary"], ensure_ascii=False, indent=2))
    print(json.dumps(output["verification"], ensure_ascii=False))


if __name__ == "__main__":
    main()
