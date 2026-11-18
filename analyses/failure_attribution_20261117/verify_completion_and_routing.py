"""Verify saved native completion and replay no-new-evidence routing without API calls."""
from contextlib import closing
import argparse
import json
from pathlib import Path
import sqlite3

from analyze import ROOT, NATIVE, STAGE_B, INPUT_HASHES, file_hash, read, tracked
from edara_mvp_eval.feedback import FeedbackMiddleware
from edara_mvp_eval.schemas import FullAuditBundle, TaskSpec


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence", type=Path, default=Path(__file__).with_name("evidence.json"))
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    evidence = read(args.evidence)
    completion = []
    for match in evidence["native"]["matches"]:
        db = tracked(Path(match["database"]))
        with closing(sqlite3.connect(db.as_uri() + "?mode=ro", uri=True)) as conn:
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA query_only=ON")
            runs = [dict(r) for r in conn.execute("SELECT policy_run_id,status,error FROM policy_runs ORDER BY rowid DESC LIMIT 2")]
            plans = []
            for row in conn.execute("SELECT v.version_id,b.bundle_id,b.policy_json,a.status,a.result_json FROM research_versions v JOIN research_review_bundles b ON json_extract(b.binding_json,'$.version_id')=v.version_id LEFT JOIN research_review_attempts a ON a.bundle_id=b.bundle_id WHERE v.kind='experiment_plan'"):
                policy = json.loads(row["policy_json"])
                result = json.loads(row["result_json"] or "{}")
                plans.append({"version_id": row["version_id"], "bundle_id": row["bundle_id"],
                              "policy_version": policy.get("policy_version"), "required_reviewers": policy.get("required_reviewers"),
                              "rubric_protocol": policy.get("rubric_protocol"), "attempt_status": row["status"],
                              "nonpass": {k: v for k, v in result.get("items", {}).items() if v.get("verdict") != "pass"}})
            completion.append({"task_id": match["task_id"], "database": str(db), "recent_runs": runs, "plan_reviews": plans})

    # This recorded recovery contains a later completion than ws-round6.log.
    log = tracked(NATIVE / "ws-budget-resume.log")
    raw = log.read_text()
    recovery, _ = json.JSONDecoder().raw_decode(raw[raw.index("{"):])
    delivery_path = Path(recovery["delivery"]["json"])
    assert delivery_path.is_relative_to(NATIVE)
    delivery = read(delivery_path)
    receipt = {"log": str(log), "status": recovery["status"], "run_id": recovery["run_id"],
               "validation_complete": recovery["validation_complete"], "delivery_path": str(delivery_path),
               "delivery_status": delivery["delivery_status"], "qualified_plan_version_ids": delivery["qualified_plan_version_ids"],
               "snapshot_sha256": delivery["snapshot_sha256"]}
    assert receipt["status"] == "plan_reviewed" and receipt["delivery_status"] == "qualified_plan_available"

    def forbidden(*args, **kwargs):
        raise AssertionError("This saved routing path must not call a model or a generator")

    class NoModel:
        chat_json = staticmethod(forbidden)

    replays = []
    # Select all unchanged tasks whose actual feedback history was a single zero-addition search.
    for task in evidence["stage_b"]["tasks"]:
        history = task["feedback_history"]
        if task["changed_by_feedback"] or len(history) != 1 or history[0].get("action") != "seek_evidence" or history[0].get("added"):
            continue
        root = STAGE_B / "tasks" / task["task_id"]
        before = read(root / "direct.json")
        prepared = TaskSpec.model_validate(read(root / "prepared.json")["task"])
        audit = FullAuditBundle.model_validate(before["audit"])
        queries = []

        def no_new_papers(query):
            queries.append(query)
            return []

        output, _, trace, _, state = FeedbackMiddleware().apply(
            prepared, before["candidate"], audit, audit_runner=NoModel(), repair_runner=NoModel(),
            generator_callback=forbidden, search=no_new_papers)
        assert output == before["candidate"] and trace == history and state == "seek_evidence"
        replays.append({"task_id": task["task_id"], "replay_matches_recorded_history": True,
                        "candidate_unchanged": True, "blocking_spec_count": audit.spec_audit.blocking_count,
                        "search_query_count": len(queries), "generator_calls": 0, "model_calls": 0,
                        "note": "Search replay returns no additions, matching saved history; no HTTP request made."})
    assert len(replays) == 4
    assert all(file_hash(Path(p)) == sha for p, sha in INPUT_HASHES.items())
    result = {"native_completion": completion, "workshop_recovery": receipt, "routing_replays": replays,
              "input_hashes": INPUT_HASHES, "inputs_unchanged": True}
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "completion_and_replays.json").open("x") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(json.dumps({"completion": [(r["task_id"], r["recent_runs"][0]["status"]) for r in completion],
                      "routing_replays": len(replays), "model_calls": 0, "inputs_unchanged": True}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
