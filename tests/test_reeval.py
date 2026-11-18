import json

import pytest

from edara_mvp_eval.llm_client import ModelEndpoint
from edara_mvp_eval.reeval import EVAL_VERSION, ReevalError, load_run, review_run
from edara_mvp_eval.stage_b_evidence import digest, write_json

CANDIDATE = {"hypothesis": "The gate reduces selective risk under shift.",
             "method": "Stratified abstention with a calibrated threshold."}


def make_run(root, candidate=None):
    candidate = CANDIDATE if candidate is None else candidate
    task = {"task_id": "t1", "source_track": "lit2test", "title": "T", "field": "ML",
            "prompt_context": "ctx\n\nDated literature:\n[]", "cutoff_date": "2025-01-01",
            "strict_evidence": True, "prior_work": [], "forbidden_title_hashes": []}
    write_json(root / "tasks_frozen.json", [task])
    write_json(root / "manifest.json", {"frozen": True})
    work = root / "tasks" / "t1"
    write_json(work / "evaluation_evidence.json", {
        "evidence": [{"paper_id": "p1", "title": "Prior", "date": "2024", "passages": "text"}],
        "candidates": {"direct": digest(candidate)},
    })
    write_json(work / "direct.json", {"status": "produced", "candidate": candidate})
    return root


def test_review_run_recomputes_and_records_stability(tmp_path):
    run = make_run(tmp_path / "run")
    out = tmp_path / "out"
    payload = review_run(run, out, judge=ModelEndpoint(), repeat=2, arms=("direct",),
                         dry_run=True, retry=False)
    assert payload["eval_version"] == EVAL_VERSION
    assert payload["stability"]["assessed"] == 1
    assert (out / "results.json").exists()
    assert (out / "summary.md").exists()
    written = json.loads((out / "results.json").read_text())
    arm = written["results"][0]["arms"]["direct"]
    assert arm["status"] == "assessed"
    assert len(arm["repeats"]) == 2
    assert arm["primary"]["axes"]["honesty"]["raw_status"] == "unresolved"


def test_candidate_digest_mismatch_fails_closed(tmp_path):
    run = make_run(tmp_path / "run")
    write_json(run / "tasks" / "t1" / "direct.json",
               {"status": "produced", "candidate": {"hypothesis": "tampered", "method": "m"}})
    with pytest.raises(ReevalError):
        load_run(run)


def test_missing_arm_is_reported_not_scored(tmp_path):
    run = make_run(tmp_path / "run")
    payload = review_run(run, tmp_path / "out", judge=ModelEndpoint(), repeat=1,
                         arms=("direct", "direct_self_refine"), dry_run=True, retry=False)
    arms = payload["results"][0]["arms"]
    assert arms["direct_self_refine"]["status"] == "no_candidate"
    assert payload["stability"]["assessed"] == 1


def test_every_repeat_is_a_distinct_request(tmp_path):
    run = make_run(tmp_path / "run")
    out = tmp_path / "out"
    review_run(run, out, judge=ModelEndpoint(), repeat=2, arms=("direct",), dry_run=True, retry=False)
    requests = sorted(path.relative_to(out).as_posix() for path in (out / "requests").glob("**/*.json"))
    assert any("r0" in name for name in requests)
    assert any("r1" in name for name in requests)
    hashes = {json.loads((out / name).read_text())["request_sha256"] for name in requests}
    assert len(hashes) == len(requests)
