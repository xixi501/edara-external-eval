"""Compare ordinary revision, selected evidence, and structured governance feedback."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from itertools import permutations
from pathlib import Path

from pydantic import BaseModel, Field, StrictBool

from edara_mvp_eval.governance import audit_idea, evidence_payload
from edara_mvp_eval.llm_client import LLMRunner, ModelEndpoint
from edara_mvp_eval.schemas import IdeaGovernanceResult

from governance_phases import regenerate_candidate
from run_governance_pilot import PROFILE_ROOT, SOURCE, read_json, save_json, sha256, task_for

ARMS = ("ordinary", "highlight", "feedback")
ARM_ORDERS = list(permutations(ARMS))

_REVISE = """Produce one revised research proposal for the same topic. Examine the original proposal
and dated evidence. When the original has a distinct testable research question, preserve it and
give prior work precise credit for covered pieces. If the central contribution is covered and no
substantive distinct question remains, propose a materially different mechanism. Keep the proposed
test and falsifying observation concrete. Describe predictions as untested and scope originality
to the evidence. Any optional diagnosis concerns the original draft and is advice, not evidence
about your new proposal. Return JSON with these string fields: title, literature_gap, hypothesis,
method, minimal_test, decisive_metric, falsifying_result, related_work."""

_SCORE = """Score the single candidate against the original, the supplied dated papers and
the frozen reference key. Scope the score to that evidence. Assess each property separately:
- replaces_covered_core: the candidate has a materially different central contribution relative
  to the covered mechanism. It may reuse the covered mechanism with attribution as a component or
  baseline. A renamed method, a generic added condition, or changing only the benchmark is insufficient.
- attributes_covered_mechanism: the candidate explicitly credits the decisive prior for the
  covered mechanism if it uses or discusses it. Set true if it genuinely removes that mechanism
  and makes no claim about it. A generic related-work mention does not count as attribution.
- preserves_residual: for a partial case, the candidate keeps the specific reference hypothesis
  and its scientifically meaningful intervention or contrast. An unrelated novel proposal fails
  this field. In a collision case with no reference residual set true as not applicable.
- tests_hypothesis: the candidate proposes a concrete distinguishing comparison with an observable
  falsifier. In a partial case the test must address the reference residual, not a new unrelated idea.
- unsupported_claim: it makes a priority claim beyond the evidence, presents a covered contribution
  as its own, or states an untested expected finding as established fact.
Return exactly one JSON object with these five boolean keys, a short rationale naming the concrete
mechanism and comparison, and prior_paper_ids containing only the evidence IDs you rely on."""


class Verdict(BaseModel):
    replaces_covered_core: StrictBool
    attributes_covered_mechanism: StrictBool
    preserves_residual: StrictBool
    tests_hypothesis: StrictBool
    unsupported_claim: StrictBool
    rationale: str = Field(min_length=1)
    prior_paper_ids: list[str] = Field(default_factory=list)


def treatment(audit):
    """Give both treatment arms the same audit-selected excerpts."""
    highlights = {}
    findings = []
    for claim in audit.claims:
        for reference in claim.evidence:
            key = (reference.paper_id, " ".join(reference.quote.split()))
            highlights.setdefault(key, reference.model_dump())
        row = claim.model_dump(exclude={"evidence"})
        row["supporting_paper_ids"] = list(dict.fromkeys(q.paper_id for q in claim.evidence))
        findings.append(row)
    diagnosis = {
        "action": audit.action, "unresolved_reason": audit.unresolved_reason, "findings": findings,
    }
    return list(highlights.values()), diagnosis


def evaluate(task, case, generation, runner):
    if generation["idea"] is None:
        return {"verdict": None, "success": False, "error": "invalid_candidate", "usage": asdict(runner.usage)}
    papers = evidence_payload(task)
    request = {
        "topic": task.title, "cutoff_date": task.cutoff_date,
        "original_draft": case["draft"], "evidence": papers,
        "reference_key": case["evaluation_key"], "candidate": generation["idea"],
    }
    try:
        verdict = Verdict.model_validate(runner.chat_json(
            _SCORE, json.dumps(request, ensure_ascii=False), max_tokens=1600, tag="feedback_ablation_blind",
        ))
        if set(verdict.prior_paper_ids) - {paper["paper_id"] for paper in papers}:
            raise ValueError("unlisted paper in verdict")
    except ValueError:
        return {"verdict": None, "success": False, "error": "invalid_verdict", "usage": asdict(runner.usage)}
    contribution = verdict.replaces_covered_core if case["kind"] == "collision" else verdict.preserves_residual
    return {
        "verdict": verdict.model_dump(),
        "success": contribution and verdict.attributes_covered_mechanism and verdict.tests_hypothesis
        and not verdict.unsupported_claim,
        "usage": asdict(runner.usage),
    }


def audit_match(case, audit):
    prior = case["evaluation_key"]["prior_id"]
    anchored = any(
        finding.status == "covered" and any(reference.paper_id == prior for reference in finding.evidence)
        for finding in audit.claims
    )
    actions = ("regenerate",) if case["kind"] == "collision" else ("correct_attribution", "retain_and_test")
    return anchored and audit.action in actions


def aggregate_usage(items):
    return {field: sum(item[field] for item in items) for field in items[0]}


def paired_change(rows, treatment_arm, control_arm):
    changes = Counter(
        (row["arms"][treatment_arm]["evaluation"]["success"],
         row["arms"][control_arm]["evaluation"]["success"])
        for row in rows
    )
    return {"gains": changes[True, False], "losses": changes[False, True]}


def summarize(rows, out):
    lines = [
        "# Three-arm governance feedback pilot", "",
        "Primary comparison is structured feedback versus highlighted evidence.",
        "Both treatment arms receive the same audit-selected verbatim evidence.", "",
        "| Case | Type | Audit action | Grounded expected route | Highlights | Ordinary | Highlight | Feedback |",
        "|---|---|---|---|---:|---|---|---|",
    ]
    for row in rows:
        outcomes = ["pass" if row["arms"][arm]["evaluation"]["success"] else "fail" for arm in ARMS]
        lines.append(f"| {row['case_id']} | {row['kind']} | {row['audit']['result']['action']} | "
                     f"{row['audit']['expected_route']} | {len(row['highlights'])} | {' | '.join(outcomes)} |")
    lines += [
        "", "| Group | Ordinary | Highlight | Feedback | Highlight vs ordinary | Feedback vs highlight |",
        "|---|---:|---:|---:|---|---|",
    ]
    for kind in ("collision", "partial", "all"):
        subset = [r for r in rows if kind == "all" or r["kind"] == kind]
        counts = [f"{sum(r['arms'][arm]['evaluation']['success'] for r in subset)}/{len(subset)}" for arm in ARMS]
        deltas = [paired_change(subset, a, b) for a, b in (("highlight", "ordinary"), ("feedback", "highlight"))]
        lines.append(f"| {kind} | {' | '.join(counts)} | " + " | ".join(
            f"+{change['gains']} / -{change['losses']}" for change in deltas
        ) + " |")
    lines += [
        "", "| Arm | Generation calls | Prompt tokens | Completion tokens | Generation cost | Blind review cost |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for arm in ARMS:
        gen = aggregate_usage([row["arms"][arm]["generation"]["usage"] for row in rows])
        review = sum(row["arms"][arm]["evaluation"]["usage"]["cost_usd"] for row in rows)
        lines.append(f"| {arm} | {gen['calls']} | {gen['prompt_tokens']} | {gen['completion_tokens']} | "
                     f"${gen['cost_usd']:.5f} | ${review:.5f} |")
    costs = aggregate_usage([r["usage"] for r in rows])
    audit_cost = sum(row["audit"]["usage"]["cost_usd"] for row in rows)
    lines += [
        "", f"Audit matched {sum(r['audit']['expected_route'] for r in rows)}/{len(rows)} expected grounded routes.",
        f"Shared audit cost ${audit_cost:.5f}. Total metered API cost ${costs['cost_usd']:.5f}.",
        "All rows are included regardless of audit and generation outcome.",
        "The test uses paired variants across six research topics and a single model judge.",
    ]
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--gen-profile", type=Path, default=PROFILE_ROOT / "p1b-idea-profile.json")
    parser.add_argument("--judge-profile", type=Path, default=PROFILE_ROOT / "p1b-critic-profile.json")
    args = parser.parse_args()

    data = read_json(args.cases)
    cases = data["cases"]
    if data["protocol"] != "feedback-ablation-cases/v1" or Counter(c["kind"] for c in cases) != {
        "collision": 6, "partial": 6,
    }:
        raise ValueError("expected the frozen six-by-two case set")
    case_ids = [c["case_id"] for c in cases]
    if len(set(case_ids)) != len(case_ids) or any(not re.fullmatch(r"[a-z0-9-]+", x) for x in case_ids):
        raise ValueError("invalid or duplicate case ID")
    gen_endpoint = ModelEndpoint.from_env_or_profile(profile_path=args.gen_profile, prefix="EDARA_GEN")
    judge_endpoint = ModelEndpoint.from_env_or_profile(profile_path=args.judge_profile, prefix="EDARA_JUDGE")
    args.out.mkdir(parents=True, exist_ok=True)
    paths = [
        *sorted(SOURCE.glob("*.py")), Path(__file__),
        Path(__file__).with_name("governance_phases.py"), Path(__file__).with_name("run_governance_pilot.py"),
    ]
    arm_order = {}
    kind_index = Counter({"partial": 3})
    for case in cases:
        kind = case["kind"]
        arm_order[case["case_id"]] = list(ARM_ORDERS[kind_index[kind] % len(ARM_ORDERS)])
        kind_index[kind] += 1
    manifest = {
        "protocol": "governance-feedback-ablation/v1",
        "cases_sha256": sha256(args.cases.read_bytes()),
        "generator": gen_endpoint.model, "judge": judge_endpoint.model,
        "profile_sha256": {"gen": sha256(args.gen_profile.read_bytes()), "judge": sha256(args.judge_profile.read_bytes())},
        "sources": {p.name: sha256(p.read_bytes()) for p in paths},
        "generation_calls_per_arm_case": 1, "requested_output_tokens": 4096, "temperature": 0.3,
        "case_arm_order": arm_order,
    }
    manifest_path = args.out / "manifest.json"
    if manifest_path.exists() and manifest != read_json(manifest_path):
        raise ValueError("manifest differs; use a new output directory")
    save_json(manifest_path, manifest)

    def run_case(case):
        cid = case["case_id"]
        folder = args.out / cid
        folder.mkdir(exist_ok=True)
        complete = folder / "result.json"
        if complete.exists():
            return read_json(complete)
        task = task_for(case)
        audit_file = folder / "audit.json"
        if audit_file.exists():
            recorded = read_json(audit_file)
            governance = IdeaGovernanceResult.model_validate(recorded["result"])
        else:
            auditor = LLMRunner(judge_endpoint)
            governance = audit_idea(task, case["draft"], auditor)
            recorded = {
                "result": governance.model_dump(), "expected_route": audit_match(case, governance),
                "usage": asdict(auditor.usage),
            }
            save_json(audit_file, recorded)
        highlights, diagnosis = treatment(governance)
        arms = {}
        for arm in manifest["case_arm_order"][cid]:
            gen_path = folder / f"{arm}.candidate.json"
            if gen_path.exists():
                generation = read_json(gen_path)
            else:
                generation = regenerate_candidate(
                    task, case, LLMRunner(gen_endpoint), diagnosis if arm == "feedback" else None,
                    highlights=highlights if arm != "ordinary" else None, system=_REVISE,
                )
                save_json(gen_path, generation)
            verdict_path = folder / f"{arm}.verdict.json"
            if verdict_path.exists():
                evaluation = read_json(verdict_path)
            else:
                evaluation = evaluate(task, case, generation, LLMRunner(judge_endpoint))
                save_json(verdict_path, evaluation)
            arms[arm] = {"generation": generation, "evaluation": evaluation}
        costs = [result["usage"] for results in arms.values() for result in results.values()]
        row = {
            "case_id": cid, "kind": case["kind"], "family": case["family"],
            "audit": recorded, "highlights": highlights, "diagnosis": diagnosis,
            "arms": arms, "usage": aggregate_usage([recorded["usage"], *costs]),
        }
        save_json(complete, row)
        print(f"{cid}: {row['usage']['calls']} calls, ${row['usage']['cost_usd']:.5f}", flush=True)
        return row

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        completed = list(pool.map(run_case, cases))
    save_json(args.out / "results.json", completed)
    summarize(completed, args.out)


if __name__ == "__main__":
    main()
