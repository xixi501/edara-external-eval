"""Run the fixed-evidence governance pilot and persist every observed case."""

import argparse
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path

from edara_mvp_eval.epistemic_audit import run_full_audit
from edara_mvp_eval.llm_client import LLMRunner, ModelEndpoint
from edara_mvp_eval.schemas import TaskSpec

_HERE = Path(__file__).resolve()
ROOT = _HERE.parents[2] if len(_HERE.parents) >= 3 else _HERE.parents[1]
PROFILE_ROOT = (ROOT.parents[1] / "data") if len(ROOT.parents) >= 2 else (ROOT / "data")
SOURCE = Path(__file__).resolve().parents[1] / "src/edara_mvp_eval"


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def save_json(path, value):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temp.replace(path)


def task_for(case):
    return TaskSpec(
        task_id=case["case_id"], source_track="aiib", field="computer science",
        title=case["topic"], prompt_context=case["topic"], cutoff_date=case["cutoff_date"],
        prior_work=case["evidence"],
    )


def prior_phase(path, phase, cases_hash):
    manifest = read_json(path / "manifest.json")
    if manifest["phase"] != phase or manifest["cases_sha256"] != cases_hash:
        raise ValueError(f"{path} does not contain the expected phase and frozen case set")
    return {row["case_id"]: row for row in read_json(path / "results.json")}


def render_summary(out, cases, rows, phase):
    by_case = {case["case_id"]: case for case in cases}
    lines = [f"# Governance pilot · {phase}", ""]
    if phase == "baseline":
        lines += [
            "| Case | ClaimNov veto | Covered components | Composition flag | Gap statuses |",
            "|---|---|---:|---|---|",
        ]
        for row in rows:
            audit = row["audit"]
            gap = ", ".join(i["status"] for i in audit["gap_audit"]["items"])
            comp = audit["compositional"]
            lines.append(f"| {row['case_id']} | {audit['claimnov']['veto']} | "
                         f"{len(comp['covered_components'])} | {comp['is_compositional_novelty']} | {gap} |")
    elif phase == "audit":
        lines += [
            "| Case | Expected | Action | Core outcomes | Match |",
            "|---|---|---|---|---|",
        ]
        for row in rows:
            result = row["governance"]
            core = ", ".join(c["status"] for c in result["claims"] if c["role"] == "core")
            lines.append(f"| {row['case_id']} | {by_case[row['case_id']]['expected']} | {result['action']} | "
                         f"{core} | {row['matched_expectation']} |")
        lines += ["", f"Matched {sum(r['matched_expectation'] for r in rows)}/{len(rows)} expectations."]
        counts = {}
        for row in rows:
            for item in row["gap"]["items"]:
                counts[item["status"]] = counts.get(item["status"], 0) + 1
        lines.extend(["", f"Gap status counts `{json.dumps(counts, sort_keys=True)}`."])
    else:
        lines += [
            "| Case | Feedback used | Ordinary success | Feedback success |",
            "|---|---|---|---|",
        ]
        for row in rows:
            lines.append(f"| {row['case_id']} | {row['feedback_used']} | "
                         f"{row['ordinary']['evaluation']['success']} | {row['feedback']['evaluation']['success']} |")
        for arm in ("ordinary", "feedback"):
            selected = [r[arm] for r in rows]
            calls = sum(r["generation"]["usage"]["calls"] for r in selected)
            total = sum(r["generation"]["usage"]["cost_usd"] for r in selected)
            lines += ["", f"`{arm}` succeeded on {sum(r['evaluation']['success'] for r in selected)}/{len(rows)} "
                      f"cases. Generation used {calls} calls and cost ${total:.5f}."]
    lines += ["", f"Metered API cost ${sum(row['usage']['cost_usd'] for row in rows):.5f}.",
              "The fixed evidence pool bounds all novelty conclusions. This is a diagnostic pilot."]
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["baseline", "audit", "regenerate"])
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--prior-phase", type=Path)
    parser.add_argument("--gen-profile", type=Path, default=PROFILE_ROOT / "p1b-idea-profile.json")
    parser.add_argument("--judge-profile", type=Path, default=PROFILE_ROOT / "p1b-critic-profile.json")
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    data = args.cases.read_bytes()
    cases = json.loads(data)["cases"]
    files = sorted(SOURCE.glob("*.py"))
    sources = {p.name: sha256(p.read_bytes()) for p in files}
    judge_endpoint = ModelEndpoint.from_env_or_profile(profile_path=args.judge_profile, prefix="EDARA_JUDGE")
    args.out.mkdir(parents=True, exist_ok=True)
    cases_hash = sha256(data)
    manifest = {
        "phase": args.phase, "cases_sha256": cases_hash, "sources": sources,
        "judge": judge_endpoint.model,
        "judge_profile_sha256": sha256(args.judge_profile.read_bytes()),
        "pilot_sha256": sha256(Path(__file__).read_bytes()),
    }
    earlier = {}
    if args.phase != "baseline":
        from governance_phases import audit_case, blind_evaluate, feedback_for, regenerate_candidate

        if args.prior_phase is None:
            parser.error("--prior-phase is required for audit and regenerate")
        prior_phase_name = "baseline" if args.phase == "audit" else "audit"
        earlier = prior_phase(args.prior_phase, prior_phase_name, cases_hash)
        manifest["prior_results_sha256"] = sha256((args.prior_phase / "results.json").read_bytes())
        manifest["phases_sha256"] = sha256(Path(__file__).with_name("governance_phases.py").read_bytes())
    if args.phase == "regenerate":
        audit_manifest = read_json(args.prior_phase / "manifest.json")
        if audit_manifest["sources"] != sources:
            raise ValueError("audit source does not match regeneration source; finish a new audit first")
        gen_endpoint = ModelEndpoint.from_env_or_profile(profile_path=args.gen_profile, prefix="EDARA_GEN")
        manifest["generator"] = gen_endpoint.model
        manifest["gen_profile_sha256"] = sha256(args.gen_profile.read_bytes())
        cases = [case for case in cases if case["decisive_paper_id"]]
    manifest_path = args.out / "manifest.json"
    if manifest_path.exists() and read_json(manifest_path) != manifest:
        raise ValueError(f"existing manifest differs at {manifest_path}; use a new output directory")
    save_json(manifest_path, manifest)

    def run_case(case):
        case_id = case["case_id"]
        path = args.out / f"{case_id}.json"
        if path.exists():
            return read_json(path)
        task = task_for(case)
        judge = LLMRunner(judge_endpoint)
        if args.phase == "baseline":
            audit = run_full_audit(task, args.phase, case["draft"], judge)
            result = {"audit": audit.model_dump(mode="json"), "usage": asdict(judge.usage)}
        elif args.phase == "audit":
            result = audit_case(task, case, earlier[case_id], judge)
        else:
            feedback = feedback_for(earlier[case_id])
            result = {"feedback_used": feedback is not None, "feedback_input": feedback}
            # The scoring model sees each candidate independently without the treatment label.
            order = ("feedback", "ordinary") if hashlib.sha256(case_id.encode()).digest()[0] % 2 else (
                "ordinary", "feedback",
            )
            for arm in order:
                draft_path = args.out / f"{case_id}.{arm}.candidate.json"
                if draft_path.exists():
                    candidate = read_json(draft_path)
                else:
                    candidate = regenerate_candidate(
                        task, case, LLMRunner(gen_endpoint), feedback if arm == "feedback" else None,
                    )
                    save_json(draft_path, candidate)
                verdict_path = args.out / f"{case_id}.{arm}.verdict.json"
                if verdict_path.exists():
                    verdict = read_json(verdict_path)
                else:
                    verdict = blind_evaluate(task, case, candidate, LLMRunner(judge_endpoint))
                    save_json(verdict_path, verdict)
                result[arm] = {"generation": candidate, "evaluation": verdict}
            usage = [artifact["usage"] for arm in order for artifact in result[arm].values()]
            result["usage"] = {key: sum(item[key] for item in usage) for key in usage[0]}
        result["case_id"] = case_id
        save_json(path, result)
        print(f"{case_id}: {result['usage']['calls']} calls, ${result['usage']['cost_usd']:.5f}", flush=True)
        return result

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(run_case, cases))
    save_json(args.out / "results.json", results)
    render_summary(args.out, cases, results, args.phase)


if __name__ == "__main__":
    main()
