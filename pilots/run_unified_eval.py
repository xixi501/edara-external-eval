"""Run Lit2Test + AlphaIdeaBench + ClaimNov unified calibration or read-only re-evaluation."""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from edara_mvp_eval.llm_client import LLMRunner, ModelEndpoint
from edara_mvp_eval.paired_inputs import Evidence
from edara_mvp_eval.paired_trial import COMPARISONS, ROOT, TrialRunner, WORKSPACE, shared_full_pool
from edara_mvp_eval.schemas import PriorWorkItem, TaskSpec
from edara_mvp_eval.stage_b_evidence import write_json
from edara_mvp_eval.unified_eval import (
    CORE_AIB_DIMENSIONS,
    build_lit2test_calibration_suite,
    calibrate_unified_evaluator,
    evaluate_unified_pair,
    summarize_unified_rows,
)


def _make_runner(endpoint: ModelEndpoint, log_dir: Path, job: str, stage: str, limit: int, *, dry_run: bool):
    if dry_run:
        return LLMRunner(endpoint, dry_run=True)
    return TrialRunner(endpoint, log_dir, task_id=job, arm="blind", stage=stage, limit=limit)


def run_calibrate(args: argparse.Namespace, endpoint: ModelEndpoint) -> int:
    cases = build_lit2test_calibration_suite(ROOT, limit_per_kind=args.limit_per_kind)

    def make_runners(case_id: str):
        base = args.out / "calibration" / case_id
        return [
            _make_runner(endpoint, base / order, case_id, order, 6, dry_run=args.dry_run)
            for order in ("forward", "reverse")
        ]

    report = calibrate_unified_evaluator(cases, make_runners)
    write_json(args.out / "calibration_unified.json", report)
    print(
        f"[unified-calibrate] substantive={report['substantive_correct']}/{report['substantive_total']} "
        f"sham={report['sham_ties']}/{report['sham_total']} qualified={report['qualified']}",
        flush=True,
    )
    print(f"[unified-calibrate] target_dim_deltas={report['mean_target_dimension_deltas']}", flush=True)
    return 0 if report["qualified"] else 3


def run_reeval(args: argparse.Namespace, endpoint: ModelEndpoint) -> int:
    source = args.source_run.resolve()
    inputs_path = source / f"inputs_{args.cohort}.json"
    if not inputs_path.is_file():
        raise FileNotFoundError(f"Missing frozen cohort inputs: {inputs_path}")
    topics = json.loads(inputs_path.read_text(encoding="utf-8"))
    allowed_hosts = set(args.hosts.split(",")) if args.hosts else {"direct", "ai_scientist", "edara"}

    jobs = []
    for task_id, seed in topics.items():
        for host, rec in seed["originals"].items():
            if rec["status"] != "ready" or host not in allowed_hosts:
                continue
            vdir = source / "versions" / task_id / host
            if all((vdir / f"{arm}.json").is_file() for arm in ("B", "C", "D", "E")):
                jobs.append((task_id, host, rec, seed, vdir))
    if args.limit:
        jobs = jobs[: args.limit]

    cmps = (args.comparison,) if args.comparison != "all" else tuple(COMPARISONS.keys())

    def one(item):
        task_id, host, rec, seed, vdir = item
        job = f"{task_id}/{host}"
        task = TaskSpec.model_validate(seed["task"])
        versions = {arm: json.loads((vdir / f"{arm}.json").read_text(encoding="utf-8")) for arm in ("B", "C", "D", "E")}
        full_packet, full_drafts = shared_full_pool(task, seed["evidence"], versions)
        local_ev = Evidence.restore(task, seed["evidence"])
        local_packet = local_ev.packet()
        local_corpus = local_ev.audit_task().prior_work
        full_corpus = list(local_corpus)
        seen_ids = {p.id for p in full_corpus}
        for arm in ("D", "E"):
            for r in versions[arm].get("retrieved") or []:
                p = PriorWorkItem.model_validate(r["paper"])
                if p.id not in seen_ids:
                    full_corpus.append(p)
                    seen_ids.add(p.id)

        unified = {}
        for label in cmps:
            arms = COMPARISONS[label]
            packet = local_packet if label == "C_vs_B" else full_packet
            drafts = ({arm: versions[arm]["draft"] for arm in arms} if label == "C_vs_B" else full_drafts)
            corpus = local_corpus if label == "C_vs_B" else full_corpus
            rdir = args.out / "versions" / task_id / host / "review" / label
            pair_runners = [
                _make_runner(endpoint, rdir / "lit2test" / o, job, o, 6, dry_run=args.dry_run)
                for o in ("forward", "reverse")
            ]
            aib_runner = (
                None if args.skip_aib
                else _make_runner(endpoint, rdir / "alphaideabench", job, "alphaideabench", 24, dry_run=args.dry_run)
            )
            cn_runner = (
                None if args.skip_claimnov
                else _make_runner(endpoint, rdir / "claimnov", job, "claimnov", 64, dry_run=args.dry_run)
            )
            unified[label] = evaluate_unified_pair(
                drafts[arms[0]], drafts[arms[1]], task, packet,
                pair_runners=pair_runners, aib_runner=aib_runner, claimnov_runner=cn_runner,
                prior_corpus=corpus, aib_dimensions=CORE_AIB_DIMENSIONS,
            )
        print(f"[unified-eval] {job} done", flush=True)
        return {
            "task_id": task_id,
            "host": host,
            "stage": rec.get("stage", "proposal"),
            "development": seed.get("development", False),
            "unified_comparisons": unified,
        }

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(one, jobs))

    summaries = {cmp: summarize_unified_rows(rows, cmp) for cmp in cmps}
    write_json(args.out / f"{args.cohort}_unified_results.json", {"results": rows, "summaries": summaries})
    print(json.dumps(summaries, ensure_ascii=False, indent=2), flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--calibrate", action="store_true")
    parser.add_argument("--limit-per-kind", type=int, default=4)
    parser.add_argument("--source-run", type=Path)
    parser.add_argument("--cohort", default="core")
    parser.add_argument("--comparison", choices=("C_vs_B", "E_vs_D", "all"), default="C_vs_B")
    parser.add_argument("--hosts", default="")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--skip-aib", action="store_true")
    parser.add_argument("--skip-claimnov", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--judge-profile", type=Path, default=WORKSPACE / "data/p1b-critic-profile.json")
    args = parser.parse_args()
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    endpoint = ModelEndpoint() if args.dry_run else ModelEndpoint.from_env_or_profile(profile_path=args.judge_profile)
    if not args.dry_run and not endpoint.resolve_api_key():
        raise RuntimeError(f"Missing judge model credential in {endpoint.credential_env}")
    if args.calibrate:
        return run_calibrate(args, endpoint)
    if not args.source_run:
        parser.error("Provide --calibrate or --source-run")
    return run_reeval(args, endpoint)


if __name__ == "__main__":
    raise SystemExit(main())
