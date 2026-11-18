"""Re-evaluate frozen candidates without regenerating them.

A run directory already contains frozen candidates, a frozen evidence pool and a
request ledger. This entry point replays only the final review under a named
evaluation version, so a change in scores can be attributed to the review rather
than to a new generation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from itertools import combinations
from pathlib import Path

from .llm_client import ModelEndpoint
from .schemas import TaskSpec
from .stage_b_evidence import digest, write_json
from .stage_b_meter import RecordedRunner
from .stage_b_review import AXES, RUBRICS, review_candidate, review_packet

EVAL_VERSION = "stage-b-eval/v2"
DEFAULT_ARMS = ("direct", "direct_self_refine", "direct_retrieve_revise", "direct_mw", "direct_feedback_mw")


class ReevalError(RuntimeError):
    pass


def load(path: Path):
    return json.loads(path.read_text())


def rubric_hash(name: str = "v2") -> str:
    rubric_text, format_text = RUBRICS[name]
    return hashlib.sha256((rubric_text + format_text).encode()).hexdigest()


def load_run(run: Path) -> tuple[list[TaskSpec], dict[str, dict]]:
    tasks = [TaskSpec.model_validate(row) for row in load(run / "tasks_frozen.json")]
    payloads = {}
    for task in tasks:
        root = run / "tasks" / task.task_id
        frozen = load(root / "evaluation_evidence.json")
        rows = {}
        for arm in DEFAULT_ARMS:
            path = root / f"{arm}.json"
            if not path.exists():
                rows[arm] = {"status": "missing"}
                continue
            row = load(path)
            expected = frozen.get("candidates", {}).get(arm)
            if expected != digest(row.get("candidate")):
                raise ReevalError(f"{task.task_id}/{arm}: candidate does not match the frozen digest")
            rows[arm] = row
        payloads[task.task_id] = {"frozen": frozen, "rows": rows}
    return tasks, payloads


def review_run(run: Path, out: Path, *, judge: ModelEndpoint, repeat: int, arms: tuple[str, ...],
               dry_run: bool, retry: bool, workers: int = 1, rubric: str = "v2") -> dict:
    tasks, payloads = load_run(run)
    usage = []

    def review_task(task):
        entry = payloads[task.task_id]
        packet = review_packet(task, entry["frozen"]["evidence"])
        arm_rows = {}
        for arm in arms:
            row = entry["rows"][arm]
            if row.get("status") != "produced":
                arm_rows[arm] = {"status": "no_candidate"}
                continue
            draws = []
            for index in range(repeat):
                runner = RecordedRunner(judge, out / "requests" / task.task_id / arm / f"r{index}",
                                        task_id=task.task_id, arm=arm, stage=f"final_verdict_r{index}",
                                        dry_run=dry_run)
                draws.append(review_candidate(packet, row["candidate"], runner, retry=retry, rubric=rubric))
            arm_rows[arm] = {"status": "assessed", "primary": draws[0], "repeats": draws}
        print(f"[reeval] {task.task_id}: {len(arm_rows)} arms x {repeat} repeats", flush=True)
        return {"task_id": task.task_id, "source": task.source_track, "arms": arm_rows}

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        results = list(pool.map(review_task, tasks))
    stability = summarize_stability(results, arms)
    payload = {"eval_version": EVAL_VERSION, "rubric": rubric, "results": results,
               "stability": stability, "arms": list(arms), "repeat": repeat}
    write_json(out / "results.json", payload)
    for path in sorted((out / "requests").glob("**/*.json")):
        record = load(path)
        usage.append({key: record.get(key) for key in
                      ("task_id", "arm", "stage", "tag", "request_number", "request_sha256",
                       "status", "usage")})
    write_json(out / "usage.json", usage)
    total = sum((row["usage"] or {}).get("cost_usd", 0) for row in usage)
    (out / "summary.md").write_text(render_summary(payload) + f"\nTotal metered API cost: ${total:.6f}.\n")
    return payload


def summarize_stability(results, arms) -> dict:
    joint_flips = axis_flips = dims = assessed = 0
    for result in results:
        for arm in arms:
            entry = result["arms"][arm]
            if entry.get("status") != "assessed":
                continue
            assessed += 1
            draws = entry["repeats"]
            joint = {draw.get("joint_pass") for draw in draws}
            if len(joint) > 1:
                joint_flips += 1
            for axis in AXES:
                dims += 1
                statuses = {draw.get("axes", {}).get(axis, {}).get("status") for draw in draws}
                if len(statuses) > 1:
                    axis_flips += 1
    return {"assessed": assessed, "joint_flips": joint_flips, "axis_flips": axis_flips, "axis_dims": dims}


def _counts(entry) -> dict:
    axes = (entry.get("primary") or {}).get("axes", {})
    return {axis: axes.get(axis, {}).get("status", "missing") for axis in AXES}


def render_summary(payload: dict) -> str:
    results = payload["results"]
    arms = payload["arms"]
    text = ["# Stage B re-evaluation (frozen candidates)", "",
            f"Evaluation version: {EVAL_VERSION}. Repeats per verdict: {payload['repeat']}.",
            "Joint pass requires all three axes to pass. Status changes that follow a failed "
            "anchor check are reported separately from scientific unresolved labels.", ""]
    text.extend(["| Arm | Joint (primary) | Honesty P/F/U | Credibility P/F/U | Novelty P/F/U |",
                 "|---|---:|---|---|---|"])
    for arm in arms:
        rows = [_counts(result["arms"][arm]) for result in results
                if result["arms"][arm].get("status") == "assessed"]
        joint = sum(1 for result in results if result["arms"][arm].get("status") == "assessed"
                    and (result["arms"][arm]["primary"] or {}).get("joint_pass"))
        cells = []
        for axis in AXES:
            cells.append(" / ".join(str(sum(1 for row in rows if row[axis] == status))
                                    for status in ("pass", "fail", "unresolved")))
        text.append(f"| {arm} | {joint}/{len(results)} | " + " | ".join(cells) + " |")
    validations: dict[str, int] = {}
    modes: dict[str, int] = {}
    for result in results:
        for arm in arms:
            entry = result["arms"][arm]
            if entry.get("status") != "assessed":
                continue
            for axis in AXES:
                axis_row = (entry["primary"] or {}).get("axes", {}).get(axis, {})
                if axis_row.get("validation"):
                    validations[axis_row["validation"]] = validations.get(axis_row["validation"], 0) + 1
                if axis_row.get("span_mode"):
                    modes[axis_row["span_mode"]] = modes.get(axis_row["span_mode"], 0) + 1
    stability = payload["stability"]
    text.extend(["", "## Anchor and stability diagnostics", "",
                 f"- anchor modes: {json.dumps(modes, sort_keys=True)}",
                 f"- review failures: {json.dumps(validations, sort_keys=True)}",
                 f"- assessed verdicts: {stability['assessed']}",
                 f"- joint verdicts that changed across repeats: {stability['joint_flips']}",
                 f"- axis labels that changed across repeats: {stability['axis_flips']}"
                 f"/{stability['axis_dims']}", ""])
    return "\n".join(text)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--judge-profile", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--only-arms", default=",".join(DEFAULT_ARMS))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-retry", action="store_true")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--rubric", choices=sorted(RUBRICS), default="v2")
    args = parser.parse_args(argv)

    arms = tuple(arm.strip() for arm in args.only_arms.split(",") if arm.strip())
    unknown = [arm for arm in arms if arm not in DEFAULT_ARMS]
    if unknown:
        parser.error(f"unknown arms: {unknown}")
    if args.out.resolve() == args.run.resolve() or args.run.resolve() in args.out.resolve().parents:
        parser.error("write the re-evaluation outside the original run directory")
    judge = ModelEndpoint.from_env_or_profile(profile_path=args.judge_profile, prefix="EDARA_JUDGE")
    manifest = {"eval_version": EVAL_VERSION, "rubric": args.rubric,
                "rubric_sha256": rubric_hash(args.rubric),
                "normalization_version": "v2", "judge": asdict(judge), "repeat": args.repeat,
                "arms": list(arms), "dry_run": args.dry_run,
                "source_manifest_sha256": digest(load(args.run / "manifest.json"))
                if (args.run / "manifest.json").exists() else None}
    args.out.mkdir(parents=True, exist_ok=True)
    existing = args.out / "eval_manifest.json"
    if existing.exists() and digest(load(existing)) != digest(manifest):
        raise ReevalError("this output directory was produced under a different evaluation configuration")
    write_json(existing, manifest)
    payload = review_run(args.run, args.out, judge=judge, repeat=args.repeat, arms=arms,
                         dry_run=args.dry_run, retry=not args.no_retry, workers=args.workers,
                         rubric=args.rubric)
    stability = payload["stability"]
    print(f"[reeval] assessed={stability['assessed']} joint_flips={stability['joint_flips']} "
          f"axis_flips={stability['axis_flips']}/{stability['axis_dims']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
