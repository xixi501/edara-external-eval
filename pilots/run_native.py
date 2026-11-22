"""Run the native E-DARA Policy loop for one frozen benchmark task.

Generalises `pilot3_runs/edara/run_topic.py` to any task in a task cache. AIIB cases
seed from the released PDFs, every other track seeds from the exported markdown
reference bundle. Each attempt gets a fresh workspace because the native bootstrap
never adopts a historical run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

from edara.framework.bench.ideabench_run import initialize_research
from edara.framework.policy.bootstrap import bootstrap
from edara.framework.policy.delivery import export
from edara.framework.policy.loop import PolicyLoop
from edara.framework.policy.runtime import build, freeze_runtime_config, load_profile
from edara.framework.scout.boundary import FILE, validate
from edara.framework.ui.policy_worker import validate_research_preparation

WORKSPACE = Path(os.environ.get("EDARA_WORKSPACE") or Path(__file__).resolve().parents[3])
AIIB_INPUTS = Path(os.environ.get("AIIB_INPUTS_DIR") or (WORKSPACE / "data/aiib-pilot-20261105/prepared/inputs"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pick_documents(task_id: str, source_track: str, case_dir: Path) -> list[Path]:
    markdown = sorted((case_dir / "inputs").glob("ref_*.md"))
    if markdown:
        return markdown
    if source_track == "aiib":
        return sorted((AIIB_INPUTS / task_id).glob("*.pdf"))
    return []


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--tasks-cache", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, required=True, help="Directory of exported edara_cases")
    parser.add_argument("--config", type=Path, required=True, help="edara_live_config.json")
    parser.add_argument("--case-root", type=Path, required=True, help="Fresh attempt directory")
    parser.add_argument("--parser-config", type=Path, help="Config carrying parser_config for PDF seeds")
    parser.add_argument("--budget-usd", type=float, default=1.5)
    parser.add_argument("--completion-target", default="plan_reviewed")
    parser.add_argument("--supplement-source", default=None,
                        help="Override the supplementary literature provider (openalex or arxiv).")
    parser.add_argument("--no-supplementary-search", action="store_true",
                        help="Skip the mandatory supplementary-search requirement when providers are unavailable.")
    args = parser.parse_args(argv)

    tasks = json.loads(args.tasks_cache.read_text())
    task = next((row for row in tasks if row["task_id"] == args.task_id), None)
    if task is None:
        raise SystemExit(f"unknown task {args.task_id}")
    case = json.loads((args.input_root / f"{args.task_id}.json").read_text())
    case_dir = args.input_root / args.task_id
    boundary = validate(json.loads((case_dir / FILE).read_text()))
    documents = pick_documents(args.task_id, task["source_track"], case_dir)
    if not documents:
        raise SystemExit(f"no seed documents for {args.task_id}")
    hashes = [sha256(path) for path in documents]
    if hashes != boundary["seed_document_sha256s"]:
        raise SystemExit("seed document hashes do not match the literature boundary")
    if boundary["cutoff_date"] != task["cutoff_date"]:
        raise SystemExit("boundary cutoff differs from the task cutoff")

    config = json.loads(args.config.read_text())
    if args.supplement_source:
        config["supplement_source"] = args.supplement_source
    if any(path.suffix == ".pdf" for path in documents) and "parser_config" not in config:
        if args.parser_config is None:
            raise SystemExit("PDF seeds require parser_config")
        config["parser_config"] = json.loads(args.parser_config.read_text())["parser_config"]
    config.update(mode="live", execute=True, allow_new_version=True,
                  authorization_scope="research_plan",
                  research_tree={"protocol": "priority-dfs/v1",
                                 "completion_target": args.completion_target},
                  evidence_search_policy="idea-first-evidence/v3",
                  policy_protocol="research/offers-v2", policy_guidance_version="goal-context/v5",
                  rubric_policy="stage-rubrics/v9", author_review_routing="author-review/v1",
                  literature_boundary=boundary)
    budget = int(round(args.budget_usd * 1_000_000))
    config["budget_microusd"] = budget
    validate_research_preparation(freeze_runtime_config(config))

    args.case_root.mkdir(parents=True, exist_ok=False)
    constraints = [
        f"Supplementary literature cutoff: {boundary['cutoff_date']} (inclusive).",
        f"Benchmark experiment budget and constraints: {json.dumps(task['budget'], ensure_ascii=False)}",
        "Deliver one concrete, falsifiable design with a baseline, a decisive metric, and a disconfirming outcome.",
    ]
    start = bootstrap(
        args.case_root / "topics", case["topic"], files=documents, mode="live",
        budget_microusd=budget, idempotency_key=f"native15-startup:{args.task_id}",
        user_constraints=constraints, out_of_scope=["Experiment execution and private resources."],
        require_supplementary_search=not args.no_supplementary_search,
        evidence_search_policy=config["evidence_search_policy"],
        planning_context=task["prompt_context"],
        seed_document_titles={digest: ref["title"]
                              for digest, ref in zip(hashes, case.get("references") or [], strict=False)},
        planner_profile=load_profile(config["topic_model_config"]),
        audit_profile=load_profile(config["audit_model_config"]),
    )
    (args.case_root / "bootstrap-result.json").write_text(json.dumps(start, ensure_ascii=False, indent=2))
    print("Bootstrap status:", start["status"], flush=True)
    if start["status"] != "bootstrapped":
        return 2

    workspace = Path(start["workspace"])
    with (workspace / FILE).open("x", encoding="utf-8") as stream:
        json.dump(boundary, stream)
    config = initialize_research(
        workspace, config, case_id=args.task_id, contract_version=start["contract_version"],
        seed_hashes=hashes, artifact_root=args.case_root,
        reason="Start one frozen benchmark case from its supplied references and research budget.",
    )
    runtime = build(workspace, config)
    try:
        result = PolicyLoop(
            workspace, runtime.policy, runtime.dispatcher, mode="live",
            budget_microusd=budget, allow_new_version=True, execute=True,
            audit_model=runtime.audit, config=runtime.config,
        ).run()
    finally:
        runtime.close()
    (args.case_root / "policy-result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print("Policy status:", result["status"])
    receipt = export(workspace, args.case_root / "delivery")
    (args.case_root / "delivery.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2))
    print("Delivery status:", receipt["delivery_status"])
    return 0 if result["status"] in ("plan_reviewed", "proposal_reviewed") else 3


if __name__ == "__main__":
    sys.exit(main())
