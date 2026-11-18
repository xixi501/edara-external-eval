"""Supervise native E-DARA runs for the twelve core tasks that have no artifact yet.

Each task gets at most two fresh attempts plus one arXiv continuation. Every attempt
runs through `run_logged.py`, so credentials are injected and redacted from the log.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import signal
import sqlite3
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path("/workspace/edara/staging/04_external_eval_refs")
PY = Path("/workspace/edara/runtime/core/bin/python")
LOGGED = REPO / "pilot3_runs/run_logged.py"
WORK = Path("/workspace/edara/staging/04_external_eval_refs/edara_mvp_eval")
CACHE = WORK / "outputs/tasks_mvp15.json"
INPUTS = WORK / "outputs/hosts_v1/edara_cases"
CONFIG = REPO / "pilot3_runs/edara/aiib82-arxiv-config.json"
PARSER = REPO / "pilot3_runs/edara/aiib82-config.json"
OUT = WORK / "outputs/hosts_v1/native15"
CONTINUE = REPO / "pilot3_runs/edara/switch_arxiv_and_run.py"
FRAMEWORK = "/workspace/edara/repos/framework/src"
ENV = {**os.environ, "PYTHONPATH": FRAMEWORK, "PYTHONDONTWRITEBYTECODE": "1"}

TASKS = (
    "smallari_expansion40_029_bottleneck_concept_concepts_information_cbms__v02_expansion40",
    "smallari_expansion40_040_critic_reinforcement_actor_critic_actor_continuo__v02_expansion40",
    "smallari_expansion40_011_test_time_adaptation_active_tta_test__v02_expansion40",
    "survey_06_llm_safety_jailbreak",
    "survey_08_kv_cache_compression",
    "ws_18_scientific_foundation_models",
    "ws_09_world_models_simulation",
)
FINAL = {"plan_reviewed", "proposal_reviewed", "design_ready", "inconclusive"}
CONTINUABLE = {"paused", "paused_budget", "needs_user_decision", "dispatch_failed", "needs_resources"}


def run(cmd: list[str], log: Path) -> int:
    return subprocess.call([str(PY), str(LOGGED), str(log), *cmd], cwd=str(REPO), env=ENV)


def research_db(root: Path) -> str | None:
    found = glob.glob(str(root / "topics" / "*" / "research.db"))
    return found[0] if found else None


def proposal_ready(root: Path) -> bool:
    db = research_db(root)
    if not db:
        return False
    try:
        connection = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        count = connection.execute(
            "select count(*) from research_versions where kind in ('proposal','experiment_plan')").fetchone()[0]
        connection.close()
        return count > 0
    except sqlite3.Error:
        return False


def attempt(task: str, index: int, budget: float) -> dict:
    root = OUT / task / f"attempt{index}"
    log = REPO / f"pilot3_runs/edara/native15_{task[:40]}_a{index}.log"
    cmd = [str(PY), str(LOGGED), str(log), str(PY), str(WORK / "pilots/run_native.py"),
           "--task-id", task, "--tasks-cache", str(CACHE),
           "--input-root", str(INPUTS), "--config", str(CONFIG), "--case-root", str(root),
           "--budget-usd", str(budget), "--supplement-source", "openalex"]
    process = subprocess.Popen(cmd, cwd=str(REPO), env=ENV,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                               start_new_session=True)
    stopped = False
    deadline = time.time() + 3600
    while process.poll() is None and time.time() < deadline:
        if proposal_ready(root):
            try:
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            except ProcessLookupError:
                pass
            stopped = True
            break
        time.sleep(15)
    if process.poll() is None:
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        except ProcessLookupError:
            pass
    try:
        code = process.wait(timeout=90)
    except subprocess.TimeoutExpired:
        os.killpg(os.getpgid(process.pid), signal.SIGKILL)
        code = -9
    result = root / "policy-result.json"
    payload = json.loads(result.read_text()) if result.exists() else {}
    return {"attempt": index, "exit": code, "status": payload.get("status") or
            ("stopped_after_proposal" if stopped and proposal_ready(root) else "no_proposal"),
            "proposal": proposal_ready(root), "root": str(root), "log": str(log),
            "error_code": payload.get("error_code")}


def continue_arxiv(task: str, previous: dict) -> dict:
    bootstrap = Path(previous["root"]) / "bootstrap-result.json"
    if not bootstrap.exists():
        return {"continued": False, "reason": "no bootstrap result"}
    workspace = json.loads(bootstrap.read_text()).get("workspace")
    if not workspace:
        return {"continued": False, "reason": "no workspace"}
    log = REPO / f"pilot3_runs/edara/native15_{task[:40]}_arxiv.log"
    code = run([str(PY), str(CONTINUE), task, workspace], log)
    result = Path(previous["root"]) / "policy-result.json"
    payload = json.loads(result.read_text()) if result.exists() else {}
    return {"continued": True, "exit": code, "status": payload.get("status"), "log": str(log)}


def handle(task: str, budget: float) -> dict:
    record = {"task_id": task, "attempts": []}
    import re
    attempts = len(list((OUT / task).glob("attempt*"))) if (OUT / task).exists() else 0
    logs = [int(match.group(1)) for path in (REPO / "pilot3_runs/edara").glob(f"native15_{task[:40]}_a*.log")
            if (match := re.search(r"_a(\d+)\.log$", path.name))]
    used = max([attempts, max(logs, default=0)])
    start = used + 1
    for index in (start, start + 1):
        if (OUT / task / f"attempt{index}" / "policy-result.json").exists():
            continue
        info = attempt(task, index, budget)
        record["attempts"].append(info)
        if info.get("proposal") or info["status"] in FINAL:
            break
        if index == start and info["status"] in CONTINUABLE:
            record["continuation"] = continue_arxiv(task, info)
            if record["continuation"].get("status") in FINAL:
                break
    return record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--budget-usd", type=float, default=2.0)
    parser.add_argument("--only", default=",".join(TASKS))
    args = parser.parse_args(argv)
    tasks = [task.strip() for task in args.only.split(",") if task.strip()]
    OUT.mkdir(parents=True, exist_ok=True)
    started = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        records = list(pool.map(lambda task: handle(task, args.budget_usd), tasks))
    (OUT / "supervisor.json").write_text(json.dumps(
        {"tasks": records, "elapsed_seconds": round(time.time() - started, 1)}, ensure_ascii=False, indent=2))
    for record in records:
        statuses = [item.get("status") for item in record["attempts"]]
        print(f"[supervisor] {record['task_id'][:50]:50s} attempts={statuses} "
              f"cont={record.get('continuation', {}).get('status')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
