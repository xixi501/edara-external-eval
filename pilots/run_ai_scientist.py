"""Run AI-Scientist-v2 ideation on every core task and convert the outputs to arm JSONL."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

WORKSPACE = Path(os.environ.get("EDARA_WORKSPACE") or Path(__file__).resolve().parents[3])
REPO = Path(os.environ.get("EDARA_REFS_ROOT") or Path(__file__).resolve().parents[2])
WRAPPER = REPO / ".venv/bin/python"
LOGGED = REPO / "pilot3_runs/run_logged.py"
NATIVE = Path(os.environ.get("AI_SCIENTIST_NATIVE_ROOT") or (WORKSPACE / "staging/03_active_benchmarks/external-native-eval-20261109"))
SOURCE = NATIVE / "sources/ai-scientist-v2"
AISCI = Path(os.environ.get("AI_SCIENTIST_PYTHON") or (WORKSPACE / "runtime/ai-scientist-v2-ideation/bin/python"))
SCRIPT = SOURCE / "ai_scientist/perform_ideation_temp_free.py"
TOPICS = REPO / "edara_mvp_eval/outputs/hosts_v1/ai_scientist_topics"
SHIM = REPO / "edara_mvp_eval/outputs/hosts_v1/ai_scientist_shim"
OUT = REPO / "edara_mvp_eval/outputs/hosts_v1/ai_scientist_v1"

MISSING = (
    "aiib-10541", "aiib-33", "aiib-4717", "aiib-7144",
    "smallari_expansion40_002_unlearning_machine_language_use_algorithms__v02_expansion40",
    "smallari_expansion40_029_bottleneck_concept_concepts_information_cbms__v02_expansion40",
    "smallari_expansion40_040_critic_reinforcement_actor_critic_actor_continuo__v02_expansion40",
    "smallari_expansion40_011_test_time_adaptation_active_tta_test__v02_expansion40",
    "survey_06_llm_safety_jailbreak",
    "survey_08_kv_cache_compression",
    "ws_18_scientific_foundation_models",
    "ws_09_world_models_simulation",
)


def one(task: str) -> dict:
    work = OUT / task
    work.mkdir(parents=True, exist_ok=True)
    produced = work / f"{task}.json"
    if produced.exists() and json.loads(produced.read_text()):
        return {"task_id": task, "exit": 0, "ideas": len(json.loads(produced.read_text())),
                "log": "cached", "json": str(produced)}
    md = work / f"{task}.md"
    boundary = work / f"{task}.boundary.json"
    if not md.exists():
        shutil.copy2(TOPICS / f"{task}.md", md)
        shutil.copy2(TOPICS / f"{task}.boundary.json", boundary)
    last = {}
    for attempt, reflections in ((1, 3), (2, 1)):
        env = {**os.environ,
               "PYTHONPATH": f"{SHIM}:{NATIVE / 'common'}:{SOURCE}",
               "PYTHONDONTWRITEBYTECODE": "1",
               "AI_SCIENTIST_CASE_ID": task,
               "AI_SCIENTIST_BOUNDARY_PATH": str(boundary),
               "AI_SCIENTIST_SEARCH_LEDGER": str(work / f"search_{attempt}.jsonl"),
               "AI_SCIENTIST_TOOL_LEDGER": str(work / f"tool_{attempt}.jsonl"),
               "AI_SCIENTIST_LLM_LEDGER": str(work / f"llm_{attempt}.jsonl")}
        log = work / f"run_{attempt}.log"
        code = subprocess.call(
            [str(WRAPPER), str(LOGGED), str(log), str(AISCI), str(SCRIPT),
             "--model", "deepseek-flash", "--max-num-generations", "1",
             "--num-reflections", str(reflections), "--workshop-file", str(md)],
            cwd=str(REPO), env=env)
        ideas = json.loads(produced.read_text()) if produced.exists() else []
        last = {"task_id": task, "exit": code, "ideas": len(ideas), "log": str(log),
                "json": str(produced) if produced.exists() else None, "reflections": reflections}
        if ideas:
            return last
    return last


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--only", default=",".join(MISSING))
    args = parser.parse_args(argv)
    tasks = [task.strip() for task in args.only.split(",") if task.strip()]
    OUT.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        records = list(pool.map(one, tasks))
    rows = []
    seed = REPO / "pilot3_runs/ai_scientist_v2/ai_scientist_outputs.jsonl"
    for line in seed.read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
    for record in records:
        if not record["json"]:
            print(f"[ai-scientist] FAILED {record['task_id'][:50]} exit={record['exit']}")
            continue
        ideas = json.loads(Path(record["json"]).read_text())
        if not ideas:
            print(f"[ai-scientist] EMPTY {record['task_id'][:50]}")
            continue
        payload = ideas[0] if isinstance(ideas, list) else ideas
        rows.append({**payload, "task_id": record["task_id"]})
        print(f"[ai-scientist] ok {record['task_id'][:50]} keys={len(payload)}")
    (OUT / "ai_scientist_15.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n")
    print(f"[ai-scientist] rows={len(rows)} -> {OUT / 'ai_scientist_15.jsonl'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
