"""Frozen five-arm Stage B run using the existing evaluation framework."""

import argparse
import hashlib
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from itertools import combinations
from pathlib import Path

import numpy as np
from openai import APIError

from .baselines import run_retrieve_and_revise, run_self_refine
from .cli import _GENERATE_SYSTEM, _generate_direct, _load_imported_arm
from .epistemic_audit import run_full_audit
from .feedback import FeedbackMiddleware
from .governance import evidence_payload
from .literature_scout import _local_bm25_search, stage_search
from .llm_client import ModelEndpoint
from .middleware import TrustworthyNoveltyMiddleware
from .schemas import FullAuditBundle, PriorWorkItem, TaskSpec
from .stage_b_evidence import EvidenceStore, digest, native_prior_work, write_json
from .stage_b_meter import RecordedRunner
from .stage_b_review import AXES, compare, review_candidate, review_packet

ARMS = ("direct", "direct_self_refine", "direct_retrieve_revise", "direct_mw", "direct_feedback_mw")


def load(path):
    return json.loads(path.read_text())


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cached(path, produce):
    if path.exists():
        return load(path)
    answer = produce()
    write_json(path, answer)
    return answer


_RECOVERABLE = (APIError, ValueError, TimeoutError)


def _result(draft, audit, history=(), state=None):
    return {"candidate": draft, "audit": audit.model_dump(), "history": list(history),
            "governance_state": state, "status": "produced"}


class Experiment:
    def __init__(self, args):
        self.args = args
        self.gen = ModelEndpoint.from_env_or_profile(profile_path=args.gen_profile, prefix="EDARA_GEN")
        self.judge = ModelEndpoint.from_env_or_profile(profile_path=args.judge_profile, prefix="EDARA_JUDGE")
        self.out = args.out
        source = Path(__file__).parent
        protocol = source.parent.parent / "STAGE_B_PROTOCOL.md"
        assets = {
            str(p): file_hash(p)
            for p in (args.tasks_cache, args.development_cache, args.direct_inputs)
            if p is not None
        }
        manifest = {"sources": {p.name: file_hash(p) for p in sorted(source.glob("*.py"))},
                    "protocol": file_hash(protocol), "inputs": assets, "dry_run": args.dry_run,
                    "endpoints": {"generator": asdict(self.gen), "judge": asdict(self.judge)}, "arms": ARMS}
        path = self.out / "manifest.json"
        if path.exists() and digest(load(path)) != digest(manifest):
            raise RuntimeError("Frozen run signature differs. Use the matching inputs and code or a new output directory.")
        if not path.exists():
            write_json(path, manifest)
        self.development = {r["task_id"] for r in load(args.development_cache)}
        output_root = args.development_cache.parent
        historical = [
            output_root / "pilot3_full_compare_v3/results.json",
            output_root / "pilot3_full_compare_v4/results.json",
            output_root / "governance_pilot_v1/audit_final/results.json",
            output_root / "governance_feedback_v2/live/results.json",
            output_root / "tasks_mvp15.json",
            output_root.parent.parent / "pilot3_runs/direct/direct_outputs.jsonl",
        ]
        archive = {str(p): file_hash(p) for p in historical if p.is_file()}
        old_assets = self.out / "historical_inputs.json"
        if old_assets.exists() and load(old_assets) != archive:
            raise RuntimeError("Historical source hash changed during the run")
        if not old_assets.exists():
            write_json(old_assets, archive)
        self.tasks = [TaskSpec.model_validate(r).model_copy(update={"strict_evidence": True})
                      for r in load(args.tasks_cache)]
        self.native_provenance: dict[str, list] = {}
        if getattr(args, "citation_maps", None):
            self.tasks = self._inject_native(self.tasks, Path(args.citation_maps))
        self.imported = _load_imported_arm(args.direct_inputs) if args.direct_inputs else None
        ids = [task.task_id for task in self.tasks]
        if len(set(ids)) != len(ids) or (self.imported is not None and set(ids) != set(self.imported)):
            raise ValueError("Task IDs must be unique and match the frozen drafts exactly")

    def workdir(self, task):
        return self.out / "tasks" / task.task_id

    def _inject_native(self, tasks, root: Path):
        """Merge reconciled native sources into the shared per-task evidence pool."""
        injected = []
        for task in tasks:
            path = root / f"{task.task_id}.citation_map.json"
            if not path.exists():
                injected.append(task)
                continue
            artifact = root / f"{task.task_id}.artifact.json"
            meta = json.loads(artifact.read_text()) if artifact.exists() else {}
            items, provenance = native_prior_work(
                json.loads(path.read_text()),
                stage=meta.get("stage", "proposal"),
                source_db=meta.get("source_db", "native-export"),
                cutoff_date=task.cutoff_date,
            )
            self.native_provenance[task.task_id] = provenance
            known = {paper.id for paper in task.prior_work}
            merged = list(task.prior_work) + [item for item in items if item.id not in known]
            injected.append(task.model_copy(update={"prior_work": merged}))
        return injected

    def runner(self, task, arm, stage, *, generator=False):
        return RecordedRunner(self.gen if generator else self.judge,
                              self.workdir(task) / "requests" / arm / stage,
                              task_id=task.task_id, arm=arm, stage=stage, dry_run=self.args.dry_run)

    def prepare(self, task):
        path = self.workdir(task) / "prepared.json"

        def prepare_task():
            store = EvidenceStore(self.workdir(task), task, offline=self.args.dry_run)
            prepared = store.prepare()
            return {"task": prepared.model_dump(), "evidence_events": store.events}

        prepared = TaskSpec.model_validate(cached(path, prepare_task)["task"])
        print(f"[prepare] {task.task_id}: {len(prepared.prior_work)} admitted references", flush=True)
        return prepared

    def draft(self, task):
        def generate():
            try:
                raw = self.imported[task.task_id] if self.imported is not None else _generate_direct(
                    task, self.runner(task, "direct", "initial_generation", generator=True))[0]
                if not any(raw.get(k) for k in ("hypothesis", "method")):
                    return {"status": "generation_failed", "candidate": raw}
                return {"status": "produced", "candidate": raw, "sha256": digest(raw)}
            except _RECOVERABLE as exc:
                return {"status": "generation_failed", "error": type(exc).__name__}
        return cached(self.workdir(task) / "draft.json", generate)

    def produce(self, task):
        root = self.workdir(task)
        initial = self.draft(task)
        for arm in ARMS:
            row = cached(root / f"{arm}.json", lambda: self._produce_arm(root, task, arm, initial))
            print(f"[produce] {task.task_id} {arm}: {row['status']}", flush=True)
        return task

    def _produce_arm(self, root, task, arm, initial):
        if initial["status"] != "produced":
            return {"status": "no_initial_draft"}
        raw = initial["candidate"]
        if arm != "direct":
            direct = load(root / "direct.json")
            if not direct.get("audit"):
                return {"status": "initial_audit_failed", "candidate": raw}
            before = FullAuditBundle.model_validate(direct["audit"])
        store = EvidenceStore(root, task, offline=self.args.dry_run)
        found = {}

        def search(query):
            papers = store.search(query)
            found.update((p.id, p) for p in papers)
            return papers

        token = stage_search.set(search)
        audit = self.runner(task, arm, "governance_audit")
        gen = self.runner(task, arm, "revision", generator=True)
        history = []
        try:
            options = {"repair_runner": gen, "audit_runner": audit, "target_arm_name": arm}
            if arm == "direct":
                row = _result(raw, run_full_audit(task, arm, raw, audit, live_search=True))
            elif arm == "direct_self_refine":
                draft, scored, _ = run_self_refine(task, raw, before_bundle=before, **options)
                row = _result(draft, scored)
            elif arm == "direct_retrieve_revise":
                draft, scored, _ = run_retrieve_and_revise(task, before_bundle=before,
                                                           live_search=True, **options)
                row = _result(draft, scored)
            elif arm == "direct_mw":
                draft, _, scored, _ = TrustworthyNoveltyMiddleware(live_search=True).apply(
                    task, raw, initial_audit=before, history=history, **options)
                row = _result(draft, scored, history)
            else:
                def generate_from_feedback(packet):
                    return gen.chat_json(
                        _GENERATE_SYSTEM + "\nUse the diagnostic as advice about the old candidate. Preserve "
                        "any evidence-supported valid distinction. For regenerate, develop a substantively "
                        "new core relative to the closest dated work. For clarify, resolve the specific "
                        "mechanism or distinguishing-test defect. Do not present proposed results as observed.",
                        json.dumps(packet, ensure_ascii=False), temperature=0.3,
                        max_tokens=4096, tag="stage_b_feedback")
                draft, scored, history, papers, state = FeedbackMiddleware().apply(
                    task, raw, before, audit_runner=audit, repair_runner=gen,
                    generator_callback=generate_from_feedback, search=search)
                found.update((p.id, p) for p in papers)
                row = _result(draft, scored, history, state)
        except _RECOVERABLE as exc:
            row = {"status": "execution_failed", "error": type(exc).__name__, "history": history}
        finally:
            stage_search.reset(token)
        return {**row, "evidence": [p.model_dump() for p in found.values()], "search_events": store.events}

    def evaluate(self, task):
        root = self.workdir(task)
        rows = {arm: load(root / f"{arm}.json") for arm in ARMS}

        def freeze_pool():
            store = EvidenceStore(root, task, offline=self.args.dry_run)
            papers = list(task.prior_work)
            for row in rows.values():
                papers.extend(PriorWorkItem.model_validate(p) for p in row.get("evidence", []))
                if row["status"] != "produced":
                    continue
                proposal = row["audit"]["proposal"]
                for query in (proposal["title"], proposal["hypothesis"]):
                    papers.extend(store.search(query))
            admitted = store.admit(papers)
            selected = {p.id: p for p in task.prior_work}
            for row in rows.values():
                if row["status"] == "produced":
                    proposal = row["audit"]["proposal"]
                    query = proposal["title"] + " " + proposal["hypothesis"]
                    selected.update((p.id, p) for p in _local_bm25_search(query, admitted, top_k=4))
            return {"evidence": evidence_payload(task, list(selected.values())),
                    "all_admitted": [p.model_dump() for p in admitted], "events": store.events,
                    "candidates": {a: digest(r.get("candidate")) for a, r in rows.items()}}

        frozen = cached(root / "evaluation_evidence.json", freeze_pool)
        if frozen["candidates"] != {a: digest(r.get("candidate")) for a, r in rows.items()}:
            raise RuntimeError("Final candidate changed after the evaluation pool was frozen")
        packet = review_packet(task, frozen["evidence"])

        def verdict(arm):
            if rows[arm]["status"] != "produced":
                return {"status": "execution_failed"}
            try:
                return review_candidate(packet, rows[arm]["candidate"], self.runner(task, arm, "final_verdict"))
            except _RECOVERABLE as exc:
                return {"status": "invalid_review", "error": type(exc).__name__}

        verdicts = {arm: cached(root / "reviews" / f"{arm}.json", lambda: verdict(arm)) for arm in ARMS}

        def pair(a, b):
            if any(rows[arm]["status"] != "produced" for arm in (a, b)):
                return {"status": "missing_candidate"}
            try:
                return compare(packet, rows[a]["candidate"], rows[b]["candidate"],
                               self.runner(task, f"{a}--{b}", "pairwise"))
            except _RECOVERABLE as exc:
                return {"status": "invalid_review", "error": type(exc).__name__}

        pairs = []
        for a, b in combinations(ARMS, 2):
            key = f"{a}--{b}"
            pairs.append({"x": a, "y": b, **cached(root / "reviews" / f"{key}.json", lambda: pair(a, b))})
        print(f"[review] {task.task_id}: {len(verdicts)} verdicts, {len(pairs)} comparisons", flush=True)
        return {"task_id": task.task_id, "source": task.source_track,
                "development": task.task_id in self.development,
                "draft_sha256": load(root / "draft.json").get("sha256"),
                "arms": rows, "verdicts": verdicts, "pairwise": pairs}

    def run(self):
        with ThreadPoolExecutor(max_workers=self.args.workers) as pool:
            prepared = list(pool.map(self.prepare, self.tasks))
            write_json(self.out / "tasks_frozen.json", [t.model_dump() for t in prepared])
            drafts = list(pool.map(self.draft, prepared))
            write_json(self.out / "direct_frozen.json", drafts)
            list(pool.map(self.produce, prepared))
            if getattr(self.args, "produce_only", False):
                print("[produce] frozen all arm candidates; skipping final review", flush=True)
                return
            results = list(pool.map(self.evaluate, prepared))
        usage = [load(p) for p in sorted(self.out.glob("tasks/*/requests/*/*/*.json"))]
        write_json(self.out / "usage.json", [{k: v for k, v in row.items() if k != "response"} for row in usage])
        write_json(self.out / "results.json", results)
        if self.native_provenance:
            write_json(self.out / "evidence_provenance.json", self.native_provenance)
        total = sum((row["usage"] or {}).get("cost_usd", 0) for row in usage)
        (self.out / "summary.md").write_text(summarize(results, usage) + f"\nTotal metered API cost: ${total:.6f}.\n")


def summarize(results, usage):
    text = ["# Stage B five-arm evaluation", "",
            "Joint pass requires all three axes to pass. Denominators include scheduled tasks.",
            "One GLM model assessed candidates without seeing the internal audit.", ""]
    for title, subset in [
        ("All tasks", results), ("Development tasks", [r for r in results if r["development"]]),
        ("Expansion tasks", [r for r in results if not r["development"]]),
        *((s, [r for r in results if r["source"] == s]) for s in sorted({r["source"] for r in results})),
    ]:
        if not subset:
            continue
        text.extend([f"## {title} ({len(subset)})", "",
                      "| Arm | Joint | Honesty P/F/U | Credibility P/F/U | Novelty P/F/U | Execution/review failures |",
                      "|---|---:|---:|---:|---:|---:|"])
        for arm in ARMS:
            rows = [r["verdicts"][arm] for r in subset]
            joint = sum(r.get("joint_pass", False) for r in rows)
            counts = [Counter(r["axes"][axis]["status"] for r in rows if r.get("axes")) for axis in AXES]
            formatted = [" / ".join(str(c[s]) for s in ("pass", "fail", "unresolved")) for c in counts]
            failures = sum(r["status"] != "assessed" for r in rows)
            text.append(f"| {arm} | {joint}/{len(rows)} | " + " | ".join(formatted) + f" | {failures} |")
        text.extend(["", "| Feedback vs | Joint gain/loss | Delta | Task bootstrap 95% interval |",
                     "|---|---:|---:|---:|"])
        for arm in ARMS[:-1]:
            changes = np.array([int(r["verdicts"][ARMS[-1]].get("joint_pass", False))
                                - int(r["verdicts"][arm].get("joint_pass", False)) for r in subset])
            draws = np.random.default_rng(20261117).choice(changes, (10000, len(changes))).mean(axis=1)
            bounds = np.quantile(draws, [.025, .975])
            text.append(f"| {arm} | {(changes == 1).sum()}/{(changes == -1).sum()} | "
                        f"{changes.mean():+.1%} | [{bounds[0]:+.1%}, {bounds[1]:+.1%}] |")
        text.append("")
    text.extend(["## Usage", "", "| Arm | Stage | Attempts | Prompt tokens | Completion tokens | Metered USD | Unknown usage |",
                 "|---|---|---:|---:|---:|---:|---:|"])
    for a, s in sorted({(u["arm"], u["stage"]) for u in usage if u["stage"] != "pairwise"} | {("all", "pairwise")}):
        entries = [u for u in usage if u["stage"] == s and (s == "pairwise" or u["arm"] == a)]
        totals = Counter()
        for u in entries:
            totals.update(u["usage"] or {})
        text.append(f"| {a} | {s} | {len(entries)} | {totals['prompt_tokens']} | "
                    f"{totals['completion_tokens']} | {totals['cost_usd']:.6f} | {sum(u['usage'] is None for u in entries)} |")
    text.extend(["", "## Overall blind preferences", ""])
    for a, b in combinations(ARMS, 2):
        pairs = [p for r in results for p in r["pairwise"] if (p["x"], p["y"]) == (a, b)]
        count = Counter(p.get("winner", "invalid") for p in pairs)
        sensitive = sum(p.get("order_sensitive", False) for p in pairs)
        text.append(f"- {a} / {b}: {count['A']} wins, {count['B']} losses, {count['tie']} ties "
                    f"({sensitive} order-sensitive), {count['invalid']} unavailable.")
    text.extend(["", "## Feedback failures and adverse comparisons", ""])
    for r in results:
        verdict = r["verdicts"][ARMS[-1]]
        lost = [p["x"] for p in r["pairwise"] if p["y"] == ARMS[-1] and p.get("winner") == "A"]
        if not verdict.get("joint_pass") or lost:
            axes = ", ".join(f"{axis}={v['status']}" for axis, v in verdict.get("axes", {}).items())
            text.append(f"- {r['task_id']}: {axes or verdict['status']}; preferred alternatives: {', '.join(lost) or 'none'}.")
    return "\n".join(text) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks-cache", type=Path, required=True)
    parser.add_argument("--development-cache", type=Path, default=Path("edara_mvp_eval/outputs/tasks_pilot3.json"))
    parser.add_argument("--direct-inputs", type=Path)
    parser.add_argument("--citation-maps", type=Path,
                        help="Directory of reconciled citation maps; merges native sources "
                             "into the shared per-task evidence pool.")
    parser.add_argument("--gen-profile", type=Path)
    parser.add_argument("--judge-profile", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--produce-only", action="store_true",
                        help="Freeze every arm candidate without running the final review.")
    args = parser.parse_args()
    Experiment(args).run()


if __name__ == "__main__":
    main()
