"""Versioned direction-known and style-control cases for the v1 judge gate."""

from __future__ import annotations

from pathlib import Path

from .paired_inputs import read_rows

TOPICS = ("cache eviction", "sparse decoding", "batch packing", "vector ranking", "query scheduling",
          "gradient sampling", "queue admission", "retrieval filtering", "data shuffling",
          "quantization calibration", "beam pruning", "checkpoint selection")
RESERVE_TOPICS = ("page prefetch", "tree traversal", "task placement", "index compaction", "priority queues",
                  "memory allocation", "adaptive buffering", "graph partitioning", "error recovery",
                  "load balancing", "incremental parsing", "index rebuilding")
CONTEXT = {"title": "Controlled evaluation of a computational method", "cutoff_date": "2024-01-01"}


def source(passage: str) -> list[dict]:
    return [{"paper_id": "S001", "title": "Supplied calibration reference",
             "attribution_passages": [passage], "novelty_passages": [passage], "dates": ["2022"]}]


def honesty_case(topic: str, idx: int, sham: bool) -> tuple[dict, dict, list[dict], str]:
    count = 12 + idx
    if idx % 3 == 0:
        paper = f"The study of {topic} used {count} workloads."
        before = f"The published study of {topic} used {count + 19} workloads [S001]."
        after = f"The published study of {topic} used {count} workloads [S001]."
        style = f"The published study used {count + 19} workloads to evaluate {topic} [S001]."
    elif idx % 3 == 1:
        paper = f"The reported benefit for {topic} was measured only in offline simulations."
        before = f"The prior study measured the benefit of {topic} in production deployments [S001]."
        after = f"The prior study measured the benefit of {topic} only in offline simulations [S001]."
        style = f"Production deployments were where the prior study measured the benefit of {topic} [S001]."
    else:
        paper = f"The controller for {topic} was originally introduced by the team that wrote this paper."
        before = f"We introduce a new controller for {topic}, namely the controller described in [S001]."
        after = f"We reuse the controller for {topic} introduced in [S001]."
        style = f"Our novel contribution is the controller for {topic} that [S001] describes."
    return {"claim": before}, {"claim": style if sham else after}, source(paper), "tie" if sham else "win"


def novelty_case(topic: str, idx: int, sham: bool) -> tuple[dict, dict, list[dict], str]:
    if idx % 2:
        paper = (f"Our {topic} policy uses a fixed global threshold under uniform workloads. "
                 "The policy does not adapt to changes in workload groups.")
        original = f"Use a fixed global threshold for {topic} under uniform workloads."
        gain = (f"Use a fixed global threshold for {topic} as the baseline. "
                "Add a policy that estimates each workload group's change rate independently "
                "and adapts the threshold in response to a shift in that group.")
        style = f"For {topic}, use one global, fixed threshold when workloads are uniform."
    else:
        paper = (f"The published {topic} method makes decisions only from the current observation. "
                 "It stores no observations from preceding time windows.")
        original = f"Use the current observation alone to make each {topic} decision."
        gain = (f"Use the current-observation {topic} method as the baseline. "
                "Add a second decision rule that compares the last two windows and waits for "
                "directional agreement before changing the decision.")
        style = f"Make each {topic} decision solely from the current observation."
    return {"contribution": original}, {"contribution": style if sham else gain}, source(paper), "tie" if sham else "win"


def benchmark_designs(root: Path, reserve: bool) -> list[dict]:
    path = root.parent / "lit2test-benchmark/results/proposals/fifth40_gpt52_proposals.jsonl"
    fields = ("literature_gap", "hypothesis", "minimal_test", "decisive_metric", "supporting_result",
              "falsifying_result")
    start = 18 if reserve else 0
    return [row for row in read_rows(path) if row["lit2test_valid"] and all(row.get(k) for k in fields)][start:start + 18]


def credibility_case(row: dict, idx: int, sham: bool) -> tuple[dict, dict, list[dict], str]:
    design = {key: row[key] for key in ("hypothesis", "minimal_test", "decisive_metric",
                                        "supporting_result", "falsifying_result")}
    prior = design.copy()
    if sham:
        # Matched content with changed paragraph formatting and order.
        revised = {key: "\n" + text.strip() + "\n" for key, text in reversed(list(design.items()))}
        return prior, revised, [], "tie"
    if idx % 2:
        prior["minimal_test"] = ""
        prior["decisive_metric"] = ""
        prior["supporting_result"] = ""
        prior["falsifying_result"] = ""
    else:
        # A deliberately mismatched observable, with the credible test left untouched.
        prior["decisive_metric"] = "Count the characters in the project title."
    return prior, design, [], "win"


def calibration_cases(root: Path, *, reserve: bool = False) -> list[dict]:
    designs = benchmark_designs(root, reserve)
    if len(designs) != 18:
        raise ValueError("Missing public Lit2Test calibration inputs")
    rows = []
    for axis in ("honesty", "credibility", "novelty"):
        for i in range(18):
            sham = i >= 12
            if axis == "credibility":
                before, after, packet, expected = credibility_case(designs[i], i, sham)
                provenance = f"Lit2Test fifth40 GPT-5.2 / {designs[i]['context_id']}"
            else:
                construct = honesty_case if axis == "honesty" else novelty_case
                topics = RESERVE_TOPICS if reserve else TOPICS
                before, after, packet, expected = construct(topics[i % 12], i + 12 if reserve else i, sham)
                provenance = "constructed from v1 evidence and comparison rules"
            prefix = "reserve-" if reserve else ""
            rows.append({"id": f"{prefix}{axis}-{i + 1:02}", "axis": axis, "kind": "style" if sham else "substantive",
                         "expected": expected, "original": before, "candidate": after, "evidence": packet,
                         "task": CONTEXT, "provenance": provenance})
    return rows
