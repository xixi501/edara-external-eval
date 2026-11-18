"""Freeze six paired contribution cases before running the three-arm pilot."""

from __future__ import annotations

import argparse
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from html import unescape
from pathlib import Path

import requests

from freeze_governance_cases import ABSTRACT, digest, paper, read_json

# Cases were chosen by mechanism and evidence structure before any model calls.
# Each pair deliberately shares a topic and evidence pool.
GROUPS = [
    {
        "id": "preference", "topic": "Preference tuning from paired text without on-policy sampling",
        "cutoff": "2025-01-01", "papers": ["2203.02155", "2305.18290", "2310.12036", "2402.01306"],
        "prior": "2305.18290",
        "terms": [("dpo update", "probability"), ("directly optimize", "reward modeling")],
        "mechanism": "a logistic pairwise classification loss on the difference between the trainable/reference log probability ratios of chosen and rejected completions, without a separate learned reward model or on-policy RL",
        "draft": (
            "We propose offline preference tuning with a frozen starting policy as a reference. For each "
            "chosen and rejected completion, subtract the starting model's sequence log likelihood from "
            "the trainable model's. Scale the difference between the two ratios and optimize its logistic "
            "pairwise loss directly, omitting reward-model fitting and policy-rollout optimization. "
            "We claim that this objective is our contribution. Evaluate held-out pair accuracy at matched compute."
        ),
        "residual": (
            "does a prompt-level disagreement estimate from independently repeated preference labels "
            "identify a regime where adapting the reference-ratio temperature beats a global temperature, "
            "at equal label and compute budgets?"
        ),
        "test": (
            "Repeat annotation only in a training/calibration split, compare group-conditioned and global "
            "temperature on held-out prompts at matched extra labels, and reject the hypothesis if there "
            "is no held-out high-disagreement benefit relative to the low-disagreement stratum."
        ),
    },
    {
        "id": "cache", "topic": "Serving long language-model requests with shared prompts",
        "cutoff": "2025-01-01", "papers": ["2205.14135", "2303.06865", "2309.06180", "2305.13245"],
        "prior": "2309.06180", "terms": [("copy-on-write", "block"), ("logical", "physical")],
        "mechanism": (
            "fixed-size KV pages mapped from logical sequence indices to physical pages, with shared prefix pages "
            "and copy-on-write when two continuations diverge"
        ),
        "draft": (
            "Our serving contribution replaces per-sequence contiguous attention storage with fixed-size "
            "segments and a per-request logical-to-physical map. Requests with a shared prefix point to the "
            "same physical attention segments. A branch makes a private copy when a write would mutate a "
            "shared segment, preserving other continuations. Compare peak KV bytes and throughput to a "
            "contiguous-cache baseline using the same token traces."
        ),
        "residual": (
            "does page sharing amplify tenant-level p95 queueing disparity when long-prompt tenants reuse "
            "prefixes more often, and does a physical-page-aware admission budget reduce that disparity?"
        ),
        "test": (
            "Replay matched arrivals under balanced and asymmetric prefix reuse. Fix batch limits and "
            "physical memory, compare default and page-aware admission, and reject the interaction if the "
            "between-tenant p95 latency gap does not change preferentially under asymmetric reuse."
        ),
    },
    {
        "id": "retrieval-traces", "topic": "Grounded multi-hop question answering over Wikipedia",
        "cutoff": "2025-01-01", "papers": ["2201.11903", "2302.04761", "2210.03350", "2210.03629"],
        "prior": "2210.03629", "terms": [("action", "observation", "thought"), ("wikipedia", "search")],
        "mechanism": (
            "interleaving free-text reasoning with Wikipedia actions and their observations, "
            "then selecting further actions or a final answer conditional on the combined history"
        ),
        "draft": (
            "For multi-hop QA, our new controller emits a text scratchpad step, then an encyclopedia "
            "query or page-reading action. It appends the returned observation and uses the combined "
            "trace to revise its plan, select another action, or answer. The interleaving of language "
            "deliberation and retrieved observations is the contribution. Compare to a fixed query "
            "sequence with matched query and prompt budgets."
        ),
        "residual": (
            "does causal order in prior reasoning carry value beyond its lexical content when an "
            "intermediate Wikipedia result is stale?"
        ),
        "test": (
            "Replay identical retrieved observations and action histories. Shuffle only past thought "
            "segments versus leaving them ordered, hold prompt length and query budget fixed, and test "
            "the ordered-minus-shuffled accuracy interaction between clean and stale-result inputs."
        ),
    },
    {
        "id": "visual-contrast", "topic": "Self-supervised visual learning without labels or a memory bank",
        "cutoff": "2025-01-01", "papers": ["2002.05709", "1911.05722", "2006.07733", "1807.03748"],
        "prior": "2002.05709", "terms": [("nonlinear", "projection"), ("minibatch", "negative")],
        "mechanism": (
            "two augmented views per image, nonlinear projection, and a temperature-scaled "
            "contrastive loss using the other views within the batch as negatives"
        ),
        "draft": (
            "Our representation-learning contribution samples two independently transformed views per "
            "image, maps the encoder output through a small nonlinear projection network, and predicts "
            "its paired view using a temperature-scaled similarity objective with all other images in "
            "the minibatch as distractors. Evaluate a linear probe on encoder representations before "
            "projection and hold the training budget fixed."
        ),
        "residual": (
            "does correlation between the two views' background perturbations, at fixed marginal "
            "augmentation severity, cause the encoder to rely on spurious visual backgrounds?"
        ),
        "test": (
            "Randomize only the coupling across paired views, keeping each augmentation's marginal "
            "distribution, labels and compute equal. Compare accuracy on background-swapped images "
            "against clean images. Reject the hypothesis if coupling does not change the swap penalty."
        ),
    },
    {
        "id": "continuous-kernels", "topic": "Efficient convolution for distant dependencies across input resolutions",
        "cutoff": "2022-06-01", "papers": ["1803.01271", "2102.02611", "2111.00396", "2203.06717"],
        "prior": "2102.02611", "terms": [("formulate", "kernel", "small neural network"), ("arbitrarily large convolutional kernel",)],
        "annotation": ("ZW5aK4yCRqU", "CKConv:", "2301.10540"),
        "mechanism": (
            "parameterizing convolution kernels as coordinate-conditioned continuous functions, "
            "so kernel samples and supported sequence lengths need not fix the learned parameter count"
        ),
        "draft": (
            "Our basic contribution is a length-independent parameterization for long-range convolution. "
            "A small coordinate network provides filter weights on demand across the available sequence "
            "extent. Sample that same function at the needed points rather than maintaining a coefficient "
            "for each lag. The kernel can cover a longer or more finely sampled sequence without adding "
            "new parameters. Compare to a fixed-grid filter on held-out lengths."
        ),
        "residual": (
            "can a single resolution-agnostic convolutional architecture transfer its structural "
            "choices across different spatial dimensionalities, rather than tuning depth, stride and "
            "downsampling per task?"
        ),
        "test": (
            "Use the same backbone depth and downsampling policy for sequences, images and point-cloud "
            "tasks with their appropriate coordinate dimensionalities. Compare to equally budgeted, "
            "dimension-specific architectures and falsify the idea if the fixed recipe repeatedly "
            "requires task-specific structural changes to match them."
        ),
    },
    {
        "id": "hypervolume", "topic": "Fast neural hypervolume prediction for Pareto front selection",
        "cutoff": "2022-10-01", "papers": ["1703.06114", "2006.05078", "2203.02185", "2005.00515"],
        "prior": "2203.02185", "terms": [("DeepSets", "permutation"), ("hypervolume", "invariant")],
        "annotation": ("fSa5IjNMmmi", "HV-Net:", "2210.02177"),
        "mechanism": (
            "approximating the hypervolume of an unordered non-dominated point set with a "
            "permutation-invariant DeepSets-style neural predictor"
        ),
        "draft": (
            "We introduce a neural approximation to the dominated volume of an unordered Pareto front. "
            "Encode its non-dominated vectors with shared weights, aggregate the per-point features by "
            "a symmetric operator, and map the aggregate to a scalar volume. This makes predictions "
            "invariant to the ordering of solutions. Measure volume error and cost against exact computation."
        ),
        "residual": (
            "does imposing exact product-of-scales equivariance under independent positive rescaling "
            "of objectives, in addition to set permutation invariance, improve extrapolation and "
            "preserve Pareto candidate preferences in shifted objective units?"
        ),
        "test": (
            "Train both set predictors on the same unit range, rescale each objective independently at "
            "test time, and compare exact-normalized error and selection ranking. The hypothesis fails "
            "if the scale-equivariant variant does not lower extrapolation error or ranking violations."
        ),
    },
]


def plain(text):
    text = re.sub(r'<math\b[^>]*alttext="([^"]*)"[^>]*>.*?</math>', r"\1", text, flags=re.S)
    return " ".join(unescape(re.sub(r"<[^>]+>", "", text)).split())


def get(url):
    for attempt in range(3):
        try:
            result = requests.get(url, timeout=12)
            result.raise_for_status()
            return result.text
        except requests.RequestException:
            if attempt == 2:
                raise
            time.sleep(1)


def extract_paper(ident, terms):
    url = f"https://arxiv.org/abs/{ident}v1"
    html = get(url)
    title = plain(re.search(r'<meta name="citation_title" content="([^"]+)"', html)[1])
    date = re.search(r'<meta name="citation_date" content="([^"]+)"', html)[1].replace("/", "-")
    abstract = ABSTRACT.search(html)
    if not abstract:
        abstract = re.search(r'<blockquote class="abstract[^"]*">.*?</span>(.*?)</blockquote>', html, re.S)
    abstract_text = plain(abstract[1])
    snippets = []
    body_url = f"https://arxiv.org/html/{ident}v1"
    if terms:
        try:
            fulltext = get(body_url)
        except requests.RequestException:
            fulltext = ""
        paragraphs = [
            plain(match) for match in re.findall(r'<p\b[^>]*class="ltx_p"[^>]*>(.*?)</p>', fulltext, re.S)
        ]
        for lookup in terms:
            sentences = (s for para in paragraphs for s in re.split(r"(?<=[.!?])\s+", para))
            matches = [
                sentence for sentence in sentences
                if 30 <= len(sentence) <= 600 and all(word.lower() in sentence.lower() for word in lookup)
            ]
            if matches:
                snippets.append(matches[0])
        snippets = list(dict.fromkeys(snippets))
    result = paper(f"arxiv:{ident}v1", abstract_text, title, date, "\n".join(snippets), url)
    result["text_sha256"] = digest(abstract_text + "\n" + result["fulltext_snippet"])
    result["method_source"] = body_url if snippets else None
    return result


def public_label(group, labels):
    if "annotation" not in group:
        return None
    pair_id, title_prefix, extension_id = group["annotation"]
    row = next(x for x in labels if x["pair_id"] == pair_id and x["B_title"].startswith(title_prefix))
    if (row["q1"], row["q2"]) != ("yes", "yes"):
        raise ValueError(f"Unexpected ClaimNov majority label for {pair_id}")
    return {**row, "later_paper": f"https://arxiv.org/abs/{extension_id}"}


def freeze(out):
    data = Path(__file__).resolve().parents[2] / "claimnov/data"
    labels = read_json(data / "annotation/n2_majority_labels.json")["rows"]
    ids = list(dict.fromkeys(ident for group in GROUPS for ident in group["papers"]))
    method_terms = {group["prior"]: group["terms"] for group in GROUPS}
    with ThreadPoolExecutor(max_workers=8) as pool:
        papers = dict(zip(ids, pool.map(lambda ident: extract_paper(ident, method_terms.get(ident)), ids), strict=True))
    cases = []
    for group in GROUPS:
        sources = [papers[ident] for ident in group["papers"]]
        assert len(sources) == 4 and all(source["year"] < group["cutoff"] for source in sources)
        prior = papers[group["prior"]]
        if not prior["fulltext_snippet"]:
            raise ValueError(f"No decisive methods excerpt for {group['id']}")
        label = public_label(group, labels)
        for kind in ("collision", "partial"):
            partial = kind == "partial"
            draft = group["draft"]
            if partial:
                draft += f' A separate hypothesis is {group["residual"]} {group["test"]}'
            cases.append({
                "case_id": f'{group["id"]}-{kind}', "family": group["id"], "kind": kind,
                "topic": group["topic"], "cutoff_date": group["cutoff"], "draft": draft,
                "draft_sha256": digest(draft), "evidence": sources,
                "evaluation_key": {
                    "prior_id": prior["id"], "covered_mechanism": group["mechanism"],
                    "preserve_residual": group["residual"] if partial else None,
                },
                "provenance": {
                    "type": "public_claimnov_pair_adaptation" if partial and label else "constructed_from_primary_sources",
                    "public_label": label if partial else None,
                },
            })
        print(f"{group['id']}: 4 prior papers, decisive method passages {len(prior['fulltext_snippet'].splitlines())}", flush=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x", encoding="utf-8") as handle:
        json.dump({"protocol": "feedback-ablation-cases/v1", "cases": cases}, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print(f"Froze {len(cases)} cases at {out}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, type=Path)
    freeze(parser.parse_args().out)
