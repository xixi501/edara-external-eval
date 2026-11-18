"""Build the frozen governance pilot from public papers and ClaimNov releases."""

import argparse
import csv
import hashlib
import json
import re
from concurrent.futures import ThreadPoolExecutor
from html import unescape
from pathlib import Path

import requests

PAPER_IDS = {
    "star_kv": "2606.08382",
    "videoflextok": "2604.12887",
    "lora": "2106.09685",
    "gptq": "2210.17323",
    "qlora": "2305.14314",
    "dpr": "2004.04906",
    "bart": "1910.13461",
    "rag": "2005.11401",
}
ABSTRACT = re.compile(r'<meta name="citation_abstract" content="([^"]+)"', re.I)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def paper(key, abstract, title, date, snippet="", url=None):
    return {
        "id": key, "title": title, "abstract": abstract, "fulltext_snippet": snippet,
        "year": date, "url": url, "source": "arxiv" if url and "arxiv.org" in url else "semantic_scholar",
    }


def load_papers():
    papers = {}
    for name, ident in PAPER_IDS.items():
        url = f"https://arxiv.org/abs/{ident}v1"
        response = requests.get(url, timeout=25)
        response.raise_for_status()
        title = re.search(r'<meta name="citation_title" content="([^"]+)"', response.text, re.I)
        date = re.search(r'<meta name="citation_date" content="([^"]+)"', response.text, re.I)
        abstract = ABSTRACT.search(response.text)
        if not abstract:
            abstract = re.search(r'<blockquote class="abstract[^"]*">.*?</span>(.*?)</blockquote>', response.text, re.S)
        if not all((title, date, abstract)):
            raise ValueError(f"missing citation metadata for {ident}")
        snippet = ""
        if name == "rag":
            methods = requests.get(f"https://arxiv.org/html/{ident}v1", timeout=20)
            methods.raise_for_status()
            excerpts = []
            for element in ("S2.SS2.p3.1", "S2.SS3.p1.1"):
                passage = re.search(rf'<p id="{element}"[^>]*>(.*?)</p>', methods.text, re.S)
                if not passage:
                    raise ValueError(f"missing RAG method passage {element}")
                plain = re.sub(r'<math\b[^>]*alttext="([^"]*)"[^>]*>.*?</math>', r'\1', passage.group(1), flags=re.S)
                excerpts.append(" ".join(unescape(re.sub(r"<[^>]+>", "", plain)).split()))
            snippet = "\n".join(excerpts)
        day = date.group(1).replace("/", "-")
        papers[name] = paper(f"arxiv:{ident}v1", " ".join(unescape(abstract.group(1)).split()),
                             unescape(title.group(1)), day, snippet, url)
    return papers


def _arxiv_by_title(title, cutoff):
    try:
        result = requests.get(
            "https://arxiv.org/search/", params={"query": title, "searchtype": "title"}, timeout=8,
        )
        result.raise_for_status()
        ids = list(dict.fromkeys(re.findall(r"https://arxiv.org/abs/(\d{4}\.\d{4,5})", result.text)))
        clean = lambda text: re.sub(r"[^a-z0-9]", "", unescape(text).lower())
        for ident in ids[:3]:
            url = f"https://arxiv.org/abs/{ident}v1"
            response = requests.get(url, timeout=8)
            response.raise_for_status()
            found = re.search(r'<meta name="citation_title" content="([^"]+)"', response.text, re.I)
            date = re.search(r'<meta name="citation_date" content="([^"]+)"', response.text, re.I)
            abstract = ABSTRACT.search(response.text)
            if found and date and abstract and clean(found[1]) == clean(title) and date[1].replace("/", "-") < cutoff:
                return {"abstract": " ".join(unescape(abstract[1]).split()), "url": url}
    except requests.RequestException:
        pass
    return {}


def load_audited_ideas(data):
    archive = data / "annotation/later_realized_audit"
    key = read_json(archive / "key.json")["items"]
    labels = {}
    for path in sorted(archive.glob("labels_*.csv")):
        with path.open(encoding="utf-8") as handle:
            labels.update((row["item_id"], row) for row in csv.DictReader(handle))

    sources = {
        pid: title for name in ("Q36", "Q42")
        for pid, title in zip(key[name]["top20_ids"], key[name]["top20_titles"], strict=True)
    }
    cutoff = "2026-03-01"

    def fetch(item):
        pid, title = item
        try:
            response = requests.get(
                f"https://api.semanticscholar.org/graph/v1/paper/{pid}",
                params={"fields": "title,abstract,externalIds"}, timeout=5,
            )
            response.raise_for_status()
            record = response.json()
            if record.get("abstract"):
                return record
        except requests.RequestException:
            pass
        return _arxiv_by_title(title, cutoff)

    with ThreadPoolExecutor(max_workers=8) as pool:
        records = dict(zip(sources, pool.map(fetch, sources.items())))

    for name in ("Q36", "Q42"):
        item = key[name]
        label = labels[name]
        if (label["done_by_listed_paper"], label["contribution_established"], label["confidence"]) != ("no", "no", "3"):
            raise ValueError(f"unexpected public label for {name}")
        anchor, index = item["item"].rsplit("::", 1)
        base = data / "later_realized/anchors_full" / anchor
        idea = next(row for row in read_json(base / "ideas.json") if row["idx"] == int(index))
        dated = {row["id"]: row for row in read_json(base / "pool_preT.json")}
        prior = []
        missing = []
        for pid in item["top20_ids"]:
            record = records.get(pid, {})
            if not record.get("abstract"):
                missing.append(pid)
                continue
            frozen = dated[pid]
            if not frozen["date"] < cutoff or (record.get("publicationDate") or frozen["date"]) >= cutoff:
                raise ValueError(f"post-cutoff evidence for {name}: {pid}")
            prior.append(paper(pid, record["abstract"], frozen["title"], frozen["date"],
                               url=record.get("url") or f"https://www.semanticscholar.org/paper/{pid}"))
        if not prior:
            raise ValueError(f"{name} has no matched prior abstracts")
        print(f"{name}: {len(prior)}/20 abstracts recovered", flush=True)
        yield case(
            name, idea["title"], f'{idea["title"]}\n\n{idea["description"]}', prior,
            "retain", cutoff, provenance={
                "claimnov_item": item["item"], "public_label": label, "later_papers": item["F_titles"],
                "audited_prior_ids": item["top20_ids"], "missing_abstract_ids": missing,
            },
        )


def case(case_id, topic, draft, papers, expected, cutoff, *, decisive=None, provenance=None):
    return {
        "case_id": case_id,
        "topic": topic, "draft": draft, "evidence": papers, "cutoff_date": cutoff,
        "expected": expected, "decisive_paper_id": decisive["id"] if decisive else None,
        "provenance": provenance or {},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    data = Path(__file__).resolve().parents[2] / "claimnov/data"
    sources = load_papers()
    cases = [
        case("star-kv", "Reduce inference memory while preserving generation quality",
             "We introduce a new KV cache compression mechanism: learn differentiable soft thresholds to select low-rank projections per attention head and block, allocate different factorizations to key and value projections according to sensitivity, and use low-rank-aware mixed precision. Our sole methodological contribution is this compression mechanism. Test quality at matched KV-cache memory against fixed-rank projection.",
             [sources["star_kv"]], "covered", "2026-07-01", decisive=sources["star_kv"]),
        case("videoflextok", "Video tokenization under a flexible token budget",
             "We introduce variable-length coarse-to-fine video tokenization: prefix tokens encode semantics and motion, later tokens add detail, and a generative flow decoder reconstructs realistic videos from any prefix token count. This tokenizer is our sole technical novelty. Compare reconstruction at matched token budgets against a fixed 3D token grid.",
             [sources["videoflextok"]], "covered", "2026-05-01", decisive=sources["videoflextok"]),
    ]
    cases.extend(load_audited_ideas(data))
    for key, topic, draft, components, combined, cutoff in [
        ("quant-adapters", "Reduce the memory cost of adapting pretrained language models",
         "Our sole methodological novelty is to adapt a pretrained language model by backpropagating through its frozen 4-bit quantized weights into trainable low-rank adapters. Compare adapter tuning in 4-bit and 16-bit at a matched rank and training set, measuring peak memory and held-out performance.",
         ("lora", "gptq"), "qlora", "2023-06-01"),
        ("retrieval-bart", "Improve knowledge-intensive generation with retrieved passages",
         "Our sole methodological novelty is to combine a DPR dense retriever over Wikipedia with a pretrained BART generator by conditioning BART on retrieved passages concatenated with the input. Compare with BART alone on open-domain QA at matched training budget, measuring exact match.",
         ("dpr", "bart"), "rag", "2020-06-01"),
    ]:
        parts = [sources[name] for name in components]
        cases.append(case(f"{key}-parts", topic, draft, parts, "unresolved", cutoff))
        cases.append(case(f"{key}-prior", topic, draft, [*parts, sources[combined]], "covered", cutoff,
                          decisive=sources[combined]))
    for row in cases:
        row["draft_sha256"] = digest(row["draft"])
        for evidence in row["evidence"]:
            evidence["text_sha256"] = digest(evidence["abstract"] + "\n" + evidence["fulltext_snippet"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as handle:
        json.dump({"version": 1, "cases": cases}, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(f"Froze {len(cases)} cases at {args.out}")


if __name__ == "__main__":
    main()
