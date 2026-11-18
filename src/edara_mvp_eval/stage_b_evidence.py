"""Versioned dated evidence preparation for the Stage B protocol."""

import hashlib
import html
import json
import re
import urllib.parse
import urllib.request
from pathlib import Path

from .governance import admissible, evidence_payload
from .schemas import PriorWorkItem, TaskSpec
from .tasks_loader import normalize_title_hash


def digest(data):
    return hashlib.sha256(json.dumps(data, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(path)


def _text(value):
    return " ".join(html.unescape(re.sub(r"<[^>]*>", " ", value or "")).split())


_COMMON_WORDS = frozenset("""
about across after against also among and are based before between could does each
for from have into more only other over same should some such than that their them
there these they this those through under using use used very was were what when
where which while will with within without would towards toward research study
paper method methods approach approaches framework systems analysis
""".split())

# Terms that routinely match across unrelated fields (page layout vs KV page,
# housing eviction vs token eviction, publishing piracy vs model piracy).
_POLYSEMOUS = frozenset("""
page pages eviction evictions settlement settlements copyright publishing
video videos agent agents
""".split())


def _keywords(text):
    return list(dict.fromkeys(t.lower() for t in re.findall(r"[a-zA-Z][a-zA-Z0-9-]{2,}", text)
                              if t.lower() not in _COMMON_WORDS))


def _relevant(query, title, abstract):
    """Require a lexical neighbourhood, not a single shared word.

    Two content hits, at least one of them domain-bearing, is what keeps genuine
    neighbours while rejecting the polysemy collisions (a sociology monograph on
    eviction, a page-layout GAN, a copyright-piracy proposal) that a single shared
    token used to let into a KV-cache evidence pool.
    """
    target = _keywords(query)
    if not target:
        return False
    corpus = set(_keywords(f"{title} {abstract}"))
    if len(target) == 1:
        return target[0] in corpus and target[0] not in _POLYSEMOUS
    hits = [token for token in target if token in corpus]
    if len(hits) < 2:
        return False
    return any(token not in _POLYSEMOUS for token in hits)


def native_prior_work(citation_map, *, stage, source_db, cutoff_date=None):
    """Convert a reconciled citation map into prior-work items for the shared pool.

    Sources with a recorded year are ordinary dated prior work. Sources whose date is
    absent from the native record are marked ``participant_provenance``: they may be used
    to verify that a quoted sentence exists, but not as novelty comparators.
    """
    items, provenance = [], []
    for marker, row in sorted(citation_map.items(), key=lambda pair: int(pair[0][1:])):
        quotes = [quote for quote in (row.get("quotes") or {}).values() if quote]
        passage = "\n".join(dict.fromkeys(quotes))
        year = (row.get("year") or "").strip() or None
        cutoff_year = int(cutoff_date[:4]) if cutoff_date and cutoff_date[:4].isdigit() else None
        dated = bool(year and year.isdigit() and cutoff_year and int(year) < cutoff_year)
        item = PriorWorkItem(
            id=f"native:{row['paper_id']}",
            title=row.get("title") or row["paper_id"],
            abstract=passage,
            year=year,
            version=f"native-export/{stage}",
            date_source=f"{source_db}#{marker}",
            url=(row.get("external_id") or None),
            source="native_export",
            date_basis="publication" if dated else "participant_provenance",
        )
        items.append(item)
        provenance.append({"marker": marker, "paper_id": row["paper_id"],
                           "title": item.title, "year": year,
                           "date_basis": item.date_basis,
                           "evidence_ids": row.get("evidence_ids", [])})
    return items, provenance


class EvidenceStore:
    def __init__(self, root, task, *, offline=False):
        self.root = root
        self.task = task
        self.offline = offline
        self.events = []

    def request(self, url):
        path = self.root / "lookup" / f"{digest(url)}.json"
        cached = path.exists()
        if cached:
            record = json.loads(path.read_text())
        elif self.offline:
            record = {"status": "offline", "data": None}
        else:
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "EDARA-Stage-B/1.0"})
                with urllib.request.urlopen(req, timeout=9) as response:
                    record = {"status": "ok", "data": json.load(response)}
            except (OSError, ValueError) as exc:
                record = {"status": "error", "error": str(exc), "data": None}
            write_json(path, record)
        self.events.append({"url": url, "status": record["status"],
                            "response_sha256": digest(record["data"])})
        return record["data"] or {}

    def _arxiv(self, query, *, title=None):
        cleaned = re.sub(r'["\\]', ' ', query)
        terms = f'"{cleaned}"' if title else " OR ".join(_keywords(cleaned)[:9])
        year = self.task.cutoff_date[:4]
        params = urllib.parse.urlencode({"query": f'titles.title:({terms}) AND prefix:10.48550 AND publicationYear:[1900 TO {year}]',
                                        "page[size]": 6})
        rows = self.request(f"https://api.datacite.org/dois?{params}").get("data", [])
        for row in rows:
            data = row["attributes"]
            titles = data.get("titles") or []
            name = titles[0].get("title", "") if titles else ""
            if title and normalize_title_hash(name) != normalize_title_hash(title):
                continue
            versions = sorted(
                (d["date"][:10], d.get("dateInformation", "")) for d in data.get("dates", [])
                if d.get("dateType") == "Submitted" and re.fullmatch(r"v\d+", d.get("dateInformation", ""))
            )
            abstract = next((_text(d["description"]) for d in data.get("descriptions", [])
                             if d.get("descriptionType") == "Abstract"), "")
            pid = row["id"]
            # DataCite's abstract describes the current record. Use it only if
            # the latest recorded version is itself inside the cutoff.
            if versions and abstract and (title or _relevant(query, name, abstract)):
                day, version = versions[-1]
                yield PriorWorkItem(id=pid, title=name, abstract=abstract, publication_date=day,
                                    year=day[:4], version=version, source="arxiv",
                                    url="https://doi.org/" + pid,
                                    date_source=f"https://api.datacite.org/dois/{pid}")

    def _crossref(self, query, *, title=None):
        params = urllib.parse.urlencode({"query.title" if title else "query.bibliographic": query,
                                         "rows": 6, "filter": f"until-pub-date:{self.task.cutoff_date}"})
        rows = self.request(f"https://api.crossref.org/works?{params}").get("message", {}).get("items", [])
        for row in rows:
            names = row.get("title") or []
            name = names[0] if names else ""
            if not name or (title and normalize_title_hash(name) != normalize_title_hash(title)):
                continue
            parts = (row.get("published", {}).get("date-parts") or [[]])[0]
            if len(parts) not in (1, 3):
                continue
            stamp = str(parts[0]) if len(parts) == 1 else f"{parts[0]:04d}-{parts[1]:02d}-{parts[2]:02d}"
            pid = row["DOI"]
            abstract = _text(row.get("abstract"))
            if not title and not _relevant(query, name, abstract):
                continue
            yield PriorWorkItem(id=pid, title=name, abstract=abstract, year=stamp[:4],
                                publication_date=stamp if len(stamp) > 4 else None,
                                url="https://doi.org/" + pid, source="crossref", version="published",
                                date_source="https://api.crossref.org/works/" + pid)

    def admit(self, papers):
        selected = []
        seen = set()
        for paper in papers:
            key = normalize_title_hash(paper.title)
            if not admissible(self.task, paper):
                reason = "date_identity_or_version"
            elif not (paper.abstract or paper.fulltext_snippet):
                reason = "no_passage"
            elif key in seen:
                reason = "duplicate_title"
            else:
                reason = "accepted"
                selected.append(paper)
                seen.add(key)
            self.events.append({"paper_id": paper.id, "title": paper.title, "reason": reason,
                                "sha256": digest(paper.model_dump())})
        return selected

    def search(self, query):
        query = " ".join(re.findall(r"[\w-]+", query)[:15])
        if not _keywords(query):
            return []
        arxiv = self.admit(self._arxiv(query))
        if len(arxiv) >= 2:
            return arxiv[:4]
        return self.admit([*arxiv, *self._crossref(query)])[:4]

    def prepare(self):
        from .tasks_loader import AIIB_ROOT

        task = self.task
        accepted = []
        case = AIIB_ROOT / "public" / f"{task.task_id}.json"
        raw = json.loads(case.read_text()) if task.source_track == "aiib" and case.exists() else {}
        refs = {r["reference_id"]: r for r in raw.get("references", [])}
        # Keep the existing non-literature task description. Replace embedded
        # source excerpts with the versioned evidence section below.
        if task.source_track == "aiib":
            context = task.prompt_context.split("Supplied Seed References")[0]
        elif task.source_track == "lit2test":
            context = task.prompt_context.split("Literature Neighborhood:")[0]
        else:
            context = task.prompt_context
        for paper in task.prior_work:
            if paper.source == "topic_doc":
                self.events.append({"paper_id": paper.id, "reason": "background_only"})
                continue
            eligible = self.admit([paper])
            if not eligible:
                eligible = self.admit(self._arxiv(paper.title, title=paper.title))
            if not eligible:
                published = list(self._crossref(paper.title, title=paper.title))
                ref = refs.get(paper.id, {})
                pdf = AIIB_ROOT / "inputs" / task.task_id / f"{paper.id}.pdf"
                pinned = (raw.get("source_boundary") or {}).get("seed_document_sha256s", [])
                if (ref.get("version") == "official-distributed" and pdf.is_file()
                        and hashlib.sha256(pdf.read_bytes()).hexdigest() in pinned):
                    # The released task provenance identifies this as the
                    # officially distributed paper, matched by exact title.
                    published = [
                        p.model_copy(update={"fulltext_snippet": paper.fulltext_snippet,
                                             "version": "official-distributed"})
                        for p in published
                    ]
                eligible = self.admit(published)
            accepted.extend(eligible[:1])
        if len(accepted) < 3:
            accepted.extend(self.search(task.title))
            if len(accepted) < 3 and task.prior_work:
                seed = task.prior_work[0]
                second = seed.title if seed.source != "topic_doc" else seed.abstract[:350]
                if second != task.title:
                    accepted.extend(self.search(second))
        accepted = self.admit(accepted)
        derived = task.model_copy(update={"prior_work": accepted})
        context = context.strip() + "\n\nDated literature:\n" + json.dumps(evidence_payload(derived), ensure_ascii=False)
        return derived.model_copy(update={"prompt_context": context})
