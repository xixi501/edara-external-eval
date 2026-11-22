"""Collect the works AI-Scientist retrieved, so the judge pool covers its citations."""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

WORK = Path(os.environ.get("EDARA_EVAL_ROOT") or Path(__file__).resolve().parents[1])
AI = WORK / "outputs/hosts_v1/ai_scientist_v1"
OUT = WORK / "outputs/hosts_v1/ai_scientist_15_evidence.json"


def reconstruct(inverted) -> str:
    if not isinstance(inverted, dict) or not inverted:
        return ""
    pairs = []
    for word, positions in inverted.items():
        for position in positions if isinstance(positions, list) else []:
            pairs.append((int(position), str(word)))
    pairs.sort()
    return " ".join(word for _, word in pairs)[:1800]


def fetch(work_id: str, cache_dir: Path) -> dict | None:
    short = work_id.rstrip("/").split("/")[-1]
    cache = cache_dir / f"{short}.json"
    if cache.exists():
        return json.loads(cache.read_text())
    url = f"https://api.openalex.org/works/{short}?mailto=edara@example.org"
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=20) as response:
                data = json.load(response)
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps(data, ensure_ascii=False))
            return data
        except (urllib.error.HTTPError, OSError, ValueError):
            time.sleep(2 + attempt * 3)
    return None


def main() -> int:
    collected: dict[str, list[dict]] = {}
    total = fetched = 0
    for task_dir in sorted(AI.glob("*/")):
        task = task_dir.name
        ids: list[str] = []
        for ledger in sorted(task_dir.glob("search*.jsonl")):
            for line in ledger.read_text().splitlines():
                if not line.strip():
                    continue
                for work_id in json.loads(line).get("kept_ids") or []:
                    if work_id not in ids:
                        ids.append(work_id)
        if not ids:
            continue
        rows = []
        for work_id in ids:
            total += 1
            data = fetch(work_id, task_dir / "openalex")
            if not data:
                continue
            fetched += 1
            rows.append({
                "paper_id": data.get("id") or work_id,
                "title": (data.get("title") or data.get("display_name") or work_id),
                "quotes": [reconstruct(data.get("abstract_inverted_index"))],
                "doi": data.get("doi"),
                "year": str(data.get("publication_year") or ""),
            })
        collected[task] = rows
        print(f"[ai-sci-sources] {task[:46]:46s} ids={len(ids)} fetched={len(rows)}")
    OUT.write_text(json.dumps(collected, ensure_ascii=False, indent=2) + "\n")
    print(f"[ai-sci-sources] works={total} fetched={fetched} tasks={len(collected)} -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
