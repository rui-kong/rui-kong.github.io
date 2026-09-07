#!/usr/bin/env python3
"""Build per-date history snapshots for the vendored AI News Radar.

Two input sources:

1. ``blog/data/archive.json`` — the rolling archive the upstream pipeline keeps
   (about three weeks). Items carry ``first_seen_at`` / ``last_seen_at``, so they
   can be regrouped by the day the radar saw them.
2. ``--brief-dir`` — optional directory of historical ``daily-brief.json``
   snapshots mined out of the upstream git history by ``tools/mine_upstream_briefs.py``.
   These give the real curated 20 items per past day, which ``archive.json``
   cannot reconstruct on its own.

Output (served as static files, no backend):
  blog/data/history/<date>.json   one file per day
  blog/data/history/index.json    manifest consumed by blog/history/index.html
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO = Path(__file__).resolve().parents[1]
DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")


def day_of(item: Dict[str, Any]) -> Optional[str]:
    """The day the radar observed the item, falling back to its publish date."""
    for field in ("first_seen_at", "last_seen_at", "published_at", "time"):
        value = item.get(field)
        if isinstance(value, str) and DATE_RE.match(value[:10]):
            return value[:10]
    return None


def normalize(item: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": item.get("id") or item.get("url"),
        "title": item.get("title") or "",
        "url": item.get("url"),
        "source": item.get("source") or item.get("site_name") or "",
        "site_id": item.get("site_id") or "",
        "published_at": item.get("published_at"),
        "first_seen_at": item.get("first_seen_at"),
        "score": item.get("score"),
        "topic": item.get("topic") or item.get("category"),
        "reason": item.get("reason") or item.get("why") or item.get("recommend_reason"),
        "summary": item.get("summary") or item.get("desc") or "",
    }


def collect_archive(archive_path: Path) -> Dict[str, List[Dict[str, Any]]]:
    payload = load_json(archive_path)
    items = payload.get("items", payload) if isinstance(payload, dict) else payload
    buckets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for item in items:
        day = day_of(item)
        if day:
            buckets[day].append(normalize(item))
    return buckets

def normalize_brief(item: Dict[str, Any]) -> Dict[str, Any]:
    """Keep the fields the upstream daily-brief carries per story."""
    reasons = item.get("reasons")
    if isinstance(reasons, list):
        reason_text = "；".join(str(r) for r in reasons if r)
    else:
        reason_text = str(reasons or "")
    persona = item.get("persona_review")
    if isinstance(persona, dict):
        persona = persona.get("text") or persona.get("review") or ""
    return {
        "id": item.get("story_id") or item.get("url"),
        "title": item.get("title") or "",
        "url": item.get("url") or item.get("primary_url"),
        "source": item.get("source_name") or item.get("source") or "",
        "source_names": item.get("source_names") or [],
        "source_count": item.get("source_count") or 1,
        "duplicate_count": item.get("duplicate_count") or 0,
        "category": item.get("category"),
        "topic": item.get("category"),
        "score": round(float(item["score"]), 3) if isinstance(item.get("score"), (int, float)) else item.get("score"),
        "importance": item.get("importance_label") or item.get("importance"),
        "published_at": item.get("latest_at") or item.get("earliest_at"),
        "first_seen_at": item.get("earliest_at"),
        "reason": reason_text,
        "persona_review": persona or "",
        "persona_id": item.get("persona_id"),
        "summary": "",
    }


def brief_items(payload: Any) -> List[Dict[str, Any]]:
    if isinstance(payload, dict):
        return payload.get("items") or []
    return payload if isinstance(payload, list) else []


def collect_briefs(brief_dir: Optional[Path]) -> Dict[str, Dict[str, Any]]:
    """Group mined daily-brief snapshots by day.

    Each day keeps the last snapshot of that day (the real end-of-day curation)
    plus the deduplicated union of every snapshot taken during the day.
    """
    if not brief_dir or not brief_dir.exists():
        return {}
    by_day: Dict[str, Dict[str, Any]] = {}
    for path in sorted(brief_dir.glob("*.json")):
        try:
            payload = load_json(path)
        except json.JSONDecodeError:
            continue
        generated = payload.get("generated_at") if isinstance(payload, dict) else None
        day = (generated or path.name)[:10]
        if not DATE_RE.fullmatch(day):
            continue
        items = [normalize_brief(item) for item in brief_items(payload)]
        entry = by_day.setdefault(day, {"snapshots": 0, "last": [], "union": {}, "generated_at": generated})
        entry["snapshots"] += 1
        entry["last"] = items
        entry["generated_at"] = generated or entry["generated_at"]
        for item in items:
            key = item.get("url") or item.get("id") or item.get("title")
            if key and key not in entry["union"]:
                entry["union"][key] = item
    return by_day

def source_counts(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    counts: Dict[str, int] = defaultdict(int)
    for item in items:
        counts[item.get("source") or "未知"] += 1
    ranked = sorted(counts.items(), key=lambda pair: pair[1], reverse=True)
    return [{"name": name, "count": count} for name, count in ranked[:12]]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=REPO / "blog/data/archive.json")
    parser.add_argument("--brief-dir", type=Path, default=None,
                        help="directory of mined daily-brief snapshots")
    parser.add_argument("--out", type=Path, default=REPO / "blog/data/history")
    parser.add_argument("--max-items-per-day", type=int, default=1500)
    parser.add_argument("--min-items", type=int, default=20,
                        help="drop days with fewer items and no curated brief "
                             "(feeds with bogus dates create sparse phantom days)")
    args = parser.parse_args()

    archive_days = collect_archive(args.archive) if args.archive.exists() else {}
    brief_days = collect_briefs(args.brief_dir)
    all_days = sorted(set(archive_days) | set(brief_days), reverse=True)
    all_days = [day for day in all_days
                if brief_days.get(day, {}).get("snapshots")
                or len(archive_days.get(day, [])) >= args.min_items]

    manifest_days = []
    for day in all_days:
        brief_entry = brief_days.get(day, {})
        brief = brief_entry.get("last") or []
        brief_union = list(brief_entry.get("union", {}).values())
        items = archive_days.get(day, [])
        items.sort(key=lambda item: item.get("first_seen_at") or "", reverse=True)
        items = items[: args.max_items_per_day]
        payload = {
            "date": day,
            "generated_at": brief_entry.get("generated_at"),
            "brief_snapshots": brief_entry.get("snapshots", 0),
            "brief": brief,
            "brief_union": brief_union,
            "items": items,
            "counts": {"brief": len(brief), "brief_union": len(brief_union), "items": len(items)},
            "sources": source_counts(items or brief_union or brief),
        }
        write_json(args.out / f"{day}.json", payload)
        manifest_days.append({
            "date": day,
            "brief_count": len(brief),
            "brief_union_count": len(brief_union),
            "item_count": len(items),
            "brief_snapshots": brief_entry.get("snapshots", 0),
            "url": f"data/history/{day}.json",
        })

    write_json(args.out / "index.json", {
        "days": manifest_days,
        "total_days": len(manifest_days),
        "source": "LearnPrompt/ai-news-radar git history + local archive.json",
    })

    # Drop day files that a previous run wrote but that no longer qualify.
    keep = {f"{entry['date']}.json" for entry in manifest_days} | {"index.json"}
    pruned = 0
    for stale in args.out.glob("*.json"):
        if stale.name not in keep:
            stale.unlink()
            pruned += 1

    print(json.dumps({"days": len(manifest_days),
                      "with_brief": sum(1 for d in manifest_days if d["brief_count"]),
                      "with_items": sum(1 for d in manifest_days if d["item_count"]),
                      "pruned": pruned,
                      "out": str(args.out)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
