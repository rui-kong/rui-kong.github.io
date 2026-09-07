#!/usr/bin/env python3
"""Build per-date data directories so the main radar page can browse history.

The upstream front end funnels every data read through one helper::

    function dataUrl(path) { ... return `${base}/${basename(path)}`; }

where ``base`` comes from ``?data=`` or ``localStorage.dataBaseUrl``. So if a
directory holds the same filenames as ``blog/data/``, the whole upstream UI —
category tabs, 精选/全量 toggle, 多源 folding, search — works on that snapshot
with no code changes. This script produces those directories:

    blog/data/history/<date>/daily-brief.json
                             stories-merged.json
                             latest-24h.json
                             latest-24h-all.json
                             source-status.json      (stub)
                             top3-personas.json      (stub)
                             waytoagi-7d.json        (stub)
    blog/data/history/index.json    date manifest for the picker

Inputs are snapshots mined out of the upstream git history by
``tools/mine_upstream_briefs.py``, plus today's live files under ``blog/data/``.

Payloads are compacted: fields the front end never reads are dropped, duplicate
arrays are removed (``items_ai`` shadows ``items``), and per-day item counts are
capped. Without this a single day costs ~1.4 MB.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO = Path(__file__).resolve().parents[1]
DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}")
STUB_FILES = ("source-status.json", "top3-personas.json", "waytoagi-7d.json")

def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    blob = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    path.write_text(blob, encoding="utf-8")
    return len(blob.encode("utf-8"))


def day_of_snapshot(path: Path, payload: Any) -> Optional[str]:
    """Snapshot day: prefer the payload's own generated_at, fall back to filename."""
    if isinstance(payload, dict):
        generated = payload.get("generated_at")
        if isinstance(generated, str) and DATE_RE.match(generated[:10]):
            return generated[:10]
    return path.name[:10] if DATE_RE.match(path.name[:10]) else None


def latest_per_day(directory: Optional[Path]) -> Dict[str, Path]:
    """Map date -> newest snapshot file for that date."""
    if not directory or not directory.exists():
        return {}
    chosen: Dict[str, Path] = {}
    for path in sorted(directory.glob("*.json")):
        day = path.name[:10]
        if DATE_RE.fullmatch(day):
            chosen[day] = path  # sorted ascending, so the last wins
    return chosen


def all_per_day(directory: Optional[Path]) -> Dict[str, List[Path]]:
    grouped: Dict[str, List[Path]] = {}
    if not directory or not directory.exists():
        return grouped
    for path in sorted(directory.glob("*.json")):
        day = path.name[:10]
        if DATE_RE.fullmatch(day):
            grouped.setdefault(day, []).append(path)
    return grouped

# Item fields the upstream front end actually reads (see assets/app.js). Anything
# else — scoring internals, raw signal dumps, translation caches — is dropped:
# keeping them makes a single archived day cost megabytes.
ITEM_KEEP = (
    "id", "url", "primary_url", "title", "title_zh", "title_en", "title_enhanced_zh",
    "title_original", "source", "source_name", "site_id", "site_name", "source_tier",
    "source_tier_rank", "ai_label", "ai_score", "ai_is_related", "published_at",
    "first_seen_at", "last_seen_at", "type", "recommend_reason_zh", "aihot_score",
)


def trim_item(item: Any) -> Any:
    if not isinstance(item, dict):
        return item
    out = {key: item[key] for key in ITEM_KEEP if key in item}
    signals = item.get("ai_signals")
    if isinstance(signals, list) and signals:
        out["ai_signals"] = signals[:3]
    return out


def compact_story(story: Dict[str, Any], max_sources: int) -> Dict[str, Any]:
    """Keep every field the UI reads; drop the duplicate items array and scoring internals."""
    out = {key: value for key, value in story.items()
           if key not in ("items", "importance_breakdown", "sources", "primary_item")}
    sources = story.get("sources") or story.get("items") or []
    out["sources"] = [trim_item(src) for src in sources[:max_sources]]
    if story.get("primary_item"):
        out["primary_item"] = trim_item(story["primary_item"])
    return out


def compact_stories_payload(payload: Dict[str, Any], cap: int, max_sources: int) -> Dict[str, Any]:
    stories = payload.get("stories") or []
    stories = sorted(stories, key=lambda s: s.get("score") or 0, reverse=True)[:cap]
    out = dict(payload)
    out["stories"] = [compact_story(story, max_sources) for story in stories]
    out["total_stories"] = len(out["stories"])
    return out


def compact_news_payload(payload: Dict[str, Any], cap: int) -> Dict[str, Any]:
    """latest-24h.json: the UI reads items_ai (falling back to items) and
    creator_items_all (falling back to creator_items_ai), so the duplicates go."""
    items = payload.get("items_ai") or payload.get("items") or []
    out = {key: value for key, value in payload.items()
           if key not in ("items", "items_ai", "creator_items_all")}
    out["items_ai"] = [trim_item(item) for item in items[:cap]]
    out["creator_items_ai"] = [trim_item(item) for item in (payload.get("creator_items_ai") or [])[:cap]]
    out["all_mode_data_url"] = "data/latest-24h-all.json"
    out["stories_data_url"] = "data/stories-merged.json"
    return out


def compact_all_payload(payload: Dict[str, Any], cap: int) -> Dict[str, Any]:
    """latest-24h-all.json: items_all_raw falls back to items_all in the UI."""
    items_all = payload.get("items_all") or payload.get("items_all_raw") or []
    out = {key: value for key, value in payload.items()
           if key not in ("items_all", "items_all_raw")}
    out["items_all"] = [trim_item(item) for item in items_all[:cap]]
    return out


def synth_all_from_news(news: Dict[str, Any], cap: int) -> Dict[str, Any]:
    """Fallback 全量 pool for days whose latest-24h-all.json was never committed."""
    items = (news.get("items_ai") or [])[:cap]
    return {
        "generated_at": news.get("generated_at"),
        "window_hours": news.get("window_hours"),
        "topic_filter": news.get("topic_filter"),
        "total_items_raw": len(items),
        "total_items_all_mode": len(items),
        "items_all": items,
        "synthesized_from": "latest-24h.json",
    }

def merge_brief_snapshots(paths: List[Path]) -> Dict[str, Any]:
    """Last snapshot of the day is the day's final curation; also union all of them."""
    last: Dict[str, Any] = {}
    union: Dict[str, Any] = {}
    count = 0
    for path in paths:
        try:
            payload = load_json(path)
        except json.JSONDecodeError:
            continue
        items = payload.get("items") or []
        if not items:
            continue
        count += 1
        last = payload
        for item in items:
            key = item.get("url") or item.get("primary_url") or item.get("story_id") or item.get("title")
            if key and key not in union:
                union[key] = item
    if not last:
        return {}
    merged = dict(last)
    merged["snapshot_count"] = count
    merged["union_items"] = list(union.values())
    return merged


def stub_payloads(date: str) -> Dict[str, Any]:
    """Minimal files so the UI's optional fetches resolve instead of erroring."""
    return {
        "source-status.json": {"generated_at": f"{date}T00:00:00Z", "sites": [],
                               "note": "历史归档不含当日源健康明细"},
        "top3-personas.json": {"generated_at": f"{date}T00:00:00Z", "items": []},
        "waytoagi-7d.json": {"generated_at": f"{date}T00:00:00Z", "updates_today": [],
                             "updates_7d": [], "count_today": 0, "count_7d": 0,
                             "has_error": False, "error": None},
    }

def build_day(date: str, out_dir: Path, brief: Dict[str, Any],
              stories_path: Optional[Path], news_path: Optional[Path],
              all_path: Optional[Path], args: argparse.Namespace) -> Optional[Dict[str, Any]]:
    day_dir = out_dir / date
    written = 0
    entry = {"date": date, "brief_count": 0, "brief_union_count": 0,
             "story_count": 0, "item_count": 0, "brief_snapshots": 0, "bytes": 0}

    if brief:
        payload = dict(brief)
        payload["items"] = [compact_story(s, args.max_sources)
                            for s in (brief.get("items") or [])[: args.brief_cap]]
        # union_items was only used by the standalone archive page; the upstream UI
        # ignores it and it cost ~150 KB per archived day, so it is not written.
        payload.pop("union_items", None)
        payload["total_items"] = len(payload["items"])
        written += write_json(day_dir / "daily-brief.json", payload)
        entry["brief_count"] = len(payload["items"])
        entry["brief_union_count"] = len(brief.get("union_items") or [])
        entry["brief_snapshots"] = brief.get("snapshot_count", 0)

    news_payload: Optional[Dict[str, Any]] = None
    if stories_path:
        stories = compact_stories_payload(load_json(stories_path), args.story_cap, args.max_sources)
        written += write_json(day_dir / "stories-merged.json", stories)
        entry["story_count"] = len(stories.get("stories") or [])
    if news_path:
        news_payload = compact_news_payload(load_json(news_path), args.item_cap)
        written += write_json(day_dir / "latest-24h.json", news_payload)
        entry["item_count"] = len(news_payload.get("items_ai") or [])
    if all_path:
        written += write_json(day_dir / "latest-24h-all.json",
                              compact_all_payload(load_json(all_path), args.all_cap))
    elif news_payload is not None:
        written += write_json(day_dir / "latest-24h-all.json",
                              synth_all_from_news(news_payload, args.all_cap))

    # latest-24h.json is the only hard requirement of the upstream boot sequence.
    if not (day_dir / "latest-24h.json").exists():
        if not brief:
            if day_dir.exists():
                shutil.rmtree(day_dir)
            return None
        fallback = {
            "generated_at": brief.get("generated_at") or f"{date}T00:00:00Z",
            "window_hours": brief.get("window_hours") or 24,
            "total_items": 0,
            "items_ai": brief.get("union_items") or brief.get("items") or [],
            "site_stats": [],
            "creator_items_ai": [],
            "all_mode_data_url": "data/latest-24h-all.json",
            "stories_data_url": "data/stories-merged.json",
            "synthesized_from": "daily-brief.json",
        }
        written += write_json(day_dir / "latest-24h.json", fallback)
        entry["item_count"] = len(fallback["items_ai"])
        if not (day_dir / "latest-24h-all.json").exists():
            written += write_json(day_dir / "latest-24h-all.json",
                                  synth_all_from_news(fallback, args.all_cap))

    for name, payload in stub_payloads(date).items():
        if not (day_dir / name).exists():
            written += write_json(day_dir / name, payload)

    entry["bytes"] = written
    return entry

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=REPO / "blog/data/history")
    parser.add_argument("--live-data", type=Path, default=REPO / "blog/data",
                        help="today's live data directory, folded in as the newest day")
    parser.add_argument("--brief-dir", type=Path, default=None,
                        help="mined data/daily-brief.json snapshots (all commits)")
    parser.add_argument("--stories-dir", type=Path, default=None,
                        help="mined data/stories-merged.json snapshots (one per day)")
    parser.add_argument("--news-dir", type=Path, default=None,
                        help="mined data/latest-24h.json snapshots (one per day)")
    parser.add_argument("--all-dir", type=Path, default=None,
                        help="mined data/latest-24h-all.json snapshots (one per day)")
    parser.add_argument("--days", type=int, default=90,
                        help="only build the newest N days (0 = all available)")
    parser.add_argument("--story-cap", type=int, default=150)
    parser.add_argument("--item-cap", type=int, default=300)
    parser.add_argument("--all-cap", type=int, default=500)
    parser.add_argument("--brief-cap", type=int, default=20)
    parser.add_argument("--max-sources", type=int, default=5)
    return parser.parse_args()


def fold_live_day(args: argparse.Namespace, briefs: Dict[str, List[Path]],
                  stories: Dict[str, Path], news: Dict[str, Path],
                  alls: Dict[str, Path]) -> Optional[str]:
    """Treat blog/data/*.json as the snapshot for its own generated_at day."""
    live = args.live_data
    brief_file = live / "daily-brief.json"
    if not brief_file.exists():
        return None
    try:
        payload = load_json(brief_file)
    except json.JSONDecodeError:
        return None
    day = day_of_snapshot(brief_file, payload)
    if not day:
        return None
    briefs.setdefault(day, [])
    if brief_file not in briefs[day]:
        briefs[day].append(brief_file)
    for name, table in (("stories-merged.json", stories), ("latest-24h.json", news),
                        ("latest-24h-all.json", alls)):
        candidate = live / name
        if candidate.exists():
            table[day] = candidate
    return day

def main() -> int:
    args = parse_args()

    briefs = all_per_day(args.brief_dir)
    stories = latest_per_day(args.stories_dir)
    news = latest_per_day(args.news_dir)
    alls = latest_per_day(args.all_dir)
    live_day = fold_live_day(args, briefs, stories, news, alls)

    # Days already on disk stay in the manifest even when this run has no input
    # for them, so an incremental CI run never drops mined history.
    existing = {path.name for path in args.out.glob("*") if path.is_dir()
                and DATE_RE.fullmatch(path.name)}
    candidates = sorted(set(briefs) | set(stories) | set(news) | set(alls), reverse=True)
    if args.days:
        candidates = candidates[: args.days]

    entries: Dict[str, Dict[str, Any]] = {}
    for date in candidates:
        entry = build_day(date, args.out, merge_brief_snapshots(briefs.get(date, [])),
                          stories.get(date), news.get(date), alls.get(date), args)
        if entry:
            entries[date] = entry

    for date in sorted(existing - set(entries), reverse=True):
        day_dir = args.out / date
        if not (day_dir / "latest-24h.json").exists():
            continue
        try:
            brief_payload = load_json(day_dir / "daily-brief.json")
        except (FileNotFoundError, json.JSONDecodeError):
            brief_payload = {}
        try:
            stories_payload = load_json(day_dir / "stories-merged.json")
        except (FileNotFoundError, json.JSONDecodeError):
            stories_payload = {}
        try:
            news_payload = load_json(day_dir / "latest-24h.json")
        except (FileNotFoundError, json.JSONDecodeError):
            news_payload = {}
        entries[date] = {
            "date": date,
            "brief_count": len(brief_payload.get("items") or []),
            "brief_union_count": len(brief_payload.get("union_items") or []),
            "story_count": len(stories_payload.get("stories") or []),
            "item_count": len(news_payload.get("items_ai") or []),
            "brief_snapshots": brief_payload.get("snapshot_count", 0),
            "bytes": sum(f.stat().st_size for f in day_dir.glob("*.json")),
        }

    days = [entries[date] for date in sorted(entries, reverse=True)]
    for entry in days:
        entry["base"] = f"./data/history/{entry['date']}"
    write_json(args.out / "index.json", {
        "generated_at": (days[0]["date"] if days else None),
        "live_day": live_day,
        "total_days": len(days),
        "days": days,
        "source": "LearnPrompt/ai-news-radar git history + local data/",
    })

    total_bytes = sum(entry["bytes"] for entry in days)
    print(json.dumps({
        "days": len(days),
        "live_day": live_day,
        "with_stories": sum(1 for d in days if d["story_count"]),
        "with_brief": sum(1 for d in days if d["brief_count"]),
        "total_mb": round(total_bytes / 1024 / 1024, 1),
        "out": str(args.out),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
