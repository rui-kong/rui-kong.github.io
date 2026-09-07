#!/usr/bin/env python3
"""Generate an RSS 2.0 feed from the curated daily brief.

Output lands at data/feed.xml so the workflow's ``git add data/`` publishes it
with every hourly snapshot. Public URL: https://news.learnprompt.pro/data/feed.xml

Feed content is the curated layer (daily-brief clusters, persona-reviewed);
falls back to top AI-scored items from latest-24h.json when the brief is
missing. Standard library only.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from xml.sax.saxutils import escape

SITE = "https://news.learnprompt.pro"
FEED_PATH = "data/feed.xml"
MAX_ITEMS = 20


def rfc822(iso: str | None) -> str:
    if not iso:
        return ""
    try:
        dt = datetime.fromisoformat(iso.replace("Z", "+00:00"))
    except ValueError:
        return ""
    return dt.strftime("%a, %d %b %Y %H:%M:%S %z")


def brief_entries(data_dir: Path) -> tuple[list[dict], str]:
    path = data_dir / "daily-brief.json"
    if not path.exists():
        return [], ""
    brief = json.loads(path.read_text(encoding="utf-8"))
    entries = []
    for cluster in brief.get("items", [])[:MAX_ITEMS]:
        primary = cluster.get("primary_item") or {}
        url = cluster.get("primary_url") or primary.get("url") or ""
        title_zh = primary.get("title_zh") or primary.get("title") or ""
        title_en = primary.get("title_en") or ""
        title = title_zh if title_zh else title_en
        if title_en and title_zh and title_en != title_zh:
            title = f"{title_zh} / {title_en}"
        desc = (
            cluster.get("persona_review")
            or primary.get("recommend_reason_zh")
            or primary.get("summary")
            or ""
        )
        label = cluster.get("importance_label") or ""
        source = cluster.get("source") or primary.get("source_name") or ""
        if not (title and url):
            continue
        entries.append(
            {
                "title": title,
                "url": url,
                "desc": desc,
                "category": label,
                "source": source,
                "date": cluster.get("latest_at") or cluster.get("earliest_at") or "",
            }
        )
    return entries, brief.get("generated_at", "")


def latest_entries(data_dir: Path) -> tuple[list[dict], str]:
    path = data_dir / "latest-24h.json"
    if not path.exists():
        return [], ""
    data = json.loads(path.read_text(encoding="utf-8"))
    items = sorted(
        data.get("items", []),
        key=lambda it: float(it.get("ai_score") or 0),
        reverse=True,
    )[:MAX_ITEMS]
    entries = []
    for it in items:
        title = it.get("title_zh") or it.get("title") or ""
        if not (title and it.get("url")):
            continue
        entries.append(
            {
                "title": title,
                "url": it["url"],
                "desc": it.get("recommend_reason_zh") or "",
                "category": it.get("source_tier_label") or "",
                "source": it.get("source") or "",
                "date": it.get("published_at") or it.get("first_seen_at") or "",
            }
        )
    return entries, data.get("generated_at", "")


def build_feed(entries: list[dict], generated_at: str) -> str:
    now = rfc822(generated_at) or rfc822(datetime.now(timezone.utc).isoformat())
    parts = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<rss version="2.0">',
        "<channel>",
        "<title>AI 更新雷达 · 每日精选</title>",
        f"<link>{SITE}</link>",
        "<description>24 小时 AI 更新雷达的策展精选：官方发布、模型上新、高信号讨论。</description>",
        "<language>zh-cn</language>",
        f"<lastBuildDate>{now}</lastBuildDate>",
        f'<atom10:link xmlns:atom10="http://www.w3.org/2005/Atom" rel="self" type="application/rss+xml" href="{SITE}/{FEED_PATH}"/>',
    ]
    for e in entries:
        desc = e["desc"]
        if e["source"]:
            desc = f"{desc}（{e['source']}）" if desc else e["source"]
        parts += [
            "<item>",
            f"<title>{escape(e['title'])}</title>",
            f"<link>{escape(e['url'])}</link>",
            f'<guid isPermaLink="true">{escape(e["url"])}</guid>',
            f"<description>{escape(desc)}</description>",
        ]
        if e["category"]:
            parts.append(f"<category>{escape(e['category'])}</category>")
        pub = rfc822(e["date"])
        if pub:
            parts.append(f"<pubDate>{pub}</pubDate>")
        parts.append("</item>")
    parts += ["</channel>", "</rss>", ""]
    return "\n".join(parts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    args = ap.parse_args()
    data_dir = Path(args.data_dir)

    entries, generated_at = brief_entries(data_dir)
    source = "daily-brief"
    if not entries:
        entries, generated_at = latest_entries(data_dir)
        source = "latest-24h"
    if not entries:
        raise SystemExit("no feed entries: neither daily-brief.json nor latest-24h.json usable")

    out = data_dir / "feed.xml"
    out.write_text(build_feed(entries, generated_at), encoding="utf-8")
    print(f"feed.xml: {len(entries)} items from {source} -> {out}")


if __name__ == "__main__":
    main()
