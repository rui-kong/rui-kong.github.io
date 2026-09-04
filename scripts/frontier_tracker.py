#!/usr/bin/env python3
"""Collect frontier-model signals and render a static daily tracker.

This script is designed to run locally or inside GitHub Actions without any
third-party Python dependencies. It gathers recent candidates from arXiv,
Hacker News, Reddit, GitHub, and Hugging Face, scores them, writes JSON
artifacts, and renders a daily HTML digest.
"""

from __future__ import annotations

import argparse
import html
import json
import math
import os
import re
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlparse
from urllib.request import Request, urlopen
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]
BLOG_DIR = ROOT / "blog"
DATA_DIR = BLOG_DIR / "data"
POSTS_DIR = BLOG_DIR / "posts"

NOW = datetime.now(timezone.utc)
TODAY = NOW.date().isoformat()

USER_AGENT = "rui-kong-frontier-tracker/1.0 (+https://rui-kong.github.io)"
TIMEOUT = 25

SOURCE_WEIGHTS = {
    "arXiv": 0.96,
    "Hugging Face": 0.90,
    "GitHub": 0.86,
    "Hacker News": 0.74,
    "Reddit": 0.66,
}

SOURCE_KIND_WEIGHTS = {
    "arxiv": 0.96,
    "huggingface": 0.90,
    "github": 0.86,
    "hn": 0.74,
    "reddit": 0.66,
}

TOPIC_RULES = [
    (
        "推理与系统",
        [
            "inference",
            "serving",
            "kv cache",
            "speculative",
            "latency",
            "throughput",
            "quantization",
            "compression",
            "routing",
            "compiler",
            "kernel",
            "decoding",
            "cache",
            "memory",
        ],
    ),
    (
        "训练/预训练",
        [
            "pretrain",
            "training",
            "pre-training",
            "optimizer",
            "scaling",
            "moe",
            "transformer",
            "architecture",
            "looped",
            "recurrent",
        ],
    ),
    (
        "对齐/后训练",
        [
            "alignment",
            "rlhf",
            "rlaif",
            "post-training",
            "dpo",
            "ppo",
            "sft",
            "preference",
            "reward",
        ],
    ),
    (
        "Agent/工具",
        [
            "agent",
            "tool use",
            "tool-use",
            "workflow",
            "orchestration",
            "memory",
            "browser",
            "function calling",
        ],
    ),
    (
        "多模态",
        [
            "vision",
            "image",
            "video",
            "audio",
            "multimodal",
            "vlm",
        ],
    ),
    (
        "数据/评测",
        [
            "benchmark",
            "evaluation",
            "eval",
            "dataset",
            "leaderboard",
            "swebench",
            "test set",
            "robustness",
        ],
    ),
    (
        "开源/发布",
        [
            "release",
            "open source",
            "open-source",
            "weights",
            "checkpoint",
            "repo",
            "github",
            "hugging face",
            "hf",
        ],
    ),
]

NOVELTY_TERMS = {
    "paper",
    "preprint",
    "technical report",
    "release",
    "open source",
    "open-source",
    "weights",
    "checkpoint",
    "benchmark",
    "dataset",
    "code",
    "repository",
    "model",
    "new",
    "first",
    "launch",
    "introduce",
    "introduces",
    "released",
}

OFFICIAL_SOURCE_HINTS = {
    "arxiv",
    "github",
    "huggingface",
    "papers",
    "openai",
    "anthropic",
    "google",
    "meta",
    "microsoft",
    "deepmind",
    "together",
    "cohere",
}


def fetch_text(url: str, headers: Optional[Dict[str, str]] = None) -> str:
    request_headers = {"User-Agent": USER_AGENT}
    if headers:
        request_headers.update(headers)
    request = Request(url, headers=request_headers)
    with urlopen(request, timeout=TIMEOUT) as response:
        return response.read().decode("utf-8", errors="replace")


def fetch_json(url: str, headers: Optional[Dict[str, str]] = None) -> Any:
    return json.loads(fetch_text(url, headers=headers))


def clean_html_text(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"<[^>]+>", " ", text)
    text = html.unescape(text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def parse_iso(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    value = value.strip()
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def format_iso(dt: Optional[datetime]) -> Optional[str]:
    if dt is None:
        return None
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def normalize_title(title: str) -> str:
    title = title.lower()
    title = re.sub(r"https?://\S+", " ", title)
    title = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", " ", title)
    title = re.sub(r"\b(the|a|an|of|for|and|to|in|on|with|by|via|from)\b", " ", title)
    title = re.sub(r"\s+", " ", title)
    return title.strip()


def canonical_url(url: Optional[str]) -> Optional[str]:
    if not url:
        return None
    parsed = urlparse(url)
    if not parsed.scheme or not parsed.netloc:
        return url
    path = parsed.path.rstrip("/")
    query = ""
    if "github.com" in parsed.netloc:
        path = re.sub(r"/(?:issues|pulls|pull|discussions)/\d+.*$", "", path)
    return f"{parsed.scheme}://{parsed.netloc}{path}{query}"


def domain_from_url(url: Optional[str]) -> str:
    if not url:
        return ""
    parsed = urlparse(url)
    return parsed.netloc.lower()


def topic_for_text(text: str) -> Tuple[str, List[str]]:
    lowered = text.lower()
    scores: List[Tuple[int, str]] = []
    tags: List[str] = []
    for topic, keywords in TOPIC_RULES:
        matched = sum(1 for keyword in keywords if keyword in lowered)
        if matched:
            scores.append((matched, topic))
            for keyword in keywords:
                if keyword in lowered and keyword not in tags:
                    tags.append(keyword)
    if scores:
        scores.sort(key=lambda item: (item[0], len(item[1])), reverse=True)
        return scores[0][1], tags[:5]
    return "其他", tags[:5]


def engagement_score(candidate: Dict[str, Any]) -> float:
    raw = candidate.get("engagement") or 0
    if raw <= 0:
        return 0.0
    return min(0.12, math.log10(1.0 + float(raw)) / 10.0)


def recency_score(candidate: Dict[str, Any]) -> float:
    published = parse_iso(candidate.get("published_at"))
    if published is None:
        return 0.45
    age_hours = max(0.0, (NOW - published).total_seconds() / 3600.0)
    return max(0.0, 1.0 - min(age_hours, 168.0) / 168.0)


def novelty_score(text: str) -> float:
    lowered = text.lower()
    score = 0.0
    for term in NOVELTY_TERMS:
        if term in lowered:
            score += 0.08
    if "technical report" in lowered:
        score += 0.08
    return min(score, 0.36)


def source_weight(candidate: Dict[str, Any]) -> float:
    if candidate.get("source_kind") in SOURCE_KIND_WEIGHTS:
        return SOURCE_KIND_WEIGHTS[candidate["source_kind"]]
    return SOURCE_WEIGHTS.get(candidate.get("source", ""), 0.6)


def cluster_bonus(cluster_size: int) -> float:
    if cluster_size <= 1:
        return 0.0
    return min(0.15, 0.05 * (cluster_size - 1))


def official_bonus(candidate: Dict[str, Any]) -> float:
    source = (candidate.get("source") or "").lower()
    url = (candidate.get("url") or "").lower()
    if any(hint in source for hint in OFFICIAL_SOURCE_HINTS):
        return 0.06
    if any(hint in url for hint in OFFICIAL_SOURCE_HINTS):
        return 0.05
    return 0.0


def score_candidate(candidate: Dict[str, Any], cluster_size: int) -> Tuple[float, Dict[str, float], List[str]]:
    title = candidate.get("title", "")
    text = f"{title} {candidate.get('summary', '')} {candidate.get('tags_text', '')}"
    topic, _ = topic_for_text(text)
    candidate["topic"] = topic
    source_component = source_weight(candidate)
    recency_component = recency_score(candidate)
    novelty_component = novelty_score(text)
    topic_component = 0.0 if topic == "其他" else 0.18
    engagement_component = engagement_score(candidate)
    cluster_component = cluster_bonus(cluster_size)
    official_component = official_bonus(candidate)

    raw_score = (
        0.34 * source_component
        + 0.24 * recency_component
        + 0.16 * novelty_component
        + 0.12 * topic_component
        + 0.08 * engagement_component
        + 0.06 * cluster_component
        + 0.02 * official_component
    )
    score = round(min(1.0, raw_score) * 140.0, 1)

    reasons: List[str] = []
    if source_component >= 0.9:
        reasons.append("高可信原始来源")
    elif source_component >= 0.74:
        reasons.append("高信号社区来源")
    if recency_component >= 0.8:
        reasons.append("发布时间很新")
    if topic != "其他":
        reasons.append(f"主题属于{topic}")
    if novelty_component >= 0.16:
        reasons.append("带有明确技术增量信号")
    if cluster_size > 1:
        reasons.append(f"被{cluster_size}个来源交叉提及")
    if engagement_component >= 0.05:
        reasons.append("社区互动较高")
    if not reasons:
        reasons.append("综合信号足够强")

    details = {
        "source": round(source_component * 100.0, 1),
        "recency": round(recency_component * 100.0, 1),
        "novelty": round(novelty_component * 100.0, 1),
        "topic": round(topic_component * 100.0, 1),
        "engagement": round(engagement_component * 100.0, 1),
        "cluster": round(cluster_component * 100.0, 1),
        "official": round(official_component * 100.0, 1),
    }
    return score, details, reasons


def make_item(
    *,
    title: str,
    url: Optional[str],
    source: str,
    source_kind: str,
    published_at: Optional[str],
    summary: str = "",
    authors: Optional[List[str]] = None,
    tags: Optional[List[str]] = None,
    engagement: int = 0,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    tags = tags or []
    summary = clean_html_text(summary)
    title = clean_html_text(title)
    text = f"{title} {summary} {' '.join(tags)}"
    topic, topic_tags = topic_for_text(text)
    normalized = normalize_title(title)
    return {
        "id": "",
        "title": title,
        "url": canonical_url(url),
        "source": source,
        "source_kind": source_kind,
        "published_at": published_at,
        "summary": summary,
        "authors": authors or [],
        "tags": list(dict.fromkeys([tag for tag in tags if tag] + topic_tags)),
        "tags_text": " ".join(tags + topic_tags),
        "topic": topic,
        "engagement": int(engagement),
        "normalized_title": normalized,
        "extra": extra or {},
    }


def fetch_arxiv_candidates() -> List[Dict[str, Any]]:
    queries = [
        ("cs.CL", "cat:cs.CL"),
        ("cs.LG", "cat:cs.LG"),
        ("cs.AI", "cat:cs.AI"),
        ("stat.ML", "cat:stat.ML"),
    ]
    items: List[Dict[str, Any]] = []
    for label, query in queries:
        url = "https://export.arxiv.org/api/query?" + urlencode(
            {
                "search_query": query,
                "sortBy": "submittedDate",
                "sortOrder": "descending",
                "start": 0,
                "max_results": 25,
            }
        )
        try:
            xml_text = fetch_text(url)
        except (HTTPError, URLError, TimeoutError):
            continue
        root = ET.fromstring(xml_text)
        ns = {
            "atom": "http://www.w3.org/2005/Atom",
            "arxiv": "http://arxiv.org/schemas/atom",
        }
        for entry in root.findall("atom:entry", ns):
            title = (entry.findtext("atom:title", default="", namespaces=ns) or "").strip()
            summary = (entry.findtext("atom:summary", default="", namespaces=ns) or "").strip()
            published = entry.findtext("atom:published", default="", namespaces=ns)
            authors = [
                (author.findtext("atom:name", default="", namespaces=ns) or "").strip()
                for author in entry.findall("atom:author", ns)
            ]
            links = entry.findall("atom:link", ns)
            pdf_url = None
            abs_url = None
            for link in links:
                href = link.attrib.get("href")
                rel = link.attrib.get("rel")
                title_attr = link.attrib.get("title", "")
                if rel == "alternate":
                    abs_url = href
                if title_attr.lower() == "pdf" or link.attrib.get("type") == "application/pdf":
                    pdf_url = href
            tags = [label]
            categories = [item.attrib.get("term", "") for item in entry.findall("atom:category", ns)]
            tags.extend([category for category in categories if category])
            items.append(
                make_item(
                    title=title,
                    url=pdf_url or abs_url,
                    source="arXiv",
                    source_kind="arxiv",
                    published_at=published,
                    summary=summary,
                    authors=authors,
                    tags=tags,
                    extra={"category": label, "alternate_url": abs_url, "pdf_url": pdf_url},
                )
            )
    return items


def fetch_reddit_candidates() -> List[Dict[str, Any]]:
    subreddits = [
        "MachineLearning",
        "LocalLLaMA",
        "artificial",
        "singularity",
        "OpenAI",
        "LanguageTechnology",
    ]
    sorts = [("new", "day"), ("top", "week")]
    items: List[Dict[str, Any]] = []
    for subreddit in subreddits:
        for sort, timeframe in sorts:
            url = f"https://www.reddit.com/r/{subreddit}/{sort}.json?limit=25&t={timeframe}"
            try:
                payload = fetch_json(url, headers={"Accept": "application/json"})
            except (HTTPError, URLError, TimeoutError, json.JSONDecodeError):
                continue
            for child in payload.get("data", {}).get("children", []):
                data = child.get("data", {})
                title = data.get("title", "")
                selftext = data.get("selftext", "")
                permalink = data.get("permalink", "")
                created_utc = data.get("created_utc")
                published_at = (
                    datetime.fromtimestamp(created_utc, tz=timezone.utc).isoformat().replace("+00:00", "Z")
                    if created_utc
                    else None
                )
                items.append(
                    make_item(
                        title=title,
                        url=f"https://www.reddit.com{permalink}" if permalink else data.get("url"),
                        source=f"Reddit/r/{subreddit}",
                        source_kind="reddit",
                        published_at=published_at,
                        summary=selftext[:800],
                        tags=[subreddit, sort, timeframe],
                        engagement=int(data.get("score") or 0) + int(data.get("num_comments") or 0),
                        extra={
                            "subreddit": subreddit,
                            "score": data.get("score", 0),
                            "num_comments": data.get("num_comments", 0),
                        },
                    )
                )
    return items


def fetch_hn_candidates() -> List[Dict[str, Any]]:
    try:
        top_ids = fetch_json("https://hacker-news.firebaseio.com/v0/topstories.json")
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError):
        return []
    items: List[Dict[str, Any]] = []
    for item_id in top_ids[:50]:
        try:
            item = fetch_json(f"https://hacker-news.firebaseio.com/v0/item/{item_id}.json")
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError):
            continue
        if not item:
            continue
        title = item.get("title", "")
        url = item.get("url") or f"https://news.ycombinator.com/item?id={item_id}"
        published_at = (
            datetime.fromtimestamp(item.get("time", 0), tz=timezone.utc)
            .isoformat()
            .replace("+00:00", "Z")
            if item.get("time")
            else None
        )
        items.append(
            make_item(
                title=title,
                url=url,
                source="Hacker News",
                source_kind="hn",
                published_at=published_at,
                summary=f"{item.get('descendants', 0)} comments on HN",
                tags=["Hacker News", "community"],
                engagement=int(item.get("score") or 0) + int(item.get("descendants") or 0),
                extra={"hn_id": item_id, "score": item.get("score", 0), "comments": item.get("descendants", 0)},
            )
        )
    return items


def fetch_github_candidates() -> List[Dict[str, Any]]:
    queries = [
        "llm pushed:>=%s" % (NOW.date() - timedelta(days=3)).isoformat(),
        "transformer pushed:>=%s" % (NOW.date() - timedelta(days=3)).isoformat(),
        "agent pushed:>=%s" % (NOW.date() - timedelta(days=3)).isoformat(),
        "multimodal pushed:>=%s" % (NOW.date() - timedelta(days=3)).isoformat(),
        "inference pushed:>=%s" % (NOW.date() - timedelta(days=3)).isoformat(),
    ]
    items: List[Dict[str, Any]] = []
    token = os.environ.get("GITHUB_TOKEN", "").strip()
    headers = {"Accept": "application/vnd.github+json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    for query in queries:
        url = "https://api.github.com/search/repositories?" + urlencode(
            {"q": query, "sort": "updated", "order": "desc", "per_page": 25}
        )
        try:
            payload = fetch_json(url, headers=headers)
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError):
            continue
        for repo in payload.get("items", []):
            items.append(
                make_item(
                    title=repo.get("full_name", ""),
                    url=repo.get("html_url"),
                    source="GitHub",
                    source_kind="github",
                    published_at=repo.get("updated_at") or repo.get("created_at"),
                    summary=repo.get("description", "") or "",
                    authors=[repo.get("owner", {}).get("login", "")] if repo.get("owner") else [],
                    tags=[repo.get("language", "") or "GitHub", "repo"],
                    engagement=int(repo.get("stargazers_count") or 0) + int(repo.get("forks_count") or 0),
                    extra={
                        "stars": repo.get("stargazers_count", 0),
                        "forks": repo.get("forks_count", 0),
                        "language": repo.get("language"),
                    },
                )
            )
    return items


def fetch_huggingface_candidates() -> List[Dict[str, Any]]:
    url = "https://huggingface.co/api/models?" + urlencode(
        {"sort": "lastModified", "direction": -1, "limit": 100, "full": "true"}
    )
    try:
        payload = fetch_json(url, headers={"Accept": "application/json"})
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError):
        return []
    items: List[Dict[str, Any]] = []
    for model in payload:
        tags = [tag for tag in model.get("tags", []) if tag]
        pipeline_tag = model.get("pipeline_tag")
        if pipeline_tag:
            tags.append(pipeline_tag)
        card_data = model.get("cardData") or {}
        summary = card_data.get("model_name") or card_data.get("language") or ""
        if not summary:
            summary = ", ".join(tags[:4])
        items.append(
            make_item(
                title=model.get("modelId", ""),
                url=f"https://huggingface.co/{model.get('modelId', '')}",
                source="Hugging Face",
                source_kind="huggingface",
                published_at=model.get("lastModified"),
                summary=summary,
                authors=[model.get("author", "")] if model.get("author") else [],
                tags=tags[:8],
                engagement=int(model.get("likes") or 0) + int(model.get("downloads") or 0),
                extra={
                    "pipeline_tag": pipeline_tag,
                    "likes": model.get("likes", 0),
                    "downloads": model.get("downloads", 0),
                },
            )
        )
    return items


def collect_candidates() -> List[Dict[str, Any]]:
    collectors = [
        fetch_arxiv_candidates,
        fetch_reddit_candidates,
        fetch_hn_candidates,
        fetch_github_candidates,
        fetch_huggingface_candidates,
    ]
    candidates: List[Dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=len(collectors)) as pool:
        futures = [pool.submit(collector) for collector in collectors]
        for future in as_completed(futures):
            try:
                candidates.extend(future.result())
            except Exception as exc:  # pragma: no cover - defensive for cron execution
                print(f"[warn] collector failed: {exc}", file=sys.stderr)
    return candidates


def merge_candidates(candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    clusters: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for candidate in candidates:
        title_key = candidate.get("normalized_title") or normalize_title(candidate.get("title", ""))
        if not title_key:
            continue
        clusters[title_key].append(candidate)

    merged: List[Dict[str, Any]] = []
    for cluster_key, group in clusters.items():
        group = [item for item in group if item.get("title")]
        if not group:
            continue
        group.sort(
            key=lambda item: (
                source_weight(item),
                recency_score(item),
                engagement_score(item),
            ),
            reverse=True,
        )
        primary = group[0].copy()
        sources = []
        source_kinds = []
        tags = []
        authors = []
        summaries = []
        urls = []
        published_values = []
        extra = {}
        for item in group:
            sources.append(item.get("source", ""))
            source_kinds.append(item.get("source_kind", ""))
            tags.extend(item.get("tags", []))
            authors.extend(item.get("authors", []))
            if item.get("summary"):
                summaries.append(item["summary"])
            if item.get("url"):
                urls.append(item["url"])
            if item.get("published_at"):
                published_values.append(item["published_at"])
            extra.setdefault("variants", []).append(
                {
                    "source": item.get("source"),
                    "url": item.get("url"),
                    "summary": item.get("summary"),
                }
            )
        primary["id"] = cluster_key
        primary["sources"] = list(dict.fromkeys([source for source in sources if source]))
        primary["source_kinds"] = list(dict.fromkeys([kind for kind in source_kinds if kind]))
        primary["authors"] = list(dict.fromkeys([author for author in authors if author]))
        primary["tags"] = list(dict.fromkeys([tag for tag in tags if tag]))
        primary["summary"] = summaries[0] if summaries else primary.get("summary", "")
        primary["url"] = urls[0] if urls else primary.get("url")
        primary["published_at"] = max(
            (published_values or [primary.get("published_at")]),
            key=lambda value: parse_iso(value) or datetime.min.replace(tzinfo=timezone.utc),
        )
        primary["cluster_size"] = len(group)
        primary["extra"] = {
            **primary.get("extra", {}),
            **extra,
            "cluster_key": cluster_key,
        }
        score, details, reasons = score_candidate(primary, len(group))
        primary["score"] = score
        primary["score_breakdown"] = details
        primary["why"] = "；".join(reasons)
        primary["cluster_sources"] = [item.get("source") for item in group if item.get("source")]
        primary["cluster_urls"] = list(dict.fromkeys([item.get("url") for item in group if item.get("url")]))
        merged.append(primary)

    merged.sort(
        key=lambda item: (
            item.get("score", 0.0),
            parse_iso(item.get("published_at")) or datetime.min.replace(tzinfo=timezone.utc),
        ),
        reverse=True,
    )
    return merged


def topic_summary(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    counts = Counter(item.get("topic", "其他") for item in items)
    return [
        {"name": topic, "count": count}
        for topic, count in counts.most_common()
    ]


def source_summary(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    counts = Counter(item.get("source", "") for item in items)
    return [
        {"name": source, "count": count}
        for source, count in counts.most_common()
    ]


def selected_items(items: List[Dict[str, Any]], limit: int, threshold: float) -> List[Dict[str, Any]]:
    filtered = [item for item in items if item.get("score", 0.0) >= threshold]
    if len(filtered) < limit:
        filtered = items[:limit]
    return filtered[:limit]


def build_archive_index() -> List[Dict[str, Any]]:
    archive: List[Dict[str, Any]] = []
    if not DATA_DIR.exists():
        return archive
    for path in sorted(DATA_DIR.glob("*.json"), reverse=True):
        if path.name in {"latest.json", "index.json"}:
            continue
        if not re.match(r"\d{4}-\d{2}-\d{2}\.json$", path.name):
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        archive.append(
            {
                "date": payload.get("meta", {}).get("date", path.stem),
                "candidate_count": payload.get("meta", {}).get("candidate_count", 0),
                "selected_count": payload.get("meta", {}).get("selected_count", 0),
                "threshold": payload.get("meta", {}).get("threshold", 0),
                "json_url": f"data/{path.name}",
                "post_url": f"posts/{path.stem}.html",
                "generated_at": payload.get("meta", {}).get("generated_at"),
            }
        )
    archive.sort(key=lambda item: item["date"], reverse=True)
    return archive


def render_daily_html(payload: Dict[str, Any]) -> str:
    meta = payload["meta"]
    selected = payload["selected"]
    candidate_total = meta["candidate_count"]
    selected_total = meta["selected_count"]
    archive = payload.get("archive", [])
    topics = payload.get("topics", [])
    sources = payload.get("sources", [])

    def render_list(items: List[Dict[str, Any]], empty_text: str) -> str:
        if not items:
            return f'<div class="alert alert-light border">{html.escape(empty_text)}</div>'
        blocks = []
        for item in items:
            tags = "".join(
                f'<span class="badge text-bg-secondary me-1 mb-1">{html.escape(tag)}</span>'
                for tag in item.get("tags", [])[:6]
            )
            sources_text = " / ".join(item.get("sources", [item.get("source", "")]))
            score = item.get("score", 0.0)
            published = item.get("published_at") or ""
            published_text = published[:10] if published else "未知时间"
            summary = html.escape(item.get("summary", "")[:260])
            why = html.escape(item.get("why", ""))
            url = html.escape(item.get("url") or "#")
            blocks.append(
                f"""
                <article class="border rounded-2 p-3 mb-3 bg-white shadow-sm">
                  <div class="d-flex justify-content-between align-items-start gap-3">
                    <div class="me-auto">
                      <h5 class="mb-1">
                        <a href="{url}" target="_blank" rel="noreferrer">{html.escape(item.get('title', ''))}</a>
                      </h5>
                      <div class="text-muted small mb-2">
                        {html.escape(item.get('source', ''))} · {html.escape(item.get('topic', ''))} · {published_text} · {sources_text}
                      </div>
                    </div>
                    <span class="badge text-bg-primary fs-6">{score:.1f}</span>
                  </div>
                  <div class="mb-2">{summary}</div>
                  <div class="small text-muted mb-2">{why}</div>
                  <div>{tags}</div>
                </article>
                """
            )
        return "".join(blocks)

    top_topics = "".join(
        f'<li class="d-flex justify-content-between"><span>{html.escape(item["name"])}</span><strong>{item["count"]}</strong></li>'
        for item in topics[:6]
    ) or '<li class="text-muted">暂无</li>'
    top_sources = "".join(
        f'<li class="d-flex justify-content-between"><span>{html.escape(item["name"])}</span><strong>{item["count"]}</strong></li>'
        for item in sources[:6]
    ) or '<li class="text-muted">暂无</li>'
    archive_links = "".join(
        f'<li><a href="../{html.escape(entry["post_url"])}">{html.escape(entry["date"])}'
        f' <span class="text-muted">({entry["selected_count"]}/{entry["candidate_count"]})</span></a></li>'
        for entry in archive[:12]
    ) or '<li class="text-muted">暂无归档</li>'
    selected_html = render_list(selected, "今天还没有进入精选的条目。")

    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Frontier Tracker · {html.escape(meta['date'])}</title>
  <link rel="stylesheet" href="https://bootswatch.com/5/flatly/bootstrap.min.css">
  <style>
    body {{ background: #f7f9fb; }}
    .page-shell {{ max-width: 1180px; }}
    .section-title {{ font-size: 1.05rem; letter-spacing: 0; }}
    article a {{ text-decoration: none; }}
  </style>
</head>
<body>
  <div class="container page-shell py-4">
    <div class="d-flex flex-wrap align-items-end justify-content-between gap-3 mb-4">
      <div>
        <p class="text-uppercase text-muted small mb-1">Frontier Tracker</p>
        <h1 class="h3 mb-2">每天自动筛选前沿大模型信号</h1>
        <div class="text-muted">日期：{html.escape(meta['date'])} · 候选 {candidate_total} 条 · 精选 {selected_total} 条 · 阈值 {meta['threshold']:.1f}</div>
      </div>
      <div class="text-end">
        <a class="btn btn-outline-primary btn-sm" href="../../index.html">返回主页</a>
        <a class="btn btn-primary btn-sm" href="../index.html">打开追踪页</a>
      </div>
    </div>

    <div class="row g-3 mb-4">
      <div class="col-md-4">
        <div class="border rounded-2 bg-white p-3 h-100">
          <div class="section-title text-muted mb-2">主题分布</div>
          <ul class="list-unstyled mb-0">{top_topics}</ul>
        </div>
      </div>
      <div class="col-md-4">
        <div class="border rounded-2 bg-white p-3 h-100">
          <div class="section-title text-muted mb-2">来源分布</div>
          <ul class="list-unstyled mb-0">{top_sources}</ul>
        </div>
      </div>
      <div class="col-md-4">
        <div class="border rounded-2 bg-white p-3 h-100">
          <div class="section-title text-muted mb-2">历史归档</div>
          <ul class="list-unstyled mb-0">{archive_links}</ul>
        </div>
      </div>
    </div>

    <div class="row g-3">
      <div class="col-lg-8">
        <h2 class="section-title mb-3">今日精选</h2>
        {selected_html}
      </div>
      <div class="col-lg-4">
        <div class="border rounded-2 bg-white p-3 sticky-top" style="top: 1rem;">
          <div class="section-title text-muted mb-2">判断口径</div>
          <ul class="small mb-0">
            <li>原始来源优先于二手转述。</li>
            <li>先按技术增量与可信度打分，再看社区热度。</li>
            <li>同题多源交叉提及会小幅加分。</li>
            <li>低分条目仍保留在 JSON 候选池里，便于回溯。</li>
          </ul>
        </div>
      </div>
    </div>
  </div>
</body>
</html>
"""


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def build_payload(candidates: List[Dict[str, Any]], limit: int, threshold: float) -> Dict[str, Any]:
    merged = merge_candidates(candidates)
    selected = selected_items(merged, limit=limit, threshold=threshold)
    payload = {
        "meta": {
            "date": TODAY,
            "generated_at": format_iso(NOW),
            "candidate_count": len(candidates),
            "merged_count": len(merged),
            "selected_count": len(selected),
            "threshold": threshold,
            "limit": limit,
        },
        "selected": selected,
        "candidates": merged,
        "topics": topic_summary(selected),
        "sources": source_summary(selected),
    }
    payload["archive"] = build_archive_index()
    return payload


def render_index_page(manifest: Dict[str, Any]) -> str:
    latest = manifest.get("latest", {})
    archive = manifest.get("archive", [])
    latest_date = latest.get("date", TODAY)
    latest_post = latest.get("post_url", f"posts/{latest_date}.html")
    archive_links = "".join(
        f'<li class="mb-1"><a href="./{html.escape(entry["post_url"])}">{html.escape(entry["date"])}</a> '
        f'<span class="text-muted">({entry["selected_count"]}/{entry["candidate_count"]})</span></li>'
        for entry in archive[:18]
    ) or '<li class="text-muted">还没有归档</li>'
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Frontier Tracker</title>
  <link rel="stylesheet" href="https://bootswatch.com/5/flatly/bootstrap.min.css">
  <style>
    body {{ background: #f7f9fb; }}
    .tracker-shell {{ max-width: 1280px; }}
    .control-bar select, .control-bar input {{ min-width: 140px; }}
    .item-card {{ border: 1px solid rgba(0,0,0,.1); border-radius: 8px; background: #fff; }}
    .item-card a {{ text-decoration: none; }}
    .item-summary {{ white-space: pre-wrap; }}
    .sticky-panel {{ position: sticky; top: 1rem; }}
  </style>
</head>
<body>
  <div class="container tracker-shell py-4">
    <div class="d-flex flex-wrap align-items-end justify-content-between gap-3 mb-4">
      <div>
        <p class="text-uppercase text-muted small mb-1">Frontier Tracker</p>
        <h1 class="h3 mb-2">前沿大模型技术追踪</h1>
        <div class="text-muted">每天收集 200-500 条候选，自动压到 20 条以内，并生成可筛选的博客归档。</div>
      </div>
      <div class="text-end">
        <a class="btn btn-outline-primary btn-sm me-2" href="../index.html">返回主页</a>
        <a class="btn btn-primary btn-sm" href="./{html.escape(latest_post)}">打开最新博客</a>
      </div>
    </div>

    <div class="row g-3 mb-3 control-bar">
      <div class="col-lg-8">
        <div class="border rounded-2 bg-white p-3">
          <div class="d-flex flex-wrap gap-2 align-items-center">
            <label class="form-label mb-0">日期</label>
            <select id="dateSelect" class="form-select form-select-sm w-auto"></select>
            <label class="form-label mb-0 ms-2">视图</label>
            <div class="btn-group btn-group-sm" role="group" aria-label="view">
              <button class="btn btn-outline-primary active" data-view="selected" type="button">精选</button>
              <button class="btn btn-outline-primary" data-view="all" type="button">候选池</button>
            </div>
            <label class="form-label mb-0 ms-2">主题</label>
            <select id="topicSelect" class="form-select form-select-sm w-auto"></select>
            <label class="form-label mb-0 ms-2">来源</label>
            <select id="sourceSelect" class="form-select form-select-sm w-auto"></select>
            <label class="form-check-label ms-2">
              <input id="highScoreOnly" class="form-check-input me-1" type="checkbox" checked>只看高分
            </label>
            <label class="form-label mb-0 ms-2">最低分</label>
            <input id="minScore" type="range" class="form-range w-auto" min="0" max="100" step="1" value="75" style="width: 180px;">
            <span id="minScoreValue" class="text-muted small">75</span>
          </div>
        </div>
      </div>
      <div class="col-lg-4">
        <div class="border rounded-2 bg-white p-3 sticky-panel">
          <div class="d-flex justify-content-between">
            <div>
              <div class="text-muted small">最新归档</div>
              <div class="fw-semibold" id="currentDate">{html.escape(latest_date)}</div>
            </div>
            <div class="text-end">
              <div class="text-muted small">精选 / 候选</div>
              <div class="fw-semibold"><span id="selectedCount">0</span> / <span id="candidateCount">0</span></div>
            </div>
          </div>
          <div class="mt-2 text-muted small" id="dataMeta">正在加载数据。</div>
        </div>
      </div>
    </div>

    <div class="row g-3">
      <div class="col-lg-8">
        <div class="mb-3">
          <div class="row g-2" id="statsRow"></div>
        </div>
        <div id="items"></div>
      </div>
      <div class="col-lg-4">
        <div class="border rounded-2 bg-white p-3 mb-3">
          <div class="fw-semibold mb-2">归档</div>
          <ul id="archiveList" class="list-unstyled small mb-0">{archive_links}</ul>
        </div>
        <div class="border rounded-2 bg-white p-3">
          <div class="fw-semibold mb-2">说明</div>
          <ul class="small mb-0">
            <li>高分代表“值得优先追踪”，不等于绝对事实判断。</li>
            <li>“候选池”保留了更广的输入，用于回溯和手工审阅。</li>
            <li>你可以继续往 `blog/feeds.json` 增加 RSS/Atom 源。</li>
          </ul>
        </div>
      </div>
    </div>
  </div>

  <script>
    const state = {{
      view: 'selected',
      minScore: 75,
      highScoreOnly: true,
      topic: '全部',
      source: '全部',
      date: null,
      payload: null,
      archive: [],
    }};

    const elements = {{
      dateSelect: document.getElementById('dateSelect'),
      topicSelect: document.getElementById('topicSelect'),
      sourceSelect: document.getElementById('sourceSelect'),
      highScoreOnly: document.getElementById('highScoreOnly'),
      minScore: document.getElementById('minScore'),
      minScoreValue: document.getElementById('minScoreValue'),
      items: document.getElementById('items'),
      selectedCount: document.getElementById('selectedCount'),
      candidateCount: document.getElementById('candidateCount'),
      dataMeta: document.getElementById('dataMeta'),
      statsRow: document.getElementById('statsRow'),
      currentDate: document.getElementById('currentDate'),
    }};

    function escapeHtml(value) {{
      return String(value)
        .replaceAll('&', '&amp;')
        .replaceAll('<', '&lt;')
        .replaceAll('>', '&gt;')
        .replaceAll('"', '&quot;')
        .replaceAll("'", '&#39;');
    }}

    function pct(value) {{
      return `${{Number(value).toFixed(1)}}`;
    }}

    function sortedUnique(values) {{
      return Array.from(new Set(values.filter(Boolean))).sort((a, b) => a.localeCompare(b, 'zh-Hans-CN'));
    }}

    function scoreBadge(score) {{
      const tone = score >= 85 ? 'primary' : score >= 72 ? 'success' : score >= 60 ? 'warning' : 'secondary';
      return `<span class="badge text-bg-${{tone}} fs-6">${{pct(score)}}</span>`;
    }}

    function buildOptionList(values, current) {{
      return values.map(value => `<option value="${{escapeHtml(value)}}">${{escapeHtml(value)}}</option>`).join('');
    }}

    function metricCard(title, value, subtitle = '') {{
      return `
        <div class="col-md-4">
          <div class="border rounded-2 bg-white p-3 h-100">
            <div class="text-muted small">${{escapeHtml(title)}}</div>
            <div class="h4 mb-1">${{escapeHtml(value)}}</div>
            <div class="text-muted small">${{escapeHtml(subtitle)}}</div>
          </div>
        </div>`;
    }}

    function itemMatches(item) {{
      if (state.topic !== '全部' && item.topic !== state.topic) return false;
      if (state.source !== '全部') {{
        const sources = item.sources || [item.source];
        if (!sources.includes(state.source) && item.source !== state.source) return false;
      }}
      if (state.highScoreOnly && item.score < state.minScore) return false;
      return true;
    }}

    function currentItems() {{
      const source = state.payload || {{}};
      const items = state.view === 'selected' ? (source.selected || []) : (source.candidates || []);
      return items.filter(itemMatches);
    }}

    function renderStats(items) {{
      const topTopics = (state.payload?.topics || []).slice(0, 3);
      const topSources = (state.payload?.sources || []).slice(0, 3);
      elements.statsRow.innerHTML = [
        metricCard('当前视图', state.view === 'selected' ? '精选' : '候选池', '过滤结果会随主题与来源联动'),
        metricCard('当前条目', String(items.length), '满足筛选条件的记录数'),
        metricCard('高分阈值', state.highScoreOnly ? `≥ ${{state.minScore}}` : '关闭', '分数越高越优先展示'),
      ].join('');
      const meta = state.payload?.meta || {{}};
      elements.dataMeta.textContent = `生成于 ${{meta.generated_at || '未知'}}，聚合后 ${{meta.merged_count || 0}} 条，原始候选 ${{meta.candidate_count || 0}} 条。`;
      elements.selectedCount.textContent = String(meta.selected_count || 0);
      elements.candidateCount.textContent = String(meta.candidate_count || 0);
      elements.currentDate.textContent = meta.date || state.date || '';
    }}

    function renderItem(item) {{
      const tags = (item.tags || []).slice(0, 6).map(tag => `<span class="badge text-bg-secondary me-1 mb-1">${{escapeHtml(tag)}}</span>`).join('');
      const sources = (item.sources || [item.source]).map(escapeHtml).join(' / ');
      const published = item.published_at ? String(item.published_at).slice(0, 10) : '未知';
      const score = scoreBadge(item.score || 0);
      const url = escapeHtml(item.url || '#');
      return `
        <article class="item-card p-3 mb-3">
          <div class="d-flex gap-3 align-items-start">
            <div class="flex-grow-1">
              <div class="d-flex align-items-start gap-2">
                <h3 class="h6 mb-1 flex-grow-1">
                  <a href="${{url}}" target="_blank" rel="noreferrer">${{escapeHtml(item.title || '')}}</a>
                </h3>
                ${{score}}
              </div>
              <div class="text-muted small mb-2">
                ${{escapeHtml(item.source || '')}} · ${{escapeHtml(item.topic || '')}} · ${{published}} · ${{sources}}
              </div>
            </div>
          </div>
          <div class="item-summary mb-2">${{escapeHtml((item.summary || '').slice(0, 320))}}</div>
          <div class="small text-muted mb-2">${{escapeHtml(item.why || '')}}</div>
          <div>${{tags}}</div>
        </article>`;
    }}

    function renderItems() {{
      const items = currentItems();
      renderStats(items);
      if (!items.length) {{
        elements.items.innerHTML = '<div class="alert alert-light border">没有匹配到条目，请放宽筛选条件。</div>';
        return;
      }}
      elements.items.innerHTML = items.map(renderItem).join('');
    }}

    function rebuildFilters() {{
      const payload = state.payload || {{}};
      const allItems = [...(payload.selected || []), ...(payload.candidates || [])];
      const topics = ['全部', ...sortedUnique(allItems.map(item => item.topic))];
      const sources = ['全部', ...sortedUnique(allItems.flatMap(item => item.sources && item.sources.length ? item.sources : [item.source]).filter(Boolean))];
      elements.topicSelect.innerHTML = buildOptionList(topics);
      elements.sourceSelect.innerHTML = buildOptionList(sources);
      elements.topicSelect.value = state.topic;
      elements.sourceSelect.value = state.source;
    }}

    async function loadPayload(date) {{
      const response = await fetch(`./data/${{date}}.json`, {{ cache: 'no-store' }});
      if (!response.ok) throw new Error(`无法加载 ${{date}} 的数据`);
      return response.json();
    }}

    async function loadArchive() {{
      const response = await fetch('./data/index.json', {{ cache: 'no-store' }});
      if (!response.ok) return [];
      const payload = await response.json();
      return payload.archive || [];
    }}

    async function initialize() {{
      state.archive = await loadArchive();
      const dates = state.archive.map(item => item.date);
      if (!dates.length) dates.push('{TODAY}');
      elements.dateSelect.innerHTML = dates.map(date => `<option value="${{escapeHtml(date)}}">${{escapeHtml(date)}}</option>`).join('');
      state.date = dates[0];
      elements.dateSelect.value = state.date;
      await refreshData(state.date);

      elements.dateSelect.addEventListener('change', async () => {{
        await refreshData(elements.dateSelect.value);
      }});

      document.querySelectorAll('[data-view]').forEach(button => {{
        button.addEventListener('click', async () => {{
          document.querySelectorAll('[data-view]').forEach(btn => btn.classList.remove('active'));
          button.classList.add('active');
          state.view = button.dataset.view;
          renderItems();
        }});
      }});

      elements.topicSelect.addEventListener('change', () => {{
        state.topic = elements.topicSelect.value;
        renderItems();
      }});
      elements.sourceSelect.addEventListener('change', () => {{
        state.source = elements.sourceSelect.value;
        renderItems();
      }});
      elements.highScoreOnly.addEventListener('change', () => {{
        state.highScoreOnly = elements.highScoreOnly.checked;
        renderItems();
      }});
      elements.minScore.addEventListener('input', () => {{
        state.minScore = Number(elements.minScore.value);
        elements.minScoreValue.textContent = String(state.minScore);
        renderItems();
      }});
    }}

    async function refreshData(date) {{
      state.date = date;
      elements.currentDate.textContent = date;
      try {{
        state.payload = await loadPayload(date);
        const meta = state.payload.meta || {{}};
        state.minScore = Math.max(60, Math.min(90, Math.round(meta.threshold || 75)));
        elements.minScore.value = String(state.minScore);
        elements.minScoreValue.textContent = String(state.minScore);
        elements.highScoreOnly.checked = true;
        state.highScoreOnly = true;
        rebuildFilters();
        renderItems();
      }} catch (error) {{
        elements.items.innerHTML = `<div class="alert alert-danger">${{escapeHtml(error.message)}}</div>`;
      }}
    }}

    initialize();
  </script>
</body>
</html>
"""


def build_manifest() -> Dict[str, Any]:
    archive = build_archive_index()
    latest = archive[0] if archive else {}
    return {"latest": latest, "archive": archive, "generated_at": format_iso(NOW)}


def ensure_placeholder_files() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    POSTS_DIR.mkdir(parents=True, exist_ok=True)


def main() -> int:
    global BLOG_DIR, DATA_DIR, POSTS_DIR
    parser = argparse.ArgumentParser(description="Build the frontier tracker blog")
    parser.add_argument("--output-dir", default=str(BLOG_DIR), help="Output directory for the blog")
    parser.add_argument("--selected-limit", type=int, default=20, help="Maximum selected items")
    parser.add_argument("--threshold", type=float, default=72.0, help="Selection score threshold")
    parser.add_argument("--write-index-html", action="store_true", help="Rewrite blog/index.html from the manifest")
    args = parser.parse_args()
    output_dir = Path(args.output_dir).resolve()
    BLOG_DIR = output_dir
    DATA_DIR = BLOG_DIR / "data"
    POSTS_DIR = BLOG_DIR / "posts"

    ensure_placeholder_files()
    candidates = collect_candidates()
    payload = build_payload(candidates, limit=args.selected_limit, threshold=args.threshold)

    daily_json = DATA_DIR / f"{TODAY}.json"
    latest_json = DATA_DIR / "latest.json"
    post_html = POSTS_DIR / f"{TODAY}.html"

    write_json(daily_json, payload)
    write_json(latest_json, payload)
    archive = build_archive_index()
    payload["archive"] = archive
    manifest = {
        "latest": archive[0] if archive else {},
        "archive": archive,
        "generated_at": format_iso(NOW),
    }
    write_json(DATA_DIR / "index.json", manifest)
    write_text(post_html, render_daily_html(payload))

    if args.write_index_html or not (BLOG_DIR / "index.html").exists():
        write_text(BLOG_DIR / "index.html", render_index_page(manifest))

    print(
        json.dumps(
            {
                "date": TODAY,
                "candidate_count": payload["meta"]["candidate_count"],
                "selected_count": payload["meta"]["selected_count"],
                "output_dir": str(BLOG_DIR),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
