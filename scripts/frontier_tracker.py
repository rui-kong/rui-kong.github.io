#!/usr/bin/env python3
"""Frontier model news radar.

Pipeline: collect -> score AI relevance -> merge into stories -> rank -> render.

The shape follows LearnPrompt/ai-news-radar: a stable pipeline that emits static
JSON, plus a single-layer front end (category tabs x brief/all toggle x
timeline) that only reads those JSON files. No backend, no API key, no login.

Outputs under blog/data:
  daily-brief.json      curated items for the latest run
  latest-24h.json       strong AI signal inside the recent window
  latest-24h-all.json   broader AI-related pool (relevance >= 0.3)
  stories-merged.json   full merged story set
  source-status.json    per-source fetch health and AI ratio
  merge-log.json        how stories were merged, for auditing
  index.json            archive manifest
  daily/<date>.json     per-day snapshot consumed by the web UI
"""

from __future__ import annotations

import argparse
import html
import json
import math
import os
import re
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlparse
from urllib.request import Request, urlopen
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BLOG_DIR = ROOT / "blog"
SOURCES_FILE = Path(__file__).resolve().parent / "sources.json"

NOW = datetime.now(timezone.utc)
TODAY = NOW.date().isoformat()
WINDOW_HOURS = 48
USER_AGENT = "frontier-tracker/2.0 (+https://rui-kong.github.io/blog/)"
TIMEOUT = 25

CATEGORIES = ["模型", "产品", "开发者", "行业", "论文", "社区", "自媒体"]

STRONG_TERMS = {
    "large language model": 0.36,
    "language model": 0.30,
    "llm": 0.34,
    "transformer": 0.30,
    "gpt": 0.28,
    "claude": 0.28,
    "gemini": 0.26,
    "llama": 0.26,
    "qwen": 0.26,
    "deepseek": 0.28,
    "mixture-of-experts": 0.32,
    "mixture of experts": 0.32,
    "moe": 0.26,
    "pretraining": 0.32,
    "pre-training": 0.32,
    "post-training": 0.30,
    "fine-tuning": 0.24,
    "reinforcement learning": 0.26,
    "rlhf": 0.30,
    "diffusion model": 0.26,
    "attention": 0.22,
    "inference": 0.22,
    "quantization": 0.26,
    "kv cache": 0.32,
    "speculative decoding": 0.34,
    "agent": 0.20,
    "multimodal": 0.24,
    "benchmark": 0.18,
    "reasoning": 0.22,
    "scaling law": 0.34,
    "tokenizer": 0.24,
    "context window": 0.26,
    "大模型": 0.30,
    "预训练": 0.32,
    "推理加速": 0.30,
    "智能体": 0.24,
    "长上下文": 0.26,
}

WEAK_TERMS = {
    "artificial intelligence": 0.12,
    "machine learning": 0.14,
    "deep learning": 0.16,
    "neural": 0.12,
    "dataset": 0.10,
    "training": 0.12,
    "open-weight": 0.16,
    "open source": 0.08,
    "gpu": 0.12,
    "cuda": 0.14,
    "人工智能": 0.12,
    "模型": 0.10,
}

TOPIC_RULES = [
    ("推理与系统", ["inference", "serving", "kv cache", "speculative", "latency", "throughput",
                "quantization", "kernel", "cuda", "vllm", "sglang", "decoding", "triton"]),
    ("训练与架构", ["pretrain", "pre-training", "training", "optimizer", "scaling", "moe",
                "transformer", "architecture", "looped", "recurrent", "attention", "tokenizer"]),
    ("对齐与后训练", ["alignment", "rlhf", "rlaif", "post-training", "dpo", "grpo", "ppo",
                 "sft", "preference", "reward", "distillation"]),
    ("Agent 与工具", ["agent", "tool use", "tool-use", "workflow", "orchestration", "memory",
                   "browser", "function calling", "mcp", "coding agent"]),
    ("多模态", ["vision", "image", "video", "audio", "speech", "multimodal", "vlm", "omni"]),
    ("数据与评测", ["benchmark", "evaluation", "eval", "dataset", "leaderboard", "swebench",
                "contamination", "robustness"]),
    ("发布与生态", ["release", "open source", "open-source", "weights", "checkpoint", "launch",
                "preview", "api"]),
]

STOPWORDS = {
    "the", "a", "an", "of", "for", "and", "to", "in", "on", "with", "by", "via", "from",
    "is", "are", "we", "our", "that", "this", "using", "towards", "toward", "how", "why",
    "new", "can", "its", "it", "at", "as", "be", "into", "about", "over", "more",
}


def load_sources() -> Dict[str, Any]:
    if SOURCES_FILE.exists():
        try:
            return json.loads(SOURCES_FILE.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            print(f"[warn] sources.json is invalid: {exc}", file=sys.stderr)
    return {"feeds": [], "arxiv_categories": ["cs.CL", "cs.LG", "cs.AI"],
            "reddit_subreddits": ["MachineLearning", "LocalLLaMA"],
            "github_queries": ["llm", "transformer"]}


BROWSER_UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
              "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")


def fetch_text(url: str, headers: Optional[Dict[str, str]] = None, attempts: int = 3) -> str:
    """Fetch a URL, retrying transient failures and falling back to a browser UA."""
    last_error: Optional[Exception] = None
    for attempt in range(attempts):
        request_headers = {"User-Agent": BROWSER_UA if attempt else USER_AGENT}
        if headers:
            request_headers.update(headers)
        try:
            with urlopen(Request(url, headers=request_headers), timeout=TIMEOUT) as response:
                return response.read().decode("utf-8", errors="replace")
        except (HTTPError, URLError, TimeoutError) as exc:
            last_error = exc
            if isinstance(exc, HTTPError) and exc.code in {401, 404, 410}:
                break
            time.sleep(1.5 * (attempt + 1))
    raise last_error if last_error else RuntimeError(f"failed to fetch {url}")


def fetch_json(url: str, headers: Optional[Dict[str, str]] = None) -> Any:
    return json.loads(fetch_text(url, headers=headers))


def clean_text(text: Optional[str]) -> str:
    if not text:
        return ""
    text = re.sub(r"<[^>]+>", " ", text)
    text = html.unescape(text)
    return re.sub(r"\s+", " ", text).strip()

def parse_date(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    value = value.strip()
    candidate = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(candidate)
    except ValueError:
        try:
            parsed = parsedate_to_datetime(value)
        except (TypeError, ValueError, IndexError):
            return None
    if parsed is None:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def to_iso(value: Optional[datetime]) -> Optional[str]:
    if value is None:
        return None
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def canonical_url(url: Optional[str]) -> Optional[str]:
    if not url:
        return None
    parsed = urlparse(url)
    if not parsed.scheme or not parsed.netloc:
        return url
    path = parsed.path.rstrip("/")
    if "arxiv.org" in parsed.netloc:
        path = re.sub(r"^/pdf/", "/abs/", path)
        path = re.sub(r"v\d+$", "", path)
    return f"{parsed.scheme}://{parsed.netloc}{path}"


def title_tokens(title: str) -> set:
    lowered = title.lower()
    lowered = re.sub(r"https?://\S+", " ", lowered)
    raw = re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]{2,}", lowered)
    return {token for token in raw if len(token) > 2 and token not in STOPWORDS}


def jaccard(left: set, right: set) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)

def ai_relevance(text: str) -> float:
    lowered = text.lower()
    score = 0.0
    for term, weight in STRONG_TERMS.items():
        if term in lowered:
            score += weight
    for term, weight in WEAK_TERMS.items():
        if term in lowered:
            score += weight
    return round(min(1.0, score), 3)


def detect_topic(text: str) -> str:
    lowered = text.lower()
    best_topic = "其他"
    best_hits = 0
    for topic, keywords in TOPIC_RULES:
        hits = sum(1 for keyword in keywords if keyword in lowered)
        if hits > best_hits:
            best_topic, best_hits = topic, hits
    return best_topic


def recency_weight(published: Optional[datetime]) -> float:
    if published is None:
        return 0.4
    age_hours = max(0.0, (NOW - published).total_seconds() / 3600.0)
    return max(0.0, 1.0 - min(age_hours, float(WINDOW_HOURS)) / float(WINDOW_HOURS))


def engagement_weight(value: int) -> float:
    if value <= 0:
        return 0.0
    return min(1.0, math.log10(1.0 + float(value)) / 4.0)


def novelty_weight(text: str) -> float:
    lowered = text.lower()
    markers = ("release", "released", "introduc", "launch", "technical report", "open-weight",
               "we present", "open source", "preview", "首发", "发布", "开源")
    hits = sum(1 for marker in markers if marker in lowered)
    return min(1.0, 0.25 * hits)

def make_item(*, title: str, url: Optional[str], source: str, source_kind: str,
              category: str, published_at: Optional[str], summary: str = "",
              weight: float = 0.7, engagement: int = 0,
              tags: Optional[List[str]] = None,
              extra: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    title = clean_text(title)
    if not title:
        return None
    summary = clean_text(summary)
    tags = [tag for tag in (tags or []) if tag]
    text = f"{title} {summary} {' '.join(tags)}"
    published = parse_date(published_at)
    return {
        "title": title,
        "url": canonical_url(url),
        "source": source,
        "source_kind": source_kind,
        "source_weight": weight,
        "category": category,
        "published_at": to_iso(published),
        "summary": summary[:600],
        "tags": tags[:8],
        "engagement": int(engagement or 0),
        "ai_score": ai_relevance(text),
        "topic": detect_topic(text),
        "tokens": title_tokens(title),
        "extra": extra or {},
    }


def score_story(story: Dict[str, Any]) -> Tuple[float, Dict[str, float], str]:
    published = parse_date(story.get("published_at"))
    source_component = story["source_weight"]
    recency_component = recency_weight(published)
    novelty_component = novelty_weight(f"{story['title']} {story.get('summary', '')}")
    relevance_component = story["ai_score"]
    multi_component = min(1.0, 0.5 * (story["cluster_size"] - 1))
    engagement_component = engagement_weight(story.get("engagement", 0))

    raw = (0.30 * source_component
           + 0.22 * recency_component
           + 0.16 * novelty_component
           + 0.14 * relevance_component
           + 0.10 * multi_component
           + 0.08 * engagement_component)
    score = round(min(100.0, raw * 115.0), 1)

    reasons: List[str] = []
    if source_component >= 0.9:
        reasons.append("一手来源")
    elif source_component >= 0.8:
        reasons.append("高信号来源")
    if story["cluster_size"] > 1:
        reasons.append(f"{story['cluster_size']} 家信源同时报道")
    if recency_component >= 0.75:
        reasons.append("刚刚发生")
    if novelty_component >= 0.25:
        reasons.append("有明确发布或技术增量")
    if relevance_component >= 0.7:
        reasons.append("与大模型技术强相关")
    if engagement_component >= 0.5:
        reasons.append("社区讨论热度高")
    if not reasons:
        reasons.append("综合信号达到入选线")

    breakdown = {
        "source": round(source_component * 100, 1),
        "recency": round(recency_component * 100, 1),
        "novelty": round(novelty_component * 100, 1),
        "relevance": round(relevance_component * 100, 1),
        "multi_source": round(multi_component * 100, 1),
        "engagement": round(engagement_component * 100, 1),
    }
    return score, breakdown, "、".join(reasons)

ATOM_NS = {"atom": "http://www.w3.org/2005/Atom"}


def collect_feed(feed: Dict[str, Any]) -> List[Dict[str, Any]]:
    xml_text = fetch_text(feed["url"], headers={"Accept": "application/rss+xml, application/xml"})
    root = ET.fromstring(xml_text)
    items: List[Dict[str, Any]] = []
    nodes = root.findall(".//item") or root.findall(".//atom:entry", ATOM_NS)
    for node in nodes[:30]:
        title = node.findtext("title") or node.findtext("atom:title", namespaces=ATOM_NS) or ""
        link = node.findtext("link") or ""
        if not link:
            link_node = node.find("atom:link", ATOM_NS)
            if link_node is not None:
                link = link_node.attrib.get("href", "")
        published = (node.findtext("pubDate")
                     or node.findtext("atom:published", namespaces=ATOM_NS)
                     or node.findtext("atom:updated", namespaces=ATOM_NS))
        summary = (node.findtext("description")
                   or node.findtext("atom:summary", namespaces=ATOM_NS)
                   or node.findtext("atom:content", namespaces=ATOM_NS)
                   or "")
        item = make_item(title=title, url=link, source=feed["name"], source_kind="feed",
                         category=feed.get("category", "行业"), published_at=published,
                         summary=summary, weight=float(feed.get("weight", 0.7)),
                         tags=[feed.get("category", "行业")])
        if item:
            items.append(item)
    return items


def collect_arxiv(categories: List[str]) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    for category in categories:
        url = "https://export.arxiv.org/api/query?" + urlencode({
            "search_query": f"cat:{category}",
            "sortBy": "submittedDate",
            "sortOrder": "descending",
            "start": 0,
            "max_results": 30,
        })
        root = ET.fromstring(fetch_text(url))
        for entry in root.findall("atom:entry", ATOM_NS):
            title = entry.findtext("atom:title", default="", namespaces=ATOM_NS)
            summary = entry.findtext("atom:summary", default="", namespaces=ATOM_NS)
            published = entry.findtext("atom:published", default="", namespaces=ATOM_NS)
            authors = [clean_text(node.findtext("atom:name", default="", namespaces=ATOM_NS))
                       for node in entry.findall("atom:author", ATOM_NS)]
            abs_url = ""
            for link in entry.findall("atom:link", ATOM_NS):
                if link.attrib.get("rel") == "alternate":
                    abs_url = link.attrib.get("href", "")
            item = make_item(title=title, url=abs_url, source="arXiv", source_kind="arxiv",
                             category="论文", published_at=published, summary=summary,
                             weight=0.92, tags=[category],
                             extra={"authors": authors[:8]})
            if item:
                items.append(item)
    return items

def collect_hf_papers() -> List[Dict[str, Any]]:
    payload = fetch_json("https://huggingface.co/api/daily_papers?limit=40",
                         headers={"Accept": "application/json"})
    items: List[Dict[str, Any]] = []
    for entry in payload:
        paper = entry.get("paper") or {}
        paper_id = paper.get("id") or ""
        item = make_item(title=paper.get("title", ""),
                         url=f"https://huggingface.co/papers/{paper_id}" if paper_id else None,
                         source="HF Daily Papers", source_kind="hf-papers", category="论文",
                         published_at=entry.get("publishedAt") or paper.get("publishedAt"),
                         summary=paper.get("summary", ""), weight=0.95,
                         engagement=int(paper.get("upvotes") or 0),
                         tags=["daily papers"])
        if item:
            items.append(item)
    return items


JUNK_MODEL_ID = re.compile(r"(?:[0-9a-f]{12,}|-\d{5,}|checkpoint[-_]?\d+|step[-_]?\d+)", re.I)


def collect_hf_models() -> List[Dict[str, Any]]:
    url = "https://huggingface.co/api/models?" + urlencode(
        {"sort": "lastModified", "direction": -1, "limit": 120, "full": "true"})
    payload = fetch_json(url, headers={"Accept": "application/json"})
    items: List[Dict[str, Any]] = []
    for model in payload:
        model_id = model.get("modelId", "")
        likes = int(model.get("likes") or 0)
        downloads = int(model.get("downloads") or 0)
        # Skip private-looking experiment dumps and models nobody has touched yet.
        if not model_id or JUNK_MODEL_ID.search(model_id):
            continue
        if likes < 3 and downloads < 500:
            continue
        tags = [tag for tag in (model.get("tags") or []) if tag][:8]
        pipeline_tag = model.get("pipeline_tag")
        summary = ", ".join(tags[:5]) or (pipeline_tag or "")
        item = make_item(title=model_id, url=f"https://huggingface.co/{model_id}",
                         source="Hugging Face", source_kind="hf-models", category="模型",
                         published_at=model.get("lastModified"), summary=summary, weight=0.82,
                         engagement=likes * 20 + downloads // 50,
                         tags=tags + ([pipeline_tag] if pipeline_tag else []))
        if item:
            items.append(item)
    return items

def collect_github(queries: List[str]) -> List[Dict[str, Any]]:
    since = (NOW.date() - timedelta(days=3)).isoformat()
    headers = {"Accept": "application/vnd.github+json"}
    token = os.environ.get("GITHUB_TOKEN", "").strip()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    items: List[Dict[str, Any]] = []
    for query in queries:
        url = "https://api.github.com/search/repositories?" + urlencode({
            "q": f"{query} pushed:>={since} stars:>200",
            "sort": "updated",
            "order": "desc",
            "per_page": 20,
        })
        payload = fetch_json(url, headers=headers)
        for repo in payload.get("items", []):
            item = make_item(title=repo.get("full_name", ""), url=repo.get("html_url"),
                             source="GitHub", source_kind="github", category="开发者",
                             published_at=repo.get("pushed_at") or repo.get("updated_at"),
                             summary=repo.get("description") or "", weight=0.70,
                             engagement=int(repo.get("stargazers_count") or 0),
                             tags=[repo.get("language") or "repo"])
            if item:
                items.append(item)
    return items


def collect_hn() -> List[Dict[str, Any]]:
    url = ("https://hn.algolia.com/api/v1/search_by_date?"
           + urlencode({"tags": "story", "query": "AI", "hitsPerPage": 60}))
    payload = fetch_json(url, headers={"Accept": "application/json"})
    items: List[Dict[str, Any]] = []
    for hit in payload.get("hits", []):
        story_id = hit.get("objectID")
        link = hit.get("url") or f"https://news.ycombinator.com/item?id={story_id}"
        item = make_item(title=hit.get("title") or hit.get("story_title") or "", url=link,
                         source="Hacker News", source_kind="hn", category="社区",
                         published_at=hit.get("created_at"),
                         summary=f"{hit.get('points') or 0} points, {hit.get('num_comments') or 0} comments",
                         weight=0.68,
                         engagement=int(hit.get("points") or 0) + int(hit.get("num_comments") or 0),
                         tags=["hacker news"],
                         extra={"hn_url": f"https://news.ycombinator.com/item?id={story_id}"})
        if item:
            items.append(item)
    return items

def collect_reddit(subreddits: List[str]) -> List[Dict[str, Any]]:
    """Reddit blocks the JSON API from datacenters, so read the public RSS feed."""
    items: List[Dict[str, Any]] = []
    for subreddit in subreddits:
        xml_text = fetch_text(f"https://www.reddit.com/r/{subreddit}/top/.rss?t=day",
                              headers={"Accept": "application/atom+xml",
                                       "User-Agent": BROWSER_UA})
        root = ET.fromstring(xml_text)
        for entry in root.findall("atom:entry", ATOM_NS)[:25]:
            link_node = entry.find("atom:link", ATOM_NS)
            link = link_node.attrib.get("href", "") if link_node is not None else ""
            item = make_item(title=entry.findtext("atom:title", default="", namespaces=ATOM_NS),
                             url=link, source=f"r/{subreddit}", source_kind="reddit",
                             category="社区",
                             published_at=(entry.findtext("atom:updated", namespaces=ATOM_NS)
                                           or entry.findtext("atom:published", namespaces=ATOM_NS)),
                             summary=entry.findtext("atom:content", default="", namespaces=ATOM_NS),
                             weight=0.62, tags=[subreddit])
            if item:
                items.append(item)
        time.sleep(2.0)
    return items


def build_collectors(config: Dict[str, Any]) -> List[Tuple[str, Callable[[], List[Dict[str, Any]]]]]:
    collectors: List[Tuple[str, Callable[[], List[Dict[str, Any]]]]] = [
        ("arXiv", lambda: collect_arxiv(config.get("arxiv_categories", []))),
        ("HF Daily Papers", collect_hf_papers),
        ("Hugging Face", collect_hf_models),
        ("GitHub", lambda: collect_github(config.get("github_queries", []))),
        ("Hacker News", collect_hn),
        ("Reddit", lambda: collect_reddit(config.get("reddit_subreddits", []))),
    ]
    for feed in config.get("feeds", []):
        collectors.append((feed["name"], lambda feed=feed: collect_feed(feed)))
    return collectors

def collect_all(config: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    collectors = build_collectors(config)
    items: List[Dict[str, Any]] = []
    status: List[Dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(collector): name for name, collector in collectors}
        for future in as_completed(futures):
            name = futures[future]
            try:
                fetched = future.result()
            except (HTTPError, URLError, TimeoutError, ET.ParseError, json.JSONDecodeError,
                    ValueError, KeyError) as exc:
                status.append({"name": name, "ok": False, "fetched": 0, "ai_related": 0,
                               "ai_ratio": 0.0, "error": f"{type(exc).__name__}: {exc}"[:160]})
                continue
            except Exception as exc:  # pragma: no cover - cron safety net
                status.append({"name": name, "ok": False, "fetched": 0, "ai_related": 0,
                               "ai_ratio": 0.0, "error": f"{type(exc).__name__}: {exc}"[:160]})
                continue
            ai_related = sum(1 for item in fetched if item["ai_score"] >= 0.3)
            status.append({
                "name": name,
                "ok": True,
                "fetched": len(fetched),
                "ai_related": ai_related,
                "ai_ratio": round(ai_related / len(fetched), 3) if fetched else 0.0,
                "error": None,
            })
            items.extend(fetched)
    status.sort(key=lambda entry: (not entry["ok"], -entry["fetched"], entry["name"]))
    return items, status


def in_window(item: Dict[str, Any]) -> bool:
    published = parse_date(item.get("published_at"))
    if published is None:
        return False
    return (NOW - published) <= timedelta(hours=WINDOW_HOURS)

def merge_stories(items: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Group items that describe the same event into a single story."""
    ordered = sorted(items, key=lambda item: (item["source_weight"], item["ai_score"]), reverse=True)
    stories: List[Dict[str, Any]] = []
    by_url: Dict[str, Dict[str, Any]] = {}
    merge_log: List[Dict[str, Any]] = []

    for item in ordered:
        target: Optional[Dict[str, Any]] = None
        reason = ""
        url = item.get("url")
        if url and url in by_url:
            target = by_url[url]
            reason = "same-url"
        else:
            for story in stories:
                similarity = jaccard(story["tokens"], item["tokens"])
                if similarity >= 0.55:
                    target = story
                    reason = f"title-similarity={similarity:.2f}"
                    break
        if target is None:
            story = dict(item)
            story["cluster_size"] = 1
            story["sources"] = [item["source"]]
            story["variants"] = [{"source": item["source"], "title": item["title"],
                                  "url": item["url"], "published_at": item["published_at"]}]
            stories.append(story)
            if url:
                by_url[url] = story
            continue
        target["cluster_size"] += 1
        if item["source"] not in target["sources"]:
            target["sources"].append(item["source"])
        target["variants"].append({"source": item["source"], "title": item["title"],
                                   "url": item["url"], "published_at": item["published_at"]})
        target["engagement"] = max(target.get("engagement", 0), item.get("engagement", 0))
        target["ai_score"] = max(target["ai_score"], item["ai_score"])
        target["tokens"] = target["tokens"] | item["tokens"]
        if not target.get("summary") and item.get("summary"):
            target["summary"] = item["summary"]
        merge_log.append({"into": target["title"], "from": item["title"],
                          "source": item["source"], "reason": reason})
    return stories, merge_log

def finalize_story(story: Dict[str, Any]) -> Dict[str, Any]:
    score, breakdown, why = score_story(story)
    published = story.get("published_at")
    return {
        "id": re.sub(r"[^a-z0-9]+", "-", " ".join(sorted(story["tokens"]))[:80]).strip("-"),
        "title": story["title"],
        "url": story.get("url"),
        "source": story["source"],
        "sources": story["sources"],
        "source_kind": story["source_kind"],
        "category": story["category"],
        "topic": story["topic"],
        "published_at": published,
        "date": (published or "")[:10],
        "summary": story.get("summary", "")[:400],
        "tags": story.get("tags", [])[:8],
        "engagement": story.get("engagement", 0),
        "ai_score": story["ai_score"],
        "cluster_size": story["cluster_size"],
        "variants": story["variants"][:8],
        "score": score,
        "score_breakdown": breakdown,
        "why": why,
        "extra": story.get("extra", {}),
    }


def pick_brief(stories: List[Dict[str, Any]], limit: int, threshold: float,
               per_source_cap: int, per_category_cap: int) -> List[Dict[str, Any]]:
    """Rank by score with source/category caps so one firehose cannot fill the page.

    Two passes: strict caps first, then relaxed caps (doubled) to fill leftovers.
    """
    ranked = sorted(stories, key=lambda story: (story["score"], story["cluster_size"]), reverse=True)
    selected: List[Dict[str, Any]] = []
    chosen_ids: set = set()
    source_counts: Counter = Counter()
    category_counts: Counter = Counter()

    def sweep(source_cap: int, category_cap: int, min_score: float) -> None:
        for story in ranked:
            if len(selected) >= limit:
                return
            marker = id(story)
            if marker in chosen_ids or story["score"] < min_score:
                continue
            if source_counts[story["source"]] >= source_cap:
                continue
            if category_counts[story["category"]] >= category_cap:
                continue
            selected.append(story)
            chosen_ids.add(marker)
            source_counts[story["source"]] += 1
            category_counts[story["category"]] += 1

    sweep(per_source_cap, per_category_cap, threshold)
    sweep(per_source_cap * 2, per_category_cap * 2, threshold)
    sweep(limit, limit, 0.0)
    selected.sort(key=lambda story: story["score"], reverse=True)
    return selected[:limit]

def count_by(stories: List[Dict[str, Any]], key: str) -> List[Dict[str, Any]]:
    counts = Counter(story.get(key) or "其他" for story in stories)
    return [{"name": name, "count": count} for name, count in counts.most_common()]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def build_archive(data_dir: Path) -> List[Dict[str, Any]]:
    archive: List[Dict[str, Any]] = []
    daily_dir = data_dir / "daily"
    if not daily_dir.exists():
        return archive
    for path in sorted(daily_dir.glob("*.json"), reverse=True):
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}\.json", path.name):
            continue
        try:
            meta = json.loads(path.read_text(encoding="utf-8")).get("meta", {})
        except json.JSONDecodeError:
            continue
        archive.append({
            "date": meta.get("date", path.stem),
            "candidate_count": meta.get("candidate_count", 0),
            "story_count": meta.get("story_count", 0),
            "brief_count": meta.get("brief_count", 0),
            "generated_at": meta.get("generated_at"),
            "data_url": f"data/daily/{path.name}",
            "post_url": f"posts/{path.stem}.html",
        })
    archive.sort(key=lambda entry: entry["date"], reverse=True)
    return archive

def render_card(story: Dict[str, Any]) -> str:
    tags = "".join(f'<span class="tag">{html.escape(tag)}</span>' for tag in story["tags"][:5])
    multi = ""
    if story["cluster_size"] > 1:
        variants = "".join(
            f'<li><span class="muted">{html.escape(variant["source"] or "")}</span> '
            f'<a href="{html.escape(variant["url"] or "#")}" target="_blank" rel="noreferrer">'
            f'{html.escape(variant["title"] or "")}</a></li>'
            for variant in story["variants"])
        multi = (f'<details class="multi"><summary>多源 {story["cluster_size"]}</summary>'
                 f'<ul>{variants}</ul></details>')
    return f"""
        <article class="card">
          <div class="card-head">
            <a class="card-title" href="{html.escape(story.get('url') or '#')}" target="_blank" rel="noreferrer">{html.escape(story['title'])}</a>
            <span class="score">{story['score']:.1f}</span>
          </div>
          <div class="meta">
            <span class="cat">{html.escape(story['category'])}</span>
            <span>{html.escape(story['source'])}</span>
            <span>{html.escape(story['topic'])}</span>
            <span>{html.escape((story.get('published_at') or '')[:16].replace('T', ' '))}</span>
          </div>
          <p class="summary">{html.escape(story.get('summary', '')[:260])}</p>
          <p class="why">为什么值得看：{html.escape(story['why'])}</p>
          <div class="tags">{tags}</div>
          {multi}
        </article>"""

PAGE_CSS = """
:root { color-scheme: light; --bg:#f6f7f9; --panel:#fff; --line:#e5e7eb; --ink:#111827;
  --muted:#6b7280; --accent:#2563eb; --accent-soft:#eff6ff; }
* { box-sizing: border-box; }
body { margin:0; background:var(--bg); color:var(--ink); font:15px/1.6 -apple-system,
  BlinkMacSystemFont,"Segoe UI","PingFang SC","Hiragino Sans GB","Microsoft YaHei",sans-serif; }
a { color:var(--accent); text-decoration:none; }
a:hover { text-decoration:underline; }
.wrap { max-width:1080px; margin:0 auto; padding:28px 20px 64px; }
.top { display:flex; flex-wrap:wrap; gap:16px; align-items:flex-end; justify-content:space-between;
  margin-bottom:24px; }
.kicker { font-size:12px; letter-spacing:.14em; text-transform:uppercase; color:var(--muted); margin:0 0 6px; }
h1 { font-size:24px; margin:0 0 8px; }
.lede { color:var(--muted); margin:0; }
.btn { display:inline-block; border:1px solid var(--line); background:var(--panel); color:var(--ink);
  border-radius:8px; padding:7px 12px; font-size:13px; }
.btn.primary { background:var(--accent); border-color:var(--accent); color:#fff; }
.grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(200px,1fr)); gap:12px; margin-bottom:24px; }
.stat { background:var(--panel); border:1px solid var(--line); border-radius:10px; padding:14px 16px; }
.stat .k { color:var(--muted); font-size:12px; }
.stat .v { font-size:22px; font-weight:600; margin-top:2px; }
.layout { display:grid; grid-template-columns:minmax(0,1fr) 288px; gap:20px; align-items:start; }
@media (max-width:880px) { .layout { grid-template-columns:minmax(0,1fr); } }
.panel { background:var(--panel); border:1px solid var(--line); border-radius:10px; padding:16px; }
.panel h2, .panel h3 { margin:0 0 10px; font-size:14px; }
.rail { display:grid; gap:16px; position:sticky; top:16px; }
.card { background:var(--panel); border:1px solid var(--line); border-radius:10px; padding:16px;
  margin-bottom:12px; }
.card-head { display:flex; gap:12px; align-items:flex-start; justify-content:space-between; }
.card-title { font-size:16px; font-weight:600; line-height:1.45; }
.score { flex:none; background:var(--accent-soft); color:var(--accent); border-radius:999px;
  padding:2px 10px; font-size:13px; font-weight:600; }
.meta { display:flex; flex-wrap:wrap; gap:10px; color:var(--muted); font-size:12px; margin:8px 0; }
.cat { background:#f3f4f6; border-radius:6px; padding:1px 8px; }
.summary { margin:8px 0; color:#374151; }
.why { margin:6px 0 10px; color:var(--muted); font-size:13px; }
.tags { display:flex; flex-wrap:wrap; gap:6px; }
.tag { background:#f3f4f6; color:#4b5563; border-radius:6px; padding:1px 8px; font-size:12px; }
.multi { margin-top:10px; font-size:13px; }
.multi summary { cursor:pointer; color:var(--accent); }
.multi ul { margin:8px 0 0; padding-left:18px; }
.muted { color:var(--muted); }
.rowlist { list-style:none; margin:0; padding:0; font-size:13px; }
.rowlist li { display:flex; justify-content:space-between; gap:10px; padding:3px 0; }
.day { margin:22px 0 10px; font-size:13px; color:var(--muted); font-weight:600; }
.bad { color:#b91c1c; }
"""

def render_post(payload: Dict[str, Any], archive: List[Dict[str, Any]]) -> str:
    meta = payload["meta"]
    brief_html = "".join(render_card(story) for story in payload["brief"]) or \
        '<div class="panel">今天没有条目通过筛选。</div>'
    hot_html = "".join(render_card(story) for story in payload["hot"][:5])
    hot_block = f'<h2 class="day">当前热点（多信源同时报道）</h2>{hot_html}' if hot_html else ""
    categories = "".join(
        f'<li><span>{html.escape(entry["name"])}</span><strong>{entry["count"]}</strong></li>'
        for entry in payload["categories"]) or '<li class="muted">暂无</li>'
    topics = "".join(
        f'<li><span>{html.escape(entry["name"])}</span><strong>{entry["count"]}</strong></li>'
        for entry in payload["topics"][:8]) or '<li class="muted">暂无</li>'
    health = "".join(
        f'<li><span class="{"" if entry["ok"] else "bad"}">{html.escape(entry["name"])}</span>'
        f'<strong>{entry["fetched"]}</strong></li>'
        for entry in payload["source_status"][:14]) or '<li class="muted">暂无</li>'
    archive_html = "".join(
        f'<li><a href="./{html.escape(entry["date"])}.html">{html.escape(entry["date"])}</a>'
        f'<span class="muted">{entry["brief_count"]}/{entry["story_count"]}</span></li>'
        for entry in archive[:12]) or '<li class="muted">暂无归档</li>'
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>前沿追踪 · {html.escape(meta['date'])}</title>
<style>{PAGE_CSS}</style>
</head>
<body>
<div class="wrap">
  <div class="top">
    <div>
      <p class="kicker">Frontier Radar</p>
      <h1>{html.escape(meta['date'])} 每日精选</h1>
      <p class="lede">候选 {meta['candidate_count']} 条，合并为 {meta['story_count']} 个事件，精选 {meta['brief_count']} 条。</p>
    </div>
    <div>
      <a class="btn primary" href="../index.html">打开追踪面板</a>
      <a class="btn" href="../../index.html">返回主页</a>
    </div>
  </div>
  <div class="grid">
    <div class="stat"><div class="k">原始候选</div><div class="v">{meta['candidate_count']}</div></div>
    <div class="stat"><div class="k">合并事件</div><div class="v">{meta['story_count']}</div></div>
    <div class="stat"><div class="k">精选</div><div class="v">{meta['brief_count']}</div></div>
    <div class="stat"><div class="k">强相关</div><div class="v">{meta['strong_count']}</div></div>
  </div>
  <div class="layout">
    <main>
      {hot_block}
      <h2 class="day">每日精选</h2>
      {brief_html}
    </main>
    <aside class="rail">
      <div class="panel"><h3>栏目分布</h3><ul class="rowlist">{categories}</ul></div>
      <div class="panel"><h3>主题分布</h3><ul class="rowlist">{topics}</ul></div>
      <div class="panel"><h3>信源抓取</h3><ul class="rowlist">{health}</ul></div>
      <div class="panel"><h3>归档</h3><ul class="rowlist">{archive_html}</ul></div>
    </aside>
  </div>
</div>
</body>
</html>
"""

def build_payload(args: argparse.Namespace) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    config = load_sources()
    raw_items, status = collect_all(config)
    windowed = [item for item in raw_items if in_window(item)]
    relevant = [item for item in windowed if item["ai_score"] >= args.min_relevance]

    stories_raw, merge_log = merge_stories(relevant)
    stories = [finalize_story(story) for story in stories_raw]
    stories.sort(key=lambda story: (story["published_at"] or "", story["score"]), reverse=True)

    strong = [story for story in stories if story["ai_score"] >= 0.55]
    brief = pick_brief(stories, limit=args.limit, threshold=args.threshold,
                       per_source_cap=args.per_source_cap,
                       per_category_cap=args.per_category_cap)
    hot = sorted([story for story in stories if story["cluster_size"] > 1],
                 key=lambda story: (story["cluster_size"], story["score"]), reverse=True)

    payload = {
        "meta": {
            "date": TODAY,
            "generated_at": to_iso(NOW),
            "window_hours": WINDOW_HOURS,
            "candidate_count": len(raw_items),
            "in_window_count": len(windowed),
            "relevant_count": len(relevant),
            "story_count": len(stories),
            "strong_count": len(strong),
            "brief_count": len(brief),
            "threshold": args.threshold,
            "limit": args.limit,
            "min_relevance": args.min_relevance,
        },
        "brief": brief,
        "hot": hot,
        "strong": strong,
        "all": stories,
        "categories": count_by(stories, "category"),
        "topics": count_by(stories, "topic"),
        "sources": count_by(stories, "source"),
        "source_status": status,
    }
    return payload, merge_log

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the frontier news radar")
    parser.add_argument("--output-dir", default=str(DEFAULT_BLOG_DIR))
    parser.add_argument("--limit", type=int, default=20, help="max curated items")
    parser.add_argument("--threshold", type=float, default=58.0, help="curation score threshold")
    parser.add_argument("--min-relevance", type=float, default=0.3,
                        help="minimum AI relevance kept in the broad pool")
    parser.add_argument("--per-source-cap", type=int, default=3,
                        help="max curated items from a single source")
    parser.add_argument("--per-category-cap", type=int, default=5,
                        help="max curated items from a single category")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    blog_dir = Path(args.output_dir).resolve()
    data_dir = blog_dir / "data"
    posts_dir = blog_dir / "posts"
    data_dir.mkdir(parents=True, exist_ok=True)
    posts_dir.mkdir(parents=True, exist_ok=True)

    payload, merge_log = build_payload(args)

    write_json(data_dir / "daily" / f"{TODAY}.json", payload)
    write_json(data_dir / "daily-brief.json", {"meta": payload["meta"], "items": payload["brief"]})
    write_json(data_dir / "latest-24h.json", {"meta": payload["meta"], "items": payload["strong"]})
    write_json(data_dir / "latest-24h-all.json", {"meta": payload["meta"], "items": payload["all"]})
    write_json(data_dir / "stories-merged.json", {"meta": payload["meta"], "items": payload["all"]})
    write_json(data_dir / "source-status.json",
               {"generated_at": payload["meta"]["generated_at"], "sources": payload["source_status"]})
    write_json(data_dir / "merge-log.json",
               {"generated_at": payload["meta"]["generated_at"], "merges": merge_log})

    archive = build_archive(data_dir)
    write_json(data_dir / "index.json", {
        "generated_at": payload["meta"]["generated_at"],
        "latest": archive[0] if archive else {},
        "categories": CATEGORIES,
        "archive": archive,
    })
    write_text(posts_dir / f"{TODAY}.html", render_post(payload, archive))

    print(json.dumps({"date": TODAY, "candidates": payload["meta"]["candidate_count"],
                      "stories": payload["meta"]["story_count"],
                      "brief": payload["meta"]["brief_count"],
                      "sources_ok": sum(1 for entry in payload["source_status"] if entry["ok"]),
                      "sources_failed": sum(1 for entry in payload["source_status"] if not entry["ok"])},
                     ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
