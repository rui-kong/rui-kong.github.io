#!/usr/bin/env python3
"""Mine historical daily-brief snapshots out of the upstream repo's git history.

The upstream radar (LearnPrompt/ai-news-radar) overwrites data/daily-brief.json
on every run, so past days only survive inside git history. This script walks
every commit that touched that file and dumps each version to disk.

Requires a partial clone (blobs are fetched lazily, so a proxy may be needed):

    git clone --filter=blob:none --no-checkout \
        https://github.com/LearnPrompt/ai-news-radar.git /tmp/radar-pc

Usage:
    https_proxy=http://127.0.0.1:7897 http_proxy=http://127.0.0.1:7897 \
        python3 tools/mine_upstream_briefs.py --repo /tmp/radar-pc --out /tmp/mined-briefs
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple

TARGET = "data/daily-brief.json"


def git(repo: Path, *args: str, binary: bool = False):
    result = subprocess.run(["git", "-C", str(repo), *args],
                            capture_output=True, check=True)
    return result.stdout if binary else result.stdout.decode("utf-8", "replace")


def commit_list(repo: Path) -> List[Tuple[str, str]]:
    out = git(repo, "log", "--format=%H %cI", "--", TARGET)
    rows = []
    for line in out.splitlines():
        parts = line.split()
        if len(parts) == 2:
            rows.append((parts[0], parts[1]))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=0, help="only mine the newest N commits")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    commits = commit_list(args.repo)
    if args.limit:
        commits = commits[: args.limit]
    print(f"commits touching {TARGET}: {len(commits)}", flush=True)

    seen_blobs: dict = {}
    ok = failed = cached = 0
    for index, (sha, when) in enumerate(commits, start=1):
        target = args.out / f"{when.replace(':', '').replace('+0000', 'Z')}_{sha[:8]}.json"
        if target.exists():
            cached += 1
            continue
        try:
            blob = git(args.repo, "rev-parse", f"{sha}:{TARGET}").strip()
        except subprocess.CalledProcessError:
            failed += 1
            continue
        if blob in seen_blobs:
            target.write_bytes(Path(seen_blobs[blob]).read_bytes())
            ok += 1
            continue
        try:
            raw = git(args.repo, "cat-file", "-p", blob, binary=True)
            json.loads(raw.decode("utf-8", "replace"))  # validate
        except (subprocess.CalledProcessError, json.JSONDecodeError) as exc:
            print(f"[warn] {sha[:8]} {when}: {type(exc).__name__}", file=sys.stderr, flush=True)
            failed += 1
            continue
        target.write_bytes(raw)
        seen_blobs[blob] = str(target)
        ok += 1
        if index % 25 == 0:
            print(f"  {index}/{len(commits)} ok={ok} cached={cached} failed={failed}", flush=True)

    print(json.dumps({"commits": len(commits), "written": ok, "cached": cached,
                      "failed": failed, "out": str(args.out)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
