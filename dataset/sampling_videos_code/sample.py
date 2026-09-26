#!/usr/bin/env python3
"""
Sample TikTok videos per hashtag and month from the TikTok Research API.

Keeps only videos with a non-empty voice_to_text transcript, until each
(hashtag, month) has --per_period of them. Writes one JSON file per pair,
<out_dir>/<hashtag>_<month>.json, plus <out_dir>/_sampling_summary.json.

Usage:
  export TIKTOK_TOKEN=...
  python sample.py --out_dir tiktok_samples
"""

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

API_ENDPOINT = "https://open.tiktokapis.com/v2/research/video/query/"
# voice_to_text is only returned if requested explicitly
FIELDS = "id,video_description,create_time,hashtag_names,region_code,voice_to_text"

TIME_PERIODS = [
    {"month": "2025.10", "start_date": "20251001", "end_date": "20251031"},
    {"month": "2025.11", "start_date": "20251101", "end_date": "20251130"},
    {"month": "2025.12", "start_date": "20251201", "end_date": "20251231"},
    {"month": "2026.1", "start_date": "20260101", "end_date": "20260131"},
]

HASHTAGS = [
    "dance", "gym", "anime", "gaming", "strangerthings", "food",
    "funnyvideos", "booktok", "makeup", "homedecor", "craft", "dogs",
    "basketball", "tech", "travel", "wealth", "skincare", "music",
]


def query_videos(token, hashtag, start_date, end_date, max_count, cursor=0, search_id=None, debug=False):
    """One Research API query (US videos with this hashtag). Returns the parsed response or None."""
    body = {
        "query": {
            "and": [
                {"operation": "EQ", "field_name": "hashtag_name", "field_values": [hashtag]},
                {"operation": "IN", "field_name": "region_code", "field_values": ["US"]},
            ]
        },
        "max_count": max_count,
        "start_date": start_date,
        "end_date": end_date,
        "cursor": cursor if search_id else 0,
    }
    if search_id:
        body["search_id"] = search_id

    req = urllib.request.Request(
        f"{API_ENDPOINT}?fields={FIELDS}",
        data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            response = json.load(r)
    except urllib.error.HTTPError as e:
        # Error responses carry a JSON body with the API's error message
        try:
            response = json.loads(e.read())
        except json.JSONDecodeError:
            print(f"    API request failed: HTTP {e.code}")
            return None
    except (urllib.error.URLError, json.JSONDecodeError) as e:
        print(f"    API request failed: {e}")
        return None

    if debug:
        videos = (response.get("data") or {}).get("videos") or []
        if videos:
            print(f"      [debug] fields in first video: {list(videos[0].keys())}")
            print(f"      [debug] voice_to_text: {str(videos[0].get('voice_to_text'))[:100]!r}")
    return response


def has_transcript(video) -> bool:
    vtt = video.get("voice_to_text")
    return isinstance(vtt, str) and bool(vtt.strip())


def sample_period(token, hashtag, start_date, end_date, target, max_requests, debug=False):
    """Page through results until `target` videos with transcripts are found (or results run out)."""
    kept, fetched, requests_made = [], 0, 0
    cursor, search_id = 0, None

    while len(kept) < target and requests_made < max_requests:
        # Ask for ~5x what's still needed, since only ~20% of videos have a transcript
        max_count = min(100, max(20, (target - len(kept)) * 5))
        response = query_videos(token, hashtag, start_date, end_date, max_count, cursor, search_id, debug)
        requests_made += 1

        if not response:
            break
        data = response.get("data") or {}
        if "videos" not in data:
            print(f"    No data in response: {response.get('error')}")
            break

        videos = data["videos"]
        fetched += len(videos)
        with_text = [v for v in videos if has_transcript(v)]
        kept.extend(with_text)
        search_id = search_id or data.get("search_id")
        print(f"    request {requests_made}: {len(with_text)}/{len(videos)} with transcript, "
              f"total {len(kept)}/{target}")

        if not data.get("has_more", False):
            print("    no more results")
            break
        cursor = data.get("cursor", 0)
        time.sleep(1)

    rate = len(kept) / fetched * 100 if fetched else 0
    print(f"    kept {min(len(kept), target)}/{target} "
          f"(fetched {fetched}, {rate:.1f}% with transcript, {requests_made} requests)")
    return kept[:target]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out_dir", type=Path, default=Path("tiktok_samples"))
    p.add_argument("--per_period", type=int, default=30, help="Videos with transcript per hashtag and month")
    p.add_argument("--max_requests", type=int, default=100, help="Request cap per hashtag and month")
    p.add_argument("--debug", action="store_true", help="Print the fields returned by the API")
    args = p.parse_args()

    token = os.environ.get("TIKTOK_TOKEN")
    if not token:
        sys.exit("Set the TIKTOK_TOKEN environment variable")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"{len(HASHTAGS)} hashtags x {len(TIME_PERIODS)} months x {args.per_period} videos "
          f"= {len(HASHTAGS) * len(TIME_PERIODS) * args.per_period} expected")

    total, incomplete = 0, []
    for i, hashtag in enumerate(HASHTAGS, 1):
        print(f"\n[{i}/{len(HASHTAGS)}] #{hashtag}")
        for period in TIME_PERIODS:
            month = period["month"]
            print(f"  {month}")
            videos = sample_period(token, hashtag, period["start_date"], period["end_date"],
                                   args.per_period, args.max_requests, args.debug)

            out_path = args.out_dir / f"{hashtag}_{month}.json"
            with out_path.open("w", encoding="utf-8") as f:
                json.dump({
                    "hashtag": hashtag,
                    "period": month,
                    "start_date": period["start_date"],
                    "end_date": period["end_date"],
                    "requested_count": args.per_period,
                    "actual_count": len(videos),
                    "subtitles_only": True,
                    "videos": videos,
                }, f, indent=2, ensure_ascii=False)

            total += len(videos)
            if len(videos) < args.per_period:
                incomplete.append({"hashtag": hashtag, "period": month,
                                   "expected": args.per_period, "actual": len(videos)})
            time.sleep(2)

    print(f"\nSampled {total} videos -> {args.out_dir}")
    for s in incomplete:
        print(f"  incomplete: {s['hashtag']} {s['period']}: {s['actual']}/{s['expected']}")

    summary_path = args.out_dir / "_sampling_summary.json"
    summary_path.write_text(json.dumps({
        "timestamp": datetime.now().isoformat(),
        "total_sampled": total,
        "hashtags": HASHTAGS,
        "time_periods": TIME_PERIODS,
        "videos_per_period": args.per_period,
        "subtitles_only_filter": True,
        "max_requests_per_period": args.max_requests,
        "failed_samples": incomplete,
    }, indent=2))
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
