#!/usr/bin/env python3
"""
Sample videos from sampled_videos_900.csv and classify each into one of 18
hashtag categories with Gemini (raw video via OpenRouter).

Usage:
  python -m annotation.annotate_gemini [--input_csv sampled_videos_900.csv] [--video_dir tiktok_videos]
      [--sample_size 100] [--seed 42] [--output_csv annotated_gemini_100.csv] [--workers 4]
      [--resume_log logs/gemini_results_<ts>.jsonl]
"""

import argparse
import base64
import csv
import json
import logging
import os
import random
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

from openai import OpenAI

from annotation.prompts import GEMINI_PROMPT_TEMPLATE, HASHTAG_OPTIONS

MODEL = "google/gemini-2.5-flash"
LOG_DIR = Path("logs")
JSONL_LOG = LOG_DIR / f"gemini_results_{datetime.now():%Y%m%d_%H%M%S}.jsonl"
FIELDNAMES = ["relative_path", "file_name", "true_hashtag", "predicted_hashtag", "confidence", "rationale"]

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def load_samples(csv_path: Path) -> list[dict]:
    with csv_path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def sample_videos(samples: list[dict], n: int, seed: int) -> list[dict]:
    random.seed(seed)
    return random.sample(samples, min(n, len(samples)))


def log_result(record: dict):
    with open(JSONL_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_cache(jsonl_path: Path | None) -> dict:
    """Load previously completed results to allow resuming."""
    cache = {}
    if not jsonl_path or not jsonl_path.exists():
        return cache
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
                if obj.get("status") == "success":
                    cache[obj["relative_path"]] = obj
            except Exception:
                pass
    return cache


def annotate_video(client: OpenAI, video_path: str, transcript: str) -> dict:
    """Send video as base64 to Gemini via OpenRouter and get hashtag classification."""
    options_str = "\n".join(f"- {h}" for h in HASHTAG_OPTIONS)
    prompt = GEMINI_PROMPT_TEMPLATE.format(options=options_str)
    video_b64 = base64.b64encode(Path(video_path).read_bytes()).decode()

    # User content parts: video + optional transcript + prompt
    user_content = [
        {"type": "image_url", "image_url": {"url": f"data:video/mp4;base64,{video_b64}"}},
    ]
    if transcript and transcript.strip():
        user_content.append({"type": "text", "text": f"Transcript/speech from the video:\n{transcript.strip()}"})
    user_content.append({"type": "text", "text": prompt})

    response = client.chat.completions.create(
        model=MODEL,
        max_tokens=300,
        temperature=0,
        messages=[{"role": "user", "content": user_content}],
    )

    text = response.choices[0].message.content.strip()
    if text.startswith("```"):   # markdown code block
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    return json.loads(text)


def _row(sample: dict, predicted="", confidence="", rationale="") -> dict:
    return {"relative_path": sample["relative_path"], "file_name": sample["file_name"],
            "true_hashtag": sample["hashtag"], "predicted_hashtag": predicted,
            "confidence": confidence, "rationale": rationale}


def process_sample(client: OpenAI, sample: dict, video_dir: Path, cache: dict, position: str) -> dict:
    rel_path = sample["relative_path"]
    video_path = str(video_dir / rel_path)
    transcript = sample.get("voice_to_text", "") or ""

    if rel_path in cache:
        cached = cache[rel_path]
        log.info(f"[{position}] {rel_path} (cached)")
        return _row(sample, cached.get("predicted_hashtag", ""), cached.get("confidence", ""),
                    cached.get("rationale", ""))

    log.info(f"[{position}] {rel_path}")
    if not os.path.exists(video_path):
        log.warning(f"  File not found: {video_path}")
        log_result({"relative_path": rel_path, "status": "missing"})
        return _row(sample, rationale="file not found")

    try:
        result = annotate_video(client, video_path, transcript)
        predicted = result.get("hashtag", "")
        confidence = result.get("confidence", "")
        rationale = result.get("rationale", "")
        log_result({
            "relative_path": rel_path,
            "status": "success",
            "predicted_hashtag": predicted,
            "confidence": confidence,
            "rationale": rationale,
            "true_hashtag": sample["hashtag"],
        })
        match = "OK" if predicted == sample["hashtag"] else "MISS"
        log.info(f"  -> {predicted} (conf={confidence}) [{match}]")
        return _row(sample, predicted, confidence, rationale)

    except json.JSONDecodeError as e:
        log.warning(f"  JSON parse error: {e}")
        log_result({"relative_path": rel_path, "status": "json_error", "error": str(e)})
        return _row(sample, rationale=f"json_error: {e}")

    except Exception as e:
        log.warning(f"  Error: {e}")
        log_result({"relative_path": rel_path, "status": "error", "error": str(e)})
        if "429" in str(e) or "rate" in str(e).lower():   # back off on rate limits
            log.info("  Rate limited — sleeping 30s")
            time.sleep(30)
        return _row(sample, rationale=f"error: {e}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input_csv", type=Path, default=Path("sampled_videos_900.csv"))
    ap.add_argument("--video_dir", type=Path, default=Path("tiktok_videos"))
    ap.add_argument("--output_csv", type=Path, default=Path("annotated_gemini_100.csv"))
    ap.add_argument("--sample_size", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--resume_log", type=Path, default=None,
                    help="JSONL log of an earlier run to resume from")
    args = ap.parse_args()
    LOG_DIR.mkdir(exist_ok=True)

    samples = load_samples(args.input_csv)
    log.info(f"Loaded {len(samples)} videos from {args.input_csv}")
    selected = sample_videos(samples, args.sample_size, args.seed)
    log.info(f"Sampled {len(selected)} videos (seed={args.seed})")
    dist = Counter(s["hashtag"] for s in selected)
    for ht in sorted(dist):
        log.info(f"  {ht:20s}: {dist[ht]}")

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("Set OPENROUTER_API_KEY environment variable")
    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)

    cache = load_cache(args.resume_log)
    if cache:
        log.info(f"Resuming: {len(cache)} videos already done")

    total = len(selected)
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        rows = list(pool.map(lambda iv: process_sample(client, iv[1], args.video_dir, cache, f"{iv[0]}/{total}"),
                             enumerate(selected, 1)))

    with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    success = [r for r in rows if r["predicted_hashtag"]]
    correct = sum(1 for r in success if r["predicted_hashtag"] == r["true_hashtag"])
    log.info(f"\nDone. {len(rows)} videos -> {args.output_csv}")
    log.info(f"  Annotated: {len(success)}/{len(rows)}")
    if success:
        log.info(f"  Accuracy:  {correct}/{len(success)} = {correct/len(success):.1%}")


if __name__ == "__main__":
    main()
