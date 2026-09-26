"""
Video annotator.

Usage:
  python -m annotation.annotate <task> <folder> [options]

  # TikTok hashtag classification
  python -m annotation.annotate tiktok_vision  videos/ --samples_csv sampled.csv
  python -m annotation.annotate tiktok_text    videos/ --samples_csv sampled.csv

  # MELD emotion/sentiment (Friends clips)
  python -m annotation.annotate meld_3x3       videos/ --meta_csv meld.csv
  python -m annotation.annotate meld_3x3_text  videos/ --meta_csv meld.csv
  python -m annotation.annotate meld_4x4       videos/ --meta_csv meld.csv
  python -m annotation.annotate meld_text      videos/ --meta_csv meld.csv

  # Use a different model via OpenRouter
  python -m annotation.annotate meld_3x3 videos/ --meta_csv meld.csv --provider openrouter

  # Concurrent requests, or the OpenAI Batch API
  python -m annotation.annotate tiktok_2x8 videos/ --samples_csv sampled.csv --workers 8
  python -m annotation.annotate tiktok_2x8 videos/ --samples_csv sampled.csv --batch_api
"""

import argparse
import base64
import csv
import json
import logging
import multiprocessing
import os
import random
import re
import subprocess
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from functools import partial
from io import BytesIO
from pathlib import Path
from typing import Any, Callable, Iterable

import pandas as pd
from openai import OpenAI
from PIL import Image

from annotation.prompts import (
    MELD_TEXT_PROMPT,
    MELD_VISION_PROMPT_3X3,
    MELD_VISION_PROMPT_3X3_TEXT,
    MELD_VISION_PROMPT_4X4,
    MELD_VISION_PROMPT_4X4_IMAGE,
    TRANSCRIPT_PROMPT,
    VISION_PROMPT,
    VISION_PROMPT_2X8,
    VISION_PROMPT_4X4,
    VISION_PROMPT_IMAGE,
)

VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv", ".hevc", ".ts")
DEFAULT_TIMEOUT = 120

PROVIDERS = {
    "openai": {
        "base_url": None,
        "api_key_env": "OPENAI_API_KEY",
        "default_model": "gpt-4o-mini",
        "json_mode": True,
    },
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "api_key_env": "OPENROUTER_API_KEY",
        "default_model": "anthropic/claude-sonnet-4.6",
        "json_mode": False,  # Claude ignores json_object; parse fences instead
    },
}

LOG_DIR = Path("logs")
RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)
log = logging.getLogger(__name__)


@dataclass
class Task:
    name: str
    system_prompt: str
    requires_grid: bool                                    # if False, transcript-only
    extra_args: Callable[[argparse.ArgumentParser], None]
    iter_videos: Callable[[argparse.Namespace], Iterable[tuple[str, dict]]]
    user_content: Callable[[dict, str | None], Any]        # (meta, grid_path) -> str | list
    build_record: Callable[[dict, dict, float], dict]      # (meta, llm, duration) -> record
    postprocess: Callable[[dict, dict], dict] | None = None
    rows: int = 3
    cols: int = 3


MAX_IMAGE_BYTES = 5 * 1024 * 1024   # Anthropic hard limit: 5 MB per image
MAX_IMAGE_EDGE = 1568               # Anthropic recommended max long edge


def _image_data_url(path: str) -> str:
    """Return a base64 data URL, downscaling if the image exceeds provider size/dimension limits."""
    raw = Path(path).read_bytes()
    ext = Path(path).suffix.lower()
    mime = "image/jpeg" if ext in (".jpg", ".jpeg") else "image/png"
    img = Image.open(BytesIO(raw))
    byte_target = MAX_IMAGE_BYTES - 256 * 1024

    if len(raw) <= byte_target and max(img.size) <= MAX_IMAGE_EDGE:
        return f"data:{mime};base64,{base64.b64encode(raw).decode()}"

    img = img.convert("RGB")
    if max(img.size) > MAX_IMAGE_EDGE:
        s = MAX_IMAGE_EDGE / max(img.size)
        img = img.resize((max(1, int(img.width * s)), max(1, int(img.height * s))), Image.Resampling.LANCZOS)
    quality, data = 90, raw
    for _ in range(12):
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=quality)
        data = buf.getvalue()
        if len(data) <= byte_target:
            break
        if quality > 55:
            quality -= 15
        else:
            img = img.resize((max(1, int(img.width * 0.8)), max(1, int(img.height * 0.8))), Image.Resampling.LANCZOS)
    log.info(f"  resized grid {len(raw)//1024}KB -> {len(data)//1024}KB, dims now {img.size}")
    return f"data:image/jpeg;base64,{base64.b64encode(data).decode()}"


def _parse_json(text: str) -> dict:
    """Parse a model reply as JSON, tolerating ```json fences and stray prose."""
    text = (text or "").strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Fall back to the first {...} block (Claude sometimes adds a preamble).
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            return json.loads(m.group(0))
        raise


_RETRY_AFTER_RE = re.compile(r"try again in ([\d.]+)\s*(ms|s)\b", re.IGNORECASE)


def _retry_delay(msg: str, attempt: int) -> float:
    is_rate = "429" in msg or "rate limit" in msg.lower() or "rate_limit" in msg.lower()
    if not is_rate:
        return min(2.0 * (2 ** attempt), 30.0)
    m = _RETRY_AFTER_RE.search(msg)
    hinted = 0.5
    if m:
        value, unit = float(m.group(1)), m.group(2).lower()
        hinted = (value / 1000.0 if unit == "ms" else value) or 0.5
    return min(hinted * (2 ** attempt), 30.0)


def llm_request(model: str, system_prompt: str, user_content: Any, json_mode: bool = True,
                max_tokens: int = 300, temperature: float = 0.0,
                reasoning_effort: str | None = None, n: int = 1) -> dict:
    kwargs = dict(
        model=model, max_tokens=max_tokens, temperature=temperature, n=n,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
    )
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    if reasoning_effort:
        kwargs["extra_body"] = {"reasoning": {"effort": reasoning_effort}}
    return kwargs


def parse_choices(contents: list[str | None], finish_reason: str | None) -> list[dict]:
    results = []
    for content in contents:
        if not content:
            continue
        try:
            results.append(_parse_json(content))
        except json.JSONDecodeError:
            continue
    if not results:
        raise RuntimeError(f"no choice produced parseable content (finish_reason={finish_reason})")
    return results


def call_llm(client: OpenAI, model: str, system_prompt: str, user_content: Any,
             json_mode: bool = True, retries: int = 10, max_tokens: int = 300,
             temperature: float = 0.0, reasoning_effort: str | None = None,
             request_timeout: float = 90.0, n: int = 1) -> list[dict]:
    """Returns a list of `n` independently-sampled parsed JSON results (length 1 by default)."""
    kwargs = llm_request(model, system_prompt, user_content, json_mode=json_mode,
                         max_tokens=max_tokens, temperature=temperature,
                         reasoning_effort=reasoning_effort, n=n)
    kwargs["timeout"] = request_timeout

    last_err = None
    for attempt in range(retries):
        try:
            r = client.chat.completions.create(**kwargs)
            if not getattr(r, "choices", None):
                # OpenRouter may return HTTP 200 with no choices; surface the raw error payload.
                try:
                    err = r.model_dump()
                except Exception:
                    err = getattr(r, "error", None) or getattr(r, "model_extra", None)
                raise RuntimeError(f"response had no choices: {json.dumps(err, default=str)[:1000]}")
            return parse_choices([c.message.content for c in r.choices],
                                 getattr(r.choices[0], "finish_reason", None))
        except Exception as e:
            last_err = e
            log.warning(f"  LLM attempt {attempt + 1}/{retries} failed: {e}")
            if attempt < retries - 1:
                delay = _retry_delay(str(e), attempt)
                log.info(f"  backing off {delay:.2f}s before retry")
                time.sleep(delay)
    raise RuntimeError(f"LLM call failed after {retries} attempts: {last_err}")


def _grid_worker(video_path: str, grid_path: str, q, num_frames: int,
                 rows: int, cols: int, frame_method: str) -> None:
    try:
        from annotation.frame_sampling import create_comic_grid, sample_frames_from_video, silence_native_stderr
        silence_native_stderr()
        frames = sample_frames_from_video(video_path, num_frames=num_frames, method=frame_method)
        if not frames:
            q.put(("error", "no frames"))
            return
        create_comic_grid(frames, rows=rows, cols=cols, spacing=10).save(grid_path, quality=95)
        q.put(("ok", grid_path))
    except Exception as e:
        q.put(("error", str(e)))


def extract_grid(video_path: str, grid_path: str, num_frames: int, rows: int, cols: int,
                 frame_method: str, timeout: int) -> tuple[bool, str]:
    # Separate process so a hung decoder can be killed.
    ctx = multiprocessing.get_context("spawn")
    q = ctx.Queue()
    p = ctx.Process(target=_grid_worker,
                    args=(video_path, grid_path, q, num_frames, rows, cols, frame_method),
                    daemon=True)
    p.start()
    p.join(timeout=timeout)
    if p.is_alive():
        p.kill(); p.join()
        return False, f"timed out after {timeout}s"
    if q.empty():
        return False, "worker produced no result"
    status, payload = q.get_nowait()
    return (status == "ok"), ("" if status == "ok" else payload)


def ffprobe_duration(video_path: str) -> float:
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "csv=p=0", video_path],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=10,
        )
        if r.returncode == 0:
            return float(r.stdout.decode().strip())
    except Exception:
        pass
    return 0.0


def append_jsonl(path: Path, record: dict) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_success_videos(path: str | None) -> set[str]:
    done: set[str] = set()
    if not path or not os.path.exists(path):
        return done
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
                if obj.get("status") == "success" and obj.get("video"):
                    done.add(obj["video"])
            except Exception:
                continue
    return done


def _tiktok_extra_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--samples_csv", default="sampled_videos.csv",
                   help="CSV with relative_path, file_name, voice_to_text, hashtag")


def _tiktok_iter(args) -> Iterable[tuple[str, dict]]:
    df = pd.read_csv(args.samples_csv)
    options = sorted({(h or "").strip() for h in df["hashtag"].dropna() if (h or "").strip()})
    if not options:
        raise ValueError(f"No hashtags found in {args.samples_csv}")
    for _, r in df.iterrows():
        rel = r["relative_path"]
        yield rel, {
            "filename": rel,
            "transcript": (r.get("voice_to_text") or "") if pd.notna(r.get("voice_to_text")) else "",
            "options": options,
            "file_name": r.get("file_name", ""),
        }


def _options_block(meta: dict) -> str:
    return "\n".join(f"- {o}" for o in meta["options"])


def _image_part(grid_path: str) -> dict:
    return {"type": "image_url", "image_url": {"url": _image_data_url(grid_path), "detail": "high"}}


def _tiktok_vision_user(meta: dict, grid_path: str | None) -> list:
    text = f"Hashtag options:\n{_options_block(meta)}\n\nTranscript:\n{meta['transcript'].strip() or '[EMPTY]'}"
    return [{"type": "text", "text": text}, _image_part(grid_path)]


def _tiktok_image_user(meta: dict, grid_path: str | None) -> list:
    return [{"type": "text", "text": f"Hashtag options:\n{_options_block(meta)}"}, _image_part(grid_path)]


def _tiktok_text_user(meta: dict, _grid_path: str | None) -> str:
    return f"Transcript:\n{meta['transcript'].strip() or '[EMPTY]'}\n\nAllowed hashtags:\n{_options_block(meta)}"


def _tiktok_vision_record(meta: dict, llm: dict, duration: float) -> dict:
    return {"video": meta["filename"], "total_duration": duration,
            "segments": [{"segment_index": 0, "start_time": 0.0, "end_time": duration,
                          "transcript": meta["transcript"], "annotations": llm}]}


def _tiktok_text_record(meta: dict, llm: dict, _duration: float) -> dict:
    return {"relative_path": meta["filename"], "file_name": meta["file_name"],
            "predicted_hashtag": llm.get("hashtag", ""),
            "confidence": llm.get("confidence", ""),
            "rationale": llm.get("rationale", "")}


def _meld_extra_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--meta_csv", required=True,
                   help="MELD CSV with Dialogue_ID, Utterance_ID, Speaker, Utterance, Emotion")
    p.add_argument("-l", "--list", default=None,
                   help="CSV listing videos to process (column: video_path)")
    p.add_argument("-n", "--num", type=int, default=100,
                   help="Random-sample size (ignored if --list given)")
    p.add_argument("-o", "--output", default="sampled_videos.csv",
                   help="Where to write the chosen sample list")
    p.add_argument("--seed", type=int, default=42)


def _meld_iter(args) -> Iterable[tuple[str, dict]]:
    meta_df = pd.read_csv(args.meta_csv)
    meta = {}
    for key in zip(meta_df["Dialogue_ID"], meta_df["Utterance_ID"], meta_df["Speaker"], meta_df["Utterance"]):
        meta.setdefault(key[:2], key[2:])

    if args.list:
        videos = pd.read_csv(args.list)["video_path"].tolist()
        log.info(f"Using provided list: {len(videos)} videos")
    else:
        all_videos = [f for f in os.listdir(args.folder) if f.lower().endswith(VIDEO_EXTS)]
        random.seed(args.seed)
        videos = random.sample(all_videos, min(args.num, len(all_videos)))
        pd.DataFrame(videos, columns=["video_path"]).to_csv(args.output, index=False)
        log.info(f"Randomly sampled {len(videos)} videos -> {args.output}")

    for filename in videos:
        m = re.match(r"dia(\d+)_utt(\d+)\.mp4", filename)
        if not m:
            log.warning(f"Skipping (filename format): {filename}")
            continue
        did, uid = int(m.group(1)), int(m.group(2))
        if (did, uid) not in meta:
            log.warning(f"Skipping (no metadata): {filename}")
            continue
        speaker, transcript = meta[did, uid]
        yield filename, {
            "filename": filename,
            "dialogue_id": did,
            "utterance_id": uid,
            "speaker": speaker,
            "transcript": transcript,
        }


def _meld_user(meta: dict, grid_path: str | None, include_transcript: bool) -> list:
    parts: list = [{"type": "text", "text": f"Character: {meta['speaker']}"}]
    if include_transcript:
        parts.append({"type": "text", "text": f"Transcript: {meta['transcript']}"})
    parts.append(_image_part(grid_path))
    return parts


def _meld_text_user(meta: dict, _grid_path: str | None) -> str:
    return f"Character: {meta['speaker']}\nDialogue: {meta['transcript']}"


def _meld_record(meta: dict, llm: dict, _duration: float) -> dict:
    return {"dialogue_id": meta["dialogue_id"], "utterance_id": meta["utterance_id"],
            "video": meta["filename"], "speaker": meta["speaker"],
            "transcript": meta["transcript"], "annotations": llm}


def _meld_postprocess(meta: dict, llm: dict) -> dict:
    """Blank out emotion/sentiment if model marked the speaker not visible."""
    if isinstance(llm, dict):
        v = llm.get(meta["speaker"], {})
        if v.get("character_visible") is False:
            llm[meta["speaker"]]["emotion"] = {"label": "N/A", "confidence": None}
            llm[meta["speaker"]]["sentiment"] = {"label": "N/A", "confidence": None}
    return llm


def _tiktok_task(name, prompt, rows=3, cols=3, user=_tiktok_vision_user,
                 record=_tiktok_vision_record, grid=True) -> Task:
    return Task(name=name, system_prompt=prompt, requires_grid=grid, rows=rows, cols=cols,
                extra_args=_tiktok_extra_args, iter_videos=_tiktok_iter,
                user_content=user, build_record=record)


def _meld_task(name, prompt, rows=3, cols=3, transcript=False, postprocess=_meld_postprocess,
               user=None, grid=True) -> Task:
    return Task(name=name, system_prompt=prompt, requires_grid=grid, rows=rows, cols=cols,
                extra_args=_meld_extra_args, iter_videos=_meld_iter,
                user_content=user or partial(_meld_user, include_transcript=transcript),
                build_record=_meld_record, postprocess=postprocess)


TASKS: dict[str, Task] = {t.name: t for t in [
    _tiktok_task("tiktok_vision",    VISION_PROMPT),
    _tiktok_task("tiktok_4x4_image", VISION_PROMPT_IMAGE, 4, 4, user=_tiktok_image_user),
    _tiktok_task("tiktok_4x4",       VISION_PROMPT_4X4,   4, 4),
    _tiktok_task("tiktok_2x8",       VISION_PROMPT_2X8,   2, 8),
    _tiktok_task("tiktok_2x8_image", VISION_PROMPT_IMAGE, 2, 8, user=_tiktok_image_user),
    _tiktok_task("tiktok_text",      TRANSCRIPT_PROMPT, grid=False,
                 user=_tiktok_text_user, record=_tiktok_text_record),
    _meld_task("meld_3x3",       MELD_VISION_PROMPT_3X3),
    _meld_task("meld_text",      MELD_TEXT_PROMPT, grid=False, user=_meld_text_user, postprocess=None),
    _meld_task("meld_3x3_text",  MELD_VISION_PROMPT_3X3_TEXT, transcript=True),
    _meld_task("meld_4x4",       MELD_VISION_PROMPT_4X4, 4, 4, transcript=True, postprocess=None),
    _meld_task("meld_4x4_image", MELD_VISION_PROMPT_4X4_IMAGE, 4, 4),
]}


def prepare_video(task: Task, filename: str, args) -> tuple[float, str | None]:
    if not task.requires_grid:
        return 0.0, None
    video_path = os.path.join(args.folder, filename)
    duration = ffprobe_duration(video_path)
    if duration <= 0:
        raise RuntimeError("could not determine duration (corrupt or missing)")

    grid_path = str(Path(args.grid_dir) / f"{Path(filename).stem}.jpg")
    if args.reuse_grids and os.path.exists(grid_path):
        log.info(f"  reusing existing grid {grid_path}")
    else:
        ok, err = extract_grid(video_path, grid_path,
                               num_frames=task.rows * task.cols,
                               rows=task.rows, cols=task.cols,
                               frame_method=args.frame_method,
                               timeout=args.video_timeout)
        if not ok:
            raise RuntimeError(f"frame extraction failed: {err}")
    return duration, grid_path


def _request_args(args) -> dict:
    return dict(json_mode=args.json_mode, max_tokens=args.max_tokens, temperature=args.temperature,
                reasoning_effort=args.reasoning_effort, n=args.n_samples)


def records_from_samples(task: Task, meta: dict, samples: list[dict], duration: float) -> list[dict]:
    records = []
    for i, llm in enumerate(samples):
        if task.postprocess:
            llm = task.postprocess(meta, llm)
        rec = task.build_record(meta, llm, duration)
        rec["sample_index"] = i
        records.append(rec)
    return records


def process_video(client: OpenAI, task: Task, filename: str, meta: dict, args) -> list[dict]:
    duration, grid_path = prepare_video(task, filename, args)
    samples = call_llm(client, args.model, task.system_prompt, task.user_content(meta, grid_path),
                       **_request_args(args))
    return records_from_samples(task, meta, samples, duration)


def _annotate_one(client, task, filename, meta, args, position: str) -> tuple[dict, list[dict]]:
    log.info(f"  [{position}] {filename}")
    record = {"video": filename, "status": "unknown", "error": None, "result": None}
    try:
        result = process_video(client, task, filename, meta, args)
        record["status"] = "success"
        record["result"] = result
        return record, result
    except Exception as e:
        logging.exception(f"Error on {filename}")
        record["status"] = "exception"
        record["error"] = str(e)
        return record, []


def run_batch_api(client, task: Task, items: list, pending: list[int], args) -> dict[int, tuple[dict, list]]:
    """Returns {index: (log record, output records)}. Failed batch requests are retried synchronously."""
    from annotation import openai_batch

    state_path = Path(args.batch_state) if args.batch_state else LOG_DIR / f"{task.name}_{RUN_ID}_batch.json"
    if state_path.exists():
        state = json.loads(state_path.read_text())
        log.info(f"Resuming batch run from {state_path}: batches {state['batch_ids']}")
    else:
        prepared, prep_errors = {}, {}
        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
            futures = {i: pool.submit(prepare_video, task, items[i][0], args) for i in pending}
            for i, fut in futures.items():
                try:
                    prepared[i] = fut.result()
                except Exception as e:
                    logging.exception(f"Error on {items[i][0]}")
                    prep_errors[i] = str(e)

        writer = openai_batch.ChunkWriter(Path(args.batch_dir), f"{task.name}_{RUN_ID}_requests",
                                          int(args.batch_max_mb * 1024 * 1024))
        for i, (_duration, grid_path) in prepared.items():
            _filename, meta = items[i]
            body = llm_request(args.model, task.system_prompt, task.user_content(meta, grid_path),
                               **_request_args(args))
            body.update(body.pop("extra_body", {}))
            writer.add(openai_batch.request_line(str(i), body))
        paths = writer.close()

        state = {
            "batch_ids": openai_batch.submit(client, paths) if paths else [],
            "videos": {str(i): items[i][0] for i in pending},
            "prepared": {str(i): list(v) for i, v in prepared.items()},
            "prep_errors": {str(i): e for i, e in prep_errors.items()},
        }
        state_path.write_text(json.dumps(state, indent=1))
        log.info(f"Batch state: {state_path} (resume with --batch_state)")

    batches = openai_batch.wait(client, state["batch_ids"], args.poll_interval)
    responses = openai_batch.collect(client, batches)

    results = {}
    for key, filename in state["videos"].items():
        i = int(key)
        if items[i][0] != filename:
            raise RuntimeError(f"Input list changed since submission: #{i} is {items[i][0]}, "
                               f"state has {filename}")
        meta = items[i][1]
        record = {"video": filename, "status": "unknown", "error": None, "result": None}
        try:
            if key in state["prep_errors"]:
                raise RuntimeError(state["prep_errors"][key])
            duration, grid_path = state["prepared"][key]
            body, err = responses.get(key, (None, "no result returned by the batch"))
            try:
                if err:
                    raise RuntimeError(err)
                samples = parse_choices([c["message"]["content"] for c in body.get("choices") or []],
                                        (body.get("choices") or [{}])[0].get("finish_reason"))
            except Exception as e:
                log.warning(f"  {filename}: {e}; retrying synchronously")
                samples = call_llm(client, args.model, task.system_prompt,
                                   task.user_content(meta, grid_path), **_request_args(args))
            result = records_from_samples(task, meta, samples, duration)
            record["status"], record["result"] = "success", result
            results[i] = (record, result)
        except Exception as e:
            logging.exception(f"Error on {filename}")
            record["status"], record["error"] = "exception", str(e)
            results[i] = (record, [])
    return results


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("task", choices=list(TASKS.keys()), help="Which annotation task to run")
    p.add_argument("folder", help="Folder containing video files")
    p.add_argument("--provider", choices=list(PROVIDERS.keys()), default="openai",
                   help="LLM provider")
    p.add_argument("--model", default=None,
                   help="Model id (default: provider preset)")
    p.add_argument("--max_tokens", type=int, default=300,
                   help="Max completion tokens")
    p.add_argument("--temperature", type=float, default=0.0,
                   help="Sampling temperature")
    p.add_argument("--n_samples", type=int, default=1,
                   help="Samples per video (n= param)")
    p.add_argument("--reasoning_effort", choices=["none", "low", "medium", "high"], default=None,
                   help="OpenRouter reasoning effort")
    p.add_argument("--frame_method", choices=["uniform", "transitions", "adaptive"], default="transitions")
    p.add_argument("--video_timeout", type=int, default=DEFAULT_TIMEOUT,
                   help="Timeout (s) for frame extraction per video")
    p.add_argument("--grid_dir", default="comic_grids", help="Where to write grid images")
    p.add_argument("--reuse_grids", action="store_true",
                   help="Use existing grids in --grid_dir")
    p.add_argument("--workers", type=int, default=1,
                   help="Videos processed concurrently")
    p.add_argument("--batch_api", action="store_true",
                   help="Use the OpenAI Batch API (openai provider only)")
    p.add_argument("--batch_state", default=None,
                   help="State file of an interrupted --batch_api run to resume")
    p.add_argument("--poll_interval", type=float, default=60.0,
                   help="Seconds between batch status checks")
    p.add_argument("--batch_max_mb", type=float, default=190.0,
                   help="Max size per batch request file")
    p.add_argument("--combined_out", default="annotations/annotations.json",
                   help="Combined JSON output path")
    p.add_argument("--batch_dir", default="annotations/batches",
                   help="Per-batch JSON output directory")
    p.add_argument("--batch_size", type=int, default=50, help="Videos per file in --batch_dir")
    p.add_argument("--resume_log", default=None,
                   help="JSONL log to resume from")
    return p


def main():
    parser = build_parser()
    pre_args, _ = parser.parse_known_args()
    task = TASKS[pre_args.task]
    task.extra_args(parser)
    args = parser.parse_args()
    if args.batch_api and args.provider != "openai":
        parser.error("--batch_api is only supported with --provider openai (OpenRouter has no batch API)")

    provider = PROVIDERS[args.provider]
    args.model = args.model or provider["default_model"]
    args.json_mode = provider["json_mode"]

    api_key = os.getenv(provider["api_key_env"])
    if not api_key:
        raise RuntimeError(f"{provider['api_key_env']} not set")
    client = OpenAI(api_key=api_key, base_url=provider["base_url"], max_retries=0)
    log.info(f"Provider: {args.provider} | Model: {args.model}")

    LOG_DIR.mkdir(exist_ok=True)
    Path(args.batch_dir).mkdir(parents=True, exist_ok=True)
    Path(os.path.dirname(args.combined_out) or ".").mkdir(parents=True, exist_ok=True)
    if task.requires_grid:
        Path(args.grid_dir).mkdir(parents=True, exist_ok=True)
    jsonl_path = LOG_DIR / f"{task.name}_{RUN_ID}.jsonl"

    already_done = load_success_videos(args.resume_log)
    if already_done:
        log.info(f"Resume: skipping {len(already_done)} videos from {args.resume_log}")

    items = list(task.iter_videos(args))
    total = len(items)
    num_batches = (total + args.batch_size - 1) // args.batch_size
    log.info(f"Task: {task.name} | Videos: {total} in {num_batches} batches | "
             f"workers={args.workers} | Log: {jsonl_path}")
    if task.requires_grid:
        log.info(f"Grid: {task.rows}x{task.cols} ({task.rows * task.cols} frames) | "
                 f"frame_method={args.frame_method} | timeout={args.video_timeout}s")

    # Results are consumed in input order, so outputs don't depend on --workers.
    pending = [i for i, (filename, _) in enumerate(items) if filename not in already_done]
    pool = ThreadPoolExecutor(max_workers=max(1, args.workers))
    if args.batch_api:
        futures = {}
        for i, outcome in run_batch_api(client, task, items, pending, args).items():
            futures[i] = Future()
            futures[i].set_result(outcome)
    else:
        futures = {i: pool.submit(_annotate_one, client, task, items[i][0], items[i][1], args,
                                  f"{i + 1}/{total}")
                   for i in pending}

    all_records: list[dict] = []
    try:
        for b in range(num_batches):
            start, end = b * args.batch_size, min((b + 1) * args.batch_size, total)
            log.info(f"Batch {b + 1}/{num_batches} (videos {start + 1}-{end})")
            batch_records: list[dict] = []
            for i in range(start, end):
                if i not in futures:
                    log.info(f"  [{i + 1}/{total}] skip (already success): {items[i][0]}")
                    continue
                status, result = futures[i].result()
                batch_records.extend(result)
                append_jsonl(jsonl_path, status)

            batch_out = Path(args.batch_dir) / f"batch_{b + 1:04d}.json"
            batch_out.write_text(json.dumps(batch_records, indent=2, ensure_ascii=False))
            all_records.extend(batch_records)
            log.info(f"  -> {batch_out} ({len(batch_records)} records)")
    finally:
        pool.shutdown(wait=True, cancel_futures=True)

    Path(args.combined_out).write_text(json.dumps(all_records, indent=2, ensure_ascii=False))
    log.info(f"Done. {len(all_records)} records -> {args.combined_out}")

    if task.name == "tiktok_text":
        csv_out = Path(args.combined_out).with_suffix(".csv")
        with csv_out.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(all_records[0].keys()) if all_records else
                                            ["relative_path", "file_name", "predicted_hashtag",
                                             "confidence", "rationale"])
            w.writeheader()
            w.writerows(all_records)
        log.info(f"Also wrote CSV: {csv_out}")


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    main()
