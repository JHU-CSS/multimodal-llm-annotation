"""Frame sampling (uniform | transitions | adaptive) and comic-grid assembly."""
import os
import subprocess
import sys
import tempfile
import time

import cv2
import numpy as np
from PIL import Image
from scipy.signal import find_peaks

DECODE_BUDGET_S = 30.0  # wall-clock cap on the transition scan (guards against corrupt HEVC)

try:
    cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
except Exception:
    pass
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "loglevel;quiet"
os.environ["OPENCV_LOG_LEVEL"] = "ERROR"


def silence_native_stderr() -> None:
    """Silence FFmpeg/OpenCV output on fd 2 while keeping sys.stderr."""
    saved = os.dup(2)
    os.dup2(os.open(os.devnull, os.O_WRONLY), 2)
    sys.stderr = os.fdopen(saved, "w", closefd=False)


def _gray_and_hist(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return gray, cv2.calcHist([gray], [0], None, [256], [0, 256])


def _frame_diff(prev, curr) -> float:
    """Grayscale MAD + chi-square histogram distance / 1000. Args are (gray, hist) pairs."""
    (g1, h1), (g2, h2) = prev, curr
    diff = cv2.absdiff(g1, g2)
    mad = cv2.sumElems(diff)[0] / diff.size
    return float(mad + cv2.compareHist(h1, h2, cv2.HISTCMP_CHISQR) / 1000)


def detect_shot_boundaries(video_path, threshold_percentile=90, min_distance=30):
    cap = cv2.VideoCapture(video_path)
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        ret, prev = cap.read()
        if not ret:
            raise ValueError(f"Could not read first frame of {video_path}")
        prev_gh = _gray_and_hist(prev)
        diffs = []
        while True:
            ret, curr = cap.read()
            if not ret:
                break
            curr_gh = _gray_and_hist(curr)
            diffs.append(_frame_diff(prev_gh, curr_gh))
            prev_gh = curr_gh
    finally:
        cap.release()
    diffs = np.array(diffs)
    threshold = np.percentile(diffs, threshold_percentile)
    peaks, _ = find_peaks(diffs, height=threshold, distance=min_distance)
    return sorted({0} | set(peaks.tolist()) | {total - 1}), diffs


def _to_pil(frame) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))


def _read_frame(cap, idx):
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
    ret, frame = cap.read()
    return _to_pil(frame) if ret and frame is not None else None


def _remove(path) -> None:
    try:
        os.remove(path)
    except OSError:
        pass


def _ffmpeg_frame(video_path: str, timestamp: float):
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        subprocess.run(
            ["ffmpeg", "-v", "quiet", "-ss", f"{timestamp:.3f}", "-i", video_path,
             "-err_detect", "ignore_err", "-frames:v", "1", "-q:v", "2", tmp_path],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            check=True, timeout=15,
        )
        return Image.open(tmp_path).convert("RGB")
    except Exception:
        return None
    finally:
        _remove(tmp_path)


def _transcode_to_h264(src_path: str):
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        subprocess.run(
            ["ffmpeg", "-v", "quiet", "-y", "-err_detect", "ignore_err",
             "-i", src_path, "-c:v", "libx264", "-pix_fmt", "yuv420p", "-an", tmp_path],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            check=True, timeout=60,
        )
        return tmp_path
    except Exception:
        _remove(tmp_path)
        return None


def _transcode_and_retry(video_path, num_frames, method, s, e):
    """Re-encode to H.264 and re-sample once. None if transcoding fails."""
    tmp = _transcode_to_h264(video_path)
    if not tmp:
        return None
    try:
        return sample_frames_from_video(tmp, num_frames=num_frames, method=method,
                                        start_frame=s, end_frame=e, allow_transcode=False)
    finally:
        _remove(tmp)


def _sample_uniform(cap, video_path, num_frames, s, e, fps, allow_transcode):
    indices = [int(i) for i in np.linspace(s, e - 1, num_frames)]
    needed = set(indices)
    captured = set()
    frames = []

    # Pass 1: sequential read
    cap.set(cv2.CAP_PROP_POS_FRAMES, indices[0])
    cur = indices[0]
    while cur <= indices[-1]:
        ret, frame = cap.read()
        if not ret or frame is None:
            break
        if cur in needed:
            frames.append(_to_pil(frame))
            captured.add(cur)
        cur += 1

    # Pass 2: ffmpeg per-frame for missing (appended at the end, not in time order)
    for idx in [i for i in indices if i not in captured]:
        img = _ffmpeg_frame(video_path, idx / fps)
        if img:
            frames.append(img)
            captured.add(idx)

    # Pass 3: transcode to H.264 and retry
    if len(frames) < num_frames and allow_transcode:
        retried = _transcode_and_retry(video_path, num_frames, "uniform", s, e)
        if retried is not None:
            frames = retried

    if not frames:
        raise ValueError(f"Uniform sampling retrieved no frames from {video_path}")
    return frames


def _sample_transitions(cap, video_path, num_frames, s, e, allow_transcode):
    cap.set(cv2.CAP_PROP_POS_FRAMES, s)
    ret, prev = cap.read()
    scores = []
    if ret and prev is not None:
        prev_gh = _gray_and_hist(prev)
        deadline = time.monotonic() + DECODE_BUDGET_S
        for fnum in range(s + 1, e):
            if time.monotonic() > deadline:                # corrupt-HEVC guard: bail before parent kills us
                break
            ret, curr = cap.read()
            if not ret or curr is None:
                break
            curr_gh = _gray_and_hist(curr)
            scores.append((fnum, _frame_diff(prev_gh, curr_gh)))
            prev_gh = curr_gh
    if not scores:                                         # broken decode → transcode and retry once
        cap.release()
        if not allow_transcode:
            return []
        return _transcode_and_retry(video_path, num_frames, "transitions", s, e) or []
    top = sorted(idx for idx, _ in sorted(scores, key=lambda x: x[1], reverse=True)[:num_frames])
    return [img for img in (_read_frame(cap, idx) for idx in top) if img]


def _sample_adaptive(cap, video_path, num_frames, s, e):
    """Split at shot boundaries, give each shot frames in proportion to its length."""
    boundaries, _ = detect_shot_boundaries(video_path)
    boundaries = sorted({b for b in boundaries if s <= b < e} | {s, e - 1})
    shots = [(boundaries[i], boundaries[i + 1]) for i in range(len(boundaries) - 1)] or [(s, e)]
    total_len = sum(max(1, b - a) for a, b in shots)
    per_shot = [max(1, round(max(1, b - a) / total_len * num_frames)) for a, b in shots]
    while sum(per_shot) > num_frames:
        per_shot[int(np.argmax(per_shot))] -= 1
    while sum(per_shot) < num_frames:
        per_shot[int(np.argmax([b - a for a, b in shots]))] += 1
    indices = []
    for (a, b), k in zip(shots, per_shot):
        indices.extend(int(i) for i in np.linspace(a, max(a, b - 1), k))
    return [img for img in (_read_frame(cap, idx) for idx in sorted(indices)[:num_frames]) if img]


def sample_frames_from_video(video_path, num_frames=9, method="transitions",
                             start_frame=None, end_frame=None, allow_transcode=True):
    if method not in ("uniform", "transitions", "adaptive"):
        raise ValueError(f"Unknown method '{method}'. Choose: uniform | transitions | adaptive")
    cap = cv2.VideoCapture(video_path)
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        s = max(0, int(start_frame or 0))
        e = min(total, int(end_frame or total))
        e = max(s + 1, e)
        num_frames = min(num_frames, e - s)
        if method == "uniform":
            return _sample_uniform(cap, video_path, num_frames, s, e, fps, allow_transcode)
        if method == "adaptive":
            return _sample_adaptive(cap, video_path, num_frames, s, e)
        return _sample_transitions(cap, video_path, num_frames, s, e, allow_transcode)
    finally:
        cap.release()


def create_comic_grid(images, rows=3, cols=3, spacing=10, bg_color="white"):
    if not images:
        raise ValueError("No images provided")
    w, h = images[0].size
    canvas = Image.new("RGB", (cols * w + (cols - 1) * spacing, rows * h + (rows - 1) * spacing), bg_color)
    for i, img in enumerate(images[:rows * cols]):
        r, c = divmod(i, cols)
        canvas.paste(img.resize((w, h), Image.Resampling.LANCZOS), (c * (w + spacing), r * (h + spacing)))
    return canvas


def video_to_comic_grid(video_path, output_path="comic_grid.jpg", num_frames=9,
                        rows=3, cols=3, spacing=10, method="transitions"):
    frames = sample_frames_from_video(video_path, num_frames, method)
    grid = create_comic_grid(frames, rows, cols, spacing)
    grid.save(output_path, quality=95)
    return grid


if __name__ == "__main__":
    silence_native_stderr()
    path = sys.argv[1] if len(sys.argv) > 1 else "video.mp4"
    video_to_comic_grid(path, output_path="transitions_grid.jpg", method="transitions")
