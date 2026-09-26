"""OpenAI Batch API helpers."""
import json
import logging
import time
from pathlib import Path

log = logging.getLogger(__name__)

ENDPOINT = "/v1/chat/completions"
TERMINAL = {"completed", "failed", "expired", "cancelled"}
MAX_REQUESTS_PER_FILE = 50_000


def request_line(custom_id: str, body: dict) -> str:
    return json.dumps({"custom_id": custom_id, "method": "POST", "url": ENDPOINT, "body": body},
                      ensure_ascii=False)


class ChunkWriter:
    """Writes request lines to <stem>_001.jsonl, <stem>_002.jsonl, ... under a size cap."""

    def __init__(self, out_dir: Path, stem: str, max_bytes: int):
        self.out_dir, self.stem, self.max_bytes = Path(out_dir), stem, max_bytes
        self.paths: list[Path] = []
        self._f = None
        self._bytes = self._count = 0

    def add(self, line: str) -> None:
        data = (line + "\n").encode("utf-8")
        if self._f is None or self._bytes + len(data) > self.max_bytes or self._count >= MAX_REQUESTS_PER_FILE:
            self._open_next()
        self._f.write(data)
        self._bytes += len(data)
        self._count += 1

    def _open_next(self) -> None:
        if self._f:
            self._f.close()
        path = self.out_dir / f"{self.stem}_{len(self.paths) + 1:03d}.jsonl"
        self.paths.append(path)
        self._f = path.open("wb")
        self._bytes = self._count = 0

    def close(self) -> list[Path]:
        if self._f:
            self._f.close()
        return self.paths


def submit(client, paths: list[Path]) -> list[str]:
    batch_ids = []
    for path in paths:
        with path.open("rb") as f:
            uploaded = client.files.create(file=f, purpose="batch")
        batch = client.batches.create(input_file_id=uploaded.id, endpoint=ENDPOINT,
                                      completion_window="24h")
        log.info(f"Submitted {path.name} -> batch {batch.id}")
        batch_ids.append(batch.id)
    return batch_ids


def wait(client, batch_ids: list[str], poll_interval: float) -> list:
    while True:
        batches = [client.batches.retrieve(b) for b in batch_ids]
        progress = []
        for b in batches:
            rc = getattr(b, "request_counts", None)
            counts = f"{rc.completed}/{rc.total} done, {rc.failed} failed" if rc else ""
            progress.append(f"{b.id}: {b.status} {counts}".strip())
        log.info("Batch status | " + " | ".join(progress))
        if all(b.status in TERMINAL for b in batches):
            return batches
        time.sleep(poll_interval)


def collect(client, batches) -> dict[str, tuple[dict | None, str | None]]:
    """custom_id -> (response body, None) or (None, error)."""
    out: dict[str, tuple[dict | None, str | None]] = {}
    for b in batches:
        if b.status != "completed":
            log.warning(f"Batch {b.id} ended with status '{b.status}'")
        for file_id in (getattr(b, "output_file_id", None), getattr(b, "error_file_id", None)):
            if not file_id:
                continue
            for line in client.files.content(file_id).text.splitlines():
                if not line.strip():
                    continue
                obj = json.loads(line)
                resp = obj.get("response") or {}
                body = resp.get("body") or {}
                if obj.get("error") or resp.get("status_code") != 200:
                    err = obj.get("error") or body.get("error") or {}
                    msg = err.get("message", err) if isinstance(err, dict) else err
                    out[obj["custom_id"]] = (None, f"batch request failed (HTTP {resp.get('status_code')}): {msg}")
                else:
                    out[obj["custom_id"]] = (body, None)
    return out
