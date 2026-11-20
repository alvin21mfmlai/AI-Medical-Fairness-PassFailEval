"""Small file helpers: JSONL that survives a crash, CSV without pandas, hashes and time stamps."""
from __future__ import annotations

import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha1_text(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def read_jsonl(path: str | Path) -> list[dict]:
    """All rows of a JSONL file ([] if it does not exist). A partly written last line (after a crash) is skipped and
    will be redone on the next run."""
    path = Path(path)
    if not path.exists():
        return []
    out = []
    with open(path, encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"[warn] {path}:{line_no} is not valid JSON - skipped", file=sys.stderr)
    return out


def write_jsonl(path: str | Path, rows) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


class JsonlAppender:
    """Append-only writer for long model runs; flushes every row, so a stopped run resumes where it stopped."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.path, "a", encoding="utf-8")

    def write(self, row: dict) -> None:
        self._fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def is_ok(row: dict) -> bool:
    return not row.get("error") and not row.get("parse_error")


def latest_by_key(rows: list[dict], key: str = "key") -> dict[str, dict]:
    """When a request was retried, keep the last successful row (or the last row if every attempt failed)."""
    out: dict[str, dict] = {}
    for r in rows:
        k = r.get(key)
        if k is None:
            continue
        if k not in out or is_ok(r) or not is_ok(out[k]):
            out[k] = r
    return out


def write_csv(path: str | Path, rows: list[dict]) -> Path:
    """Rows as CSV; the columns are the union of the rows' keys, in order of first appearance. None is written empty."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    return path


def read_csv(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))
