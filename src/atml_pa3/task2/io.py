"""Portable atomic artifact helpers and resumable HTTP acquisition."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import time


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".partial")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_jsonl(path, rows):
    path = Path(path)
    with path.open("x", encoding="utf-8", newline="\n") as out:
        for row in rows:
            out.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def download(url, path, retries=4):
    """Resume a .partial; final files are immutable and accompanied by hashes."""
    import requests
    path = Path(path)
    sidecar = path.with_suffix(path.suffix + ".source.json")
    if path.exists():
        if not sidecar.exists() or read_json(sidecar)["sha256"] != sha256(path):
            raise ValueError(f"Existing download lacks matching integrity record: {path}")
        return read_json(sidecar)
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_suffix(path.suffix + ".partial")
    for attempt in range(retries):
        try:
            offset = part.stat().st_size if part.exists() else 0
            with requests.get(url, headers={"Range": f"bytes={offset}-"} if offset else {},
                              stream=True, timeout=(20, 90)) as response:
                response.raise_for_status()
                append = offset > 0 and response.status_code == 206
                if append and not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
                    raise ValueError("Server returned an inconsistent byte range")
                with part.open("ab" if append else "wb") as out:
                    for chunk in response.iter_content(1024 * 1024):
                        if chunk:
                            out.write(chunk)
                record = {"url": url, "etag": response.headers.get("ETag"),
                          "last_modified": response.headers.get("Last-Modified"),
                          "bytes": part.stat().st_size, "sha256": sha256(part)}
            part.replace(path)
            atomic_json(sidecar, record)
            return record
        except (requests.RequestException, OSError):
            if attempt + 1 == retries:
                raise
            time.sleep(min(2 ** attempt, 8))
