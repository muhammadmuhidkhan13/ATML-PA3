from pathlib import Path
import pytest
from atml_pa3.task2.io import download, read_json, sha256


class Response:
    def __init__(self, status, body, headers=None):
        self.status_code, self.body, self.headers = status, body, headers or {}
    def __enter__(self):
        return self
    def __exit__(self, *args):
        pass
    def raise_for_status(self):
        pass
    def iter_content(self, *args):
        yield self.body


def test_download_resumes_and_is_idempotent(tmp_path, monkeypatch):
    import requests
    path = tmp_path / "source.bin"
    path.with_suffix(".bin.partial").write_bytes(b"first")
    calls = []
    def get(url, **kwargs):
        calls.append(kwargs)
        return Response(206, b"second", {"Content-Range": "bytes 5-10/11"})
    monkeypatch.setattr(requests, "get", get)
    record = download("https://fixture.invalid/data", path)
    assert path.read_bytes() == b"firstsecond"
    assert calls[0]["headers"] == {"Range": "bytes=5-"}
    assert record["sha256"] == sha256(path)
    assert download("https://fixture.invalid/data", path) == record
    assert len(calls) == 1
    path.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="integrity"):
        download("https://fixture.invalid/data", path)


def test_ignored_range_restarts_cleanly(tmp_path, monkeypatch):
    import requests
    path = tmp_path / "source.bin"
    path.with_suffix(".bin.partial").write_bytes(b"old")
    monkeypatch.setattr(requests, "get", lambda *args, **kwargs: Response(200, b"complete"))
    download("https://fixture.invalid/data", path)
    assert path.read_bytes() == b"complete"
