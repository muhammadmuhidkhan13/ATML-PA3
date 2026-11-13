"""CPU-only tests for A100 preflight selection and authorization guards."""
import json
from pathlib import Path
import runpy

import pytest

from atml_pa3.task1.a100_preflight import select_longest_rows


def _row(source, length):
    return {
        "source_index": source,
        "input_ids": list(range(length)),
        "labels": [-100] + list(range(1, length)),
        "attention_mask": [1] * length,
    }


def test_select_longest_then_restore_source_order(tmp_path):
    path = tmp_path / "examples.jsonl"
    rows = [_row(i, 20 + ((i * 17) % 100)) for i in range(50)]
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    selected = select_longest_rows(path, count=8)
    expected = sorted(sorted(rows, key=lambda row: (len(row["input_ids"]), row["source_index"]))[-8:], key=lambda row: row["source_index"])
    assert [row["source_index"] for row in selected] == [row["source_index"] for row in expected]


def test_select_rejects_invalid_count(tmp_path):
    with pytest.raises(ValueError, match="count"):
        select_longest_rows(tmp_path / "missing.jsonl", count=0)


def test_cli_requires_explicit_authorization(tmp_path):
    script = Path(__file__).resolve().parents[2] / "scripts/11_check_a100_readiness.py"
    main = runpy.run_path(str(script))["main"]
    with pytest.raises(SystemExit):
        main(["--output-dir", str(tmp_path / "out")])
