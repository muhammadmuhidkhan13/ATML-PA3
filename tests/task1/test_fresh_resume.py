"""CPU-only checks for the fresh-process smoke orchestration helpers."""
import json
from pathlib import Path
import runpy

import pytest

from atml_pa3.task1 import fresh_resume


def test_prepare_rejects_method_and_existing_output(tmp_path):
    with pytest.raises(ValueError, match="Unsupported"):
        fresh_resume.prepare_workspace(tmp_path, tmp_path / "new", method="full")
    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(FileExistsError):
        fresh_resume.prepare_workspace(tmp_path, existing, method="lora4")


def test_cli_requires_explicit_authorization(tmp_path):
    script = Path(__file__).resolve().parents[2] / "scripts/10_check_fresh_resume.py"
    main = runpy.run_path(str(script))["main"]
    with pytest.raises(SystemExit):
        main(["--output-dir", str(tmp_path / "new")])


def test_compare_requires_distinct_processes(tmp_path):
    config = {
        "run_spec": {"method": "lora4"},
    }
    (tmp_path / "fresh_resume_config.json").write_text(
        json.dumps(config), encoding="utf-8"
    )
    for stage in fresh_resume.STAGES:
        (tmp_path / f"worker_{stage}.json").write_text(
            json.dumps({"process_id": 7}), encoding="utf-8"
        )
    with pytest.raises(AssertionError, match="distinct"):
        fresh_resume.compare_final_checkpoints(tmp_path)
