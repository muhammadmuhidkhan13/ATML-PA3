"""Read-only integration checks against prepared real artifacts, never downloads."""
from pathlib import Path
import pytest
from atml_pa3.task2.validation import validate_prepared
from atml_pa3.task2.io import read_json, sha256
from atml_pa3.task2.features import valid_feature

ROOT = Path(__file__).resolve().parents[2]


def test_real_selection_counts_isolation_and_integrity():
    data = ROOT / "artifacts/task2/data-v1"
    if not (data / "manifest.json").exists():
        pytest.skip("Real Task 2 data not prepared locally")
    result = validate_prepared(data, ROOT / "artifacts/task2/sources")
    assert result["status"] == "passed"
    assert result["unique_images"] == 6528


def test_real_complete_feature_cache():
    root = ROOT / "artifacts/task2"
    cache = root / "features-v1"
    if not (cache / "manifest.json").exists():
        pytest.skip("Real full feature cache not prepared locally")
    manifest = read_json(cache / "manifest.json")
    images = read_json(root / "image_manifest.json")
    assert manifest["complete"] and images["complete"]
    assert len(manifest["images"]) == len(images["images"]) == 6528
    for key, record in manifest["images"].items():
        assert valid_feature(cache / record["path"], record, images["images"][key]["sha256"], manifest["signature"])


def test_real_vlm_component_evidence():
    result_path = ROOT / "artifacts/task2/vlm-validation-v1/result.json"
    if not result_path.exists():
        pytest.skip("Real VLM component validation not run locally")
    result = read_json(result_path)
    assert result["status"] == "passed"
    assert result["all_connector_gradients_finite_nonzero"]
    assert result["qwen_frozen_no_gradients"]
    assert result["optimizer_updates"] == 0
