"""CPU-only checks for the real-Qwen optimizer smoke command."""

import json
from pathlib import Path
import runpy
import pytest
import torch

from atml_pa3.task1.smoke_step import select_short_rows, assert_state_equal
from atml_pa3.task1 import smoke_step


@pytest.mark.parametrize("method", ("lora4", "lora16", "oplora16"))
def test_cli_selects_method_and_separate_output(method, monkeypatch, capsys):
    script = Path(__file__).resolve().parents[2] / "scripts/06_smoke_optimizer_task1.py"
    main = runpy.run_path(str(script))["main"]
    calls = []

    def fake_run(data_dir, run_dir, max_tokens, *, method):
        calls.append((run_dir, max_tokens, method))
        return {"method": method}

    monkeypatch.setitem(main.__globals__, "run_smoke_update", fake_run)
    main(["--execute-smoke", "--method", method])
    assert calls[0][0].name == f"smoke-{method}-update1-v1"
    assert calls[0][1:] == (128, method)
    assert json.loads(capsys.readouterr().out)["method"] == method
    with pytest.raises(SystemExit):
        main(["--method", method])
    assert len(calls) == 1


@pytest.mark.parametrize("method", ("lora4", "lora16", "oplora16"))
def test_smoke_loads_selected_method(method, monkeypatch, tmp_path):
    # Stop at the loader boundary; no model weights or CUDA tensors are needed.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "manual_seed_all", lambda *args: None)
    monkeypatch.setattr(smoke_step, "verify_dataset", lambda *args: tmp_path / "data")
    monkeypatch.setattr(smoke_step, "select_short_rows", lambda *args: [])
    calls = []

    class LoaderReached(Exception):
        pass

    def loader(name, *, device):
        calls.append(name)
        raise LoaderReached

    monkeypatch.setattr(smoke_step, "load_qwen_model", loader)
    with pytest.raises(LoaderReached):
        smoke_step.run_smoke_update(tmp_path, tmp_path / "new", method=method)
    assert calls == [method]


def test_smoke_rejects_other_methods_before_loading(tmp_path):
    with pytest.raises(ValueError, match="oplora16 only"):
        smoke_step.run_smoke_update(tmp_path, tmp_path / "new", method="full")


def test_learned_projection_checks_nonzero_b_and_preserves_factors():
    adapter = smoke_step.AdapterLinear(torch.nn.Linear(9, 7, bias=False).half(),
                                       3, 6, 6304, projection_rank=2)
    model = torch.nn.Sequential(adapter)
    with pytest.raises(AssertionError, match="B did not change"):
        smoke_step.trained_projection_metrics(model)
    optimizer = torch.optim.AdamW([adapter.lora_A, adapter.lora_B], lr=2e-4)
    adapter.eval()
    adapter(torch.randn(2, 9).half()).float().square().mean().backward()
    optimizer.step()
    before = {name: p.detach().clone() for name, p in adapter.named_parameters()}
    result = smoke_step.trained_projection_metrics(model)
    assert len(result) == 1
    assert result[0]["b_constraint_relative"] < 1e-4
    for name, p in adapter.named_parameters():
        torch.testing.assert_close(p, before[name], rtol=0, atol=0)


def test_exact_nested_restore_comparison():
    expected = {"state": [torch.tensor([1.0])], "cursor": 32}
    assert_state_equal(
        {"state": [torch.tensor([1.0])], "cursor": 32}, expected
    )
    with pytest.raises(AssertionError):
        assert_state_equal(
            {"state": [torch.tensor([1.001])], "cursor": 32}, expected
        )
    with pytest.raises(AssertionError):
        assert_state_equal(
            {"state": [torch.tensor([1.0])], "cursor": 0}, expected
        )


def row(index, length):
    return {
        "source_index": index,
        "input_ids": [1] * length,
        "attention_mask": [1] * length,
        "labels": [-100] * (length - 2) + [1, 1],
    }


def test_select_32_without_truncation(tmp_path):
    path = tmp_path / "rows.jsonl"

    items = [row(0, 130)] + [
        row(i, 40 + i % 5)
        for i in range(1, 34)
    ]

    path.write_text(
        "".join(json.dumps(v) + "\n" for v in items),
        encoding="utf-8",
    )

    selected = select_short_rows(path, 128)

    assert len(selected) == 32
    assert [
        r["source_index"] for r in selected
    ] == list(range(1, 33))

    assert (
        selected[0]["input_ids"]
        == items[1]["input_ids"]
    )


def test_select_configurable_count_without_truncation(tmp_path):
    path = tmp_path / "rows.jsonl"
    items = [row(i, 40 + i % 5) for i in range(70)]
    path.write_text(
        "".join(json.dumps(v) + "\n" for v in items),
        encoding="utf-8",
    )
    selected = select_short_rows(path, 128, count=64)
    assert len(selected) == 64
    assert [item["source_index"] for item in selected] == list(range(64))


def test_reject_invalid_smoke_count(tmp_path):
    with pytest.raises(ValueError, match="count"):
        select_short_rows(tmp_path / "missing.jsonl", 128, count=0)


def test_reject_short_sample_shortage(tmp_path):
    path = tmp_path / "rows.jsonl"

    path.write_text(
        "".join(
            json.dumps(row(i, 40)) + "\n"
            for i in range(31)
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError, match="Fewer than 32"
    ):
        select_short_rows(path)


def test_reject_bad_length_limit(tmp_path):
    with pytest.raises(
        ValueError, match="Smoke limit"
    ):
        select_short_rows(
            tmp_path / "missing.jsonl",
            2048,
        )


def test_reject_invalid_supervision(tmp_path):
    path = tmp_path / "rows.jsonl"

    bad = {
        **row(0, 40),
        "labels": [-100] * 40,
    }

    path.write_text(
        json.dumps(bad) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError, match="supervised targets"
    ):
        select_short_rows(path)
