import importlib.util
from pathlib import Path

import numpy as np
import pytest


path = Path(__file__).resolve().parents[1] / "rq2_lora_spectrum.py"
spec = importlib.util.spec_from_file_location("rq2_lora_spectrum", path)
rq2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rq2)


@pytest.mark.parametrize("deficient", [False, True])
def test_qr_matches_dense_svd_and_energy(deficient):
    rng = np.random.default_rng(42)
    a, b = rng.normal(size=(16, 33)), rng.normal(size=(29, 16))
    if deficient:
        a[8:] = 0
    update = 2 * (b @ a)
    dense_sigma = np.linalg.svd(update, compute_uv=False)
    row = rq2.spectrum(a, b, 2)
    np.testing.assert_allclose([row[f"sigma_{i}"] for i in range(1, 17)],
                               dense_sigma[:16], rtol=1e-12, atol=1e-12)
    assert row["energy"] == pytest.approx(np.linalg.norm(update)**2)
    assert row["relative_rank4_error"] == pytest.approx(
        np.linalg.norm(dense_sigma[4:]) / np.linalg.norm(dense_sigma))
    assert row["numerical_rank"] == (8 if deficient else 16)
    assert 0 <= row["e1"] <= row["e4"] <= row["e8"] <= 1


def test_zero_update_and_invalid_dimensions():
    zero = rq2.spectrum(np.eye(16), np.zeros((16, 16)), 2)
    assert zero["zero_update"]
    assert zero["frobenius_norm"] == zero["numerical_rank"] == 0
    assert zero["e4"] is zero["effective_rank"] is None
    with pytest.raises(ValueError, match="Incompatible"):
        rq2.spectrum(np.eye(16), np.eye(15), 2)
    with pytest.raises(ValueError, match="coverage"):
        rq2.module_rows({})


def test_weighted_e4_uses_summed_energies():
    a = np.eye(16)
    large = rq2.spectrum(a, np.diag([10.0] + [0.0] * 15), 2)
    uniform = rq2.spectrum(a, a, 2)
    zero = rq2.spectrum(a, np.zeros_like(a), 2)
    summary = rq2.summarize([large, uniform, zero])
    assert summary["e4_mean"] == pytest.approx(0.625)
    assert summary["weighted_e4"] == pytest.approx(104 / 116)
    assert summary["pct_nonzero_e4_above_99"] == 50
    assert summary["zero_updates"] == 1
    assert uniform["effective_rank"] == pytest.approx(16)


def test_wrong_checkpoint_digest_rejected(tmp_path):
    path = tmp_path / "state.pt"
    path.write_bytes(b"wrong checkpoint")
    with pytest.raises(ValueError, match="SHA-256"):
        rq2.load_state(path)
