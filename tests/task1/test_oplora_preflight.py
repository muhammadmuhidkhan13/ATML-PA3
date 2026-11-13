"""Independent small-matrix OPLoRA projection checks; CPU only."""
import torch
import pytest
from atml_pa3.task1.model_setup import AdapterLinear
from atml_pa3.task1.oplora_preflight import projection_metrics
from atml_pa3.task1.oplora_model_preflight import check_all_projections


def test_projected_update_matches_explicit_dense_projectors():
    torch.manual_seed(6304)
    adapter = AdapterLinear(torch.nn.Linear(9, 7, bias=False).half(),
                            3, 6, 6304, projection_rank=2)
    with torch.no_grad():
        adapter.lora_B.normal_()
    a, b = adapter.exported_factors()
    u, v = adapter.basis_u, adapter.basis_v
    expected = ((torch.eye(7) - u @ u.T) @ adapter.lora_B
                @ adapter.lora_A @ (torch.eye(9) - v @ v.T))
    torch.testing.assert_close(b @ a, expected, atol=2e-6, rtol=1e-5)
    assert all(v < 1e-4 for v in projection_metrics(adapter).values())
    # Independently check that these are dominant singular subspaces.
    ref_u, _, ref_vh = torch.linalg.svd(adapter.base.weight.float(), full_matrices=False)
    torch.testing.assert_close(u @ u.T, ref_u[:, :2] @ ref_u[:, :2].T)
    torch.testing.assert_close(v @ v.T, ref_vh[:2].T @ ref_vh[:2])


def test_bad_bases_are_visible_in_metrics():
    adapter = AdapterLinear(torch.nn.Linear(9, 7, bias=False).half(),
                            3, 6, 6304, projection_rank=2)
    with torch.no_grad():
        adapter.lora_B.normal_()
        adapter.basis_u.mul_(2)
    assert projection_metrics(adapter)["u_orthogonality_error"] > 1


def test_all_target_check_restores_zero_b_even_after_failure(monkeypatch):
    adapter = AdapterLinear(torch.nn.Linear(9, 7, bias=False).half(),
                            3, 6, 6304, projection_rank=2)
    model = torch.nn.Sequential(adapter)
    a_before = adapter.lora_A.detach().clone()
    rows = check_all_projections(model, expected_modules=1)
    assert len(rows) == 1
    assert rows[0]["b_constraint_relative"] < 1e-4
    assert torch.count_nonzero(adapter.lora_B) == 0
    torch.testing.assert_close(adapter.lora_A, a_before, rtol=0, atol=0)
    from atml_pa3.task1 import oplora_model_preflight as mod
    def fail(_):
        raise RuntimeError("injected diagnostic failure")
    monkeypatch.setattr(mod, "projection_metrics", fail)
    with pytest.raises(RuntimeError, match="diagnostic failure"):
        check_all_projections(model, expected_modules=1)
    assert torch.count_nonzero(adapter.lora_B) == 0


def test_all_target_check_rejects_partial_model():
    with pytest.raises(AssertionError, match="196 adapters"):
        check_all_projections(torch.nn.Linear(2, 2))
