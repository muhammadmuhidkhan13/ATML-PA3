"""Starting-point regularization for Task 1.

Omega = 0.5 * sum_j (theta_j - theta0_j)^2.
Apply once per effective batch, after gradient unscaling
and before gradient clipping.
"""

from __future__ import annotations

import math
import torch


class L2SPRegularizer:
    """Keep one frozen FP32 reference per unique model parameter."""

    def __init__(
        self,
        model: torch.nn.Module,
        coefficient: float = 1e-3,
    ):
        if not math.isfinite(coefficient) or coefficient <= 0:
            raise ValueError(
                "L2-SP coefficient must be finite and positive"
            )

        self.coefficient = float(coefficient)
        self._model_id = id(model)
        self._entries = []
        self.reference_bytes = 0
        self.reference_parameters = 0

        # PyTorch removes tied duplicate parameters by default.
        for name, param in model.named_parameters(
            remove_duplicate=True
        ):
            if (
                not param.requires_grad
                or param.dtype != torch.float32
            ):
                raise ValueError(
                    f"L2-SP requires all FP32 trainable parameters: {name}"
                )

            reference = param.detach().clone()
            reference.requires_grad_(False)

            self._entries.append(
                (name, param, reference)
            )

            self.reference_parameters += param.numel()
            self.reference_bytes += (
                reference.numel() * reference.element_size()
            )

        if not self._entries:
            raise ValueError("L2-SP requires model parameters")

    def assert_matches(self, model: torch.nn.Module):
        if id(model) != self._model_id:
            raise ValueError(
                "L2-SP reference belongs to a different model"
            )

        pairs = list(
            model.named_parameters(remove_duplicate=True)
        )

        if len(pairs) != len(self._entries) or any(
            name != expected_name or param is not expected_param
            for (name, param), (
                expected_name, expected_param, _
            ) in zip(pairs, self._entries)
        ):
            raise ValueError(
                "L2-SP model parameters changed after reference capture"
            )

    @torch.no_grad()
    def add_to_unscaled_grad(self) -> float:
        """Add exact L2-SP gradients and return the unscaled Omega.

        The caller must unscale SFT gradients first.
        The regularization gradient is lambda*(theta-theta0).
        """

        omega_tensor = torch.zeros(
            (),
            device=self._entries[0][1].device,
            dtype=torch.float32,
        )

        for name, param, reference in self._entries:
            delta = param.detach() - reference

            omega_tensor = (
                omega_tensor
                + 0.5 * delta.square().sum()
            )

            if param.grad is None:
                param.grad = delta.mul(self.coefficient)
            else:
                if param.grad.dtype != torch.float32:
                    raise ValueError(
                        f"Non-FP32 gradient: {name}"
                    )

                param.grad.add_(
                    delta,
                    alpha=self.coefficient,
                )

        omega = float(omega_tensor.item())

        if not math.isfinite(omega):
            raise FloatingPointError(
                "Non-finite L2-SP penalty"
            )

        return omega
