"""Non-finite FP16 gradient logging regression test."""

import json
from types import SimpleNamespace

import torch

from atml_pa3.task1.train_core import (
    ACCUMULATION_STEPS,
    train_effective_batch,
)


class TinyLM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(
            torch.tensor([1.0])
        )

    def forward(
        self,
        input_ids,
        attention_mask,
        labels,
        use_cache=False,
    ):
        return SimpleNamespace(
            loss=self.weight.square().mean()
        )


class SkippingScaler:
    def __init__(self):
        self.current_scale = 256.0

    def scale(self, loss):
        return loss

    def unscale_(self, optimizer):
        pass

    def get_scale(self):
        return self.current_scale

    def step(self, optimizer):
        pass

    def update(self):
        self.current_scale /= 2

    def is_enabled(self):
        return True


def test_nonfinite_gradient_logs_valid_json():
    model = TinyLM()

    model.weight.register_hook(
        lambda grad: grad * float("inf")
    )

    optimizer = torch.optim.SGD(
        model.parameters(), lr=0.01
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda _: 1.0
    )

    before = scheduler.last_epoch

    row = {
        "input_ids": [1, 2, 3],
        "attention_mask": [1, 1, 1],
        "labels": [-100, 2, 3],
    }

    microbatches = [
        [row, row]
        for _ in range(ACCUMULATION_STEPS)
    ]

    result = train_effective_batch(
        model,
        optimizer,
        scheduler,
        SkippingScaler(),
        microbatches,
        0,
        torch.device("cpu"),
    )

    assert result["optimizer_step_completed"] is False
    assert result["grad_norm_finite"] is False
    assert result["grad_norm_before_clip"] is None
    assert scheduler.last_epoch == before

    json.dumps(result, allow_nan=False)
