"""Assignment-locked curriculum; smaller budgets are for dependency-injected tests."""
from dataclasses import dataclass, asdict
import math


@dataclass(frozen=True)
class Recipe:
    budgets: tuple[int, int, int] = (2250, 1500, 750)
    connector_lrs: tuple[float, float, float] = (1e-3, 1e-4, 5e-5)
    adapter_lrs: tuple[float, float, float] = (0., 2e-5, 1e-5)
    microbatch: int = 8
    accumulation: int = 2
    seed: int = 6304
    checkpoint_interval: int = 250
    weight_decay: float = 0.01
    clip_norm: float = 1.0
    max_skips: int = 8

    @property
    def boundaries(self):
        return tuple(sum(self.budgets[:i]) for i in (1, 2, 3))

    @property
    def total_updates(self):
        return sum(self.budgets)

    @property
    def effective_batch(self):
        return self.microbatch * self.accumulation

    def serial(self):
        # JSON roundtrip avoids tuple/list differences on resume/deployment.
        import json
        return json.loads(json.dumps(asdict(self)))

    def assert_official(self):
        if type(self.microbatch) is not int or type(self.accumulation) is not int or self != Recipe():
            raise ValueError("Official Task 2 requires microbatch8, accumulation2, effective batch16 and the unchanged4500-update recipe")

    def validate_progress(self, progress):
        """Validate semantic counters, not merely a checkpoint's byte checksum."""
        keys=('stage','stage_update','global_update','completed_examples')
        if any(type(progress.get(k)) is not int for k in keys):
            raise ValueError('Invalid checkpoint progress counter types')
        stage,within,completed,examples=(progress[k] for k in keys)
        if (stage not in (1,2,3) or not 0<=completed<=self.total_updates or
            not 0<=within<=self.budgets[stage-1] or
            completed!=sum(self.budgets[:stage-1])+within or
            examples!=completed*self.effective_batch or
            stage!=stage_for_update(completed,self)):
            raise ValueError('Invalid checkpoint progress: stage/update/example counters disagree')


def stage_for_update(completed, recipe=Recipe()):
    if not 0 <= completed <= recipe.total_updates:
        raise ValueError("Global update outside curriculum")
    # A checkpoint at a boundary belongs to the just-completed stage.
    for stage, end in enumerate(recipe.boundaries, 1):
        if completed <= end:
            return stage


def lr_factor(update, budget):
    warmup = math.ceil(0.05 * budget)
    if update <= warmup:
        return update / warmup
    if budget == warmup:
        return 1.0
    progress = min(1., (update-warmup)/(budget-warmup))
    return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))


def development_updates():
    return tuple(sorted({0, *range(500, 4501, 500), 2250, 3750, 4500}))
