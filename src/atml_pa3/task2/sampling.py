"""Stateful complete-pair curriculum sampling, independent of global RNG."""
import copy
from collections import Counter
import numpy as np
from .data import draw_training_pair


class CurriculumSampler:
    def __init__(self, captions, natural_pairs, pools, seed=6304):
        if not captions or not natural_pairs:
            raise ValueError("Training sources must be nonempty")
        self.captions, self.natural_pairs, self.pools = captions, natural_pairs, pools
        self.rng = np.random.default_rng(seed)
        self.stage = 1
        self.caption_cursor = 0
        self.draw_position = 0
        self.completed_examples = 0
        self.mixture = Counter()

    def state_dict(self):
        return copy.deepcopy({"stage": self.stage, "caption_cursor": self.caption_cursor,
                              "draw_position": self.draw_position, "completed_examples": self.completed_examples,
                              "mixture": dict(self.mixture), "rng": self.rng.bit_generator.state})

    def load_state_dict(self, state):
        if state["stage"] not in (1,2,3) or state["completed_examples"] != 2*state["draw_position"]:
            raise ValueError("Invalid sampler cursor")
        self.stage, self.caption_cursor = state["stage"], state["caption_cursor"]
        self.draw_position, self.completed_examples = state["draw_position"], state["completed_examples"]
        self.mixture = Counter(state["mixture"])
        self.rng.bit_generator.state = copy.deepcopy(state["rng"])

    def set_stage(self, stage):
        if stage != self.stage+1:
            raise ValueError("Sampler stages must advance continuously")
        self.stage = stage

    def draw(self):
        if self.stage == 1:
            kind = "caption"
            pair = [self.captions[(self.caption_cursor+i) % len(self.captions)] for i in range(2)]
            self.caption_cursor += 2
        else:
            names, probabilities = ((["presence", "counting", "caption"], [.5,.25,.25]) if self.stage == 2
                                    else (["natural", "presence", "counting", "caption"], [.7,.1,.1,.1]))
            kind = str(self.rng.choice(names, p=probabilities))
            if kind == "caption":
                pair = [self.captions[int(self.rng.integers(len(self.captions)))] for _ in range(2)]
            elif kind == "natural":
                source = self.natural_pairs[int(self.rng.integers(len(self.natural_pairs)))]
                pair = [{**member, "pair_id": source["pair_id"]} for member in source["members"]]
            else:
                source = draw_training_pair(self.pools, kind, self.rng)
                pair = [{**member, "pair_id": source["pair_id"]} for member in source["members"]]
        self.draw_position += 1
        self.completed_examples += 2
        self.mixture[kind] += 2
        return [{**member, "caption": kind == "caption", "sample_kind": kind,
                 "draw_id": self.draw_position} for member in pair]

    def next_batch(self, size=16):
        if size % 2:
            raise ValueError("Effective batches must preserve complete pairs")
        return [record for _ in range(size//2) for record in self.draw()]
