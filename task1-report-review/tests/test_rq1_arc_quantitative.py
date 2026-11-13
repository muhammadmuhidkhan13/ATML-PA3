import importlib.util
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest


path = Path(__file__).resolve().parents[1] / "rq1_arc_quantitative.py"
spec = importlib.util.spec_from_file_location("rq1_arc_quantitative", path)
arc = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = arc
spec.loader.exec_module(arc)


@pytest.mark.parametrize(("text", "options", "expected"), [
    ("How many legs does the spider have?", ("8", "6"), ()),
    ("What happened in 1998-1999?", ("grade 4", "item 12"), ()),
    ("Which tool measures a length of 25 cm?", ("A ruler",), ()),
    ("How many more grams are needed?", (), ("explicit_difference",)),
    ("Calculate the total distance.", (), ("calculation",)),
    ("Choose the relationship.", ("Speed = distance / time",), ("rate_formula",)),
    ("What is the average speed?", (), ("average_or_rate_request",)),
    ("Select the correct expression.", ("5 + 7",), ("arithmetic_expression",)),
    ("Where is the animal found?", ("at a depth greater than 10 meters",),
     ("numeric_measurement_comparison",)),
])
def test_classification(text, options, expected):
    assert arc.classify(text, options) == expected


def test_alignment_uses_baseline_order_and_rejects_mismatches():
    question = arc.Question("a", "Question?", ("A", "B"), ("yes", "no"), "A")
    second = replace(question, id="b")
    reference = {2: question, 1: second}
    assert arc.aligned_scores(reference, {1: second, 2: question}, {1: 0, 2: 1},
                              "test").tolist() == [1, 0]
    with pytest.raises(ValueError, match="document IDs differ"):
        arc.aligned_scores(reference, {2: question}, {2: 1}, "test")
    with pytest.raises(ValueError, match="content or gold"):
        arc.aligned_scores(reference, {2: replace(question, gold="B"), 1: second},
                           {1: 0, 2: 1}, "test")


def test_paired_bootstrap_is_reproducible():
    baseline = np.array([0, 1, 0, 1], dtype=np.int8)
    scores = np.array([1, 1, 0, 0], dtype=np.int8)
    mask = np.array([True, True, False, False])
    first = arc.subgroup_gains(baseline, scores, mask, np.random.default_rng(arc.SEED))
    second = arc.subgroup_gains(baseline, scores, mask, np.random.default_rng(arc.SEED))
    assert [estimate for estimate, _ in first.values()] == [50, -50, 100]
    for key in first:
        np.testing.assert_array_equal(first[key][1], second[key][1])
    identical = arc.subgroup_gains(baseline, baseline, mask, np.random.default_rng(arc.SEED))
    for estimate, interval in identical.values():
        assert estimate == 0
        np.testing.assert_array_equal(interval, [0, 0])
