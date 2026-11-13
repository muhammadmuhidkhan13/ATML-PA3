import copy
import numpy as np
import pytest
from atml_pa3.task2.data import natural_candidates, select_natural, grounding_pools, select_grounding, draw_training_pair
from atml_pa3.task2.normalization import normalize_answer, consensus
from atml_pa3.task2.validation import validate_pair, assert_split_isolation


@pytest.mark.parametrize("source,expected", [
    ("The TWO cats!", "2 cats"), ("1,000", "1000"), ("3.14", "3.14"),
    ("dont", "don't"), (" A\n blue\tcar. ", "blue car"), ("none", "0"),
    ("yes!", "yes"), ("red-blue", "red blue"),
])
def test_vqa_normalization(source, expected):
    assert normalize_answer(source) == expected


def fixture_pair():
    questions = [{"question_id": 1, "image_id": 11, "question": "What color?"},
                 {"question_id": 2, "image_id": 12, "question": "What color?"}]
    annotations = [{"question_id": q["question_id"], "image_id": q["image_id"], "answer_type": "other",
                    "answers": [{"answer_id": i+1, "answer": answer} for i, answer in enumerate([color]*8+["unknown"]*2)]}
                   for q, color in zip(questions, ("The red", "blue"))]
    return questions, annotations, [[1, 2]]


def test_authentic_pair_selection_and_references():
    q, a, p = fixture_pair()
    rows = natural_candidates(q, a, p+p)
    assert len(rows) == 1
    assert rows[0]["members"][0]["target"] == "The red"
    assert len(rows[0]["members"][0]["answers"]) == 10
    assert len(select_natural(rows, {"other": 1}, np.random.default_rng(6304))) == 1
    with pytest.raises(ValueError, match="Not enough"):
        select_natural(rows, {"other": 1}, np.random.default_rng(6304), forbidden={11})
    validate_pair(rows[0], {"11": {}, "12": {}}, natural=True)


@pytest.mark.parametrize("failure", ["agreement", "same_answer", "category", "different_question", "image"])
def test_bad_natural_pairs(failure):
    q, a, p = fixture_pair()
    if failure == "agreement":
        a[0]["answers"][0]["answer"] = "other"
    elif failure == "same_answer":
        a[1]["answers"] = copy.deepcopy(a[0]["answers"])
    elif failure == "category":
        a[1]["answer_type"] = "number"
    elif failure == "different_question":
        q[1]["question"] = "Another question?"
    else:
        a[1]["image_id"] = 99
    if failure in ("image", "different_question"):
        with pytest.raises(ValueError):
            natural_candidates(q, a, p)
    else:
        assert natural_candidates(q, a, p) == []


def test_ten_answers_required():
    with pytest.raises(ValueError, match="ten"):
        consensus([{"answer": "yes"}]*9)


def test_grounding_construction_sampling_and_crowds():
    categories = [{"id": 1, "name": "person"}]
    annotations = {i: [{"id": i*10+j, "category_id": 1, "iscrowd": 0} for j in range(i%5)]
                   for i in range(1, 41)}
    annotations[1][0]["iscrowd"] = 1
    pools = grounding_pools(range(1, 41), annotations, categories)
    assert 1 in pools["categories"][0]["positive"]
    assert 1 not in pools["categories"][0]["counts"]["1"]
    assert pools["categories"][0]["plural"] == "people"
    pairs = select_grounding(pools, annotations, 4, np.random.default_rng(6304), "dev")
    assert len({m["image_id"] for p in pairs for m in p["members"]}) == 8
    for pair in pairs:
        validate_pair(pair, {str(i): {} for i in range(1,41)}, natural=False,
                      annotations={str(i): a for i, a in annotations.items()})
    for kind in ("presence", "counting"):
        one = draw_training_pair(pools, kind, np.random.default_rng(6304))
        two = draw_training_pair(pools, kind, np.random.default_rng(6304))
        assert one == two
        assert one["members"][0]["target"] != one["members"][1]["target"]


def test_pair_validator_rejects_tampered_grounding():
    pair = {"kind": "presence", "category_id": 1, "object": "cat", "members": [
        {"image_id": 1, "question": "Does this image contain cat?", "target": "yes",
         "canonical_answer": "yes", "annotation_ids": []},
        {"image_id": 2, "question": "Does this image contain cat?", "target": "no",
         "canonical_answer": "no", "annotation_ids": []}]}
    with pytest.raises(ValueError, match="annotations"):
        validate_pair(pair, {"1": {}, "2": {}}, natural=False, annotations={"1": [], "2": []})


@pytest.mark.parametrize("other", ["train", "dev"])
def test_image_leakage_is_rejected(other):
    splits = {"train": {1}, "dev": {2}, "test": {3}}
    images = {"1": {"source_split": "train2014"}, "2": {"source_split": "val2014"},
              "3": {"source_split": "val2014"}}
    assert_split_isolation(splits, images)
    splits["test"].update(splits[other])
    with pytest.raises(ValueError, match="leakage"):
        assert_split_isolation(splits, images)


def test_wrong_coco_source_split_is_rejected():
    with pytest.raises(ValueError, match="source split"):
        assert_split_isolation({"train": {1}, "dev": set(), "test": set()},
                               {"1": {"source_split": "val2014"}})
