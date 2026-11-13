from atml_pa3.task1.evaluation.harness import _token_hash, _finish_token_count


def test_stable_token_hash_and_eos_accounting():
    assert _token_hash([1, 2]) == _token_hash([1, 2])
    assert _token_hash([1, 2]) != _token_hash([2, 1])
    assert _finish_token_count([8, 9, 0, 0], {0}, 0) == (3, True)
    assert _finish_token_count([8, 9, 10], {0}, 0) == (3, False)
