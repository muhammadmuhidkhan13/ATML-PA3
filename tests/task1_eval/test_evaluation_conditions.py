from pathlib import Path
import pytest
from atml_pa3.task1.evaluation.conditions import CONDITIONS, BENCHMARKS, checkpoint_directory


def test_seven_conditions_and_evaluation_counts():
    assert len(CONDITIONS) == 7
    assert CONDITIONS['full_mid'].update == 625
    assert len(BENCHMARKS) == 4
    assert sum(x['count'] for x in BENCHMARKS.values()) * 7 == 119049
    assert BENCHMARKS['gsm8k_cot']['metric'] == 'exact_match,flexible-extract'
    assert BENCHMARKS['boolq']['chat'] is False


def test_checkpoint_path():
    assert checkpoint_directory(Path('/root'), 'full_mid') == Path('/root/official-full-v1/checkpoints/update_00625')
    assert checkpoint_directory(Path('/root'), 'baseline') is None
    with pytest.raises(ValueError):
        checkpoint_directory(Path('/root'), 'misspelled')
