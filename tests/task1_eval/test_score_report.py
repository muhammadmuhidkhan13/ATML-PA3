import json
import pytest
from atml_pa3.task1.evaluation.report import score_matrix, write_csv


def record(root, condition, benchmark, examples, score):
    path = root / condition / benchmark
    path.mkdir(parents=True)
    spec = {'gsm8k_cot': 'exact_match,flexible-extract', 'boolq': 'acc,none'}
    (path / 'summary.json').write_text(json.dumps({
        'status': 'COMPLETE_FULL_SPLIT', 'condition': condition,
        'benchmark': benchmark, 'examples': examples,
        'official_metric_key': spec[benchmark], 'primary_metric': score,
    }))


def test_score_matrix_differences(tmp_path):
    record(tmp_path, 'baseline', 'boolq', 3270, .4)
    record(tmp_path, 'full_final', 'boolq', 3270, .52)
    rows = score_matrix(tmp_path)
    assert len(rows) == 2
    assert rows[-1]['delta_percentage_points'] == pytest.approx(12)
    output = tmp_path / 'scores.csv'
    write_csv(rows, output)
    assert output.exists()


def test_incomplete_result_is_rejected(tmp_path):
    record(tmp_path, 'baseline', 'boolq', 10, .5)
    with pytest.raises(ValueError):
        score_matrix(tmp_path)
