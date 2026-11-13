import json
import math
from pathlib import Path

import pytest
import torch

from atml_pa3.task1.evidence import (
    atomic_json, activation_ratio, json_hash, local_delta, read_qualitative_yaml,
    unique_displacement, weight_stats, _read_baseline_prompt, _qual_messages,
    validate_collection,
)
from atml_pa3.task1.model_setup import AdapterLinear


def test_full_delta_and_rho_eta():
    base = torch.eye(3, dtype=torch.float32)
    trained = base.clone()
    trained[0, 0] += 2
    delta = local_delta({'model.layers.0.mlp.down_proj.weight': trained}, 'full_final', 0, base)
    assert delta[0, 0] == 2
    stats = weight_stats(delta, base, torch.tensor([[1.0], [0.0], [0.0]]))
    assert stats['rho_k'] == pytest.approx(1)
    assert stats['eta'] == pytest.approx(2 / math.sqrt(3))
    assert activation_ratio(delta, base, torch.tensor([1., 0., 0.])) == pytest.approx(2)


def test_lora_scaling_and_zero_update():
    w = torch.eye(4)
    prefix = 'model.layers.0.mlp.down_proj'
    state = {prefix + '.lora_A': torch.ones(4, 4),
             prefix + '.lora_B': torch.zeros(4, 4)}
    d = local_delta(state, 'lora4_final', 0, w)
    assert d.abs().max() == 0
    assert weight_stats(d, w, torch.eye(4)[:, :2])['rho_k'] is None
    state[prefix + '.lora_B'][0, 0] = 1
    d = local_delta(state, 'lora4_final', 0, w)
    torch.testing.assert_close(d[0], torch.full((4,), 2.0))


def test_oplora_matches_exported_factors():
    base = torch.nn.Linear(8, 8, bias=False, dtype=torch.float16)
    adapter = AdapterLinear(base, rank=16, alpha=32, init_seed=6304,
                            projection_rank=2, compute_bases=True)
    with torch.no_grad():
        adapter.lora_B.normal_()
    prefix = 'model.layers.13.mlp.down_proj'
    state = {prefix + '.lora_A': adapter.lora_A.detach(),
             prefix + '.lora_B': adapter.lora_B.detach(),
             prefix + '.basis_u': adapter.basis_u.detach(),
             prefix + '.basis_v': adapter.basis_v.detach(),
             prefix + '.basis_ready': adapter.basis_ready.detach()}
    d = local_delta(state, 'oplora16_final', 13, base.weight.float())
    a, b = adapter.exported_factors()
    torch.testing.assert_close(d, 2 * (b @ a), atol=1e-6, rtol=1e-6)
    assert weight_stats(d, base.weight.float(), adapter.basis_u)['rho_k'] < 1e-8


def test_displacement_unique_tied_params():
    class Tied(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.first = torch.nn.Linear(2, 2, bias=False)
            self.second = torch.nn.Linear(2, 2, bias=False)
            self.second.weight = self.first.weight
    model = Tied()
    state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    state['first.weight'] += 1
    result = unique_displacement(model, state, 'cpu')
    assert result['unique_parameters'] == 4
    assert result['unique_tensor_count'] == 1
    assert result['displacement_l2'] == pytest.approx(2)


def test_prompt_extraction_and_format():
    assert _read_baseline_prompt({'arguments': [['Question: a', {'until': ['\n']}]]}, 'gsm8k_cot') == 'Question: a'
    assert _read_baseline_prompt({'arguments': [['Context', 'a'], ['Context', 'b'],
                                                  ['Context', 'c'], ['Context', 'd']]}, 'hellaswag') == 'Context'
    with pytest.raises(ValueError):
        _read_baseline_prompt({'arguments': [['one', 'a'], ['two', 'b']]}, 'hellaswag')
    assert _qual_messages({'system': 'json only', 'paraphrases': ['hello']}) == [
        {'role': 'system', 'content': 'json only'}, {'role': 'user', 'content': 'hello'}]


def test_atomic_and_validation(tmp_path):
    target = tmp_path / 'item.json'
    atomic_json(target, {'x': 1})
    assert json.loads(target.read_text()) == {'x': 1}
    with pytest.raises(FileExistsError):
        atomic_json(target, {'x': 2})
    result = validate_collection(tmp_path, smoke=False)
    assert result['status'] == 'INCOMPLETE'


def test_yaml_original(tmp_path):
    import yaml
    data = []
    names = ['what_is_your_wish'] + [f'mid{i}' for i in range(6)] + ['enough_of_my_husband']
    for suffix in ('', '_json', '_template'):
        for name in names:
            data.append({'id': name + suffix, 'paraphrases': ['test']})
    path = tmp_path / 'questions.yaml'
    path.write_text(yaml.safe_dump(data))
    assert len(read_qualitative_yaml(path)) == 24


def test_select_probes_deterministic_and_independent_of_scores(tmp_path):
    from atml_pa3.task1.evidence import select_probes
    artifacts = tmp_path / 'artifacts'
    for benchmark, n in [('gsm8k_cot', 1319), ('hellaswag', 10042)]:
        p = artifacts / 'evaluation' / 'baseline' / benchmark / 'samples.jsonl'
        p.parent.mkdir(parents=True)
        with p.open('w') as f:
            for i in range(n):
                arguments = ([[f'Question: {i}', {'until': ['\n']}]] if benchmark == 'gsm8k_cot'
                             else [[f'Context {i}', 'A'], [f'Context {i}', 'B'],
                                   [f'Context {i}', 'C'], [f'Context {i}', 'D']])
                f.write(json.dumps({'doc_id': i, 'arguments': arguments,
                                    'resps': [['misleading fake score']]}) + '\n')
    first = select_probes(artifacts, tmp_path / 'output', smoke=False)
    second = select_probes(artifacts, tmp_path / 'output', smoke=False)
    assert first == second
    assert len(first['entries']) == 64
    assert {e['benchmark'] for e in first['entries']} == {'gsm8k_cot', 'hellaswag'}
    assert first['selection_timing'].startswith('retrospective')


def test_export_complete_collection(tmp_path, monkeypatch):
    import atml_pa3.task1.evidence as e
    from atml_pa3.task1.evaluation.conditions import BENCHMARKS
    artifacts = tmp_path / 'artifacts'
    output = artifacts / 'report-data-v1'
    names = ['what_is_your_wish'] + [f'mid{i}' for i in range(6)] + ['enough_of_my_husband']
    prompts = [{'id': name + suffix, 'paraphrases': ['prompt']}
               for suffix in ('', '_json', '_template') for name in names]
    yaml_path = tmp_path / 'questions.yaml'
    import yaml
    yaml_path.write_text(yaml.safe_dump(prompts))
    monkeypatch.setattr(e, 'QUALITATIVE_YAML_SHA256', e.sha256_file(yaml_path))
    ids = {'seed': 6304, 'source': 'baseline_logged_requests', 'entries': [
        {'benchmark': b, 'doc_id': i, 'prompt_sha256': 'x'}
        for b in ('gsm8k_cot', 'hellaswag') for i in range(32)]}
    e.atomic_json(output / 'activation' / 'probe_ids.json', ids)
    for c in e.TRAINED:
        e.atomic_json(output / 'weight' / (c + '.json'), {
            'weights': [{'condition': c, 'layer': layer, 'protection_rank': 128,
                         'rho_k': 0.1, 'eta': 0.2, 'delta_fro': 1, 'w0_fro': 5}
                        for layer in e.LAYERS],
            'displacement': {'displacement_l2': 1.0, 'unique_parameters': 10,
                             'unique_tensor_count': 2, 'parameter_names_sha256': 'x'}
            if c in ('full_mid', 'full_final', 'l2sp_final') else None})
        e.atomic_jsonl(output / 'activation' / (c + '.jsonl'), [
            {'condition': c, 'layer': layer, 'benchmark': b, 'doc_id': i,
             'prompt_sha256': 'x', 'r_z': 0.1}
            for layer in e.LAYERS for b in ('gsm8k_cot', 'hellaswag') for i in range(32)])
    for c in e.ALL:
        for item in prompts:
            e.atomic_json(output / 'qualitative' / c / (item['id'] + '.json'), {
                'condition': c, 'question_id': item['id'],
                'format': 'plain', 'response': 'test response', 'hit_512_token_cap': False})
        for b, spec in BENCHMARKS.items():
            e.atomic_json(artifacts / 'evaluation' / c / b / 'summary.json', {
                'status': 'COMPLETE_FULL_SPLIT', 'condition': c, 'benchmark': b,
                'examples': spec['count'], 'official_metric_key': spec['metric'],
                'primary_metric': 0.5})
            e.atomic_json(artifacts / 'evaluation' / c / b / 'harness_results.json', {
                'results': {b: {'acc,none': 0.4, 'acc_norm,none': 0.5}}})
    for method in ('full', 'l2sp', 'lora4', 'lora16', 'oplora16'):
        e.atomic_json(output / 'resources' / (method + '_final_qualitative_load.json'), {
            'load_seconds': 1.0, 'factor_export_seconds': 0.2, 'merge_seconds': 0.3})
        folder = artifacts / f'official-{method}-v1'
        folder.mkdir(parents=True)
        with (folder / 'attempts.jsonl').open('w') as f:
            for update in range(1, 1251):
                f.write(json.dumps({'event': 'optimizer_attempt', 'completed': True,
                                    'target_update': update, 'sft_loss': 0.2,
                                    'lr_used': 0.0001, 'omega': 0.0,
                                    'lambda_omega': 0.0, 'total_loss': 0.2,
                                    'elapsed_seconds': 1.0}) + '\n')
        e.atomic_json(folder / 'result_x.json', {
            'elapsed_seconds': 100, 'peak_gpu_memory_allocated_bytes': 10,
            'checkpoint_sizes_bytes': {'update_01250': 10}})
        e.atomic_json(folder / 'session_x.json', {
            'model_info': {'trainable_parameters': 10, 'total_parameters': 20,
                           'basis_preprocessing_seconds': 0},
            'model_setup_seconds': 1, 'setup_peak_gpu_memory_allocated_bytes': 10,
            'reference_creation_seconds': 0, 'reference_bytes': 0,
            'basis_storage_bytes': 0})
    result = e.export_collection(artifacts, output, yaml_path)
    assert result['status'] == 'COMPLETE'
    assert result['export_counts']['activation_rows'] == 1152
    assert len((output / 'export' / 'training_curves.csv').read_text().splitlines()) == 6251
    assert len((output / 'export' / 'qualitative_annotation_subset.csv').read_text().splitlines()) == 43
    with pytest.raises(FileExistsError):
        e.export_collection(artifacts, output, yaml_path)


def test_qualitative_tokenizer_batchencoding():
    from transformers.tokenization_utils_base import BatchEncoding
    from atml_pa3.task1.evidence import _qual_input_ids

    class DummyTokenizer:
        def __init__(self, output):
            self.output = output

        def apply_chat_template(self, messages, **kwargs):
            assert kwargs["enable_thinking"] is False
            assert kwargs["add_generation_prompt"] is True
            assert kwargs["return_tensors"] == "pt"
            return self.output

    expected = torch.tensor([[1, 2, 3]])

    for output in (
        expected,
        BatchEncoding({"input_ids": expected}),
    ):
        result = _qual_input_ids(DummyTokenizer(output), [], "cpu")
        torch.testing.assert_close(result, expected)

    with pytest.raises(ValueError):
        _qual_input_ids(
            DummyTokenizer(BatchEncoding({
                "input_ids": torch.tensor([1, 2, 3])
            })),
            [],
            "cpu",
        )
