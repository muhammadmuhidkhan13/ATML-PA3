"""Stock harness benchmark execution with per-generation token-limit audit.

Importing this module does not import lm_eval, download a model or run inference.
"""
from __future__ import annotations
from collections import defaultdict, deque
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
import time
import uuid

from .conditions import BENCHMARKS, CONDITIONS
from .model import load_model_for_evaluation


def _token_hash(token_ids) -> str:
    return hashlib.sha256(json.dumps([int(i) for i in token_ids], separators=(',', ':')).encode()).hexdigest()


def _finish_token_count(generated, eos_token_ids: set[int], pad_id: int | None) -> tuple[int, bool]:
    """Determine first observed EOS and count generated tokens through EOS.

    If batch generation pads after a finished row, ignore trailing pad.
    """
    for i, token in enumerate(generated):
        if token in eos_token_ids:
            return i + 1, True
        if pad_id is not None and token == pad_id:
            return i + 1, True
    return len(generated), False


def _make_tracing_hflm():
    from lm_eval.models.huggingface import HFLM

    class TraceHFLM(HFLM):
        """Capture actual output token counts without changing official scorer."""

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._generated_by_prompt = defaultdict(deque)
            self.generation_trace = []

        def _model_generate(self, context, max_length, stop, **generation_kwargs):
            result = super()._model_generate(context=context, max_length=max_length,
                                             stop=stop, **generation_kwargs)
            masks = generation_kwargs.get('attention_mask')
            prompt_width = context.shape[1]
            cap = max_length - prompt_width
            eos = self.model.generation_config.eos_token_id
            eos_set = ({int(eos)} if isinstance(eos, int) else
                       {int(i) for i in eos} if eos is not None else set())
            pad = self.tokenizer.pad_token_id
            for i in range(result.shape[0]):
                input_ids = context[i].detach().cpu().tolist()
                if masks is not None:
                    attention_mask = masks[i].detach().cpu().tolist()
                    input_ids = [v for v, keep in zip(input_ids, attention_mask) if keep]
                output_ids = result[i, prompt_width:].detach().cpu().tolist()
                used, ended = _finish_token_count(output_ids, eos_set, pad)
                event = {
                    'prompt_token_hash': _token_hash(input_ids),
                    'generated_tokens': used,
                    'max_new_tokens': cap,
                    'hit_token_cap': used >= cap and not ended,
                    'observed_eos_or_pad': ended,
                    'stock_stop_strings': list(stop),
                }
                self._generated_by_prompt[event['prompt_token_hash']].append(event)
            return result

        def generate_until(self, requests, disable_tqdm=False):
            outputs = super().generate_until(requests, disable_tqdm=disable_tqdm)
            if len(outputs) != len(requests):
                raise RuntimeError('Harness returned an unexpected response count')
            for request, text in zip(requests, outputs):
                # HFLM's tok_encode and tok_batch_encode must be consistent.
                token_hash = _token_hash(self.tok_encode(request.args[0]))
                available = self._generated_by_prompt[token_hash]
                if not available:
                    raise RuntimeError('Cannot associate generation token audit with harness prompt')
                event = available.popleft()
                event['generated_text'] = text
                self.generation_trace.append(event)
            return outputs

    return TraceHFLM


def run_benchmark(*, condition: str, benchmark: str, output_dir: Path,
                  device: str = 'cuda:0', checkpoint: Path | None = None,
                  export: Path | None = None, limit: int | None = None,
                  batch_size: int = 1) -> dict:
    """Evaluate ONE benchmark. Only callable after explicit CLI execution.

    `limit` is a smoke-test option; any such run is marked nonofficial.
    """
    if condition not in CONDITIONS or benchmark not in BENCHMARKS:
        raise ValueError('Unknown condition or benchmark')
    if limit is not None and (limit < 1 or isinstance(limit, bool)):
        raise ValueError('limit must be a positive integer')
    if batch_size < 1:
        raise ValueError('batch_size must be positive')
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f'Refusing to overwrite evaluation: {output_dir}')
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_dir.parent / f'.incomplete-{output_dir.name}-{uuid.uuid4().hex}'
    temporary.mkdir()

    model = None
    try:
        import torch
        import lm_eval
        started = time.perf_counter()
        is_adapter = CONDITIONS[condition].method in ('lora4', 'lora16', 'oplora16')
        model, tokenizer, model_manifest = load_model_for_evaluation(
            condition, device=device, checkpoint=checkpoint, export=export,
            merge_adapters=not is_adapter)
        if is_adapter:
            from .model import verify_merge_logits
            # Verify the real, restored, unmerged model before scoring. This
            # checks model-wide logits, not merely an isolated linear layer.
            tokens = tokenizer('What is two plus two?', return_tensors='pt')['input_ids'].to(device)
            model_manifest['merge_verification'] = verify_merge_logits(model, tokens)
            model_manifest['adapters_merged'] = True
            del tokens
        if device.startswith('cuda'):
            torch.cuda.reset_peak_memory_stats(device)
        TraceHFLM = _make_tracing_hflm()
        lm = TraceHFLM(pretrained=model, tokenizer=tokenizer,
                       batch_size=batch_size, device=device,
                       enable_thinking=False,
                       chat_template_args={'enable_thinking': False})
        is_gsm = benchmark == 'gsm8k_cot'
        # `gen_kwargs` updates the stock task's generation_kwargs rather than
        # replacing its built-in stop strings and answer-filter definitions.
        kwargs = dict(
            model=lm, tasks=[benchmark], num_fewshot=0, batch_size=batch_size,
            device=device, apply_chat_template=is_gsm,
            log_samples=True, limit=limit,
            random_seed=6304, numpy_random_seed=6304,
            torch_random_seed=6304, fewshot_random_seed=6304,
            bootstrap_iters=0,
        )
        if is_gsm:
            kwargs['gen_kwargs'] = {'max_gen_toks': 1024, 'do_sample': False}
        result = lm_eval.simple_evaluate(**kwargs)
        if result is None:
            raise RuntimeError('Harness returned no evaluation results')
        samples = result.get('samples', {}).get(benchmark)
        if not isinstance(samples, list):
            raise RuntimeError('Harness did not return per-example samples')
        expected = limit if limit is not None else BENCHMARKS[benchmark]['count']
        raw_samples = samples
        if is_gsm:
            # lm-eval may log multiple filter/metric rows per GSM8K document.
            # Count distinct documents while preserving all raw metric rows.
            by_doc = {}
            for sample in raw_samples:
                doc_id = sample.get('doc_id')
                if doc_id is None:
                    raise ValueError('GSM8K sample missing doc_id')
                by_doc.setdefault(doc_id, sample)
            samples = list(by_doc.values())
        if len(samples) != expected:
            raise ValueError(
                f'Unexpected {benchmark} unique sample count: '
                f'{len(samples)}, expected {expected}'
            )
        if benchmark not in result.get('results', {}):
            raise ValueError('Missing aggregate benchmark metrics')
        expected_metric = BENCHMARKS[benchmark]['metric']
        if expected_metric not in result['results'][benchmark]:
            raise ValueError(f'Missing official primary metric: {expected_metric}')
        if is_gsm and len(lm.generation_trace) != len(samples):
            raise RuntimeError('GSM8K token audit count does not match evaluated examples')

        for s in samples:
            if 'doc_id' not in s and 'doc' not in s:
                raise ValueError('Sample output lacks document identity')
        with (temporary / 'samples.jsonl').open('w', encoding='utf-8') as f:
            for sample in samples:
                f.write(json.dumps(sample, sort_keys=True, default=str, ensure_ascii=False) + '\n')
        if is_gsm:
            with (temporary / 'metric_filter_samples.jsonl').open('w', encoding='utf-8') as f:
                for sample in raw_samples:
                    f.write(json.dumps(sample, sort_keys=True, default=str, ensure_ascii=False) + '\n')
            with (temporary / 'generation_token_audit.jsonl').open('w', encoding='utf-8') as f:
                for record in lm.generation_trace:
                    f.write(json.dumps(record, sort_keys=True, ensure_ascii=False) + '\n')

        # Samples are stored separately to avoid duplicating potentially large outputs.
        aggregate = {key: value for key, value in result.items() if key != 'samples'}
        (temporary / 'harness_results.json').write_text(
            json.dumps(aggregate, sort_keys=True, default=str, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        summary = {
            'schema_version': 1,
            'status': 'SMOKE_ONLY' if limit is not None else 'COMPLETE_FULL_SPLIT',
            'condition': condition, 'benchmark': benchmark,
            'examples': len(samples), 'expected_full_split': BENCHMARKS[benchmark]['count'],
            'official_metric_key': expected_metric,
            'primary_metric': result['results'][benchmark][expected_metric],
            'device': device, 'batch_size': batch_size,
            'num_fewshot': 0, 'apply_chat_template': is_gsm,
            'enable_thinking': False,
            'gsm8k_max_new_tokens': 1024 if is_gsm else None,
            'model': model_manifest,
            'lm_eval_version': importlib.metadata.version('lm-eval'),
            'torch_version': torch.__version__,
            'python_version': platform.python_version(),
            'timestamp_utc': datetime.now(timezone.utc).isoformat(),
            'elapsed_seconds': time.perf_counter() - started,
            'peak_gpu_memory_allocated_bytes': (torch.cuda.max_memory_allocated(device)
                if device.startswith('cuda') else None),
        }
        (temporary / 'summary.json').write_text(
            json.dumps(summary, indent=2, sort_keys=True, default=str) + '\n', encoding='utf-8')
        temporary.rename(output_dir)
        return summary
    except BaseException:
        # Keep failure details in caller's stderr; no partial output is published.
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    finally:
        if model is not None:
            del model
