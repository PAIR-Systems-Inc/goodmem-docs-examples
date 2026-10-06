"""Regression coverage for failures found by executing the old guide."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest
import uuid

from hybrid_common import (evaluate, metrics, paired_comparison, require_ready,
                           retrieve, validate_weights)
from insert_squad_sentences_goodmem import ingest, prepare
from optimize_embedder_weights import best_ratio, parse_ratios


def event(**kwargs):
    return NS(status=None, result_set_boundary=None, retrieved_item=None, **kwargs)


def boundary(kind):
    e = event(); e.result_set_boundary = NS(kind=kind, result_set_id='r'); return e


def chunk(mid):
    e = event(); e.retrieved_item = NS(chunk=NS(chunk=NS(memory_id=mid, chunk_text='text'),
                                               relevance_score=-1.2)); return e


class FakeAPI:
    def __init__(self, events=None, memories=None):
        self.events = events if events is not None else [boundary('BEGIN'), chunk('m'), boundary('END')]
        self.stored = memories or []
        self.memories = self
        self.calls = 0

    def retrieve(self, **kwargs):
        self.calls += 1
        return self.events

    def list(self, **kwargs):
        # Like SDK Page: .data is one page but iteration crosses pages.
        stored = self.stored
        class Page:
            data = stored[:50]
            def __iter__(self):
                return iter(stored)
        return Page()


class PipelineTests(unittest.TestCase):
    def test_coverage_is_any_result_not_answer_found(self):
        r = metrics([{'rr': 0, 'rank': None, 'hits': ['wrong']},
                     {'rr': 0, 'rank': None, 'hits': []}], 5)
        self.assertEqual(r['coverage'], .5)
        self.assertEqual(r['hit_at_5'], 0)
        self.assertNotIn('hit_at_10', r)

    def test_invalid_weights_never_silently_use_defaults(self):
        for weights in ({'placeholder': 1}, {'d': 0, 's': 0}, {'d': float('nan'), 's': 1},
                        {'d': -1, 's': 1}):
            with self.assertRaises(ValueError):
                validate_weights(weights, ['d', 's'])

    def test_paginated_readiness_checks_every_memory(self):
        stored = [NS(memory_id=str(i), processing_status='COMPLETED') for i in range(150)]
        stored[-1].processing_status = 'PENDING'
        with redirect_stdout(io.StringIO()), self.assertRaises(TimeoutError):
            require_ready(FakeAPI(memories=stored), 's', [str(i) for i in range(150)])
        stored[-1].processing_status = 'COMPLETED'
        self.assertEqual(require_ready(FakeAPI(memories=stored), 's', [str(i) for i in range(150)]),
                         {'COMPLETED': 150})

    def test_failed_or_missing_memories_prevent_evaluation(self):
        for stored in ([], [NS(memory_id='m', processing_status='FAILED')]):
            with self.assertRaises(RuntimeError):
                require_ready(FakeAPI(memories=stored), 's', ['m'])

    def test_status_and_truncated_streams_are_errors(self):
        warning = event(); warning.status = NS(code='EMBEDDER_FAILED', message='provider failed')
        for events in ([warning], [boundary('BEGIN'), chunk('m')]):
            with self.assertRaises(RuntimeError):
                retrieve(FakeAPI(events=events), 's', 'q', {'d': 1}, 10)

    def test_hits_use_actual_memory_ids_and_deduplicate(self):
        hits = retrieve(FakeAPI(events=[boundary('BEGIN'), chunk('m'), chunk('m'), boundary('END')]),
                        's', 'q', {'d': 1}, 10)
        self.assertEqual([h['memory_id'] for h in hits], ['m'])

    def test_cache_keys_include_questions_weights_and_top_k(self):
        api = FakeAPI()
        manifest = {'space_id': 's', 'installation': {'models': [{'embedder_id': 'd'}]}}
        questions = [{'question_id': 'q', 'article': 'a', 'question': 'text', 'relevant_memory_ids': ['m']}]
        with tempfile.TemporaryDirectory() as temp:
            for k in (5, 5, 10):
                result = evaluate(api, manifest, questions, {'d': 1}, top_k=k, cache_dir=temp)
                self.assertEqual(result['metrics']['mrr'], 1)
            self.assertEqual(api.calls, 2)
            evaluate(api, manifest, questions, {'d': 2}, cache_dir=temp)
            self.assertEqual(api.calls, 3)

    def test_failed_evaluation_writes_no_success_cache(self):
        manifest = {'space_id': 's', 'installation': {'models': [{'embedder_id': 'd'}]}}
        questions = [{'question_id': 'q', 'article': 'a', 'question': 'text', 'relevant_memory_ids': ['m']}]
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(RuntimeError):
                evaluate(FakeAPI(events=[]), manifest, questions, {'d': 1}, cache_dir=temp)
            self.assertEqual(list(Path(temp).iterdir()), [])

    def test_paired_bootstrap_joins_ids_not_completion_order(self):
        rows = [{'question_id': str(i), 'article': str(i//2), 'rr': (i%3)/2} for i in range(10)]
        result = paired_comparison({'rows': rows}, {'rows': list(reversed(rows))}, iterations=100)
        self.assertEqual(result['mrr_difference'], 0)
        self.assertEqual(result['ci95'], [0, 0])
        self.assertFalse(result['evidence_of_improvement'])
        with self.assertRaises(ValueError):
            paired_comparison({'rows': rows}, {'rows': rows[:-1]})

    def test_pure_baselines_can_win_and_ties_prefer_simplicity(self):
        self.assertIn(0., parse_ratios('.005'))
        self.assertIn('sparse', parse_ratios('.005'))
        results = {'0.005': {'metrics': {'mrr': .6}}, '0': {'metrics': {'mrr': .6}}}
        self.assertEqual(best_ratio(results), '0')

    def test_prepare_splits_articles_and_preserves_all_valid_answers(self):
        data = {'data': [{'title': str(i), 'paragraphs': [{'context': 'Alpha is first. Beta is second.',
                 'qas': [{'id': str(i), 'question': 'Which word?', 'answers': [
                     {'text': 'Alpha', 'answer_start': 0}, {'text': 'Beta', 'answer_start': 16}]}]}]}
                        for i in range(4)]}
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'data.json'; path.write_text(json.dumps(data))
            sid = str(uuid.uuid4())
            m = prepare(path, sid, sample_size=2)
            self.assertEqual(m, prepare(path, sid, sample_size=2))
            self.assertFalse({q['article'] for q in m['questions']['tune']} &
                             {q['article'] for q in m['questions']['test']})
            self.assertTrue(all(len(q['relevant_memory_ids']) == 2 for qs in m['questions'].values() for q in qs))

    def test_partial_batch_failure_is_not_success(self):
        api = FakeAPI()
        api.batch_create = lambda **kwargs: NS(results=[NS(success=False, memory=None)])
        manifest = {'space_id': str(uuid.uuid4()), 'corpus_sha256': 'hash', 'corpus': [
            {'memory_id': str(uuid.uuid4()), 'text': 'text', 'article': 'a', 'sentence_id': 's', 'paragraph_id': 'p'}]}
        with self.assertRaises(RuntimeError):
            ingest(api, manifest)


if __name__ == '__main__':
    unittest.main()
