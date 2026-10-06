"""Shared, fail-closed helpers for the hybrid retrieval examples."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import time

import numpy as np
from goodmem import Goodmem
from goodmem.models.space_key import SpaceKey
from goodmem.models.embedder_weight import EmbedderWeight

SCHEMA_VERSION = 2


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
    temp.replace(path)


def client():
    url = os.environ.get('GOODMEM_BASE_URL') or os.environ.get('GOODMEM_SERVER_URL')
    key = os.environ.get('GOODMEM_API_KEY')
    if not url or not key:
        raise ValueError('Set GOODMEM_BASE_URL (REST URL without /v1) and GOODMEM_API_KEY.')
    return Goodmem(base_url=url.rstrip('/'), api_key=key, timeout=120)


def validate_weights(weights, embedder_ids):
    if set(weights) != set(embedder_ids):
        raise ValueError('Weights must specify every space embedder ID exactly once.')
    if any(not math.isfinite(v) or v < 0 for v in weights.values()) or not any(weights.values()):
        raise ValueError('Weights must be finite, nonnegative, and not all zero.')


def ratio_weights(dense_id, sparse_id, ratio):
    if ratio == 'sparse':
        return {dense_id: 0.0, sparse_id: 1.0}
    return {dense_id: 1.0, sparse_id: float(ratio)}


def snapshot(api, space_id):
    space = api.spaces.get(id=space_id)
    models = []
    for item in space.space_embedders:
        model = api.embedders.get(id=item.embedder_id)
        # Credentials are deliberately not included in run artifacts.
        models.append({field: getattr(model, field) for field in (
            'embedder_id', 'model_identifier', 'dimensionality', 'distribution_type',
            'provider_type', 'endpoint_url', 'api_path')})
    return {'space_id': space_id, 'models': models,
            'server': api.system.info().model_dump(mode='json'),
            'base_url': os.environ.get('GOODMEM_BASE_URL') or os.environ.get('GOODMEM_SERVER_URL')}


def all_memories(api, space_id):
    # Iterating Page follows every next-token; .data alone would read only one page.
    return {m.memory_id: m for m in api.memories.list(space_id=space_id, page_size=500)}


def require_ready(api, space_id, expected_ids, *, timeout=0, poll=10):
    deadline = time.monotonic() + timeout
    expected = set(expected_ids)
    while True:
        memories = all_memories(api, space_id)
        if set(memories) != expected:
            raise RuntimeError(f'Corpus mismatch: {len(expected-set(memories))} missing, '
                               f'{len(set(memories)-expected)} unexpected memories.')
        counts = Counter(m.processing_status for m in memories.values())
        if counts.get('FAILED'):
            raise RuntimeError(f'Embedding failed: {dict(counts)}; inspect memory processing history.')
        if counts.get('COMPLETED', 0) == len(expected):
            return dict(counts)
        print(f'Processing: {dict(counts)}', flush=True)
        if time.monotonic() >= deadline:
            raise TimeoutError('Corpus is not fully processed; no evaluation was performed.')
        time.sleep(min(poll, max(0, deadline-time.monotonic())))


def retrieve(api, space_id, question, weights, top_k):
    ids, hits, began, ended = [], [], set(), set()
    events = api.memories.retrieve(
        message=question, space_keys=[SpaceKey(space_id=space_id, embedder_weights=[
            EmbedderWeight(embedder_id=eid, weight=w) for eid, w in sorted(weights.items())])],
        requested_size=top_k, fetch_memory=True, stream=False)
    for event in events:
        if event.status is not None:
            raise RuntimeError(f'Retrieval status {event.status.code}: {event.status.message}')
        if event.result_set_boundary is not None:
            boundary = event.result_set_boundary
            (began if boundary.kind == 'BEGIN' else ended).add(boundary.result_set_id)
        if event.retrieved_item is not None:
            item = event.retrieved_item
            if item.chunk is not None:
                mid = item.chunk.chunk.memory_id
                hit = {'memory_id': mid, 'score': item.chunk.relevance_score,
                       'text': item.chunk.chunk.chunk_text}
            else:
                mid = item.memory.memory_id
                hit = {'memory_id': mid, 'score': None, 'text': None}
            if mid not in ids:
                ids.append(mid)
                hits.append(hit)
    if not began or began != ended:
        raise RuntimeError('Incomplete retrieval stream; result set END was not received.')
    return hits[:top_k]


def metrics(rows, top_k):
    if not rows:
        raise ValueError('No questions to evaluate.')
    n = len(rows)
    result = {'questions': n, 'mrr': sum(r['rr'] for r in rows)/n,
              'coverage': sum(bool(r['hits']) for r in rows)/n}
    for k in sorted({1, 5, 10, top_k}):
        if k <= top_k:
            result[f'hit_at_{k}'] = sum(r['rank'] is not None and r['rank'] <= k for r in rows)/n
    return result


def evaluate(api, manifest, questions, weights, *, top_k=10, threads=4, cache_dir=None):
    if top_k < 1 or threads < 1:
        raise ValueError('top_k and threads must be positive.')
    validate_weights(weights, [m['embedder_id'] for m in manifest['installation']['models']])
    identity = {'schema': SCHEMA_VERSION, 'manifest': digest(manifest), 'questions': digest(questions),
                'weights': weights, 'top_k': top_k}
    path = Path(cache_dir)/f'{digest(identity)}.json' if cache_dir else None
    if path and path.exists():
        cached = read_json(path)
        if cached['identity'] != identity or cached['metrics']['questions'] != len(questions):
            raise ValueError('Invalid evaluation cache.')
        return cached

    def one(q):
        hits = retrieve(api, manifest['space_id'], q['question'], weights, top_k)
        relevant = set(q['relevant_memory_ids'])
        rank = next((i for i, hit in enumerate(hits, 1) if hit['memory_id'] in relevant), None)
        return {'question_id': q['question_id'], 'article': q['article'], 'question': q['question'],
                'relevant_memory_ids': sorted(relevant), 'rank': rank,
                'rr': 1/rank if rank else 0.0, 'hits': hits}

    # map preserves input order; inference/stream errors propagate and prevent success artifacts.
    with ThreadPoolExecutor(max_workers=threads) as pool:
        rows = list(pool.map(one, questions))
    rows.sort(key=lambda r: r['question_id'])
    result = {'identity': identity, 'metrics': metrics(rows, top_k), 'rows': rows}
    if path:
        write_json(path, result)
    return result


def paired_comparison(candidate, baseline, *, seed=42, iterations=10000):
    """Paired article-cluster bootstrap; no uncorrected post-search p-value claims."""
    a = {r['question_id']: r for r in candidate['rows']}
    b = {r['question_id']: r for r in baseline['rows']}
    if len(a) != len(candidate['rows']) or len(b) != len(baseline['rows']) or a.keys() != b.keys():
        raise ValueError('Paired comparison requires identical, unique question IDs.')
    groups = {}
    for qid in sorted(a):
        if a[qid]['article'] != b[qid]['article']:
            raise ValueError('Article mismatch for paired question.')
        groups.setdefault(a[qid]['article'], []).append(a[qid]['rr']-b[qid]['rr'])
    if len(groups) < 2:
        raise ValueError('At least two article clusters are needed for an interval.')
    sums = np.array([sum(v) for v in groups.values()])
    sizes = np.array([len(v) for v in groups.values()])
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(groups), size=(iterations, len(groups)))
    boot = sums[draws].sum(axis=1)/sizes[draws].sum(axis=1)
    ci = np.quantile(boot, [.025, .975]).tolist()
    delta = float(sums.sum()/sizes.sum())
    return {'mrr_difference': delta, 'ci95': ci, 'article_clusters': len(groups),
            'bootstrap_iterations': iterations, 'seed': seed,
            'evidence_of_improvement': ci[0] > 0,
            'method': 'paired article-cluster percentile bootstrap'}


def run_main(main):
    try:
        main()
    except (Exception, KeyboardInterrupt) as exc:
        raise SystemExit(f'FAILED: {exc}') from exc
