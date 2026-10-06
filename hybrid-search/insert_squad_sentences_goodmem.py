#!/usr/bin/env python3
"""Prepare a reproducible SQuAD sentence corpus and ingest it with the maintained SDK."""
import argparse
import hashlib
from pathlib import Path
import random
import uuid

from goodmem import MemoryCreationRequest

import sb_sed
from hybrid_common import (SCHEMA_VERSION, all_memories, client, digest, read_json,
                           require_ready, run_main, snapshot, write_json)


def prepare(dataset, space_id, *, seed=42, sample_size=1000, limit=None):
    raw = Path(dataset).read_bytes()
    data = read_json(dataset)
    checksum = hashlib.sha256(raw).hexdigest()
    corpus, questions = [], []
    skipped = 0
    titles = [a['title'] for a in data['data']]
    if len(titles) != len(set(titles)) or len(titles) < 4:
        raise ValueError('Dataset must contain at least four uniquely named articles.')
    shuffled = sorted(titles)
    random.Random(seed).shuffle(shuffled)
    tuning_articles = set(shuffled[:len(shuffled)//2])
    for ai, article in enumerate(data['data']):
        for pi, paragraph in enumerate(article['paragraphs']):
            text = paragraph['context']
            spans = []
            for si, (start, end) in enumerate(sb_sed.infer_sentence_breaks(text)):
                sentence = text[start:end].strip()
                if not sentence:
                    continue
                sid = f'{ai}:{pi}:{si}'
                mid = str(uuid.uuid5(uuid.UUID(space_id), f'{checksum}:{sid}'))
                spans.append((start, end, mid))
                corpus.append({'memory_id': mid, 'text': sentence, 'article': article['title'],
                               'sentence_id': sid, 'paragraph_id': f'{ai}:{pi}'})
            for qa in paragraph['qas']:
                relevant = set()
                for answer in qa['answers']:
                    start = answer['answer_start']; end = start + len(answer['text'])
                    if text[start:end] != answer['text']:
                        raise ValueError(f'Invalid answer span in question {qa["id"]}')
                    relevant.update(mid for lo, hi, mid in spans if lo <= start and end <= hi)
                if not relevant:
                    skipped += 1
                    continue
                questions.append({'question_id': qa['id'], 'question': qa['question'],
                    'article': article['title'], 'relevant_memory_ids': sorted(relevant),
                    'split': 'tune' if article['title'] in tuning_articles else 'test'})
    if limit:
        # A diagnostic subset spread across articles, not the first article only.
        corpus = random.Random(seed).sample(corpus, min(limit, len(corpus)))
        corpus.sort(key=lambda m: m['sentence_id'])
        kept = {m['memory_id'] for m in corpus}
        questions = [q for q in questions if set(q['relevant_memory_ids']) <= kept]
    selected = {}
    for split in ('tune', 'test'):
        pool = sorted([q for q in questions if q['split'] == split], key=lambda q: q['question_id'])
        if sample_size > len(pool):
            raise ValueError(f'{split} has only {len(pool)} eligible questions; reduce --sample-size.')
        selected[split] = sorted(random.Random(seed+1).sample(pool, sample_size),
                                 key=lambda q: q['question_id'])
    return {'schema': SCHEMA_VERSION, 'space_id': space_id, 'dataset_sha256': checksum,
            'dataset_file': Path(dataset).name, 'seed': seed, 'corpus': corpus,
            'corpus_sha256': digest(corpus), 'questions': selected,
            'statistics': {'articles': len(titles), 'sentences': len(corpus),
                           'eligible_questions': len(questions), 'unmapped_questions': skipped}}


def ingest(api, manifest, *, batch_size=100, timeout=3600):
    expected = {m['memory_id']: m for m in manifest['corpus']}
    existing = all_memories(api, manifest['space_id'])
    if set(existing)-set(expected):
        raise ValueError('Use a dedicated space: it contains memories outside this manifest.')
    for mid, memory in existing.items():
        if (memory.metadata or {}).get('corpus_sha256') != manifest['corpus_sha256']:
            raise ValueError(f'Existing memory {mid} belongs to a different corpus.')
    pending = [m for mid, m in expected.items() if mid not in existing]
    for start in range(0, len(pending), batch_size):
        batch = pending[start:start+batch_size]
        response = api.memories.batch_create(requests=[MemoryCreationRequest(**{
            'space_id': manifest['space_id'], 'memory_id': m['memory_id'],
            'original_content': m['text'], 'content_type': 'text/plain',
            'chunking_config': {'none': {}},
            'metadata': {'source': 'squad_1.1', 'article': m['article'],
                         'sentence_id': m['sentence_id'], 'paragraph_id': m['paragraph_id'],
                         'corpus_sha256': manifest['corpus_sha256']}}) for m in batch])
        returned = [r.memory.memory_id for r in response.results if r.success and r.memory]
        if len(response.results) != len(batch) or set(returned) != {m['memory_id'] for m in batch}:
            raise RuntimeError('Batch creation failed partially; rerun to resume by stable memory ID.')
        print(f'Created {min(start+batch_size,len(pending))}/{len(pending)} new memories', flush=True)
    return require_ready(api, manifest['space_id'], expected, timeout=timeout)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--space-id', required=True)
    p.add_argument('--dataset', type=Path, default=Path('squad_data/dev-v1.1.json'))
    p.add_argument('--output', type=Path, default=Path('run/manifest.json'))
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--sample-size', type=int, default=1000, help='Questions per article-disjoint split')
    p.add_argument('--limit', type=int, help='Diagnostic sentence subset; omit for benchmark')
    p.add_argument('--batch-size', type=int, default=100)
    p.add_argument('--timeout', type=int, default=3600)
    p.add_argument('--dry-run', action='store_true', help='Prepare locally; no server calls')
    args = p.parse_args()
    if args.sample_size < 1 or not 1 <= args.batch_size <= 500 or args.timeout < 0 or (args.limit is not None and args.limit < 1):
        p.error('Invalid sample, limit, batch size, or timeout.')
    manifest = prepare(args.dataset, args.space_id, seed=args.seed,
                       sample_size=args.sample_size, limit=args.limit)
    print(manifest['statistics'], flush=True)
    if args.dry_run:
        print('Dry run succeeded; no server data or manifest was written.')
        return
    with client() as api:
        manifest['installation'] = snapshot(api, args.space_id)
        if args.output.exists() and read_json(args.output) != manifest:
            raise ValueError('Manifest differs from the existing run; use a new output directory/space.')
        write_json(args.output, manifest)
        print(ingest(api, manifest, batch_size=args.batch_size, timeout=args.timeout), flush=True)
    print(f'Ready. Manifest: {args.output}')


if __name__ == '__main__':
    run_main(main)
