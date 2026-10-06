#!/usr/bin/env python3
"""Inspect real retrieval misses through GoodMem; no direct database/model access."""
import argparse
from pathlib import Path

from hybrid_common import client, read_json, retrieve, run_main, validate_weights, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, default=Path('run/manifest.json'))
    p.add_argument('--evaluation', type=Path, required=True, help='Exact JSON output of the evaluator')
    p.add_argument('--limit', type=int, default=10)
    p.add_argument('--search-size', type=int, default=100)
    p.add_argument('--output', type=Path, default=Path('run/misses.json'))
    args = p.parse_args()
    manifest, evaluation = read_json(args.manifest), read_json(args.evaluation)
    from hybrid_common import digest
    if evaluation['identity']['manifest'] != digest(manifest):
        raise ValueError('Evaluation belongs to another corpus or configuration.')
    weights = evaluation['identity']['weights']
    validate_weights(weights, [m['embedder_id'] for m in manifest['installation']['models']])
    if args.limit < 1 or args.search_size < evaluation['identity']['top_k']:
        p.error('Use a positive limit and search-size at least as large as evaluated top-k.')
    misses = [r for r in evaluation['rows'] if r['rank'] is None][:args.limit]
    rows = []
    with client() as api:
        for row in misses:
            variants = {'hybrid': weights}
            for eid in weights:
                variants[eid] = {other: float(other == eid) for other in weights}
            queries = {name: retrieve(api, manifest['space_id'], row['question'], w, args.search_size)
                       for name, w in variants.items()}
            relevant = set(row['relevant_memory_ids'])
            rows.append({'question_id': row['question_id'], 'question': row['question'],
                'relevant_memory_ids': sorted(relevant),
                'retrievals': {name: {'first_relevant_rank': next((i for i, h in enumerate(hits, 1)
                    if h['memory_id'] in relevant), None), 'hits': hits} for name, hits in queries.items()}})
    write_json(args.output, {'search_size': args.search_size, 'rows': rows,
        'note': 'Scores are provider-specific weighted retrieval scores. Absence from these candidates is not proof of absence from the corpus. Increasing search size can change the candidate set and ranks.'})
    print(f'Analyzed {len(rows)} misses. Results: {args.output}')


if __name__ == '__main__':
    run_main(main)
