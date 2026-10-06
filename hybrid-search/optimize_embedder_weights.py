#!/usr/bin/env python3
"""Tune ratios on one split, freeze the choice, then compare on unseen articles."""
import argparse
import json
import math
from pathlib import Path

import numpy as np
from hybrid_common import (client, digest, evaluate, paired_comparison, ratio_weights,
                           read_json, require_ready, run_main, snapshot, write_json)

DEFAULT_RATIOS = '0,0.00001,0.00003,0.0001,0.0003,0.001,0.003,0.005,0.01,0.03,0.1,0.3,1,3,10,30,100,sparse'


def parse_ratios(text):
    values = []
    for entry in text.split(','):
        entry = entry.strip()
        value = 'sparse' if entry == 'sparse' else float(entry)
        if value != 'sparse' and (not math.isfinite(value) or value < 0):
            raise ValueError('Ratios must be finite and nonnegative, or sparse.')
        if value not in values:
            values.append(value)
    # Baselines are always eligible to win; both must be evaluated.
    for value in (0.0, 'sparse'):
        if value not in values:
            values.append(value)
    return values


def best_ratio(results):
    # Prefer a simpler pure model when scores tie, then the smaller sparse ratio.
    return max(results, key=lambda key: (results[key]['metrics']['mrr'],
        key in ('0', 'sparse'), -float(key) if key != 'sparse' else -math.inf))


def ratio_key(ratio):
    return 'sparse' if ratio == 'sparse' else format(float(ratio), '.12g')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, default=Path('run/manifest.json'))
    p.add_argument('--output-dir', type=Path, default=Path('run/search'))
    p.add_argument('--ratios', default=DEFAULT_RATIOS, help='Sparse coefficient with dense fixed at 1; sparse means 0:1')
    p.add_argument('--fine-points', type=int, default=9)
    p.add_argument('--top-k', type=int, default=10)
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--bootstrap-iterations', type=int, default=10000)
    args = p.parse_args()
    if args.fine_points < 0 or args.bootstrap_iterations < 100:
        p.error('Use nonnegative fine-points and at least 100 bootstrap iterations.')
    manifest = read_json(args.manifest)
    tune, test = manifest['questions']['tune'], manifest['questions']['test']
    if {q['question_id'] for q in tune} & {q['question_id'] for q in test}:
        raise ValueError('Tuning/test question IDs overlap.')
    if {q['article'] for q in tune} & {q['article'] for q in test}:
        raise ValueError('Tuning/test articles overlap.')
    models = manifest['installation']['models']
    dense = [m['embedder_id'] for m in models if m['distribution_type'] == 'DENSE']
    sparse = [m['embedder_id'] for m in models if m['distribution_type'] == 'SPARSE']
    if len(models) != 2 or len(dense) != 1 or len(sparse) != 1:
        raise ValueError('This ratio search requires exactly one dense and one sparse embedder.')
    coarse = parse_ratios(args.ratios)
    identity = {'manifest_sha256': digest(manifest), 'coarse_ratios': coarse,
                'fine_points': args.fine_points, 'top_k': args.top_k}
    selection_file = args.output_dir/'selection.json'
    cache = args.output_dir/'evaluations'
    with client() as api:
        if snapshot(api, manifest['space_id']) != manifest['installation']:
            raise ValueError('Server/model configuration changed; use a new manifest.')
        require_ready(api, manifest['space_id'], [m['memory_id'] for m in manifest['corpus']])
        if selection_file.exists():
            selection = read_json(selection_file)
            if selection['identity'] != identity:
                raise ValueError('Search differs from frozen selection; use a new output directory.')
        else:
            results = {}
            def assess(ratio):
                key = ratio_key(ratio)
                if key not in results:
                    results[key] = evaluate(api, manifest, tune,
                        ratio_weights(dense[0], sparse[0], ratio), top_k=args.top_k,
                        threads=args.threads, cache_dir=cache)
                    print(f'TUNE 1:{key}: {results[key]["metrics"]}', flush=True)
            for ratio in coarse:
                assess(ratio)
            winner = best_ratio(results)
            positives = sorted(float(k) for k in results if k not in ('0', 'sparse'))
            # One prespecified logarithmic refinement; never inspect held-out scores to tune.
            if args.fine_points and positives:
                if winner == '0':
                    lo, hi = positives[0]/100, positives[0]
                elif winner == 'sparse':
                    lo, hi = positives[-1], positives[-1]*100
                else:
                    index = positives.index(float(winner))
                    lo = positives[index-1] if index else positives[0]/10
                    hi = positives[index+1] if index+1 < len(positives) else positives[-1]*10
                for ratio in np.geomspace(lo, hi, args.fine_points):
                    assess(float(ratio_key(ratio)))
            winner = best_ratio(results)
            selection = {'identity': identity, 'selected_ratio': winner,
                         'weights': ratio_weights(dense[0], sparse[0], winner),
                         'tuning': {k: v['metrics'] for k, v in results.items()}}
            write_json(selection_file, selection)
            print(f'Frozen tuning choice: {winner}; now evaluating held-out articles.', flush=True)
        heldout = {}
        for ratio in dict.fromkeys([selection['selected_ratio'], '0', 'sparse', '0.005', '1']):
            heldout[ratio] = evaluate(api, manifest, test,
                ratio_weights(dense[0], sparse[0], ratio), top_k=args.top_k,
                threads=args.threads, cache_dir=cache)
            print(f'TEST 1:{ratio}: {heldout[ratio]["metrics"]}', flush=True)
        comparisons = {baseline: paired_comparison(heldout[selection['selected_ratio']], heldout[baseline],
            seed=manifest['seed'], iterations=args.bootstrap_iterations) for baseline in ('0', 'sparse', '0.005')}
        supported = all(comparisons[b]['evidence_of_improvement'] for b in ('0', 'sparse'))
        summary = {'selection': selection, 'test': {k: v['metrics'] for k, v in heldout.items()},
                   'comparisons': comparisons,
                   'conclusion': ('Held-out intervals support an improvement over both single-model scoring baselines.' if supported else
                     'Held-out intervals do not establish improvement over both single-model scoring baselines. Test a single-embedder deployment separately before drawing cost or latency conclusions.'),
                   'scope': 'Best tested ratio for this corpus, split, model pair, server, and retrieval settings; not a global optimum.'}
        write_json(args.output_dir/'summary.json', summary)
        print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    run_main(main)
