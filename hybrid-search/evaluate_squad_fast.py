#!/usr/bin/env python3
"""Evaluate a frozen question split; inference errors stop the run with a nonzero exit."""
import argparse
import json
from pathlib import Path

from hybrid_common import (client, evaluate, read_json, require_ready, run_main,
                           snapshot, write_json)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, default=Path('run/manifest.json'))
    p.add_argument('--split', choices=['tune', 'test'], default='tune')
    p.add_argument('--custom-weights', required=True, help='JSON object with every embedder ID')
    p.add_argument('--top-k', type=int, default=10)
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    manifest = read_json(args.manifest)
    with client() as api:
        if snapshot(api, manifest['space_id']) != manifest['installation']:
            raise ValueError('Server/model configuration changed; create a new benchmark run.')
        require_ready(api, manifest['space_id'], [m['memory_id'] for m in manifest['corpus']])
        result = evaluate(api, manifest, manifest['questions'][args.split],
                          json.loads(args.custom_weights), top_k=args.top_k, threads=args.threads)
    write_json(args.output, result)
    missing = [r for r in result['rows'] if r['rank'] is None]
    missing_file = args.output.with_name(args.output.stem + '.missing.json')
    write_json(missing_file, missing)
    print(json.dumps(result['metrics'], indent=2))
    print(f'Results: {args.output}\nMissing answers: {missing_file}')


if __name__ == '__main__':
    run_main(main)
