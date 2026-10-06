#!/usr/bin/env python3
"""Ingest a small corpus and compare real dense, sparse, and hybrid retrieval."""
import argparse
from pathlib import Path
import uuid

from goodmem import MemoryCreationRequest
from hybrid_common import all_memories, client, read_json, require_ready, retrieve, run_main

TEXTS = [
    'Solar panels convert sunlight into electricity using photovoltaic cells.',
    'Wind turbines generate electrical power from the movement of air.',
    'A photovoltaic inverter converts direct current from solar panels into alternating current.',
    'Lithium-ion batteries store energy for use when solar generation is low.',
    'Hydroelectric plants generate power using water flowing through turbines.',
    'A heat pump transfers thermal energy between indoor and outdoor spaces.',
]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--resources', type=Path, default=Path('run/resources.json'))
    p.add_argument('--ratio', type=float, default=0.168702397557, help='Sparse coefficient; compare before adopting')
    p.add_argument('--query', default='Which device changes the electrical output of photovoltaic cells to AC?')
    args = p.parse_args()
    state = read_json(args.resources)
    ids = [str(uuid.uuid5(uuid.UUID(state['space_id']), 'demo:'+t)) for t in TEXTS]
    with client() as api:
        existing = all_memories(api, state['space_id'])
        if set(existing)-set(ids):
            raise ValueError('Use a separate space for the demo and the SQuAD benchmark.')
        for mid, text in zip(ids, TEXTS):
            if mid not in existing:
                api.memories.create(space_id=state['space_id'], memory_id=mid,
                    original_content=text, chunking_config={'none': {}})
        require_ready(api, state['space_id'], ids, timeout=300)
        for label, dw, sw in [('dense', 1, 0), ('sparse', 0, 1), ('hybrid', 1, args.ratio)]:
            weights = {state['dense']: dw, state['sparse']: sw}
            from hybrid_common import validate_weights
            validate_weights(weights, [state['dense'], state['sparse']])
            print(f'\n{label} ({dw}:{sw})')
            for i, hit in enumerate(retrieve(api, state['space_id'], args.query, weights, 3), 1):
                print(f'{i}. {hit["text"]} [score={hit["score"]}]')


if __name__ == '__main__':
    run_main(main)
