#!/usr/bin/env python3
"""Initialize the local example and register the two models and an unchunked space."""
import argparse
import os
from pathlib import Path
import shlex
import time

from goodmem import Goodmem
from hybrid_common import read_json, run_main, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir', type=Path, default=Path('run'))
    p.add_argument('--dense-id', help='Reuse an existing dense embedder')
    p.add_argument('--sparse-id', help='Reuse an existing sparse embedder')
    p.add_argument('--name', default='MiniLM + SPLADE example')
    p.add_argument('--dense-endpoint', default='http://dense:80')
    p.add_argument('--sparse-endpoint', default='http://sparse:80')
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    url = os.environ.get('GOODMEM_BASE_URL', 'http://127.0.0.1:18080')
    key = os.environ.get('GOODMEM_API_KEY')
    creds = args.output_dir/'credentials.env'
    if not key and creds.exists():
        raise ValueError(f'Load existing credentials first: source {creds}')
    with Goodmem(base_url=url, api_key=key, timeout=120) as api:
        for attempt in range(60):
            try:
                api.system.info()
                break
            except Exception:
                if attempt == 59:
                    raise
                time.sleep(2)
        if not key:
            response = api.system.init()
            if not response.root_api_key:
                raise ValueError('Server is already initialized. Set its GOODMEM_API_KEY.')
            key = response.root_api_key
            fd = os.open(creds, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, 'w') as f:
                f.write(f'export GOODMEM_BASE_URL={shlex.quote(url)}\nexport GOODMEM_API_KEY={shlex.quote(key)}\n')
    resource_file = args.output_dir/'resources.json'
    state = read_json(resource_file) if resource_file.exists() else {'base_url': url}
    if state['base_url'] != url:
        raise ValueError('Output directory belongs to another server.')
    with Goodmem(base_url=url, api_key=key, timeout=120) as api:
        for role, endpoint, model, dimension, distribution in (
            ('dense', args.dense_endpoint, 'all-MiniLM-L6-v2', 384, 'DENSE'),
            ('sparse', args.sparse_endpoint, 'splade-cocondenser-ensembledistil', 30522, 'SPARSE')):
            if role not in state and getattr(args, role+'_id'):
                existing = api.embedders.get(id=getattr(args, role+'_id'))
                if existing.distribution_type != distribution or existing.dimensionality != dimension:
                    raise ValueError(f'Existing {role} embedder has incompatible dimensions/distribution.')
                state[role] = existing.embedder_id
                write_json(resource_file, state)
            if role not in state:
                embedder = api.embedders.create(display_name=model, provider_type='TEI',
                    endpoint_url=endpoint, model_identifier=model, dimensionality=dimension,
                    distribution_type=distribution)
                state[role] = embedder.embedder_id
                write_json(resource_file, state)
        if 'space_id' not in state:
            space = api.spaces.create(name=args.name,
                space_embedders=[{'embedder_id': state[role]} for role in ('dense', 'sparse')],
                default_chunking_config={'none': {}})
            state['space_id'] = space.space_id
            write_json(resource_file, state)
    print(f'Resources: {resource_file}\nSpace: {state["space_id"]}')
    if creds.exists():
        print(f'Load credentials: source {creds}')
    print('Wait for both TEI model /health endpoints before ingesting.')


if __name__ == '__main__':
    run_main(main)
