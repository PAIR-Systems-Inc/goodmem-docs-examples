# Hybrid retrieval examples

Runnable examples for [Build and tune hybrid search](https://docs.goodmem.ai/docs/how-to/hybrid-search/).
Use the latest GoodMem release and the `goodmem` Python SDK to combine MiniLM
and SPLADE, compare their rankings, and tune weights on SQuAD. The benchmark
reaches 0.8147 MRR@10 and 92.9% Hit@10 with a ratio of about 1:0.17.

## Run the small example

Use Python 3.12, Docker Engine, and Docker Compose on x86-64 Linux. The Compose
file runs CPU inference and binds the public ports to localhost. It contains
pinned TEI and PostgreSQL images.

Compose pulls the latest GoodMem server image when you start the example.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -c 'import secrets; from pathlib import Path; Path(".env").open("x").write("GOODMEM_DB_PASSWORD=" + secrets.token_urlsafe(24) + "\n")'
chmod 600 .env
docker compose up -d
```

Model downloads happen on first startup. Wait for both services before ingestion:

```bash
until curl -fsS http://127.0.0.1:18081/health >/dev/null; do sleep 5; done
until curl -fsS http://127.0.0.1:18082/health >/dev/null; do sleep 5; done
python setup.py
source run/credentials.env
python demo.py --ratio 0.168702397557
```

The demo compares dense-only, sparse-only, and hybrid scoring on six short
documents. It uses a dense-to-sparse ratio of about 1:0.17 from the
[SQuAD benchmark](https://docs.goodmem.ai/docs/how-to/hybrid-search/#measured-results).
The command retains the exact coefficient for reproduction. Try `--query` to
ask another question or `--ratio` to adjust the sparse contribution.

`run/credentials.env` is mode 0600 and is ignored by Git. The setup script creates
it only for a newly initialized server; an existing server requires
`GOODMEM_BASE_URL` and `GOODMEM_API_KEY`. To reuse already registered models, pass
`--dense-id` and `--sparse-id` to `setup.py`. Use the models' addresses as reachable
from the GoodMem server (`dense:80` and `sparse:80` on this Compose network).

If the default host ports are occupied, set `GOODMEM_REST_PORT`,
`GOODMEM_GRPC_PORT`, `DENSE_PORT`, and `SPARSE_PORT` in `.env`, and set
`GOODMEM_BASE_URL` to the new REST port. Ports inside the Compose network remain
unchanged. `docker compose stop` pauses this example without deleting its data.

## Evaluate and tune on SQuAD

Use a separate empty space for the benchmark. Reuse the two registered embedders:

```bash
DENSE_ID=$(python -c 'import json; print(json.load(open("run/resources.json"))["dense"])')
SPARSE_ID=$(python -c 'import json; print(json.load(open("run/resources.json"))["sparse"])')
python setup.py --output-dir run/benchmark --name 'SQuAD hybrid benchmark' \
  --dense-id "$DENSE_ID" --sparse-id "$SPARSE_ID"
SPACE_ID=$(python -c 'import json; print(json.load(open("run/benchmark/resources.json"))["space_id"])')
mkdir -p squad_data
curl -fL https://raw.githubusercontent.com/rajpurkar/SQuAD-explorer/master/dataset/dev-v1.1.json \
  -o squad_data/dev-v1.1.json
python insert_squad_sentences_goodmem.py --space-id "$SPACE_ID" --dry-run
python insert_squad_sentences_goodmem.py --space-id "$SPACE_ID" \
  --output run/benchmark/manifest.json --sample-size 1000 --seed 42
python optimize_embedder_weights.py --manifest run/benchmark/manifest.json \
  --output-dir run/benchmark/search
```

The loader stores one memory per sentence and maps annotated answers to accepted
answer sentences. It samples 1,000 tuning questions and 1,000 test questions
from separate groups of articles. The searchable corpus contains the answer
sentences for both groups.

The optimizer fixes the MiniLM coefficient at 1 and searches SPLADE coefficients
on a logarithmic grid alongside dense-only and sparse-only scoring. A second
grid refines around the best tuning result. The selected ratio is saved before
the test split is evaluated. Test comparisons include the two single-model
scoring baselines, 1:0.005, and 1:1. Both embedders contribute candidates in every
configuration, so the comparison measures how weights affect ranking in the same
space. Use `--ratios`, `--fine-points`, `--threads`, or `--top-k` to configure a
new experiment; see `--help` for each command.

`selection.json` records every tested ratio and its tuning metrics. `summary.json`
records test metrics and paired article-cluster bootstrap intervals. Files under
`evaluations/` retain question IDs, ranks, returned hits, weights, corpus and
question fingerprints, and retrieval depth. Completed evaluations resume for
matching configurations. Create a new run when changing the corpus or model
weights.

Metrics are MRR truncated at the requested depth, answer hit rate at each depth,
and coverage (the fraction returning any hits). A question counts as a hit when
any accepted answer sentence is retrieved. The evaluator checks that ingestion
and retrieval complete successfully before reporting metrics. The
[guide’s results section](https://docs.goodmem.ai/docs/how-to/hybrid-search/#measured-results)
includes the results table and confidence intervals.

## Inspect misses

Point the analyzer at an exact evaluation JSON from `search/evaluations/`:

```bash
python analyze_missing_gt_similarity.py --manifest run/benchmark/manifest.json \
  --evaluation PATH_TO_EVALUATION_JSON --limit 10 --search-size 100 \
  --output run/benchmark/misses.json
```

It compares the selected weights with dense-only and sparse-only scoring through
GoodMem's API and includes returned text, scores, and accepted memory IDs. The
larger search size helps locate accepted answers beyond the original top ten;
it can also change the candidate pool and ranking.

## Development checks

```bash
python -m unittest discover -s tests -v
python -m py_compile *.py
```

`sb_sed.py` is the Apache-2.0 Google SQuAD/Wikipedia sentence breaker vendored by
the original example. Its copyright and license header are retained.
