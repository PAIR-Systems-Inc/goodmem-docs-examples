# Hybrid retrieval examples

Runnable examples for [Hybrid search](https://docs.goodmem.ai/docs/how-to/hybrid-search/)
and [Evaluate hybrid search](https://docs.goodmem.ai/docs/how-to/evaluate-hybrid-search/).
They use the maintained `goodmem` SDK. The former `goodmem-client` scripts and their
statistical recommendations have been replaced.

## Run the small example

Use Python 3.12, Docker Engine, and Docker Compose on x86-64 Linux. The Compose
file runs CPU inference and binds the public ports to localhost. It contains
pinned TEI 1.9.4 and PostgreSQL images.

**Server release pending:** The benchmark uses the fusion fix in
[GoodMem PR #1844](https://github.com/PAIR-Systems-Inc/goodmem/pull/1844), scheduled
for v1.0.325. Set `GOODMEM_SERVER_IMAGE` to an image built with that fix before
running Compose, or wait for the corrected release. The server image has no
default while the release is pending; v1.0.324 silently omits some fusion scores
and must not be used to reproduce these results.

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

`0.168702397557` (about 0.17) is the best tested sparse coefficient from the
[recorded SQuAD experiment](results/2026-10-06/summary.json). It is specific to
this setup, not a universal optimum. The evaluation workflow below searches
for a ratio on your corpus.
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

The loader makes one memory per sentence with chunking disabled and considers all
annotator answer spans. Questions without a complete answer span in any sentence
are counted and excluded. It freezes 1,000 tuning and 1,000 test questions from
disjoint sets of articles. Both groups' source sentences are searchable: this
is a retrieval task, so test answers must exist in the corpus.

The optimizer fixes the MiniLM coefficient at 1 and searches SPLADE coefficients
on a logarithmic grid. Dense-only and sparse-only baselines can win. A second,
prespecified logarithmic grid refines around the best tuning result. The selected
ratio is saved before the held-out split is evaluated. Test comparisons include
the two single-model scoring baselines, 1:0.005, and 1:1. Both embedders still
discover candidates when one coefficient is zero; these comparisons isolate
scoring weights in one two-embedder space, not the cost or latency of deploying
a single model. No ratio is chosen using test
scores. Use `--ratios`, `--fine-points`, `--threads`, or `--top-k` to change a new
experiment; see `--help` for each command.

`selection.json` records every tested ratio and its tuning metrics. `summary.json`
records held-out metrics and paired article-cluster bootstrap intervals. Files
under `evaluations/` retain question IDs, ranks, returned hits, weights, corpus
and question fingerprints, and retrieval depth. Completed evaluations resume
only for matching configurations. Do not modify corpus contents or model weights
in place; create a new run if those change. The scripts verify the registered
server/model configuration and all corpus memory processing states.

Metrics are MRR truncated at the requested depth, answer hit rate at each depth,
and coverage (the fraction returning any hits). With multiple accepted answer
sentences, answer hit rate is not document recall. A failed request, warning
status, incomplete stream, failed memory, partial batch, or invalid weight ID
fails the command; it is not converted into a retrieval miss.

## Inspect misses

Point the analyzer at an exact evaluation JSON from `search/evaluations/`:

```bash
python analyze_missing_gt_similarity.py --manifest run/benchmark/manifest.json \
  --evaluation PATH_TO_EVALUATION_JSON --limit 10 --search-size 100 \
  --output run/benchmark/misses.json
```

It compares the selected weights with each single model through GoodMem's API,
and includes returned text, scores, and accepted memory IDs. It does not require
database credentials or assume a Qwen model. Increasing search size can change
candidate discovery and rankings; the result is a diagnostic, not an exact rank
over the whole corpus.

## Development checks

```bash
python -m unittest discover -s tests -v
python -m py_compile *.py
```

`sb_sed.py` is the Apache-2.0 Google SQuAD/Wikipedia sentence breaker vendored by
the original example. Its copyright and license header are retained.
