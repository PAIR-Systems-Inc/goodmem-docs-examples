# GoodMem Documentation Examples

This repository contains example scripts and code samples referenced in the [GoodMem documentation](https://docs.goodmem.ai).

## Contents

### Hybrid Search Examples
Located in `hybrid-search/`, these scripts demonstrate how to implement and optimize hybrid search systems using GoodMem:

- **`insert_squad_sentences_goodmem.py`** - Load and prepare test data using the SQuAD dataset
- **`evaluate_squad_fast.py`** - Test and evaluate hybrid search performance
- **`optimize_embedder_weights.py`** - Search and compare weights for your embedders
- **`analyze_missing_gt_similarity.py`** - Analyze and understand search failures

For detailed usage instructions, see the [hybrid search guide](https://docs.goodmem.ai/docs/how-to/hybrid-search/) in the documentation.

## Getting Started

Each script includes comprehensive command-line documentation. Run with `--help` to see available options:

```bash
cd hybrid-search
python insert_squad_sentences_goodmem.py --help
```

## Dependencies

### Sentence Boundary Detection (`sb_sed.py`)

The `insert_squad_sentences_goodmem.py` script uses `sb_sed.py`, a sentence boundary detection utility from Google's [`retrieval-qa-eval`](https://github.com/google-research-datasets/retrieval-qa-eval) project (Apache 2.0 License).

`sb_sed.py` is included in the repository, so no separate download is needed.

## Related Documentation

- [Hybrid search](https://docs.goodmem.ai/docs/how-to/hybrid-search/)
- [GoodMem Documentation](https://docs.goodmem.ai)
