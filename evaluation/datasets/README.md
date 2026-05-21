# Evaluation Datasets

This folder stores the original sample benchmark questions for the research agent.

The product-grade golden set now lives in `evaluation/golden/*.jsonl`.
`evaluation/golden/retrieval.jsonl` is the retrieval-focused seed set for Recall@K, MRR, nDCG, and insufficient-evidence checks.

Required fields per golden item:

- `id`
- `question`
- `expected_answer_type`
- `required_sources`
- `expected_facts`
- `forbidden_claims`
- `answer_rubric`
- `difficulty`
- `query_category`
- `top_k`

Run the CI-safe harness:

```bash
PYTHONPATH=. ./.venv/bin/python -m evaluation.run_eval --mode mock --fail-on-threshold
```

Run the live graph over the golden set:

```bash
PYTHONPATH=. ./.venv/bin/python -m evaluation.run_eval --mode live --output evaluation/reports/latest.jsonl
```
