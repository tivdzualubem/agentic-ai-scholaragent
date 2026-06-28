# Evaluation results

Files in this directory are reproducible development outputs.

`official_retrieval_comparison_calibrated.json` compares BM25, dense retrieval,
and hybrid reciprocal-rank fusion on the small official-source development
benchmark.

The dense cosine-similarity threshold of 0.67 is a development calibration. It
must be recalibrated on a separate calibration split before evaluation on an
independent final test set. These development results are not final publication
evidence.

<!-- official-rag-development:start -->
## Live official RAG development comparison

Artifact:

`official_rag_comparison_tinyllama_development.json`

Configuration:

- Generator: `tinyllama:latest`
- Embeddings: `nomic-embed-text`
- Historical development dense threshold: `0.67`
- Official development cases: 3 positive and 1 unsupported
- Retrieval budget: 2 attempts
- Generation budget: 2 attempts

| Metric | Single-pass RAG | Agentic RAG |
|---|---:|---:|
| Positive completion rate | 0.000 | 1.000 |
| Positive citation-pass rate | 0.000 | 1.000 |
| Relevant grounding rate | 1.000 | 1.000 |
| Relevant citation rate | 0.000 | 1.000 |
| No-result accuracy | 1.000 | 1.000 |
| Mean latency, seconds | 126.447 | 230.560 |
| Mean generation calls | 0.750 | 1.500 |
| Mean repair attempts | 0.000 | 0.750 |
| Positive fallback rate | 0.000 | 1.000 |

The conventional baseline retrieved the relevant scholarship in every
positive case but produced answers that failed citation verification.
Agentic RAG recovered every positive case through its verified
deterministic fallback and abstained correctly on the unsupported query.

All three positive Agentic RAG cases used fallback. Therefore, this
experiment demonstrates bounded safety recovery and citation
faithfulness, not successful TinyLlama citation repair.

These results come from a very small development benchmark and a
development-calibrated dense threshold. They must not be presented as
final publication-level evidence.
<!-- official-rag-development:end -->

## Independent calibration threshold sweep

`calibration_retrieval_threshold_sweep.json` records the predefined
13-value dense-threshold sweep over the six-scholarship, 24-case official
calibration partition.

The selection objective equally weights dense top-1 accuracy, dense
no-result accuracy, hybrid top-1 accuracy, and hybrid no-result accuracy.
The selected and frozen dense cosine-similarity threshold is `0.60`.

At the selected threshold:

- dense top-1 hit rate: `0.50`;
- dense no-result accuracy: `1.00`;
- dense MRR: `0.6667`;
- hybrid RRF top-1 hit rate: `0.90`;
- hybrid RRF no-result accuracy: `1.00`;
- hybrid RRF MRR: `0.95`.

The frozen configuration is stored in
`eval/config/frozen_retrieval_settings.json`. The held-out test partition
was not used during threshold selection, and these calibration metrics are
not final publication-level performance.

## Final held-out retrieval and eligibility evaluation

`held_out_retrieval_comparison.json` records the one-time comparison of BM25,
dense retrieval, and hybrid reciprocal-rank fusion on the six-identity,
24-case held-out benchmark.

The settings were frozen before the held-out identities and labels were
evaluated:

- dense cosine-similarity threshold: `0.60`;
- top-k: `3`;
- hybrid candidate-k: `9`;
- RRF constant: `60`;
- embedding model: `nomic-embed-text:latest`.

Final held-out retrieval metrics:

| Retriever | Precision@3 | Recall@3 | MRR | Top-1 | No-result |
|---|---:|---:|---:|---:|---:|
| BM25 | 0.3333 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Dense | 0.3000 | 0.9000 | 0.5917 | 0.4000 | 1.0000 |
| Hybrid RRF | 0.3333 | 1.0000 | 0.8750 | 0.7500 | 1.0000 |

BM25 achieved the strongest ranking performance. Hybrid retrieval preserved
perfect Recall@3 and abstention accuracy but did not exceed BM25. Dense
retrieval missed two positive cases within the top three and achieved weaker
top-one ranking performance.

`held_out_eligibility_evaluation.json` records deterministic eligibility
evaluation over 20 balanced labels. Accuracy, macro precision, macro recall,
macro F1, and weighted F1 were all `1.0000`.

No held-out threshold, prompt, query, retrieval, or RRF tuning was performed
after these results were observed.
