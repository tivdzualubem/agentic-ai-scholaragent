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
- Dense threshold: `0.67`
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
