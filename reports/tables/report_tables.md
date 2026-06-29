# ScholarAgent Report Tables

All values were generated directly from tracked evaluation artifacts. No model calls or manual metric transcription were used.

## Primary held-out retrieval at k=3

| Retriever | P@k | R@k | MRR | Top-1 | No-result |
|---|---|---|---|---|---|
| bm25 | 0.333 | 1.000 | 1.000 | 1.000 | 1.000 |
| dense | 0.300 | 0.900 | 0.592 | 0.400 | 1.000 |
| hybrid_rrf | 0.333 | 1.000 | 0.875 | 0.750 | 1.000 |

BM25 achieved the strongest primary held-out ranking results on this small, lexically distinctive corpus.

## Supplemental held-out retrieval at k=5

| Retriever | P@k | R@k | MRR | Top-1 | No-result |
|---|---|---|---|---|---|
| bm25 | 0.200 | 1.000 | 1.000 | 1.000 | 1.000 |
| dense | 0.190 | 0.950 | 0.604 | 0.400 | 1.000 |
| hybrid_rrf | 0.200 | 1.000 | 0.875 | 0.750 | 1.000 |

Recall@5 is a post-hoc descriptive supplement and was not used for tuning or model selection.

## Eligibility classification

| Status | Support | Precision | Recall | F1 |
|---|---|---|---|---|
| eligible | 5 | 1.000 | 1.000 | 1.000 |
| potentially_eligible | 5 | 1.000 | 1.000 | 1.000 |
| not_eligible | 5 | 1.000 | 1.000 | 1.000 |
| insufficient_information | 5 | 1.000 | 1.000 | 1.000 |

Overall accuracy, macro precision, macro recall, macro F1, weighted F1, and no-result accuracy were all 1.000.

## Single-pass versus Agentic RAG

| System | Completion | Citation | Grounding | Relevant citation | No-result | Mean latency (s) |
|---|---|---|---|---|---|---|
| single_pass_rag | 0.000 | 0.000 | 0.950 | 0.050 | 1.000 | 140.251 |
| agentic_rag | 1.000 | 1.000 | 0.950 | 0.700 | 1.000 | 293.064 |

## Verification-stage ablation

| Configuration | Completion/acceptance | Citation | Relevant citation | Unsafe acceptance | Generation calls | Fallback |
|---|---|---|---|---|---|---|
| without_verification | 1.000 | 0.000 | 0.050 | 1.000 | 0.833 | 0.000 |
| with_verification | 1.000 | 1.000 | 0.700 | 0.000 | 1.667 | 1.000 |

All 20 positive first-pass outputs failed citation verification. There were zero successful LLM repairs and 20 deterministic fallback recoveries.

## Runtime and direct API cost

| System | Total minutes | Mean seconds | Generation calls | Direct API fee (USD) |
|---|---|---|---|---|
| single_pass_rag | 56.101 | 140.251 | 20 | 0.00 |
| agentic_rag | 117.226 | 293.064 | 40 | 0.00 |

Agentic RAG used 2.000 times as many generation calls and 2.090 times the total latency of single-pass RAG.

The direct hosted API fee was zero because Ollama ran locally. Electricity, hardware depreciation, CPU opportunity cost, operator time, and internet usage were not monetized.

## Interpretation constraints

- The held-out corpus contains six official scholarship identities and 24 benchmark cases.
- Perfect eligibility results must not be generalized beyond this controlled benchmark.
- Citation-audit success is distinct from benchmark-relevant citation accuracy.
- Agentic completion depended on deterministic verification and fallback, not successful TinyLlama citation repair.
