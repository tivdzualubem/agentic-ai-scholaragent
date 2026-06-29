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

## Final held-out RAG comparison

`held_out_rag_comparison.json` records the final comparison between
conventional single-pass RAG and bounded Agentic RAG on 24 held-out cases:
20 positive scholarship cases and four unsupported queries.

The execution used:

- `tinyllama:latest`;
- temperature `0.0`;
- hybrid RRF retrieval with top-k `3`;
- candidate-k `9`;
- dense threshold `0.60`;
- RRF constant `60`;
- two retrieval attempts and two generation attempts;
- a documented 900-second CPU transport timeout.

Every positive single-pass response failed deterministic citation auditing.
Every positive Agentic RAG case ultimately completed through the verified
deterministic fallback. The four unsupported queries were correctly
abstained from without an LLM generation.

TinyLlama did not successfully repair any positive response. Therefore,
the result demonstrates bounded recovery, deterministic citation safety,
and correct abstention—not successful citation generation by TinyLlama.

The final Agentic RAG citation-audit pass rate was `1.00`, while the benchmark-relevant citation rate was `0.70`. These are different measurements: the citation verifier confirmed that fallback citations were supported by retrieved evidence, but only 70% of positive cases cited a scholarship identity marked as relevant by the benchmark. The result must therefore not be described as 100% relevant scholarship citation.

One citation-repair request exceeded the 900-second transport timeout. It
was recorded as a failed bounded generation attempt and proceeded through
the same verified fallback policy.

`held_out_rag_ablation.json` records the component analysis:

- no held-out case invoked query rewriting;
- citation repair produced zero successful LLM repairs;
- bypassing repair and proceeding directly to verified fallback preserved
  safety metrics while requiring fewer LLM generations;
- removing deterministic fallback reduced positive completion and citation
  success because all positive completions depended on fallback;
- BM25 remained the strongest retriever on this small, lexically distinctive
  held-out corpus.

These findings are configuration- and dataset-specific and must not be
presented as universal conclusions about sparse retrieval or agentic RAG.

## Supplemental held-out Recall@5

`held_out_retrieval_recall5_supplement.json` reports a post-hoc,
descriptive Recall@5 analysis because the submitted proposal explicitly
named Recall@5 and MRR, while the preregistered primary held-out evaluation
used Recall@3.

The same held-out corpus, benchmark, embedding model, dense threshold and
RRF constant were retained. No parameter selection, model selection or
post-test tuning was performed.

| Retriever | Precision@5 | Recall@5 | MRR | Top-1 | No-result |
|---|---:|---:|---:|---:|---:|
| BM25 | 0.2000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Dense | 0.1900 | 0.9500 | 0.6042 | 0.4000 | 1.0000 |
| Hybrid RRF | 0.2000 | 1.0000 | 0.8750 | 0.7500 | 1.0000 |

The primary confirmatory retrieval result remains Recall@3. This
supplement must not be presented as a second independent held-out
experiment. It shows that increasing the reporting cutoff from three to
five recovered one additional dense-retrieval hit, raising dense recall
from `0.90` to `0.95`. BM25 and hybrid RRF already had complete recall at
three and therefore remained at `1.00`.

## Verification-stage ablation

`held_out_verification_ablation.json` provides the proposal-promised
comparison with and without the verification stage. It is a post-hoc,
trace-derived counterfactual and made no new LLM calls.

The without-verification condition accepts each stored first-pass
TinyLlama answer immediately. The same deterministic citation auditor is
then applied only for measurement; it does not alter the counterfactual
decision. The with-verification condition uses the completed Agentic RAG
workflow: citation auditing, one bounded repair attempt, and deterministic
verified fallback.

| Metric | Without verification | With verification |
|---|---:|---:|
| Raw/verified positive completion | 1.0000 raw acceptance | 1.0000 verified completion |
| Citation-audit pass | 0.0000 | 1.0000 |
| Relevant grounding | 0.9500 | 0.9500 |
| Relevant citation | 0.0500 | 0.7000 |
| Unsafe acceptance | 1.0000 | 0.0000 |
| No-result accuracy | 1.0000 | 1.0000 |
| Mean generation calls | 0.8333 | 1.6667 |
| Positive fallback rate | 0.0000 | 1.0000 |

All 20 positive first-pass answers failed citation verification. TinyLlama
successfully repaired none of them. Deterministic fallback produced all 20
verified completions. Therefore, the observed benefit comes from bounded
verification and fallback recovery, not from successful self-repair by
the language model.

Citation-audit validity and benchmark relevance remain distinct. The
verified workflow achieved a citation-audit pass rate of `1.00`, but its
relevant-citation rate was `0.70`. The result must not be described as
100% benchmark-relevant citation.

This artifact is descriptive rather than independent confirmatory
evidence because it reuses stored held-out traces after completion of the
primary experiment.

## Held-out runtime and cost summary

`held_out_runtime_cost_summary.json` is a deterministic descriptive
summary derived from the completed held-out RAG evaluation. It performs
no new model calls and is not independent confirmatory evidence.

| Metric | Single-pass RAG | Agentic RAG |
|---|---:|---:|
| Evaluated cases | 24 | 24 |
| Total runtime, minutes | 56.1006 | 117.2255 |
| Mean latency, seconds | 140.2514 | 293.0638 |
| Retrieval calls | 24 | 24 |
| Generation calls | 20 | 40 |
| Positive completion | 0.00 | 1.00 |
| Citation-audit pass | 0.00 | 1.00 |
| Relevant citation | 0.05 | 0.70 |
| Positive fallback rate | 0.00 | 1.00 |

Agentic RAG required approximately `2.09` times the total wall-clock
runtime and twice the generation calls of single-pass RAG.

The direct hosted-model API fee was `$0.00` because generation and
embedding inference ran locally through Ollama. This does not imply zero
total computation cost. Electricity, hardware depreciation, operator
time, CPU opportunity cost and internet access were not monetized.

One Agentic RAG generation attempt reached the frozen 900-second timeout.
The failure was converted into the predefined non-factual transport-error
marker and processed as a failed bounded generation attempt before
verified deterministic fallback.

## Representative execution traces

`held_out_execution_traces.json` and
`docs/execution_traces.md` provide six trace-derived examples:

- eligible result recovered by verified fallback;
- potentially eligible result requiring manual review;
- not eligible result caused by a hard constraint;
- insufficient-information result;
- unsupported-query abstention;
- generation-timeout recovery.

The traces were extracted from the completed held-out evaluation. They
perform no new model calls, involve no post-test tuning, and are
descriptive rather than independent confirmatory evidence.

## Expired-opportunity handling

`expired_opportunity_handling.json` and
`docs/expired_handling.md` document how ScholarAgent handles relevant
scholarships whose recorded application deadline has passed.

The deterministic eligibility engine assigns `not_eligible` when the
deadline is earlier than the assessment date. The scholarship may remain
available as an explanatory match, but it is not presented as an active
recommendation.

The artifact covers:

- the KTH Scholarship expired calibration round;
- the SI Scholarship for Global Professionals expired development round;
- the SI single-pass citation failure;
- the SI Agentic RAG verified fallback recovery.

No new model calls or held-out-test evidence were used.
