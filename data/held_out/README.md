# Official-source held-out test corpus

This directory contains six scholarship records created from the official
source snapshots under `data/sources/held_out/`.

The calibration-selected retrieval settings were frozen before these
scholarship identities and records were introduced:

- dense cosine-similarity threshold: `0.60`;
- retrieval top-k: `3`;
- hybrid candidate-k: `9`;
- reciprocal-rank-fusion constant: `60`.

The corpus must not be used to select thresholds, rewrite rules, prompts,
retrieval parameters, or other model settings. Its benchmark labels and final
evaluation results will be created and executed only after the records and
provenance metadata have been validated.

Every record retains an official URL, source-check date, source snapshot, and
manual-review requirements for conditions that the current deterministic
schema cannot represent safely.

The University of Manitoba page states a minimum admission GPA of `3.0` but
does not state the GPA scale. ScholarAgent therefore preserves that condition
for manual verification rather than inventing a numeric scale. Maastricht's
minimum GPA of `7.5/10` is explicitly supported and is represented
structurally.

Current progress: six of six held-out scholarship identities, official
source records, the balanced 24-case benchmark, the frozen retrieval
comparison, and deterministic eligibility evaluation are complete. The
held-out RAG comparison and preregistered ablation evaluation are also
complete.

The five `insufficient_information` benchmark cases use Maastricht's
explicit `7.5/10` GPA requirement because it is the only held-out source
with a numeric eligibility threshold whose scale is stated directly.

## Frozen held-out retrieval results

The one-time held-out evaluation used the calibration-selected settings
without modification:

- dense threshold: `0.60`;
- retrieval top-k: `3`;
- hybrid candidate-k: `9`;
- reciprocal-rank-fusion constant: `60`;
- embedding model: `nomic-embed-text:latest`.

Retrieval results on the 20 positive and four unsupported cases were:

| Retriever | Precision@3 | Recall@3 | MRR | Top-1 | No-result accuracy |
|---|---:|---:|---:|---:|---:|
| BM25 | 0.3333 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Dense | 0.3000 | 0.9000 | 0.5917 | 0.4000 | 1.0000 |
| Hybrid RRF | 0.3333 | 1.0000 | 0.8750 | 0.7500 | 1.0000 |

BM25 was the strongest retriever on this small held-out corpus. Hybrid RRF
did not outperform BM25, and dense retrieval showed weaker ranking
generalization. These results must be reported without post-test parameter
adjustment.

The deterministic eligibility engine reproduced all 20 held-out labels:
accuracy, macro precision, macro recall, macro F1, and weighted F1 were all
`1.0000`.
