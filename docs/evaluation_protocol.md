# ScholarAgent Evaluation Protocol

## 1. Purpose

This document freezes the ScholarAgent corpus-expansion, calibration, and
held-out evaluation protocol before additional official scholarship sources
are selected.

The protocol is intended to prevent test leakage, retrospective metric
selection, and overstatement of results from the existing three-record
development corpus.

## 2. Dataset partitions

Scholarship identities, rather than only queries or student profiles, define
the partitions.

| Partition | Scholarship records | Purpose |
|---|---:|---|
| Development | 3 existing records | Software development and debugging |
| Calibration | 6 new records | Dense-threshold and fixed-parameter selection |
| Held-out test | 6 new records | Final evaluation after all parameters are frozen |
| Total | 15 records | Complete official-source research corpus |

The existing development scholarship IDs are:

- `erasmus-mundus-joint-masters`
- `si-global-professionals-2026`
- `university-twente-scholarship-2027`

No scholarship ID, programme identity, or duplicate version of the same
scholarship may appear in more than one partition.

Queries and profiles may not be copied across partitions with only minor
wording changes.

## 3. Official-source inclusion criteria

Every included scholarship must satisfy all of the following:

1. The source is an official provider, university, government,
   intergovernmental organisation, or official scholarship programme page.
2. The record has a stable canonical official URL.
3. A normalized source snapshot is committed under
   `data/sources/official/`.
4. The snapshot stores:
   - source URL;
   - source type;
   - page title;
   - source-check date;
   - normalized text;
   - byte size;
   - SHA-256 content hash.
5. The structured scholarship record is supported by the saved snapshot.
6. The manifest connects the scholarship ID to the exact snapshot and hash.
7. Ambiguous or programme-specific conditions are stored as
   `manual_review_requirements`.
8. Requirements that are not explicitly present in the official source are
   not inferred or invented.

Under the current provenance model, one canonical snapshot must provide
sufficient support for the structured record. If a scholarship requires
multiple official pages, the provenance model must be extended before that
record is included.

## 4. Diversity requirements

The 12 new calibration and test records must collectively provide meaningful
variation in:

- degree level;
- funding type;
- host country or region;
- provider type;
- field restrictions;
- nationality restrictions;
- GPA requirements;
- language requirements;
- work-experience requirements;
- deadlines;
- manual-review conditions.

Each six-record calibration or test partition should satisfy the following
minimum targets:

- at least three distinct host-country or regional settings;
- at least two provider categories;
- at least one bachelor-level opportunity;
- at least two master-level opportunities;
- at least two PhD or postdoctoral opportunities;
- at least two fully funded records;
- at least one partially funded record;
- at least one tuition-only record where an official candidate is available;
- at least two records with explicit machine-checkable academic, language,
  or work-experience requirements;
- at least two records with genuine manual-review requirements;
- at least one record capable of producing a fully `eligible` case without
  unresolved manual requirements;
- at least one record capable of producing an
  `insufficient_information` case from a deliberately incomplete profile.

A single scholarship may satisfy more than one diversity target.

## 5. Benchmark construction

Separate benchmark files will be created for calibration and held-out test
partitions.

Each partition will contain:

- 20 positive profile-scholarship cases;
- 4 unsupported or no-result cases;
- 24 total cases.

The 20 positive cases will contain exactly:

| Eligibility status | Target cases |
|---|---:|
| `eligible` | 5 |
| `potentially_eligible` | 5 |
| `not_eligible` | 5 |
| `insufficient_information` | 5 |

Each scholarship should appear in multiple profile cases, but no single
scholarship should dominate the benchmark. The target is two to four positive
cases per scholarship.

All expected statuses and relevant scholarship IDs must be manually reviewed
against the saved official snapshot before evaluation.

## 6. Eligibility-label rules

The benchmark labels follow the implemented deterministic decision order:

1. One or more hard failures produce `not_eligible`.
2. Otherwise, missing information required for a deterministic check produces
   `insufficient_information`.
3. Otherwise, unresolved official or manual-review conditions produce
   `potentially_eligible`.
4. Otherwise, the result is `eligible`.

Funding preference warnings do not independently change eligibility status.

Expired deadlines are hard failures relative to the benchmark `as_of` date.

## 7. Calibration protocol

Only the calibration partition may be used for dense-threshold selection or
other tunable retrieval parameters.

The independent calibration sweep selected and froze a dense cosine-similarity threshold of `0.60`. The sweep used only the six-scholarship, 24-case calibration partition; held-out test data was not used.

Calibration will:

1. evaluate a predefined threshold grid;
2. measure positive Recall@3 and unsupported-query no-result accuracy;
3. select a threshold using a documented objective;
4. use no-result accuracy as the first safety tie-breaker;
5. freeze the selected threshold before opening the held-out test labels or
   running final experiments.

Top-k, retry budgets, generation budgets, RRF configuration, and all other
reported settings must be recorded with the selected threshold.

The held-out test partition may be evaluated only after these values are
frozen.

## 8. Retrieval evaluation

The following systems will be compared:

- BM25;
- dense retrieval;
- hybrid reciprocal-rank fusion.

Reported metrics will include:

- Precision@3;
- Recall@3;
- mean reciprocal rank;
- top-1 hit rate;
- unsupported-query no-result accuracy;
- per-case retrieved IDs and ranks.

## 9. Eligibility evaluation

Reported eligibility metrics will include:

- evaluated label count;
- multiclass accuracy;
- macro precision;
- macro recall;
- macro F1;
- weighted F1;
- per-status support;
- per-status precision;
- per-status recall;
- per-status F1;
- per-case expected and predicted statuses.

Zero-support classes will remain visible in detailed output but will not be
included in active-class macro averages.

## 10. RAG evaluation

The conventional single-pass RAG baseline and Agentic RAG will run on
identical benchmark cases, profiles, scholarship records, retrieval settings,
and generator model.

Reported metrics will include:

- positive completion rate;
- citation-pass rate;
- relevant-grounding rate;
- relevant-citation rate;
- unsupported-query abstention accuracy;
- mean latency;
- retrieval calls;
- generation calls;
- query rewrites;
- citation-repair attempts;
- deterministic-fallback rate;
- per-case final status and answer.

A fallback-completed result must be reported separately from a direct
LLM-completed result.

## 11. Ablation analysis

The final analysis will quantify the contribution of:

1. sparse versus dense versus hybrid retrieval;
2. bounded query rewriting;
3. citation-repair attempts;
4. deterministic verified fallback.

Any ablation requiring an implementation toggle must be added before final
test execution. The held-out test data must not be used to design or tune the
ablation configuration.

### Frozen held-out RAG ablation procedure

Before running the held-out RAG comparison, the following configuration is
frozen in `eval/config/frozen_rag_settings.json`:

- generator: `tinyllama:latest`;
- generation temperature: `0.0`;
- request timeout: `240` seconds;
- top-k: `3`;
- hybrid candidate-k: `9`;
- dense threshold: `0.60`;
- RRF constant: `60`;
- maximum retrieval attempts: `2`;
- maximum generation attempts: `2`.

The retrieval ablation uses the already frozen held-out BM25, dense, and
hybrid comparison. Citation-repair and deterministic-fallback ablations are
derived from the full Agentic RAG audit traces so that they do not introduce
additional stochastic LLM samples.

Query rewriting is evaluated using a rule fixed before test execution. If the
full trace contains no rewrites, its observed contribution is reported as zero
for this benchmark. If rewrites occur, only the affected cases are rerun with
`max_retrieval_attempts=1`; every other parameter remains unchanged.

Fallback-completed outputs are always separated from direct or repaired LLM
completions. A citation-safe deterministic fallback is evidence of bounded
safety recovery, not evidence that the generator successfully followed the
citation format.

### CPU transport-timeout amendment

The held-out RAG execution began with the preregistered 240-second HTTP
transport timeout. The first TinyLlama baseline request contained 8,132
prompt characters and completed in 234.67 seconds. The immediately following
agentic request exceeded 240 seconds and was terminated by the HTTP client.

Before any aggregate RAG or ablation result was generated, the transport
timeout was increased from 240 to 900 seconds. This was an operational
feasibility amendment for CPU inference, not performance tuning.

The following remained unchanged:

- `tinyllama:latest`;
- temperature `0.0`;
- all prompts;
- the frozen benchmark and corpus;
- embedding model and dense threshold;
- top-k, candidate-k, and RRF constant;
- retrieval and generation attempt budgets;
- all preregistered ablation rules.

One successfully completed baseline result was already checkpointed and is
preserved. The timed-out agentic request produced no saved case result and
will be rerun. The final artifact must disclose this amendment and must not
claim that the complete RAG evaluation used the original 240-second timeout.

### Ollama generation-failure handling amendment

After increasing the CPU transport timeout to 900 seconds, the second
citation-repair generation for the Manitoba held-out case exceeded the
timeout on two separate executions. The corresponding agentic case was not
checkpointed, while five baseline cases and four complete agentic cases
remained preserved.

Before continuing, a deterministic transport-failure policy was recorded.
An Ollama timeout or client transport error is treated as a failed generation
attempt and converted to a fixed non-factual failure marker. The ordinary
citation audit therefore fails. The existing bounded Agentic RAG workflow
then continues within the preregistered generation budget and activates its
verified deterministic fallback when that budget is exhausted.

For the conventional baseline, the same failure marker produces a
`citation_failed` result. No additional retries are introduced beyond the
existing generation budget.

This amendment does not change the model, temperature, prompts, retrieval
configuration, benchmark, eligibility logic, citation rules, generation
budget, or deterministic fallback. Final reporting will disclose the number
of transport failures and will not count timeout-recovered fallback outputs
as successful TinyLlama generations.

## 12. Reproducibility and freezing

Before final evaluation:

- source snapshots and hashes must be committed;
- structured records must validate;
- partition IDs must be checked for disjointness;
- benchmark schemas must validate;
- the calibration-selected threshold must be recorded;
- all tests must pass;
- the working tree must be clean;
- the exact Git commit must be recorded.

Final held-out results must be saved as tracked JSON artifacts. Runtime logs
may remain ignored.

## 13. Interpretation limits

The existing three-record development corpus and its results are software
validation evidence only.

They must not be presented as final publication-level evidence.

Final conclusions will be based only on the frozen held-out test partition,
with corpus size, class support, fallback dependence, model limitations, and
latency trade-offs reported explicitly.

## Post-hoc proposal-alignment supplement

The submitted proposal named Recall@5, whereas the frozen confirmatory
retrieval evaluation used Recall@3. After all primary held-out retrieval,
eligibility and RAG results were completed, a descriptive Recall@5
supplement was generated from the same six-record held-out corpus and
24-case benchmark.

The supplement retained:

- `nomic-embed-text:latest`;
- dense threshold `0.60`;
- RRF constant `60`;
- the unchanged held-out corpus and benchmark;
- the existing no-result abstention policy.

The reporting cutoff changed from three to five solely to fulfil the
proposal's stated metric coverage. The supplemental result was not used
for tuning, parameter selection, model selection or reinterpretation of
the primary Recall@3 experiment. It is not independent confirmatory
evidence.
