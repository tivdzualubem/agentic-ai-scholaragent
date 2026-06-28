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

Current progress: six of six held-out scholarship identities and official
source records are complete. The held-out benchmark has not yet been created
or evaluated.
