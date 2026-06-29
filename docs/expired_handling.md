# Expired Opportunity Handling

ScholarAgent may retrieve an expired scholarship when it is relevant to the query, but the eligibility engine rejects the expired round as `not_eligible`.

This distinction allows the system to explain a relevant historical match without recommending that the student apply to a closed round.

| Case | Deadline | Assessment date | Result |
|---|---:|---:|---|
| KTH Scholarship | 2026-01-15 | 2026-05-01 | not_eligible |
| SI Scholarship for Global Professionals | 2026-02-25 | 2026-06-28 | not_eligible |

## Deterministic deadline rule

When the recorded deadline is earlier than the assessment date, the eligibility engine adds a hard failure stating that the deadline has passed.

### KTH expired calibration round

- Case ID: `not-eligible-kth-expired-round`
- Scholarship: KTH Scholarship
- Deadline: `2026-01-15`
- Assessment date: `2026-05-01`
- Predicted status: `not_eligible`
- Recorded failure: Deadline 2026-01-15 has passed.
- Official source: https://www.kth.se/en/studies/master/admissions/scholarships/kth-scholarship-1.72827

### SI expired development round

- Case ID: `si-global-professionals-expired-round`
- Scholarship: SI Scholarship for Global Professionals
- Deadline: `2026-02-25`
- Assessment date: `2026-06-28`
- Predicted status: `not_eligible`
- Recorded failure: Deadline 2026-02-25 has passed.
- Official source: https://si.se/en/apply/scholarships/swedish-institute-scholarships-for-global-professionals/

## Agentic RAG behavior

For the SI Global Professionals expired-round case, single-pass RAG failed citation verification. Agentic RAG completed through verified deterministic fallback.

The fallback explains the expired relevant match rather than presenting it as an active recommendation.

## Interpretation limit

Deadline decisions are based on the recorded official-source snapshot and the supplied assessment date. Applicants must confirm the current round on the official source.
