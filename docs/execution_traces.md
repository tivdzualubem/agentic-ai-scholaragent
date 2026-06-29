# Representative ScholarAgent Execution Traces

These examples were extracted from the completed held-out evaluation. No new model calls or post-test tuning were performed.

The traces are descriptive examples rather than independent confirmatory evidence.

## 1. Eligible scholarship recovered through verified fallback

Case ID: `eligible-bristol-master-verified`

Trace type: `eligible_verified_fallback`

### Input

- Query: University of Bristol Think Big scholarship international postgraduate tuition funding September 2026
- Expected no-result case: False

### Step 1: Retrieval and query planning

- Retrieval calls: 1
- Query rewrites: 0
- Grounded candidates: 3
- Candidate: Think Big Scholarships
  - Eligibility: `eligible`
  - Role: `recommendation`
  - Evidence verified: True
  - Assessment: This eligibility status is a deterministic ScholarAgent screening result, not an official admission or funding decision.
- Candidate: Global Leadership Scholarship
  - Eligibility: `potentially_eligible`
  - Role: `recommendation`
  - Evidence verified: True
  - Assessment: This eligibility status is a deterministic ScholarAgent screening result, not an official admission or funding decision.
- Candidate: Maastricht University NL-High Potential Scholarship
  - Eligibility: `potentially_eligible`
  - Role: `recommendation`
  - Evidence verified: True
  - Assessment: This eligibility status is a deterministic ScholarAgent screening result, not an official admission or funding decision.

### Step 2: Generation

- Generator: `tinyllama:latest`
- Generation calls: 2
- Repair attempts: 1
- Transport failure: none recorded.

### Step 3: Citation verification

- Audit 1: passed=False, cited=0, invalid=0, uncited bullets=4
- Audit 2: passed=False, cited=0, invalid=0, uncited bullets=0
- Audit 3: passed=True, cited=4, invalid=0, uncited bullets=0
- Final citation audit passed: True
- Deterministic fallback used: True

### Step 4: Finalization

- Final status: `completed_fallback`
- Relevant grounding: True
- Relevant citation: True
- Latency: 455.550 seconds

Final answer:

> - Think Big Scholarships is offered by University of Bristol. [bristol-think-big-scholarships-2026:source_identity]
> - The opportunity is hosted in United Kingdom. [bristol-think-big-scholarships-2026:host_countries]
> - Its listed funding type is tuition only. [bristol-think-big-scholarships-2026:funding_type]
> - The official eligibility information states: The University of Bristol Think Big scheme provides funding towards tuition fees for international students studying eligible undergraduate or postgraduate courses. Applicants must have applied for an eligible course beginning in September 2026 and must be classed as overseas students for fee purposes. Applications close at 10:00 UK time on 10 April 2026. Confirm final eligibility on the official source before applying. [bristol-think-big-scholarships-2026:eligibility_text]

---

## 2. Potentially eligible result requiring manual verification

Case ID: `potential-bristol-bachelor-unverified`

Trace type: `potentially_eligible_manual_review`

### Input

- Query: Bristol Think Big international undergraduate scholarship tuition fees
- Expected no-result case: False

### Step 1: Retrieval and query planning

- Retrieval calls: 1
- Query rewrites: 0
- Grounded candidates: 3
- Candidate: Think Big Scholarships
  - Eligibility: `potentially_eligible`
  - Role: `recommendation`
  - Evidence verified: True
  - Assessment: This eligibility status is a deterministic ScholarAgent screening result, not an official admission or funding decision.
- Candidate: Vice-Chancellor's International Excellence Scholarship
  - Eligibility: `potentially_eligible`
  - Role: `recommendation`
  - Evidence verified: True
  - Assessment: This eligibility status is a deterministic ScholarAgent screening result, not an official admission or funding decision.
- Candidate: Aalto University Excellence Scholarship
  - Eligibility: `potentially_eligible`
  - Role: `recommendation`
  - Evidence verified: True
  - Assessment: This eligibility status is a deterministic ScholarAgent screening result, not an official admission or funding decision.

### Step 2: Generation

- Generator: `tinyllama:latest`
- Generation calls: 2
- Repair attempts: 1
- Transport failure: none recorded.

### Step 3: Citation verification

- Audit 1: passed=False, cited=0, invalid=0, uncited bullets=2
- Audit 2: passed=False, cited=0, invalid=0, uncited bullets=6
- Audit 3: passed=True, cited=4, invalid=0, uncited bullets=0
- Final citation audit passed: True
- Deterministic fallback used: True

### Step 4: Finalization

- Final status: `completed_fallback`
- Relevant grounding: True
- Relevant citation: True
- Latency: 281.619 seconds

Final answer:

> - Think Big Scholarships is offered by University of Bristol. [bristol-think-big-scholarships-2026:source_identity]
> - The opportunity is hosted in United Kingdom. [bristol-think-big-scholarships-2026:host_countries]
> - Its listed funding type is tuition only. [bristol-think-big-scholarships-2026:funding_type]
> - The official eligibility information states: The University of Bristol Think Big scheme provides funding towards tuition fees for international students studying eligible undergraduate or postgraduate courses. Applicants must have applied for an eligible course beginning in September 2026 and must be classed as overseas students for fee purposes. Applications close at 10:00 UK time on 10 April 2026. Confirm final eligibility on the official source before applying. [bristol-think-big-scholarships-2026:eligibility_text]

---

## 3. Not eligible because of a hard GPA constraint

Case ID: `not-eligible-maastricht-low-gpa`

Trace type: `not_eligible_hard_constraint`

### Input

- Query: Maastricht NL High Potential full scholarship master tuition stipend GPA 7.5
- Expected no-result case: False

### Step 1: Retrieval and query planning

- Retrieval calls: 1
- Query rewrites: 0
- Grounded candidates: 3
- Candidate: Maastricht University NL-High Potential Scholarship
  - Eligibility: `not_eligible`
  - Role: `explanatory_ineligible`
  - Evidence verified: True
  - Assessment: This eligibility status is a deterministic ScholarAgent screening result, not an official admission or funding decision.
- Candidate: Aalto University Excellence Scholarship
  - Eligibility: `potentially_eligible`
  - Role: `recommendation`
  - Evidence verified: True
  - Assessment: This eligibility status is a deterministic ScholarAgent screening result, not an official admission or funding decision.
- Candidate: Global Leadership Scholarship
  - Eligibility: `potentially_eligible`
  - Role: `recommendation`
  - Evidence verified: True
  - Assessment: This eligibility status is a deterministic ScholarAgent screening result, not an official admission or funding decision.

### Step 2: Generation

- Generator: `tinyllama:latest`
- Generation calls: 2
- Repair attempts: 1
- Transport failure: none recorded.

### Step 3: Citation verification

- Audit 1: passed=False, cited=0, invalid=0, uncited bullets=4
- Audit 2: passed=False, cited=0, invalid=0, uncited bullets=8
- Audit 3: passed=True, cited=4, invalid=0, uncited bullets=0
- Final citation audit passed: True
- Deterministic fallback used: True

### Step 4: Finalization

- Final status: `completed_fallback`
- Relevant grounding: True
- Relevant citation: True
- Latency: 321.764 seconds

Final answer:

> - Explanatory result only—not a recommendation. Maastricht University NL-High Potential Scholarship is offered by Maastricht University. [maastricht-university-nl-high-potential-scholarship-2026:source_identity]
> - The opportunity is hosted in Netherlands. [maastricht-university-nl-high-potential-scholarship-2026:host_countries]
> - Its listed funding type is fully funded. [maastricht-university-nl-high-potential-scholarship-2026:funding_type]
> - The official eligibility information states: The Maastricht University NL-High Potential programme offers full scholarships including a tuition-fee waiver and monthly stipend for high-achieving applicants from outside the European Union who are admitted to a participating master's programme. Applicants must present a GPA of at least 7.5 on a 10.0 scale and must not be older than 35 on 1 September 2026. The scholarship deadline is 1 February 2026. Confirm final eligibility on the official source before applying. [maastricht-university-nl-high-potential-scholarship-2026:eligibility_text]

---

## 4. Insufficient information because GPA evidence is missing

Case ID: `insufficient-maastricht-health-missing-gpa`

Trace type: `insufficient_information`

### Input

- Query: NL High Potential Maastricht public health master monthly stipend
- Expected no-result case: False

### Step 1: Retrieval and query planning

- Retrieval calls: 1
- Query rewrites: 0
- Grounded candidates: 3
- Candidate: Maastricht University NL-High Potential Scholarship
  - Eligibility: `insufficient_information`
  - Role: `recommendation`
  - Evidence verified: True
  - Assessment: This eligibility status is a deterministic ScholarAgent screening result, not an official admission or funding decision.
- Candidate: Vice-Chancellor's International Excellence Scholarship
  - Eligibility: `potentially_eligible`
  - Role: `recommendation`
  - Evidence verified: True
  - Assessment: This eligibility status is a deterministic ScholarAgent screening result, not an official admission or funding decision.
- Candidate: University of Manitoba Graduate Fellowship
  - Eligibility: `potentially_eligible`
  - Role: `recommendation`
  - Evidence verified: True
  - Assessment: This eligibility status is a deterministic ScholarAgent screening result, not an official admission or funding decision.

### Step 2: Generation

- Generator: `tinyllama:latest`
- Generation calls: 2
- Repair attempts: 1
- Transport failure: none recorded.

### Step 3: Citation verification

- Audit 1: passed=False, cited=0, invalid=0, uncited bullets=4
- Audit 2: passed=False, cited=0, invalid=0, uncited bullets=0
- Audit 3: passed=True, cited=4, invalid=0, uncited bullets=0
- Final citation audit passed: True
- Deterministic fallback used: True

### Step 4: Finalization

- Final status: `completed_fallback`
- Relevant grounding: True
- Relevant citation: True
- Latency: 357.438 seconds

Final answer:

> - Maastricht University NL-High Potential Scholarship is offered by Maastricht University. [maastricht-university-nl-high-potential-scholarship-2026:source_identity]
> - The opportunity is hosted in Netherlands. [maastricht-university-nl-high-potential-scholarship-2026:host_countries]
> - Its listed funding type is fully funded. [maastricht-university-nl-high-potential-scholarship-2026:funding_type]
> - The official eligibility information states: The Maastricht University NL-High Potential programme offers full scholarships including a tuition-fee waiver and monthly stipend for high-achieving applicants from outside the European Union who are admitted to a participating master's programme. Applicants must present a GPA of at least 7.5 on a 10.0 scale and must not be older than 35 on 1 September 2026. The scholarship deadline is 1 February 2026. Confirm final eligibility on the official source before applying. [maastricht-university-nl-high-potential-scholarship-2026:eligibility_text]

---

## 5. Unsupported query handled through safe abstention

Case ID: `no-result-lunar-botany`

Trace type: `safe_abstention`

### Input

- Query: selenian moon botany crystal greenhouse grant
- Expected no-result case: True

### Step 1: Retrieval and query planning

- Retrieval calls: 1
- Query rewrites: 0
- Grounded candidates: 0
- No grounded scholarship candidates.

### Step 2: Generation

- Generator: `tinyllama:latest`
- Generation calls: 0
- Repair attempts: 0
- Transport failure: none recorded.

### Step 3: Citation verification

- No citation audit was required.
- Final citation audit passed: True
- Deterministic fallback used: False

### Step 4: Finalization

- Final status: `abstained`
- Relevant grounding: None
- Relevant citation: None
- Latency: 0.122 seconds

Final answer:

> ScholarAgent abstained because it could not retrieve sufficient verified evidence.

---

## 6. Generation timeout recovered through bounded fallback

Case ID: `eligible-manitoba-phd-verified`

Trace type: `transport_timeout_recovery`

### Input

- Query: University of Manitoba Graduate Fellowship doctoral PhD 25000 per year Canada all citizenships
- Expected no-result case: False

### Step 1: Retrieval and query planning

- Retrieval calls: 1
- Query rewrites: 0
- Grounded candidates: 1
- Candidate: University of Manitoba Graduate Fellowship
  - Eligibility: `eligible`
  - Role: `recommendation`
  - Evidence verified: True
  - Assessment: This eligibility status is a deterministic ScholarAgent screening result, not an official admission or funding decision.

### Step 2: Generation

- Generator: `tinyllama:latest`
- Generation calls: 2
- Repair attempts: 1
- Transport failure: Ollama request timed out after 900.0 seconds.
- Failure marker: `GENERATION_FAILED_DUE_TO_OLLAMA_TRANSPORT_ERROR`

### Step 3: Citation verification

- Audit 1: passed=False, cited=0, invalid=0, uncited bullets=4
- Audit 2: passed=False, cited=0, invalid=0, uncited bullets=0
- Audit 3: passed=True, cited=4, invalid=0, uncited bullets=0
- Final citation audit passed: True
- Deterministic fallback used: True

### Step 4: Finalization

- Final status: `completed_fallback`
- Relevant grounding: True
- Relevant citation: True
- Latency: 1023.428 seconds

Final answer:

> - University of Manitoba Graduate Fellowship is offered by University of Manitoba. [university-of-manitoba-graduate-fellowship:source_identity]
> - The opportunity is hosted in Canada. [university-of-manitoba-graduate-fellowship:host_countries]
> - Its listed funding type is partially funded. [university-of-manitoba-graduate-fellowship:funding_type]
> - The official eligibility information states: The University of Manitoba Graduate Fellowship supports eligible master's and PhD students in graduate fields that qualify for Tri-Agency funding. Master's awards are valued at CAD 20,000 per year and doctoral awards at CAD 25,000 per year. All citizenships may apply. The page states a minimum admission GPA of 3.0 but does not identify its scale, so the numeric threshold remains a manual-review condition. Application timing and nomination are managed by departments or units. Confirm final eligibility on the official source before applying. [university-of-manitoba-graduate-fellowship:eligibility_text]

---
