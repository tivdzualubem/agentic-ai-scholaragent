"""Evidence grounding and citation verification for ScholarAgent."""

from __future__ import annotations

from datetime import date
from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from scholaragent.eligibility import EligibilityStatus
from scholaragent.pipeline import (
    ScholarshipSearchReport,
    ScreenedScholarship,
)
from scholaragent.schemas import ScholarshipRecord


EvidenceField = Literal[
    "source_identity",
    "host_countries",
    "degree_levels",
    "eligible_nationalities",
    "eligible_fields",
    "manual_review_requirements",
    "funding_type",
    "deadline",
    "eligibility_text",
]


class EvidenceSnippet(BaseModel):
    """A source-linked fact extracted from one scholarship record."""

    model_config = ConfigDict(extra="forbid")

    citation_id: str = Field(min_length=3)
    scholarship_id: str = Field(min_length=3)
    field_name: EvidenceField
    text: str = Field(min_length=1)
    source_url: str = Field(min_length=1)
    source_last_checked: date


class GroundedClaim(BaseModel):
    """A factual claim linked to one or more evidence snippets."""

    model_config = ConfigDict(extra="forbid")

    claim_id: str = Field(min_length=3)
    text: str = Field(min_length=1)
    citation_ids: list[str] = Field(min_length=1)


class CitationVerification(BaseModel):
    """Result of deterministic citation-integrity checks."""

    model_config = ConfigDict(extra="forbid")

    passed: bool
    errors: list[str] = Field(default_factory=list)


class GroundedCandidateReport(BaseModel):
    """Evidence-grounded report for one screened scholarship."""

    model_config = ConfigDict(extra="forbid")

    scholarship_id: str
    title: str
    provider: str
    official_url: str
    source_last_checked: date

    eligibility_status: str
    assessment_note: str

    claims: list[GroundedClaim] = Field(min_length=1)
    evidence: list[EvidenceSnippet] = Field(min_length=1)
    verification: CitationVerification


class GroundedScholarshipReport(BaseModel):
    """Ranked source-grounded output for one ScholarAgent search."""

    model_config = ConfigDict(extra="forbid")

    query: str
    as_of: date
    candidates: list[GroundedCandidateReport]
    all_citations_verified: bool
    disclaimer: str


def _display_value(value: object) -> str:
    """Return a readable representation for enum-like values."""
    if isinstance(value, Enum):
        return str(value.value)

    return str(value)


def _join_values(values: list[object]) -> str:
    """Join structured scholarship values deterministically."""
    return ", ".join(
        _display_value(value)
        for value in values
    )


def _citation_id(
    scholarship_id: str,
    field_name: EvidenceField,
) -> str:
    """Create a stable citation identifier."""
    return f"{scholarship_id}:{field_name}"


def build_evidence_snippets(
    scholarship: ScholarshipRecord,
) -> list[EvidenceSnippet]:
    """Convert structured scholarship fields into cited evidence."""
    evidence_rows: list[tuple[EvidenceField, str]] = [
        (
            "source_identity",
            (
                f"{scholarship.title} is offered by "
                f"{scholarship.provider}."
            ),
        ),
        (
            "host_countries",
            (
                "Host country or countries: "
                f"{_join_values(scholarship.host_countries)}."
            ),
        ),
        (
            "degree_levels",
            (
                "Supported degree level or levels: "
                f"{_join_values(scholarship.degree_levels)}."
            ),
        ),
        (
            "funding_type",
            (
                "Funding type: "
                f"{scholarship.funding_type.value.replace('_', ' ')}."
            ),
        ),
    ]

    if scholarship.eligible_nationalities:
        evidence_rows.append(
            (
                "eligible_nationalities",
                (
                    "Eligible nationalities: "
                    f"{_join_values(
                        scholarship.eligible_nationalities
                    )}."
                ),
            )
        )

    if scholarship.eligible_fields:
        evidence_rows.append(
            (
                "eligible_fields",
                (
                    "Eligible fields: "
                    f"{_join_values(scholarship.eligible_fields)}."
                ),
            )
        )

    if scholarship.manual_review_requirements:
        evidence_rows.append(
            (
                "manual_review_requirements",
                (
                    "Requirements needing authoritative "
                    "manual verification: "
                    f"{_join_values(
                        scholarship.manual_review_requirements
                    )}."
                ),
            )
        )

    if scholarship.deadline is not None:
        evidence_rows.append(
            (
                "deadline",
                (
                    "Application deadline: "
                    f"{scholarship.deadline.isoformat()}."
                ),
            )
        )

    evidence_rows.append(
        (
            "eligibility_text",
            scholarship.eligibility_text,
        )
    )

    return [
        EvidenceSnippet(
            citation_id=_citation_id(
                scholarship.scholarship_id,
                field_name,
            ),
            scholarship_id=scholarship.scholarship_id,
            field_name=field_name,
            text=text,
            source_url=str(scholarship.official_url),
            source_last_checked=(
                scholarship.source_last_checked
            ),
        )
        for field_name, text in evidence_rows
    ]


def build_grounded_claims(
    scholarship: ScholarshipRecord,
) -> list[GroundedClaim]:
    """Create deterministic claims with explicit citations."""
    identifier = scholarship.scholarship_id

    claims = [
        GroundedClaim(
            claim_id=f"{identifier}:claim:identity",
            text=(
                f"{scholarship.title} is offered by "
                f"{scholarship.provider}."
            ),
            citation_ids=[
                _citation_id(identifier, "source_identity")
            ],
        ),
        GroundedClaim(
            claim_id=f"{identifier}:claim:location",
            text=(
                "The opportunity is hosted in "
                f"{_join_values(scholarship.host_countries)}."
            ),
            citation_ids=[
                _citation_id(identifier, "host_countries")
            ],
        ),
        GroundedClaim(
            claim_id=f"{identifier}:claim:degree",
            text=(
                "It supports the following degree level or levels: "
                f"{_join_values(scholarship.degree_levels)}."
            ),
            citation_ids=[
                _citation_id(identifier, "degree_levels")
            ],
        ),
        GroundedClaim(
            claim_id=f"{identifier}:claim:funding",
            text=(
                "Its listed funding type is "
                f"{scholarship.funding_type.value.replace('_', ' ')}."
            ),
            citation_ids=[
                _citation_id(identifier, "funding_type")
            ],
        ),
    ]

    if scholarship.eligible_fields:
        claims.append(
            GroundedClaim(
                claim_id=f"{identifier}:claim:fields",
                text=(
                    "The listed eligible fields include "
                    f"{_join_values(scholarship.eligible_fields)}."
                ),
                citation_ids=[
                    _citation_id(identifier, "eligible_fields")
                ],
            )
        )

    if scholarship.eligible_nationalities:
        claims.append(
            GroundedClaim(
                claim_id=f"{identifier}:claim:nationalities",
                text=(
                    "The listed nationality eligibility is "
                    f"{_join_values(
                        scholarship.eligible_nationalities
                    )}."
                ),
                citation_ids=[
                    _citation_id(
                        identifier,
                        "eligible_nationalities",
                    )
                ],
            )
        )

    if scholarship.manual_review_requirements:
        claims.append(
            GroundedClaim(
                claim_id=(
                    f"{identifier}:claim:manual_review"
                ),
                text=(
                    "The following conditions require manual "
                    "verification against the official source: "
                    f"{_join_values(
                        scholarship.manual_review_requirements
                    )}."
                ),
                citation_ids=[
                    _citation_id(
                        identifier,
                        "manual_review_requirements",
                    )
                ],
            )
        )

    if scholarship.deadline is not None:
        claims.append(
            GroundedClaim(
                claim_id=f"{identifier}:claim:deadline",
                text=(
                    "The listed application deadline is "
                    f"{scholarship.deadline.isoformat()}."
                ),
                citation_ids=[
                    _citation_id(identifier, "deadline")
                ],
            )
        )

    claims.append(
        GroundedClaim(
            claim_id=f"{identifier}:claim:eligibility",
            text=(
                "The official eligibility information states: "
                f"{scholarship.eligibility_text}"
            ),
            citation_ids=[
                _citation_id(identifier, "eligibility_text")
            ],
        )
    )

    return claims


def verify_candidate_citations(
    candidate: GroundedCandidateReport,
    scholarship: ScholarshipRecord,
) -> CitationVerification:
    """Verify that claims and evidence match the source record."""
    errors: list[str] = []

    if candidate.scholarship_id != scholarship.scholarship_id:
        errors.append(
            "Candidate scholarship identifier does not match "
            "the source record."
        )

    if candidate.title != scholarship.title:
        errors.append(
            "Candidate title does not match the source record."
        )

    if candidate.provider != scholarship.provider:
        errors.append(
            "Candidate provider does not match the source record."
        )

    if candidate.official_url != str(scholarship.official_url):
        errors.append(
            "Candidate official URL does not match the source record."
        )

    if (
        candidate.source_last_checked
        != scholarship.source_last_checked
    ):
        errors.append(
            "Candidate source-check date does not match "
            "the source record."
        )

    expected_evidence = {
        item.citation_id: item
        for item in build_evidence_snippets(scholarship)
    }

    actual_evidence = {
        item.citation_id: item
        for item in candidate.evidence
    }

    if len(actual_evidence) != len(candidate.evidence):
        errors.append(
            "Duplicate citation identifiers were detected."
        )

    missing_evidence = (
        set(expected_evidence) - set(actual_evidence)
    )
    unexpected_evidence = (
        set(actual_evidence) - set(expected_evidence)
    )

    for citation_id in sorted(missing_evidence):
        errors.append(
            f"Expected evidence is missing: {citation_id}."
        )

    for citation_id in sorted(unexpected_evidence):
        errors.append(
            f"Unexpected evidence was supplied: {citation_id}."
        )

    for citation_id in sorted(
        set(expected_evidence) & set(actual_evidence)
    ):
        expected = expected_evidence[citation_id]
        actual = actual_evidence[citation_id]

        if actual != expected:
            errors.append(
                f"Evidence does not match the source: "
                f"{citation_id}."
            )

    claim_ids = [
        claim.claim_id
        for claim in candidate.claims
    ]

    if len(claim_ids) != len(set(claim_ids)):
        errors.append(
            "Duplicate claim identifiers were detected."
        )

    for claim in candidate.claims:
        if not claim.citation_ids:
            errors.append(
                f"Claim has no citations: {claim.claim_id}."
            )
            continue

        for citation_id in claim.citation_ids:
            if citation_id not in actual_evidence:
                errors.append(
                    f"Claim {claim.claim_id} references unknown "
                    f"citation {citation_id}."
                )

    return CitationVerification(
        passed=not errors,
        errors=errors,
    )


def build_grounded_candidate(
    screened: ScreenedScholarship,
) -> GroundedCandidateReport:
    """Build and verify one grounded candidate report."""
    scholarship = screened.scholarship

    draft = GroundedCandidateReport(
        scholarship_id=scholarship.scholarship_id,
        title=scholarship.title,
        provider=scholarship.provider,
        official_url=str(scholarship.official_url),
        source_last_checked=scholarship.source_last_checked,
        eligibility_status=screened.assessment.status.value,
        assessment_note=(
            "This eligibility status is a deterministic "
            "ScholarAgent screening result, not an official "
            "admission or funding decision."
        ),
        claims=build_grounded_claims(scholarship),
        evidence=build_evidence_snippets(scholarship),
        verification=CitationVerification(
            passed=False,
            errors=["Citation verification has not run."],
        ),
    )

    verification = verify_candidate_citations(
        draft,
        scholarship,
    )

    return draft.model_copy(
        update={"verification": verification}
    )


def build_grounded_report(
    report: ScholarshipSearchReport,
    *,
    as_of: date,
    include_ineligible: bool = False,
) -> GroundedScholarshipReport:
    """Create a ranked grounded report from screened results."""
    screened_results = [
        result
        for result in report.results
        if (
            include_ineligible
            or result.assessment.status
            is not EligibilityStatus.NOT_ELIGIBLE
        )
    ]

    candidates = [
        build_grounded_candidate(result)
        for result in screened_results
    ]

    return GroundedScholarshipReport(
        query=report.query,
        as_of=as_of,
        candidates=candidates,
        all_citations_verified=(
            bool(candidates)
            and all(
                candidate.verification.passed
                for candidate in candidates
            )
        ),
        disclaimer=(
            "Scholarship information can change. Confirm every "
            "requirement and deadline on the cited official source "
            "before applying."
        ),
    )
