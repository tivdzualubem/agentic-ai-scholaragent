"""Integrated scholarship retrieval and eligibility-screening pipeline."""

from __future__ import annotations

from datetime import date

from pydantic import BaseModel, ConfigDict, Field

from scholaragent.eligibility import (
    EligibilityAssessment,
    EligibilityStatus,
    assess_eligibility,
)
from scholaragent.retrieval import ScholarshipSearchIndex
from scholaragent.schemas import ScholarshipRecord, StudentProfile


STATUS_PRIORITY = {
    EligibilityStatus.ELIGIBLE: 0,
    EligibilityStatus.POTENTIALLY_ELIGIBLE: 1,
    EligibilityStatus.INSUFFICIENT_INFORMATION: 2,
    EligibilityStatus.NOT_ELIGIBLE: 3,
}


class ScreenedScholarship(BaseModel):
    """A retrieved scholarship with its eligibility assessment."""

    model_config = ConfigDict(extra="forbid")

    final_rank: int = Field(ge=1)
    retrieval_rank: int = Field(ge=1)
    retrieval_score: float
    scholarship: ScholarshipRecord
    assessment: EligibilityAssessment


class ScholarshipSearchReport(BaseModel):
    """Structured output from the integrated screening pipeline."""

    model_config = ConfigDict(extra="forbid")

    query: str
    retrieved_count: int = Field(ge=0)
    results: list[ScreenedScholarship]


def search_and_screen(
    *,
    query: str,
    profile: StudentProfile,
    index: ScholarshipSearchIndex,
    k: int = 5,
    as_of: date | None = None,
) -> ScholarshipSearchReport:
    """Retrieve scholarships, assess eligibility, and rank useful matches.

    Ranking first separates viable candidates from records with hard
    eligibility failures. Query relevance is preserved among viable
    candidates, while eligibility status and preference warnings act as
    tie-breakers.

    This prevents an unrelated but fully eligible scholarship from
    outranking a query-specific opportunity that only needs manual
    verification or additional applicant information.
    """
    retrieval_results = index.search(query, k=k)

    screened: list[ScreenedScholarship] = []

    for result in retrieval_results:
        assessment = assess_eligibility(
            profile,
            result.scholarship,
            as_of=as_of,
        )

        screened.append(
            ScreenedScholarship(
                final_rank=1,
                retrieval_rank=result.rank,
                retrieval_score=result.score,
                scholarship=result.scholarship,
                assessment=assessment,
            )
        )

    screened.sort(
        key=lambda item: (
            item.assessment.status is EligibilityStatus.NOT_ELIGIBLE,
            -item.retrieval_score,
            STATUS_PRIORITY[item.assessment.status],
            len(item.assessment.preference_warnings),
            item.retrieval_rank,
        )
    )

    ranked_results = [
        item.model_copy(update={"final_rank": rank})
        for rank, item in enumerate(screened, start=1)
    ]

    return ScholarshipSearchReport(
        query=query.strip(),
        retrieved_count=len(ranked_results),
        results=ranked_results,
    )
