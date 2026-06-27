"""Integrated scholarship retrieval and eligibility-screening pipeline."""

from __future__ import annotations

from datetime import date

from pydantic import BaseModel, ConfigDict, Field

from scholaragent.eligibility import (
    EligibilityAssessment,
    EligibilityStatus,
    assess_eligibility,
)
from scholaragent.retrieval import BM25ScholarshipIndex
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
    index: BM25ScholarshipIndex,
    k: int = 5,
    as_of: date | None = None,
) -> ScholarshipSearchReport:
    """Retrieve scholarships, assess eligibility, and rank useful matches.

    Ranking favours:

    1. eligible records;
    2. potentially eligible records requiring verification;
    3. records missing applicant information;
    4. clearly ineligible records.

    Within the same status, fewer preference warnings and a higher BM25
    score are preferred.
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
            STATUS_PRIORITY[item.assessment.status],
            len(item.assessment.preference_warnings),
            -item.retrieval_score,
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
