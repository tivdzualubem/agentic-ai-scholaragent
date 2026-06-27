"""Structured outputs produced by the eligibility engine."""

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class EligibilityStatus(StrEnum):
    """Final result of deterministic eligibility screening."""

    ELIGIBLE = "eligible"
    POTENTIALLY_ELIGIBLE = "potentially_eligible"
    NOT_ELIGIBLE = "not_eligible"
    INSUFFICIENT_INFORMATION = "insufficient_information"


class EligibilityAssessment(BaseModel):
    """Explainable eligibility assessment for one scholarship."""

    model_config = ConfigDict(extra="forbid")

    scholarship_id: str
    scholarship_title: str
    status: EligibilityStatus

    passed_checks: list[str] = Field(default_factory=list)
    hard_failures: list[str] = Field(default_factory=list)
    missing_information: list[str] = Field(default_factory=list)
    manual_review_items: list[str] = Field(default_factory=list)
    preference_warnings: list[str] = Field(default_factory=list)
