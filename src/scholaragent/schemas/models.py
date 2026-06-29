"""Validated domain models used throughout ScholarAgent."""

from __future__ import annotations

from datetime import date
from enum import StrEnum
import re
from typing import Any, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    HttpUrl,
    field_validator,
    model_validator,
)


class DegreeLevel(StrEnum):
    """Supported target degree levels."""

    BACHELOR = "bachelor"
    MASTER = "master"
    PHD = "phd"
    POSTDOCTORAL = "postdoctoral"
    OTHER = "other"


class FundingType(StrEnum):
    """Scholarship funding coverage."""

    FULLY_FUNDED = "fully_funded"
    PARTIALLY_FUNDED = "partially_funded"
    TUITION_ONLY = "tuition_only"
    UNSPECIFIED = "unspecified"


def _clean_text_list(value: Any) -> Any:
    """Strip, remove empty values, and deduplicate a list of strings."""
    if not isinstance(value, list):
        return value

    cleaned: list[str] = []
    seen: set[str] = set()

    for item in value:
        if not isinstance(item, str):
            cleaned.append(item)
            continue

        normalized = item.strip()

        if normalized and normalized.casefold() not in seen:
            cleaned.append(normalized)
            seen.add(normalized.casefold())

    return cleaned


class StudentProfile(BaseModel):
    """Minimal student information needed for scholarship matching."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    nationality: str = Field(min_length=2, max_length=100)
    country_of_residence: str | None = Field(
        default=None,
        min_length=2,
        max_length=100,
    )
    target_degree_level: DegreeLevel
    fields_of_study: list[str] = Field(min_length=1)
    gpa: float | None = Field(default=None, ge=0)
    gpa_scale: float | None = Field(default=None, gt=0)
    language_scores: dict[str, float] = Field(default_factory=dict)
    years_work_experience: float = Field(default=0, ge=0, le=60)
    preferred_countries: list[str] = Field(default_factory=list)
    requires_full_funding: bool = True
    verified_manual_requirements: dict[str, list[str]] = Field(
        default_factory=dict
    )

    @field_validator(
        "fields_of_study",
        "preferred_countries",
        mode="before",
    )
    @classmethod
    def clean_text_lists(cls, value: Any) -> Any:
        """Normalize user-supplied string lists."""
        return _clean_text_list(value)

    @field_validator(
        "verified_manual_requirements",
        mode="before",
    )
    @classmethod
    def clean_verified_manual_requirements(
        cls,
        value: Any,
    ) -> Any:
        """Normalize scholarship-scoped verified manual evidence."""
        if value is None:
            return {}

        if not isinstance(value, dict):
            return value

        cleaned: dict[str, list[str]] = {}

        for scholarship_id, requirements in value.items():
            if not isinstance(scholarship_id, str):
                raise ValueError(
                    "Verified manual-evidence keys must be "
                    "scholarship identifiers."
                )

            normalized_id = scholarship_id.strip()

            if (
                not 3 <= len(normalized_id) <= 100
                or re.fullmatch(
                    r"[a-z0-9][a-z0-9-]*",
                    normalized_id,
                )
                is None
            ):
                raise ValueError(
                    "Verified manual-evidence keys must be "
                    "valid scholarship identifiers."
                )

            if not isinstance(requirements, list):
                raise ValueError(
                    "Verified manual evidence must be supplied "
                    "as lists of requirement strings."
                )

            normalized_requirements = _clean_text_list(
                requirements
            )

            existing = cleaned.get(
                normalized_id,
                [],
            )

            cleaned[normalized_id] = _clean_text_list(
                existing + normalized_requirements
            )

        return cleaned

    @field_validator("language_scores")
    @classmethod
    def validate_language_scores(
        cls,
        value: dict[str, float],
    ) -> dict[str, float]:
        """Reject blank test names and negative scores."""
        cleaned: dict[str, float] = {}

        for test_name, score in value.items():
            normalized_name = test_name.strip().upper()

            if not normalized_name:
                raise ValueError("Language-test names must not be blank.")

            if score < 0:
                raise ValueError("Language scores must not be negative.")

            cleaned[normalized_name] = score

        return cleaned

    @model_validator(mode="after")
    def validate_gpa_pair(self) -> Self:
        """Require GPA and GPA scale together and ensure GPA fits the scale."""
        if (self.gpa is None) != (self.gpa_scale is None):
            raise ValueError(
                "gpa and gpa_scale must either both be provided or both omitted."
            )

        if (
            self.gpa is not None
            and self.gpa_scale is not None
            and self.gpa > self.gpa_scale
        ):
            raise ValueError("gpa must not exceed gpa_scale.")

        return self


class ScholarshipRecord(BaseModel):
    """Structured scholarship information extracted from an official source."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    scholarship_id: str = Field(
        min_length=3,
        max_length=100,
        pattern=r"^[a-z0-9][a-z0-9-]*$",
    )
    title: str = Field(min_length=3, max_length=300)
    provider: str = Field(min_length=2, max_length=200)
    official_url: HttpUrl

    host_countries: list[str] = Field(min_length=1)
    degree_levels: list[DegreeLevel] = Field(min_length=1)
    eligible_nationalities: list[str] = Field(default_factory=list)
    eligible_fields: list[str] = Field(default_factory=list)
    manual_review_requirements: list[str] = Field(
        default_factory=list
    )

    minimum_gpa: float | None = Field(default=None, ge=0)
    gpa_scale: float | None = Field(default=None, gt=0)
    minimum_work_experience_years: float | None = Field(
        default=None,
        ge=0,
        le=60,
    )

    language_requirements: dict[str, float] = Field(default_factory=dict)
    funding_type: FundingType = FundingType.UNSPECIFIED
    deadline: date | None = None
    application_year: int | None = Field(default=None, ge=2000, le=2100)

    source_last_checked: date
    eligibility_text: str = Field(min_length=1)

    @field_validator(
        "host_countries",
        "eligible_nationalities",
        "eligible_fields",
        "manual_review_requirements",
        mode="before",
    )
    @classmethod
    def clean_text_lists(cls, value: Any) -> Any:
        """Normalize scholarship metadata lists."""
        return _clean_text_list(value)

    @field_validator("language_requirements")
    @classmethod
    def validate_language_requirements(
        cls,
        value: dict[str, float],
    ) -> dict[str, float]:
        """Normalize language-test names and reject negative requirements."""
        cleaned: dict[str, float] = {}

        for test_name, score in value.items():
            normalized_name = test_name.strip().upper()

            if not normalized_name:
                raise ValueError("Language-test names must not be blank.")

            if score < 0:
                raise ValueError(
                    "Language requirements must not be negative."
                )

            cleaned[normalized_name] = score

        return cleaned

    @model_validator(mode="after")
    def validate_minimum_gpa_pair(self) -> Self:
        """Require the minimum GPA and scale to be supplied together."""
        if (self.minimum_gpa is None) != (self.gpa_scale is None):
            raise ValueError(
                "minimum_gpa and gpa_scale must either both be provided "
                "or both omitted."
            )

        if (
            self.minimum_gpa is not None
            and self.gpa_scale is not None
            and self.minimum_gpa > self.gpa_scale
        ):
            raise ValueError("minimum_gpa must not exceed gpa_scale.")

        return self
