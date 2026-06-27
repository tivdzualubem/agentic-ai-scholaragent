"""Load and validate ScholarAgent evaluation benchmarks."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    model_validator,
)

from scholaragent.eligibility import EligibilityStatus
from scholaragent.schemas import StudentProfile


class BenchmarkError(ValueError):
    """Raised when an evaluation benchmark is invalid."""


class BenchmarkCase(BaseModel):
    """One information need and its manually defined ground truth."""

    model_config = ConfigDict(extra="forbid")

    case_id: str = Field(
        min_length=3,
        pattern=r"^[a-z0-9][a-z0-9-]*$",
    )
    query: str = Field(min_length=1)
    profile: StudentProfile
    relevant_ids: list[str]
    expected_statuses: dict[str, EligibilityStatus]
    expect_no_results: bool = False

    @model_validator(mode="after")
    def validate_ground_truth(self) -> Self:
        """Ensure positive and no-result cases are internally consistent."""
        if self.expect_no_results and self.relevant_ids:
            raise ValueError(
                "A no-result case must not contain relevant identifiers."
            )

        if not self.expect_no_results and not self.relevant_ids:
            raise ValueError(
                "A positive case must contain relevant identifiers."
            )

        unknown_status_ids = (
            set(self.expected_statuses) - set(self.relevant_ids)
        )

        if unknown_status_ids:
            raise ValueError(
                "Expected statuses may only be supplied for relevant records."
            )

        return self


class BenchmarkDataset(BaseModel):
    """A complete reproducible evaluation dataset."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    as_of: date
    cases: list[BenchmarkCase] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_case_ids(self) -> Self:
        """Require unique benchmark case identifiers."""
        identifiers = [case.case_id for case in self.cases]

        if len(identifiers) != len(set(identifiers)):
            raise ValueError(
                "Benchmark case identifiers must be unique."
            )

        return self


def load_benchmark(path: str | Path) -> BenchmarkDataset:
    """Load and validate a benchmark JSON file."""
    benchmark_path = Path(path)

    if not benchmark_path.is_file():
        raise BenchmarkError(
            f"Benchmark file does not exist: {benchmark_path}"
        )

    try:
        raw_data = json.loads(
            benchmark_path.read_text(encoding="utf-8")
        )
    except json.JSONDecodeError as exc:
        raise BenchmarkError(
            f"Benchmark is not valid JSON: {benchmark_path}"
        ) from exc

    try:
        return BenchmarkDataset.model_validate(raw_data)
    except ValidationError as exc:
        raise BenchmarkError(
            f"Benchmark validation failed: {exc}"
        ) from exc
