"""Load and validate scholarship records from JSON files."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import ValidationError

from scholaragent.schemas import ScholarshipRecord


class ScholarshipDataError(ValueError):
    """Raised when a scholarship dataset is invalid."""


def load_scholarships(path: str | Path) -> list[ScholarshipRecord]:
    """Load and validate a JSON list of scholarship records."""
    source_path = Path(path)

    if not source_path.is_file():
        raise ScholarshipDataError(
            f"Scholarship dataset does not exist: {source_path}"
        )

    try:
        raw_data = json.loads(source_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ScholarshipDataError(
            f"Scholarship dataset is not valid JSON: {source_path}"
        ) from exc

    if not isinstance(raw_data, list):
        raise ScholarshipDataError(
            "Scholarship dataset must contain a JSON list."
        )

    records: list[ScholarshipRecord] = []

    try:
        for position, item in enumerate(raw_data, start=1):
            if not isinstance(item, dict):
                raise ScholarshipDataError(
                    f"Record {position} must be a JSON object."
                )

            records.append(ScholarshipRecord.model_validate(item))
    except ValidationError as exc:
        raise ScholarshipDataError(
            f"Scholarship record {position} failed validation: {exc}"
        ) from exc

    identifiers = [record.scholarship_id for record in records]

    if len(identifiers) != len(set(identifiers)):
        raise ScholarshipDataError(
            "Scholarship identifiers must be unique."
        )

    return records
