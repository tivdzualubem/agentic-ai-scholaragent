"""Scholarship eligibility screening."""

from scholaragent.eligibility.engine import assess_eligibility
from scholaragent.eligibility.models import (
    EligibilityAssessment,
    EligibilityStatus,
)

__all__ = [
    "EligibilityAssessment",
    "EligibilityStatus",
    "assess_eligibility",
]
