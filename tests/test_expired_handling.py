"""Regression tests for expired-opportunity handling."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


RESULT = Path(
    "eval/results/expired_opportunity_handling.json"
)
DOCUMENT = Path("docs/expired_handling.md")


def load_result() -> dict:
    return json.loads(
        RESULT.read_text(encoding="utf-8")
    )


def sha256(path: Path) -> str:
    return hashlib.sha256(
        path.read_bytes()
    ).hexdigest()


def cases_by_id() -> dict[str, dict]:
    return {
        case["case_id"]: case
        for case in load_result()["cases"]
    }


def test_expired_artifact_metadata() -> None:
    payload = load_result()

    assert payload["artifact_type"] == (
        "expired_opportunity_handling_evidence"
    )
    assert len(payload["cases"]) == 2

    constraints = payload["evaluation_constraints"]

    assert constraints["new_model_calls_performed"] is False
    assert constraints["held_out_test_used"] is False
    assert constraints["parameter_selection_performed"] is False


def test_expired_artifact_provenance() -> None:
    for source in load_result()["provenance"].values():
        path = Path(source["path"])

        assert path.is_file()
        assert source["sha256"] == sha256(path)


def test_both_expired_cases_are_not_eligible() -> None:
    for case in load_result()["cases"]:
        assert case["expected_status"] == "not_eligible"
        assert case["predicted_status"] == "not_eligible"
        assert "deadline" in case["deadline_failure"].lower()
        assert "passed" in case["deadline_failure"].lower()


def test_kth_expired_round() -> None:
    case = cases_by_id()[
        "not-eligible-kth-expired-round"
    ]

    assert case["deadline"] == "2026-01-15"
    assert case["benchmark_as_of"] == "2026-05-01"


def test_si_agentic_fallback() -> None:
    case = cases_by_id()[
        "si-global-professionals-expired-round"
    ]

    assert case["deadline"] == "2026-02-25"
    assert case["benchmark_as_of"] == "2026-06-28"

    assert case["rag_evidence"]["single_pass_rag"][
        "status"
    ] == "citation_failed"

    agentic = case["rag_evidence"]["agentic_rag"]

    assert agentic["status"] == "completed_fallback"
    assert agentic["citation_passed"] is True
    assert agentic["fallback_used"] is True


def test_expired_handling_document() -> None:
    text = DOCUMENT.read_text(encoding="utf-8")

    assert "Expired Opportunity Handling" in text
    assert "KTH Scholarship" in text
    assert "SI Scholarship for Global Professionals" in text
    assert "not_eligible" in text
    assert "must confirm the current round" in text
