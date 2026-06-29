"""Regression tests for representative execution traces."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


RESULT = Path(
    "eval/results/held_out_execution_traces.json"
)

DOCUMENT = Path("docs/execution_traces.md")


def load_result() -> dict:
    return json.loads(
        RESULT.read_text(encoding="utf-8")
    )


def sha256(path: Path) -> str:
    return hashlib.sha256(
        path.read_bytes()
    ).hexdigest()


def traces_by_type() -> dict[str, dict]:
    return {
        trace["trace_type"]: trace
        for trace in load_result()["traces"]
    }


def test_trace_artifact_metadata() -> None:
    payload = load_result()

    assert payload["artifact_type"] == (
        "held_out_representative_execution_traces"
    )
    assert payload["partition"] == "held_out_test"
    assert payload["selected_trace_count"] == 6
    assert len(payload["traces"]) == 6

    constraints = payload["evaluation_constraints"]

    assert constraints["new_model_calls_performed"] is False
    assert constraints["held_out_tuning_performed"] is False
    assert (
        constraints["parameter_selection_after_test"]
        is False
    )
    assert (
        constraints["independent_confirmatory_evidence"]
        is False
    )


def test_trace_source_provenance() -> None:
    for source in load_result()["provenance"].values():
        path = Path(source["path"])

        assert path.is_file()
        assert source["sha256"] == sha256(path)


def test_representative_behaviors_exist() -> None:
    traces = traces_by_type()

    assert set(traces) == {
        "eligible_verified_fallback",
        "potentially_eligible_manual_review",
        "not_eligible_hard_constraint",
        "insufficient_information",
        "safe_abstention",
        "transport_timeout_recovery",
    }


def test_positive_fallback_trace() -> None:
    trace = traces_by_type()[
        "eligible_verified_fallback"
    ]

    assert trace["finalization_stage"]["status"] == (
        "completed_fallback"
    )
    assert trace["verification_stage"]["fallback_used"] is True
    assert (
        trace["verification_stage"]["final_citation_passed"]
        is True
    )


def test_safe_abstention_trace() -> None:
    trace = traces_by_type()["safe_abstention"]

    assert trace["finalization_stage"]["status"] == "abstained"
    assert trace["retrieval_stage"]["candidate_count"] == 0
    assert trace["generation_stage"]["generation_calls"] == 0
    assert trace["verification_stage"]["fallback_used"] is False
    assert trace["evaluation"]["no_result_correct"] is True


def test_timeout_recovery_trace() -> None:
    trace = traces_by_type()[
        "transport_timeout_recovery"
    ]

    failure = trace["generation_stage"]["transport_failure"]

    assert failure is not None
    assert failure["timeout_seconds"] == 900.0
    assert failure["failure_marker"] == (
        "GENERATION_FAILED_DUE_TO_OLLAMA_TRANSPORT_ERROR"
    )
    assert trace["verification_stage"]["fallback_used"] is True


def test_human_readable_document() -> None:
    text = DOCUMENT.read_text(encoding="utf-8")

    assert "Representative ScholarAgent Execution Traces" in text
    assert "Eligible scholarship recovered" in text
    assert "safe abstention" in text.lower()
    assert "Generation timeout recovered" in text
