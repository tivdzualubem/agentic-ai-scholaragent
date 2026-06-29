import json
from pathlib import Path


def test_query_rewrite_validation_artifact() -> None:
    data = json.loads(
        Path(
            "eval/results/query_rewrite_validation.json"
        ).read_text(encoding="utf-8")
    )

    assert data["evaluation_type"] == (
        "development_query_rewrite_validation"
    )
    assert data["case_count"] == 3
    assert data["passed_count"] == 3
    assert data["all_passed"] is True

    cases = {
        item["case_id"]: item
        for item in data["cases"]
    }

    assert cases[
        "generic_query_requires_rewrite"
    ]["observed_rewrites"] == 1

    assert cases[
        "out_of_domain_no_drift"
    ]["observed_rewrites"] == 0

    assert cases[
        "out_of_domain_no_drift"
    ]["observed_status"] == "abstained"
