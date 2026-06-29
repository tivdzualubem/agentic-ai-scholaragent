import json
from pathlib import Path

from scholaragent.retrieval.bm25 import SearchResult
from scholaragent.schemas import ScholarshipRecord
from scholaragent.ui.demo import ExactTargetScholarshipIndex


class DeliberatelyWrongIndex:
    def __init__(self, records: list[ScholarshipRecord]) -> None:
        self.records = records

    def search(self, query: str, *, k: int = 5):
        maastricht = next(
            item for item in self.records
            if "maastricht" in item.scholarship_id
        )
        bristol = next(
            item for item in self.records
            if "bristol" in item.scholarship_id
        )

        return [
            SearchResult(rank=1, score=10.0, scholarship=maastricht),
            SearchResult(rank=2, score=9.0, scholarship=bristol),
        ][:k]


def load_records() -> list[ScholarshipRecord]:
    raw = json.loads(
        Path("data/demo/combined_scholarships.json").read_text(
            encoding="utf-8"
        )
    )
    return [
        ScholarshipRecord.model_validate(item)
        for item in raw
    ]


def test_named_provider_is_ranked_first() -> None:
    records = load_records()
    index = ExactTargetScholarshipIndex(
        DeliberatelyWrongIndex(records),
        records,
    )

    results = index.search(
        "Find a master's scholarship at the University of Bristol.",
        k=2,
    )

    assert results[0].scholarship.scholarship_id == (
        "bristol-think-big-scholarships-2026"
    )


def test_named_title_is_ranked_first() -> None:
    records = load_records()
    index = ExactTargetScholarshipIndex(
        DeliberatelyWrongIndex(records),
        records,
    )

    results = index.search(
        "Am I eligible for the Think Big Scholarships?",
        k=2,
    )

    assert results[0].scholarship.scholarship_id == (
        "bristol-think-big-scholarships-2026"
    )


def test_generic_query_preserves_base_ranking() -> None:
    records = load_records()
    index = ExactTargetScholarshipIndex(
        DeliberatelyWrongIndex(records),
        records,
    )

    results = index.search(
        "Find a fully funded European master's scholarship.",
        k=2,
    )

    assert "maastricht" in results[0].scholarship.scholarship_id
