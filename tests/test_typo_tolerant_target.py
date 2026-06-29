import json
from pathlib import Path

from scholaragent.retrieval.bm25 import SearchResult
from scholaragent.schemas import ScholarshipRecord
from scholaragent.ui.demo import ExactTargetScholarshipIndex


def _records() -> list[ScholarshipRecord]:
    raw = json.loads(
        Path("data/demo/combined_scholarships.json").read_text(
            encoding="utf-8"
        )
    )
    return [
        ScholarshipRecord.model_validate(item)
        for item in raw
    ]


class WrongOrderIndex:
    def __init__(
        self,
        records: list[ScholarshipRecord],
    ) -> None:
        self.records = records

    def search(self, query: str, *, k: int = 5):
        maastricht = next(
            item for item in self.records
            if "maastricht" in item.scholarship_id
        )
        glasgow = next(
            item for item in self.records
            if "glasgow" in item.scholarship_id
        )
        bristol = next(
            item for item in self.records
            if "bristol" in item.scholarship_id
        )

        return [
            SearchResult(
                rank=1,
                score=10.0,
                scholarship=maastricht,
            ),
            SearchResult(
                rank=2,
                score=9.0,
                scholarship=glasgow,
            ),
            SearchResult(
                rank=3,
                score=8.0,
                scholarship=bristol,
            ),
        ][:k]


def test_misspelled_glasgow_provider_is_first() -> None:
    records = _records()
    index = ExactTargetScholarshipIndex(
        WrongOrderIndex(records),
        records,
    )

    results = index.search(
        "Am I eligible for the Univeristy of Glasgo "
        "Global Leadership Scholarship?",
        k=3,
    )

    assert "glasgow" in results[0].scholarship.scholarship_id


def test_misspelled_bristol_title_is_first() -> None:
    records = _records()
    index = ExactTargetScholarshipIndex(
        WrongOrderIndex(records),
        records,
    )

    results = index.search(
        "Find the Thnik Big Scholrships at Bristol.",
        k=3,
    )

    assert "bristol" in results[0].scholarship.scholarship_id
