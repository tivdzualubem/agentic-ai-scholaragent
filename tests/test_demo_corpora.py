import json
from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_combined_demo_corpus_contains_every_scholarship() -> None:
    paths = [
        Path("data/official/official_scholarships.json"),
        Path("data/calibration/calibration_scholarships.json"),
        Path("data/held_out/held_out_scholarships.json"),
    ]

    expected_ids = {
        item["scholarship_id"]
        for path in paths
        for item in json.loads(path.read_text(encoding="utf-8"))
    }

    combined = json.loads(
        Path("data/demo/combined_scholarships.json").read_text(
            encoding="utf-8"
        )
    )
    combined_ids = {item["scholarship_id"] for item in combined}

    assert len(combined) == 15
    assert combined_ids == expected_ids
    assert "bristol-think-big-scholarships-2026" in combined_ids


def test_app_defaults_to_complete_demo_corpus() -> None:
    app = AppTest.from_file(
        "src/scholaragent/ui/streamlit_app.py"
    ).run(timeout=30)

    assert not app.exception

    corpus_selectors = [
        item
        for item in app.selectbox
        if item.label == "Scholarship corpus"
    ]

    assert len(corpus_selectors) == 1
    assert corpus_selectors[0].value == (
        "Combined demo corpus (15 scholarships)"
    )
