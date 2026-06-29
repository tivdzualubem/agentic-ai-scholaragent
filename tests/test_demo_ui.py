"""Tests for the ScholarAgent demonstration application."""

from __future__ import annotations

from datetime import date
from pathlib import Path

from streamlit.testing.v1 import AppTest

from scholaragent.retrieval import BM25ScholarshipIndex
from scholaragent.schemas import (
    ScholarshipRecord,
    StudentProfile,
)
from scholaragent.ui import (
    DemoExecutionMode,
    DemoRetrieverMode,
    build_student_profile,
    parse_csv_values,
    run_demo_workflow,
)


def test_parse_csv_values() -> None:
    """CSV form values should be cleaned and deduplicated."""
    assert parse_csv_values(
        "AI, Data Science, ai, , Robotics"
    ) == [
        "AI",
        "Data Science",
        "Robotics",
    ]


def test_build_student_profile() -> None:
    """Form-compatible values should create a valid profile."""
    profile = build_student_profile(
        nationality="Nigerian",
        country_of_residence="Finland",
        target_degree_level="master",
        fields_of_study=(
            "Artificial Intelligence, Data Science"
        ),
        include_gpa=True,
        gpa=4.2,
        gpa_scale=5.0,
        include_language_score=True,
        language_test="IELTS",
        language_score=7.5,
        years_work_experience=2.0,
        preferred_countries="Sweden, Netherlands",
        requires_full_funding=True,
    )

    assert profile.nationality == "Nigerian"
    assert profile.fields_of_study == [
        "Artificial Intelligence",
        "Data Science",
    ]
    assert profile.language_scores == {
        "IELTS": 7.5,
    }
    assert profile.gpa == 4.2
    assert profile.gpa_scale == 5.0


def test_optional_profile_values_can_be_omitted() -> None:
    """Unknown academic evidence should remain optional."""
    profile = build_student_profile(
        nationality="Ghanaian",
        country_of_residence="",
        target_degree_level="phd",
        fields_of_study="Civil Engineering",
        include_gpa=False,
        gpa=0.0,
        gpa_scale=1.0,
        include_language_score=False,
        language_test="",
        language_score=0.0,
        years_work_experience=1.0,
        preferred_countries="",
        requires_full_funding=False,
    )

    assert profile.country_of_residence is None
    assert profile.gpa is None
    assert profile.gpa_scale is None
    assert profile.language_scores == {}
    assert profile.preferred_countries == []


def test_fast_demo_uses_verified_fallback_without_llm() -> None:
    """Fast mode should demonstrate recovery without Ollama."""
    scholarship = ScholarshipRecord(
        scholarship_id="demo-ai-scholarship",
        title="Demo AI Scholarship",
        provider="Example University",
        official_url=(
            "https://example.edu/demo-ai-scholarship"
        ),
        host_countries=["Finland"],
        degree_levels=["master"],
        eligible_nationalities=["all"],
        eligible_fields=[
            "Artificial Intelligence",
            "Data Science",
        ],
        manual_review_requirements=[],
        minimum_gpa=3.0,
        gpa_scale=4.0,
        minimum_work_experience_years=None,
        language_requirements={},
        funding_type="fully_funded",
        deadline=date(2027, 1, 31),
        application_year=2027,
        source_last_checked=date(2026, 6, 29),
        eligibility_text=(
            "Open to international master's applicants "
            "in artificial intelligence and data science."
        ),
    )

    profile = StudentProfile(
        nationality="Nigerian",
        country_of_residence="Finland",
        target_degree_level="master",
        fields_of_study=[
            "Artificial Intelligence",
        ],
        gpa=3.8,
        gpa_scale=4.0,
        years_work_experience=1.0,
        preferred_countries=["Finland"],
        requires_full_funding=True,
    )

    execution = run_demo_workflow(
        query="Demo AI Scholarship",
        profile=profile,
        index=BM25ScholarshipIndex(
            [scholarship]
        ),
        corpus_path="tests/demo-corpus.json",
        corpus_size=1,
        execution_mode=(
            DemoExecutionMode.FAST_VERIFIED
        ),
        retriever_mode=(
            DemoRetrieverMode.BM25_ONLY
        ),
        as_of=date(2026, 6, 29),
    )

    assert execution.external_llm_calls == 0
    assert execution.result.status == (
        "completed_fallback"
    )
    assert execution.result.fallback_used is True
    assert (
        execution.result.citation_audit.passed
        is True
    )
    assert execution.result.retrieval_calls == 1
    assert execution.result.generation_calls == 2


def test_streamlit_initial_render() -> None:
    """The Streamlit page should render without exceptions."""
    path = Path(
        "src/scholaragent/ui/streamlit_app.py"
    )

    app = AppTest.from_file(
        str(path)
    ).run(timeout=30)

    assert not app.exception
    assert len(app.title) == 1
    assert app.title[0].value == "🎓 ScholarAgent"
    assert len(app.selectbox) >= 4
    assert len(app.button) >= 1
