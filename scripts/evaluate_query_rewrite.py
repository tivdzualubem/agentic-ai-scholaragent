import json
from datetime import date
from pathlib import Path

from scholaragent.agents.scholar_graph import (
    ScholarAgentOutcome,
    build_scholar_agent_graph,
)
from scholaragent.retrieval import (
    BM25ScholarshipIndex,
    load_scholarships,
)
from scholaragent.schemas import StudentProfile

DATASET = Path("data/demo/synthetic_scholarships.json")
OUTPUT = Path("eval/results/query_rewrite_validation.json")
AS_OF = date(2026, 6, 27)

profile = StudentProfile(
    nationality="Nigerian",
    country_of_residence="Finland",
    target_degree_level="master",
    fields_of_study=["Artificial Intelligence", "Data Science"],
    gpa=4.2,
    gpa_scale=5.0,
    language_scores={"IELTS": 7.5},
    years_work_experience=1.0,
    preferred_countries=["Finland"],
    requires_full_funding=True,
)

graph = build_scholar_agent_graph(
    BM25ScholarshipIndex(load_scholarships(DATASET))
)

cases = [
    {
        "case_id": "specific_query",
        "query": (
            "fully funded artificial intelligence "
            "master's scholarship in Finland"
        ),
        "expected_status": "completed",
        "expected_rewrites": 0,
    },
    {
        "case_id": "generic_query_requires_rewrite",
        "query": "scholarship",
        "expected_status": "completed",
        "expected_rewrites": 1,
    },
    {
        "case_id": "out_of_domain_no_drift",
        "query": "xylophone archaeology scholarship on Mars",
        "expected_status": "abstained",
        "expected_rewrites": 0,
    },
]

results = []

for case in cases:
    state = graph.invoke(
        {
            "original_query": case["query"],
            "profile": profile,
            "as_of": AS_OF,
            "top_k": 3,
            "max_attempts": 2,
        }
    )

    outcome = ScholarAgentOutcome.model_validate(state["outcome"])

    passed = (
        outcome.status == case["expected_status"]
        and len(outcome.rewrites) == case["expected_rewrites"]
    )

    results.append(
        {
            **case,
            "observed_status": outcome.status,
            "observed_rewrites": len(outcome.rewrites),
            "final_query": outcome.final_query,
            "attempts": outcome.attempts,
            "candidate_ids": [
                item.scholarship_id
                for item in outcome.candidates
            ],
            "passed": passed,
        }
    )

artifact = {
    "evaluation_type": "development_query_rewrite_validation",
    "dataset": str(DATASET),
    "as_of": AS_OF.isoformat(),
    "case_count": len(results),
    "passed_count": sum(item["passed"] for item in results),
    "all_passed": all(item["passed"] for item in results),
    "scope_note": (
        "This validates bounded rewrite control flow on the "
        "development corpus. It is not a held-out effectiveness claim."
    ),
    "cases": results,
}

OUTPUT.write_text(
    json.dumps(artifact, indent=2) + "\n",
    encoding="utf-8",
)

assert artifact["all_passed"]
print(json.dumps(artifact, indent=2))
