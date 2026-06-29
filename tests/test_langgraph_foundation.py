"""Smoke tests for the LangGraph orchestration dependency."""

from scholaragent.agents.graph_smoke import build_smoke_graph


def test_langgraph_state_graph_executes() -> None:
    """The local LangGraph installation can compile and execute a graph."""
    graph = build_smoke_graph()

    result = graph.invoke(
        {
            "message": "ScholarAgent",
            "completed": False,
        }
    )

    assert result == {
        "message": "ScholarAgent -> LangGraph working",
        "completed": True,
    }
