from unittest.mock import patch

from backend.app.api.routes.research import research
from backend.app.schemas.research import ResearchRequest, SourceItem, SynthesisOutput


def test_research_route_returns_structured_response() -> None:
    mocked_sources = [
        SourceItem(
            source_id="vector-1",
            title="Internal document",
            snippet="Passage retrieved from the corpus.",
            source_type="vector",
            url=None,
            metadata={
                "source_file": "RAG.pdf",
                "section_title": "Abstract",
                "page_start": 1,
                "page_end": 1,
            },
        ),
        SourceItem(
            source_id="web-1",
            title="Official source",
            snippet="Official documentation passage.",
            source_type="web",
            url="https://example.com/official",
            metadata={"domain": "example.com", "retrieval_rank": 1},
        ),
    ]

    with (
        patch("agent.nodes.search_documents", return_value=[mocked_sources[0]]),
        patch("agent.nodes.search_web", return_value=[mocked_sources[1]]),
        patch("agent.nodes.query_structured_data", return_value=[]),
        patch("agent.nodes.rerank_sources_global", return_value=[mocked_sources[1], mocked_sources[0]]),
        patch(
            "agent.nodes.generate_research_answer",
            return_value=SynthesisOutput(
                answer="Final answer grounded in the evidence [1] [2].",
                confidence="high",
                cited_source_ids=["web-1", "vector-1"],
                uncertainty_note=None,
            ),
        ),
    ):
        response = research(
            ResearchRequest(
                question="What are the main points of the project?",
                top_k=3,
            )
        )

    assert response.question == "What are the main points of the project?"
    assert response.answer == "Final answer grounded in the evidence [1] [2]."
    assert len(response.sources) == 2
    assert response.sources[0].source_type == "web"
    assert response.evaluation
    assert "classified_question" in response.execution_trace
    assert "vector_search_results=1" in response.execution_trace
    assert "web_search_results=1" in response.execution_trace
    assert "global_rerank_applied" in response.execution_trace
    assert "synthesis_confidence=high" in response.execution_trace
    assert "llm_synthesis_success" in response.execution_trace


def test_research_route_rejects_out_of_scope_questions_without_calling_tools() -> None:
    with (
        patch("agent.nodes.search_documents") as mocked_vector_search,
        patch("agent.nodes.search_web") as mocked_web_search,
        patch("agent.nodes.query_structured_data") as mocked_sql_query,
        patch("agent.nodes.generate_research_answer") as mocked_llm_synthesis,
    ):
        response = research(
            ResearchRequest(
                question="what is the most beautiful animal?",
                top_k=3,
            )
        )

    mocked_vector_search.assert_not_called()
    mocked_web_search.assert_not_called()
    mocked_sql_query.assert_not_called()
    mocked_llm_synthesis.assert_not_called()
    assert response.sources == []
    assert response.answer.startswith("This assistant only answers questions about this project")
    assert any(metric.metric == "scope_compliance" for metric in response.evaluation)
    assert "classification_query_kind=off_topic" in response.execution_trace
    assert "scope_guardrail_triggered" in response.execution_trace
    assert "scope_guardrail_skipped_evidence_collection" in response.execution_trace
    assert "scope_guardrail_response" in response.execution_trace
