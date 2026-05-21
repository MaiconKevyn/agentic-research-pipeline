from unittest.mock import patch

from fastapi import HTTPException

from backend.app.api.routes.workspace import (
    export_run,
    get_run,
    get_run_source,
    list_feedback_eval_cases,
    list_runs,
    submit_feedback,
)
from backend.app.schemas.research import (
    CorpusStats,
    FeedbackRequest,
    FeedbackEvalCase,
    ResearchRequest,
    RunSourceDetail,
    RunSummary,
    SourceItem,
    WorkspaceRun,
)


def _workspace_run() -> WorkspaceRun:
    source = SourceItem(
        source_id="source-1",
        title="RAG source",
        snippet="RAG retrieval grounds answers.",
        source_type="pdf_chunk",
        metadata={"page_start": 3, "chunk_id": "chunk-1"},
    )
    return WorkspaceRun(
        run_id="run-1",
        corpus_version_id="corpus-v1",
        corpus_stats=CorpusStats(source_document_count=1, chunk_count=1, corpus_version_id="corpus-v1"),
        question="What does this project say about RAG?",
        answer="RAG retrieval grounds answers.",
        answer_mode="detailed",
        sources=[source],
        claims=[],
        evaluation=[],
        execution_trace=["answer_mode=detailed"],
        created_at="2026-05-21T12:00:00Z",
        latency_ms=123,
        feedback=[],
        replay_request=ResearchRequest(
            question="What does this project say about RAG?",
            top_k=1,
            answer_mode="detailed",
        ),
    )


def test_workspace_routes_list_runs_and_return_run_detail() -> None:
    summaries = [
        RunSummary(
            run_id="run-1",
            question="What does this project say about RAG?",
            answer_preview="RAG retrieval grounds answers.",
            answer_mode="detailed",
            source_count=1,
            created_at="2026-05-21T12:00:00Z",
            latency_ms=123,
        )
    ]
    with (
        patch("backend.app.api.routes.workspace.list_research_runs", return_value=summaries) as mocked_list,
        patch("backend.app.api.routes.workspace.get_research_run", return_value=_workspace_run()) as mocked_get,
    ):
        assert list_runs(limit=10) == summaries
        detail = get_run("run-1")

    mocked_list.assert_called_once_with(limit=10)
    mocked_get.assert_called_once_with("run-1")
    assert detail.run_id == "run-1"
    assert detail.replay_request.answer_mode == "detailed"


def test_workspace_routes_raise_404_for_missing_run() -> None:
    with patch("backend.app.api.routes.workspace.get_research_run", return_value=None):
        try:
            get_run("missing-run")
        except HTTPException as exc:
            assert exc.status_code == 404
        else:
            raise AssertionError("Expected missing run to raise HTTPException")


def test_workspace_routes_return_source_detail_with_highlights() -> None:
    detail = RunSourceDetail(
        run_id="run-1",
        source_id="source-1",
        title="RAG source",
        source_type="pdf_chunk",
        snippet="RAG retrieval grounds answers.",
        text="RAG retrieval grounds answers. More page text.",
        source_document_id="doc-1",
        chunk_id="chunk-1",
        page_start=3,
        page_end=3,
        section_title="Retrieval",
        highlights=["RAG retrieval grounds answers."],
        metadata={"page_start": 3},
    )
    with patch("backend.app.api.routes.workspace.get_research_run_source", return_value=detail):
        response = get_run_source("run-1", "source-1")

    assert response.text.startswith("RAG retrieval")
    assert response.highlights == ["RAG retrieval grounds answers."]


def test_workspace_export_route_returns_markdown_response() -> None:
    with patch("backend.app.api.routes.workspace.get_research_run", return_value=_workspace_run()):
        response = export_run("run-1", export_format="markdown")

    assert response.media_type == "text/markdown"
    assert "attachment; filename=run-1.md" in response.headers["content-disposition"]
    assert b"# Research Run run-1" in response.body


def test_workspace_feedback_route_persists_feedback() -> None:
    payload = FeedbackRequest(
        rating="up",
        comment="Useful evidence.",
        add_to_eval=True,
        corrected_answer=None,
    )
    with patch("backend.app.api.routes.workspace.record_run_feedback", return_value="feedback-1") as mocked_record:
        response = submit_feedback("run-1", payload)

    mocked_record.assert_called_once()
    assert response.feedback_id == "feedback-1"
    assert response.status == "recorded"


def test_workspace_feedback_eval_cases_route_returns_eval_ready_items() -> None:
    cases = [
        FeedbackEvalCase(
            id="feedback-1",
            question="What does this project say about RAG?",
            expected_answer_type="factual",
            required_sources=["source-1"],
            expected_facts=["retrieval"],
            forbidden_claims=[],
            answer_rubric="Use corrected researcher feedback.",
            difficulty="feedback",
            query_category="feedback",
            top_k=1,
        )
    ]
    with patch("backend.app.api.routes.workspace.list_feedback_as_eval_cases", return_value=cases) as mocked_list:
        response = list_feedback_eval_cases(limit=100)

    mocked_list.assert_called_once_with(limit=100)
    assert response == cases


def test_research_request_supports_answer_modes() -> None:
    payload = ResearchRequest(
        question="What does this project say about RAG?",
        top_k=3,
        answer_mode="concise",
    )

    assert payload.answer_mode == "concise"
