import json

from backend.app.schemas.research import CorpusStats, ResearchRequest, SourceItem, WorkspaceRun
from backend.app.services.export_service import export_run_csv, export_run_json, export_run_markdown


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
        answer_mode="evidence_table",
        sources=[source],
        claims=[],
        evaluation=[],
        execution_trace=["answer_mode=evidence_table"],
        created_at="2026-05-21T12:00:00Z",
        latency_ms=123,
        feedback=[],
        replay_request=ResearchRequest(
            question="What does this project say about RAG?",
            top_k=1,
            answer_mode="evidence_table",
        ),
    )


def test_export_run_markdown_includes_answer_sources_and_replay_request() -> None:
    output = export_run_markdown(_workspace_run())

    assert "# Research Run run-1" in output
    assert "## Answer" in output
    assert "RAG retrieval grounds answers." in output
    assert "source-1" in output
    assert "answer_mode: evidence_table" in output


def test_export_run_csv_emits_source_rows() -> None:
    output = export_run_csv(_workspace_run())

    assert "run_id,source_id,title,source_type,page_start,chunk_id,snippet" in output
    assert "run-1,source-1,RAG source,pdf_chunk,3,chunk-1,RAG retrieval grounds answers." in output


def test_export_run_json_round_trips_structured_payload() -> None:
    payload = json.loads(export_run_json(_workspace_run()))

    assert payload["run_id"] == "run-1"
    assert payload["answer_mode"] == "evidence_table"
    assert payload["sources"][0]["source_id"] == "source-1"
    assert payload["replay_request"]["top_k"] == 1
