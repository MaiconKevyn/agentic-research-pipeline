from unittest.mock import patch

from agent.tools.sql_query import query_structured_data
from backend.app.schemas.research import CorpusStats


def test_sql_query_reports_source_document_and_chunk_counts() -> None:
    with patch(
        "agent.tools.sql_query.get_corpus_stats",
        return_value=CorpusStats(
            source_document_count=4,
            chunk_count=19,
            corpus_version_id="corpus-v2",
        ),
    ):
        sources = query_structured_data("How many source documents and chunks are indexed?")

    assert len(sources) == 1
    assert sources[0].source_id == "sql-corpus-stats"
    assert "4 source documents" in sources[0].snippet
    assert "19 indexed chunks" in sources[0].snippet
    assert sources[0].metadata["source_document_count"] == 4
    assert sources[0].metadata["chunk_count"] == 19
    assert sources[0].metadata["corpus_version_id"] == "corpus-v2"
