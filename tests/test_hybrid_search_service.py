from unittest.mock import patch

from backend.app.schemas.research import SourceItem
from backend.app.services.hybrid_search import search_hybrid_documents


def test_hybrid_search_fuses_dense_and_lexical_rankings() -> None:
    dense_sources = [
        SourceItem(
            source_id="dense-only",
            title="Dense only",
            snippet="Semantic match.",
            source_type="pdf_chunk",
            url=None,
            metadata={"retrieval_rank": 1},
        ),
        SourceItem(
            source_id="shared",
            title="Shared source",
            snippet="Semantic and lexical match.",
            source_type="pdf_chunk",
            url=None,
            metadata={"retrieval_rank": 2},
        ),
    ]
    lexical_sources = [
        SourceItem(
            source_id="shared",
            title="Shared source",
            snippet="Semantic and lexical match.",
            source_type="pdf_chunk",
            url=None,
            metadata={"lexical_rank": 1},
        ),
        SourceItem(
            source_id="lexical-only",
            title="Lexical only",
            snippet="Keyword match.",
            source_type="pdf_chunk",
            url=None,
            metadata={"lexical_rank": 2},
        ),
    ]

    with (
        patch("backend.app.services.hybrid_search.generate_embedding", return_value=[0.1, 0.2]),
        patch("backend.app.services.hybrid_search.search_similar_documents", return_value=dense_sources),
        patch("backend.app.services.hybrid_search.search_lexical_documents", return_value=lexical_sources),
    ):
        results = search_hybrid_documents("rag retrieval", top_k=3, dense_top_k=30, lexical_top_k=30)

    assert [source.source_id for source in results] == ["shared", "dense-only", "lexical-only"]
    assert results[0].metadata["retrieval_path"] == "hybrid"
    assert results[0].metadata["dense_rank"] == 2
    assert results[0].metadata["lexical_rank"] == 1
    assert results[0].metadata["hybrid_rank"] == 1
    assert results[0].metadata["hybrid_score"] > results[1].metadata["hybrid_score"]
