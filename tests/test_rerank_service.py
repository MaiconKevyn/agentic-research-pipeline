from unittest.mock import patch

from backend.app.schemas.research import SourceItem
from backend.app.services.rerank_service import rerank_sources_global


def test_global_rerank_orders_candidates_by_semantic_similarity() -> None:
    sources = [
        SourceItem(
            source_id="vector-1",
            title="Documento interno",
            snippet="LangGraph coordinates stateful agent workflows.",
            source_type="pdf_chunk",
            url=None,
            metadata={"retrieval_rank": 2},
        ),
        SourceItem(
            source_id="web-1",
            title="Fonte web",
            snippet="Renewable energy market outlook and incentives.",
            source_type="web",
            url="https://example.com/energy",
            metadata={"retrieval_rank": 1},
        ),
    ]

    with patch(
        "backend.app.services.rerank_service.generate_embeddings",
        return_value=[
            [1.0, 0.0],
            [0.95, 0.05],
            [0.10, 0.90],
        ],
    ):
        reranked = rerank_sources_global(
            question="How does LangGraph orchestrate agent workflows?",
            sources=sources,
        )

    assert [source.source_id for source in reranked] == ["vector-1", "web-1"]
    assert reranked[0].metadata["global_rerank_rank"] == 1
    assert reranked[0].metadata["global_rerank_score"] > reranked[1].metadata["global_rerank_score"]
