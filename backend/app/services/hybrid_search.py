from __future__ import annotations

from backend.app.schemas.research import SourceItem
from backend.app.services.document_repository import (
    search_lexical_documents,
    search_similar_documents,
)
from backend.app.services.embedding_service import generate_embedding


DEFAULT_DENSE_TOP_K = 30
DEFAULT_LEXICAL_TOP_K = 30
RRF_K = 60


def search_hybrid_documents(
    question: str,
    top_k: int,
    dense_top_k: int = DEFAULT_DENSE_TOP_K,
    lexical_top_k: int = DEFAULT_LEXICAL_TOP_K,
) -> list[SourceItem]:
    query_embedding = generate_embedding(question)
    dense_sources = search_similar_documents(query_embedding=query_embedding, top_k=dense_top_k)
    lexical_sources = search_lexical_documents(query=question, top_k=lexical_top_k)
    fused_sources = reciprocal_rank_fuse(
        dense_sources=dense_sources,
        lexical_sources=lexical_sources,
    )
    return fused_sources[:top_k]


def reciprocal_rank_fuse(
    *,
    dense_sources: list[SourceItem],
    lexical_sources: list[SourceItem],
    rrf_k: int = RRF_K,
) -> list[SourceItem]:
    source_map: dict[str, SourceItem] = {}
    scores: dict[str, float] = {}

    for rank, source in enumerate(dense_sources, start=1):
        source_map[source.source_id] = _copy_source_with_metadata(source)
        source_map[source.source_id].metadata["dense_rank"] = rank
        scores[source.source_id] = scores.get(source.source_id, 0.0) + 1.0 / (rrf_k + rank)

    for rank, source in enumerate(lexical_sources, start=1):
        if source.source_id not in source_map:
            source_map[source.source_id] = _copy_source_with_metadata(source)
        else:
            source_map[source.source_id].metadata.update(source.metadata)
        source_map[source.source_id].metadata["lexical_rank"] = rank
        scores[source.source_id] = scores.get(source.source_id, 0.0) + 1.0 / (rrf_k + rank)

    ranked_ids = sorted(scores, key=lambda source_id: (-scores[source_id], source_id))
    fused: list[SourceItem] = []
    for hybrid_rank, source_id in enumerate(ranked_ids, start=1):
        source = source_map[source_id]
        source.metadata["retrieval_path"] = "hybrid"
        source.metadata["hybrid_rank"] = hybrid_rank
        source.metadata["hybrid_score"] = round(scores[source_id], 8)
        fused.append(source)
    return fused


def _copy_source_with_metadata(source: SourceItem) -> SourceItem:
    return source.model_copy(update={"metadata": dict(source.metadata)})
