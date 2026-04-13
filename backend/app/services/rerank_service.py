from __future__ import annotations

from math import sqrt

from backend.app.core.logging import get_logger
from backend.app.schemas.research import SourceItem
from backend.app.services.embedding_service import EmbeddingServiceError, generate_embeddings


logger = get_logger(__name__)


class RerankServiceError(RuntimeError):
    """Raised when global reranking cannot score the candidate evidence pool."""


def _build_rerank_text(source: SourceItem) -> str:
    parts = [source.title, source.snippet]
    section_title = source.metadata.get("section_title")
    if isinstance(section_title, str) and section_title.strip():
        parts.append(f"section: {section_title}")
    domain = source.metadata.get("domain")
    if isinstance(domain, str) and domain.strip():
        parts.append(f"domain: {domain}")
    return "\n".join(part.strip() for part in parts if part and part.strip())


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    numerator = sum(left_value * right_value for left_value, right_value in zip(left, right, strict=True))
    left_norm = sqrt(sum(value * value for value in left))
    right_norm = sqrt(sum(value * value for value in right))
    if not left_norm or not right_norm:
        return 0.0
    return numerator / (left_norm * right_norm)


def rerank_sources_global(question: str, sources: list[SourceItem]) -> list[SourceItem]:
    if not sources:
        return []

    try:
        rerank_payload = [question, *[_build_rerank_text(source) for source in sources]]
        embeddings = generate_embeddings(rerank_payload)
    except EmbeddingServiceError as exc:
        raise RerankServiceError(f"Global rerank embeddings failed: {exc}") from exc

    query_embedding = embeddings[0]
    scored_sources: list[SourceItem] = []
    for source, source_embedding in zip(sources, embeddings[1:], strict=True):
        rerank_score = _cosine_similarity(query_embedding, source_embedding)
        metadata = {
            **source.metadata,
            "global_rerank_score": round(rerank_score, 6),
        }
        scored_sources.append(source.model_copy(update={"metadata": metadata}))

    scored_sources.sort(
        key=lambda source: (
            -float(source.metadata.get("global_rerank_score", 0.0)),
            int(source.metadata.get("retrieval_rank", 9999)),
        )
    )

    reranked: list[SourceItem] = []
    for rank, source in enumerate(scored_sources, start=1):
        metadata = {
            **source.metadata,
            "global_rerank_rank": rank,
        }
        reranked.append(source.model_copy(update={"metadata": metadata}))

    logger.info("Global rerank applied to %s candidate sources", len(reranked))
    return reranked
