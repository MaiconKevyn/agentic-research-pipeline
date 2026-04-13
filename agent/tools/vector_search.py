from backend.app.core.logging import get_logger
from backend.app.schemas.research import SourceItem
from backend.app.services.document_repository import (
    DocumentRepositoryError,
    search_similar_documents,
)
from backend.app.services.embedding_service import (
    EmbeddingServiceError,
    generate_embedding,
)


logger = get_logger(__name__)


def search_documents(question: str, top_k: int) -> list[SourceItem]:
    try:
        query_embedding = generate_embedding(question)
        return search_similar_documents(query_embedding=query_embedding, top_k=top_k)
    except (EmbeddingServiceError, DocumentRepositoryError) as exc:
        logger.warning("Vector search unavailable: %s", exc)
        return []
