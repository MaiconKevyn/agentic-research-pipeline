from backend.app.core.logging import get_logger
from backend.app.schemas.research import SourceItem
from backend.app.services.web_search_service import (
    WebSearchServiceError,
    perform_web_search,
)


logger = get_logger(__name__)


def search_web(question: str, top_k: int) -> list[SourceItem]:
    try:
        return perform_web_search(question=question, top_k=top_k)
    except WebSearchServiceError as exc:
        logger.warning("Web search unavailable: %s", exc)
        return []
