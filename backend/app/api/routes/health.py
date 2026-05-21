from fastapi import APIRouter

from backend.app.core.config import settings
from backend.app.services.document_repository import get_corpus_stats


router = APIRouter(tags=["health"])


@router.get("/health")
def healthcheck() -> dict[str, str]:
    return {
        "status": "ok",
        "environment": settings.app_env,
        "version": settings.app_version,
    }


@router.get("/ready")
def readiness() -> dict:
    try:
        stats = get_corpus_stats()
    except Exception as exc:
        return {
            "status": "not_ready",
            "checks": {"database": "error"},
            "corpus": None,
            "errors": [str(exc)],
        }
    return {
        "status": "ready",
        "checks": {"database": "ok"},
        "corpus": {
            "source_document_count": stats.source_document_count,
            "chunk_count": stats.chunk_count,
            "corpus_version_id": stats.corpus_version_id,
        },
        "errors": [],
    }
