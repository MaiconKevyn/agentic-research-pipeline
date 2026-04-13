from __future__ import annotations

from typing import Any

import requests

from backend.app.core.config import settings
from backend.app.core.logging import get_logger


logger = get_logger(__name__)


class EmbeddingServiceError(RuntimeError):
    """Raised when the embedding provider fails or returns malformed data."""


def _request_embeddings(input_payload: str | list[str]) -> list[list[float]]:
    if not settings.openai_api_key:
        raise EmbeddingServiceError("OPENAI_API_KEY is not configured.")

    base_url = (settings.openai_base_url or "https://api.openai.com/v1").rstrip("/")
    url = f"{base_url}/embeddings"
    payload = {
        "model": settings.embedding_model,
        "input": input_payload,
        "encoding_format": "float",
    }
    headers = {
        "Authorization": f"Bearer {settings.openai_api_key}",
        "Content-Type": "application/json",
    }

    logger.info("Requesting embedding from model=%s", settings.embedding_model)
    try:
        response = requests.post(
            url,
            headers=headers,
            json=payload,
            timeout=settings.request_timeout_seconds,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        raise EmbeddingServiceError(f"Embedding request failed: {exc}") from exc

    try:
        response_payload: dict[str, Any] = response.json()
        data = response_payload["data"]
    except (ValueError, KeyError, IndexError, TypeError) as exc:
        raise EmbeddingServiceError("Embedding response was malformed.") from exc

    embeddings: list[list[float]] = []
    for item in data:
        embedding = item.get("embedding")
        if not isinstance(embedding, list) or not embedding:
            raise EmbeddingServiceError("Embedding response did not include a valid vector.")
        embeddings.append([float(value) for value in embedding])

    return embeddings


def generate_embedding(text: str) -> list[float]:
    return _request_embeddings(text)[0]


def generate_embeddings(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    return _request_embeddings(texts)
