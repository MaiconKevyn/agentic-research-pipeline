from __future__ import annotations

from typing import Any
from urllib.parse import urlparse

import requests

from backend.app.core.config import settings
from backend.app.core.logging import get_logger
from backend.app.schemas.research import SourceItem


logger = get_logger(__name__)


class WebSearchServiceError(RuntimeError):
    """Raised when the OpenAI web search tool fails or returns malformed data."""


def _build_web_metadata(url: str | None, rank: int | None = None) -> dict[str, Any]:
    metadata: dict[str, Any] = {"provider": "openai_web_search"}
    if rank is not None:
        metadata["retrieval_rank"] = rank
    if url:
        hostname = urlparse(url).hostname
        if hostname:
            metadata["domain"] = hostname
    return metadata


def _extract_output_text(payload: dict[str, Any]) -> str:
    output_text = payload.get("output_text")
    if isinstance(output_text, str) and output_text.strip():
        return output_text.strip()

    collected_text: list[str] = []
    for item in payload.get("output", []):
        for content in item.get("content", []):
            if content.get("type") == "output_text" and content.get("text"):
                collected_text.append(content["text"])

    return "\n".join(part.strip() for part in collected_text if part.strip()).strip()


def _extract_sources(payload: dict[str, Any]) -> list[SourceItem]:
    sources: list[SourceItem] = []
    fallback_snippet = _extract_output_text(payload)
    citation_by_url: dict[str, dict[str, str]] = {}
    seen_urls: set[str] = set()

    if fallback_snippet:
        sources.append(
            SourceItem(
                source_id="web-summary",
                title="OpenAI web search summary",
                snippet=fallback_snippet[:500],
                source_type="web",
                url=None,
                metadata=_build_web_metadata(url=None),
            )
        )

    for item in payload.get("output", []):
        if item.get("type") != "message":
            continue
        for content in item.get("content", []):
            for annotation in content.get("annotations", []):
                if annotation.get("type") != "url_citation":
                    continue
                url = annotation.get("url")
                if not url:
                    continue
                citation_by_url[url] = {
                    "title": annotation.get("title") or url,
                    "snippet": fallback_snippet or "Resultado de busca web.",
                }

    for index, (url, citation) in enumerate(citation_by_url.items(), start=1):
        sources.append(
            SourceItem(
                source_id=f"web-citation-{index}",
                title=citation["title"],
                snippet=citation["snippet"][:500],
                source_type="web",
                url=url,
                metadata=_build_web_metadata(url=url, rank=index),
            )
        )
        seen_urls.add(url)

    for item in payload.get("output", []):
        if item.get("type") != "web_search_call":
            continue

        action = item.get("action", {})
        for index, source in enumerate(action.get("sources", []), start=1):
            url = source.get("url")
            if url and url in seen_urls:
                continue
            citation = citation_by_url.get(url or "", {})
            title = source.get("title") or citation.get("title") or url or f"web-source-{index}"
            snippet = (
                source.get("snippet")
                or citation.get("snippet")
                or fallback_snippet
                or "Resultado de busca web."
            )
            sources.append(
                SourceItem(
                    source_id=f"web-{index}",
                    title=title,
                    snippet=snippet[:500],
                    source_type="web",
                    url=url,
                    metadata=_build_web_metadata(url=url, rank=index),
                )
            )
            if url:
                seen_urls.add(url)

    if sources:
        return sources

    if fallback_snippet:
        return [
            SourceItem(
                source_id="web-1",
                title="OpenAI web search result",
                snippet=fallback_snippet[:500],
                source_type="web",
                url=None,
                metadata=_build_web_metadata(url=None, rank=1),
            )
        ]

    raise WebSearchServiceError("Web search response did not contain usable sources.")


def perform_web_search(question: str, top_k: int) -> list[SourceItem]:
    if not settings.openai_api_key:
        raise WebSearchServiceError("OPENAI_API_KEY is not configured.")

    base_url = (settings.openai_base_url or "https://api.openai.com/v1").rstrip("/")
    url = f"{base_url}/responses"
    payload = {
        "model": settings.openai_model,
        "tools": [{"type": "web_search"}],
        "tool_choice": {"type": "web_search"},
        "include": ["web_search_call.action.sources"],
        "input": [
            {
                "role": "developer",
                "content": [
                    {
                        "type": "input_text",
                        "text": (
                            "Use web search to answer the user's question. "
                            "Prioritize official documentation, official package registries, "
                            "vendor release notes, or the official project repository. "
                            "Avoid low-authority aggregator sites when better sources exist."
                        ),
                    }
                ],
            },
            {
                "role": "user",
                "content": [{"type": "input_text", "text": question}],
            },
        ],
    }
    headers = {
        "Authorization": f"Bearer {settings.openai_api_key}",
        "Content-Type": "application/json",
    }

    logger.info("Requesting web search via model=%s", settings.openai_model)
    try:
        response = requests.post(
            url,
            headers=headers,
            json=payload,
            timeout=settings.request_timeout_seconds,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        raise WebSearchServiceError(f"Web search request failed: {exc}") from exc

    try:
        response_payload: dict[str, Any] = response.json()
    except ValueError as exc:
        raise WebSearchServiceError("Web search response was not valid JSON.") from exc

    return _extract_sources(response_payload)[:top_k]
