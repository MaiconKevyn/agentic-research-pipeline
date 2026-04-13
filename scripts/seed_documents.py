from __future__ import annotations

import json
from pathlib import Path

from backend.app.services.document_repository import upsert_documents
from backend.app.services.embedding_service import generate_embedding


CORPUS_PATH = Path(__file__).resolve().parents[1] / "data" / "corpus" / "initial_documents.json"


def main() -> None:
    raw_documents = json.loads(CORPUS_PATH.read_text(encoding="utf-8"))
    prepared_documents: list[dict] = []

    for document in raw_documents:
        enriched_document = dict(document)
        enriched_document["embedding"] = generate_embedding(document["content"])
        prepared_documents.append(enriched_document)

    inserted = upsert_documents(prepared_documents)
    print(f"Seeded {inserted} documents into PostgreSQL.")


if __name__ == "__main__":
    main()
