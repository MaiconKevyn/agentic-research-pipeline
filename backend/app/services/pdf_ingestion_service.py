from __future__ import annotations

import hashlib
import re
from pathlib import Path

from langchain_pymupdf4llm import PyMuPDF4LLMLoader
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

from backend.app.services.document_repository import delete_all_documents, upsert_documents
from backend.app.services.embedding_service import generate_embeddings


PAGES_DELIMITER = "\n----- PAGE BREAK -----\n"
MIN_CHUNK_CHARS = 120
HEADER_SPLITTER = MarkdownHeaderTextSplitter(
    headers_to_split_on=[
        ("#", "h1"),
        ("##", "h2"),
        ("###", "h3"),
    ],
    strip_headers=False,
)
TOKEN_SPLITTER = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=800,
    chunk_overlap=100,
    separators=[
        "\n## ",
        "\n### ",
        "\n\n",
        "\n",
        ". ",
        " ",
        "",
    ],
)

def _clean_text(text: str) -> str:
    cleaned = text.replace("\u00ad", "")
    cleaned = re.sub(r"(?<=\w)-\n(?=\w)", "", cleaned)
    cleaned = cleaned.replace(PAGES_DELIMITER, "\n")
    lines = [line.rstrip() for line in cleaned.splitlines()]
    normalized_lines: list[str] = []
    previous_blank = False
    for line in lines:
        is_blank = not line.strip()
        if is_blank and previous_blank:
            continue
        normalized_lines.append(line)
        previous_blank = is_blank
    return "\n".join(normalized_lines).strip()


def _normalize_heading(value: str | None) -> str:
    if not value:
        return "document-body"
    normalized = re.sub(r"[#*_`]+", " ", value)
    normalized = re.sub(r"\s+", " ", normalized).strip(" :-")
    return normalized or "document-body"


def _build_chunk(
    *,
    file_path: Path,
    title: str,
    chunk_text: str,
    section_title: str,
    section_index: int,
    chunk_index: int,
    total_pages: int | None,
    page_start: int | None,
    page_end: int | None,
) -> dict:
    chunk_identifier = hashlib.sha1(
        f"{file_path.name}:{section_index}:{chunk_index}:{page_start}:{page_end}:{chunk_text[:120]}".encode(
            "utf-8"
        )
    ).hexdigest()
    metadata = {
        "source_file": file_path.name,
        "document_title": title,
        "section_title": section_title,
        "section_index": section_index,
        "chunk_index": chunk_index,
        "page_start": page_start,
        "page_end": page_end,
        "total_pages": total_pages,
    }
    return {
        "document_id": f"{file_path.stem}#{chunk_identifier}",
        "title": f"{title} | {section_title}",
        "content": chunk_text,
        "source_type": "pdf_chunk",
        "source_url": None,
        "metadata": metadata,
    }


def _split_loaded_document(file_path: Path) -> list[dict]:
    loader = PyMuPDF4LLMLoader(
        str(file_path.resolve()),
        mode="page",
        pages_delimiter=PAGES_DELIMITER,
    )
    loaded_documents = loader.load()
    if not loaded_documents:
        return []

    chunks: list[dict] = []
    total_pages = loaded_documents[0].metadata.get("total_pages")
    section_counter = 0
    for page_document in loaded_documents:
        title = page_document.metadata.get("title") or file_path.stem
        page_number = int(page_document.metadata.get("page", 0)) + 1
        cleaned_content = _clean_text(page_document.page_content)
        if not cleaned_content:
            continue

        structured_docs = HEADER_SPLITTER.split_text(cleaned_content)
        if not structured_docs:
            structured_docs = TOKEN_SPLITTER.create_documents(
                texts=[cleaned_content],
                metadatas=[
                    {
                        "section_title": "document-body",
                    }
                ],
            )

        for section_doc in structured_docs:
            section_counter += 1
            section_title = _normalize_heading(
                section_doc.metadata.get("h3")
                or section_doc.metadata.get("h2")
                or section_doc.metadata.get("h1")
                or section_doc.metadata.get("section_title")
            )
            split_docs = TOKEN_SPLITTER.create_documents(
                texts=[section_doc.page_content],
                metadatas=[
                    {
                        "section_title": section_title,
                        "section_index": section_counter,
                    }
                ],
            )
            for chunk_index, chunk_doc in enumerate(split_docs, start=1):
                chunk_text = chunk_doc.page_content.strip()
                if len(chunk_text) < MIN_CHUNK_CHARS:
                    continue
                chunks.append(
                    _build_chunk(
                        file_path=file_path,
                        title=title,
                        chunk_text=chunk_text,
                        section_title=section_title,
                        section_index=section_counter,
                        chunk_index=chunk_index,
                        total_pages=total_pages,
                        page_start=page_number,
                        page_end=page_number,
                    )
                )
    return chunks


def build_pdf_corpus(raw_dir: Path) -> list[dict]:
    all_chunks: list[dict] = []
    for file_path in sorted(raw_dir.glob("*.pdf")):
        all_chunks.extend(_split_loaded_document(file_path))
    return all_chunks


def ingest_pdf_corpus(raw_dir: Path, replace_existing: bool = True) -> int:
    chunks = build_pdf_corpus(raw_dir)
    if not chunks:
        return 0

    batch_size = 16
    for start_index in range(0, len(chunks), batch_size):
        batch = chunks[start_index : start_index + batch_size]
        embeddings = generate_embeddings([chunk["content"] for chunk in batch])
        for chunk, embedding in zip(batch, embeddings, strict=True):
            chunk["embedding"] = embedding

    if replace_existing:
        delete_all_documents()
    return upsert_documents(chunks)
