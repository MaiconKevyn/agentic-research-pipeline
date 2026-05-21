from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from langchain_pymupdf4llm import PyMuPDF4LLMLoader
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

from backend.app.services.document_repository import delete_all_documents, upsert_documents
from backend.app.services.embedding_service import generate_embeddings


PAGES_DELIMITER = "\n----- PAGE BREAK -----\n"
MIN_CHUNK_CHARS = 120
PARSER_VERSION = "pymupdf4llm-markdown-v2"
DEFAULT_REPORT_DIR = Path("ingestion_reports")
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


@dataclass(frozen=True)
class IngestionReport:
    source_file: str
    source_document_id: str
    checksum: str
    parser_version: str
    pages_processed: int
    chunks_created: int
    skipped_chunks: int
    extraction_warnings: list[str]
    token_distribution: dict[str, float]
    duplicate_percentage: float
    baseline_embedding_text_chars: int
    contextualized_embedding_text_chars: int


@dataclass(frozen=True)
class ParsedPdfDocument:
    chunks: list[dict]
    report: IngestionReport


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


def _approx_token_count(text: str) -> int:
    return len(re.findall(r"\S+", text))


def _build_contextual_header(
    *,
    title: str,
    section_title: str,
    page_start: int | None,
    page_end: int | None,
    total_pages: int | None,
) -> str:
    if page_start is None:
        page_label = "unknown page"
    elif page_end and page_end != page_start:
        page_label = f"pages {page_start}-{page_end}"
    else:
        page_label = f"page {page_start}"
    total_label = f" of {total_pages}" if total_pages else ""
    return (
        f"Document: {title}\n"
        f"Section: {section_title}\n"
        f"Location: {page_label}{total_label}"
    )


def _parent_chunk_id(
    *,
    file_path: Path,
    file_checksum: str,
    section_index: int,
    section_title: str,
) -> str:
    parent_identifier = hashlib.sha1(
        f"{file_path.name}:{file_checksum}:{section_index}:{section_title}".encode("utf-8")
    ).hexdigest()
    return f"{file_path.stem}#parent-{parent_identifier}"


def _build_chunk(
    *,
    file_path: Path,
    file_checksum: str,
    title: str,
    chunk_text: str,
    section_title: str,
    section_index: int,
    chunk_index: int,
    total_pages: int | None,
    page_start: int | None,
    page_end: int | None,
) -> dict:
    contextual_header = _build_contextual_header(
        title=title,
        section_title=section_title,
        page_start=page_start,
        page_end=page_end,
        total_pages=total_pages,
    )
    contextualized_embedding_text = f"{contextual_header}\n\n{chunk_text}"
    parent_chunk_id = _parent_chunk_id(
        file_path=file_path,
        file_checksum=file_checksum,
        section_index=section_index,
        section_title=section_title,
    )
    chunk_identifier = hashlib.sha1(
        (
            f"{file_path.name}:{file_checksum}:{section_index}:"
            f"{chunk_index}:{page_start}:{page_end}:{chunk_text[:120]}"
        ).encode("utf-8")
    ).hexdigest()
    metadata = {
        "source_file": file_path.name,
        "source_document_id": f"{file_path.name}:{file_checksum[:16]}",
        "file_checksum": file_checksum,
        "document_title": title,
        "section_title": section_title,
        "section_index": section_index,
        "chunk_index": chunk_index,
        "chunk_role": "child",
        "parent_chunk_id": parent_chunk_id,
        "contextual_header": contextual_header,
        "page_start": page_start,
        "page_end": page_end,
        "total_pages": total_pages,
        "parser_version": PARSER_VERSION,
    }
    return {
        "document_id": f"{file_path.stem}#{chunk_identifier}",
        "title": f"{title} | {section_title}",
        "content": chunk_text,
        "contextualized_embedding_text": contextualized_embedding_text,
        "source_type": "pdf_chunk",
        "source_url": None,
        "metadata": metadata,
    }


class PdfIngestionPipeline:
    def extract(self, file_path: Path):
        loader = PyMuPDF4LLMLoader(
            str(file_path.resolve()),
            mode="page",
            pages_delimiter=PAGES_DELIMITER,
        )
        return loader.load()

    def clean(self, text: str) -> str:
        return _clean_text(text)

    def segment(self, file_path: Path) -> ParsedPdfDocument:
        return _parse_pdf_document(file_path, self)

    def contextualize(self, chunk: dict) -> str:
        return chunk["contextualized_embedding_text"]

    def embed(self, chunks: list[dict]) -> None:
        batch_size = 16
        for start_index in range(0, len(chunks), batch_size):
            batch = chunks[start_index : start_index + batch_size]
            embeddings = generate_embeddings([self.contextualize(chunk) for chunk in batch])
            for chunk, embedding in zip(batch, embeddings, strict=True):
                chunk["embedding"] = embedding

    def index(self, chunks: list[dict], *, replace_existing: bool) -> int:
        if replace_existing:
            delete_all_documents()
        return upsert_documents(chunks)

    def validate(self, parsed_documents: list[ParsedPdfDocument]) -> None:
        for parsed_document in parsed_documents:
            chunk_ids = [chunk["document_id"] for chunk in parsed_document.chunks]
            if len(chunk_ids) != len(set(chunk_ids)):
                raise ValueError(
                    f"Duplicate chunk IDs generated for {parsed_document.report.source_file}"
                )


def _split_loaded_document(file_path: Path) -> list[dict]:
    return PdfIngestionPipeline().segment(file_path).chunks


def _parse_pdf_document(file_path: Path, pipeline: PdfIngestionPipeline) -> ParsedPdfDocument:
    file_checksum = hashlib.sha256(file_path.read_bytes()).hexdigest()
    loaded_documents = pipeline.extract(file_path)
    source_document_id = f"{file_path.name}:{file_checksum[:16]}"
    extraction_warnings: list[str] = []
    if not loaded_documents:
        report = IngestionReport(
            source_file=file_path.name,
            source_document_id=source_document_id,
            checksum=file_checksum,
            parser_version=PARSER_VERSION,
            pages_processed=0,
            chunks_created=0,
            skipped_chunks=0,
            extraction_warnings=["No pages were extracted from the PDF."],
            token_distribution={"min": 0.0, "max": 0.0, "avg": 0.0},
            duplicate_percentage=0.0,
            baseline_embedding_text_chars=0,
            contextualized_embedding_text_chars=0,
        )
        return ParsedPdfDocument(chunks=[], report=report)

    chunks: list[dict] = []
    total_pages = loaded_documents[0].metadata.get("total_pages")
    section_counter = 0
    skipped_chunks = 0
    for page_document in loaded_documents:
        title = page_document.metadata.get("title") or file_path.stem
        page_number = int(page_document.metadata.get("page", 0)) + 1
        cleaned_content = pipeline.clean(page_document.page_content)
        if not cleaned_content:
            extraction_warnings.append(f"Page {page_number} produced no usable text.")
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
                    skipped_chunks += 1
                    continue
                chunks.append(
                    _build_chunk(
                        file_path=file_path,
                        file_checksum=file_checksum,
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
    report = _build_ingestion_report(
        file_path=file_path,
        file_checksum=file_checksum,
        source_document_id=source_document_id,
        pages_processed=len(loaded_documents),
        chunks=chunks,
        skipped_chunks=skipped_chunks,
        extraction_warnings=extraction_warnings,
    )
    return ParsedPdfDocument(chunks=chunks, report=report)


def _build_ingestion_report(
    *,
    file_path: Path,
    file_checksum: str,
    source_document_id: str,
    pages_processed: int,
    chunks: list[dict],
    skipped_chunks: int,
    extraction_warnings: list[str],
) -> IngestionReport:
    token_counts = [_approx_token_count(chunk["content"]) for chunk in chunks]
    content_hashes = [hashlib.sha256(chunk["content"].encode("utf-8")).hexdigest() for chunk in chunks]
    duplicate_count = len(content_hashes) - len(set(content_hashes))
    chunk_count = len(chunks)
    return IngestionReport(
        source_file=file_path.name,
        source_document_id=source_document_id,
        checksum=file_checksum,
        parser_version=PARSER_VERSION,
        pages_processed=pages_processed,
        chunks_created=chunk_count,
        skipped_chunks=skipped_chunks,
        extraction_warnings=extraction_warnings,
        token_distribution={
            "min": float(min(token_counts)) if token_counts else 0.0,
            "max": float(max(token_counts)) if token_counts else 0.0,
            "avg": (sum(token_counts) / len(token_counts)) if token_counts else 0.0,
        },
        duplicate_percentage=(duplicate_count / chunk_count) if chunk_count else 0.0,
        baseline_embedding_text_chars=sum(len(chunk["content"]) for chunk in chunks),
        contextualized_embedding_text_chars=sum(
            len(chunk["contextualized_embedding_text"]) for chunk in chunks
        ),
    )


def _write_ingestion_reports(reports: list[IngestionReport], report_dir: Path) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    for report in reports:
        report_path = report_dir / f"{Path(report.source_file).stem}.ingestion.json"
        report_path.write_text(
            json.dumps(asdict(report), indent=2, sort_keys=True),
            encoding="utf-8",
        )


def build_pdf_corpus(raw_dir: Path, *, file_pattern: str = "*.pdf") -> list[dict]:
    all_chunks: list[dict] = []
    for file_path in sorted(raw_dir.glob(file_pattern)):
        all_chunks.extend(_split_loaded_document(file_path))
    return all_chunks


def ingest_pdf_corpus(
    raw_dir: Path,
    replace_existing: bool = True,
    report_dir: Path | None = DEFAULT_REPORT_DIR,
    file_pattern: str = "*.pdf",
) -> int:
    pipeline = PdfIngestionPipeline()
    parsed_documents = [
        pipeline.segment(file_path)
        for file_path in sorted(raw_dir.glob(file_pattern))
    ]
    reports = [parsed_document.report for parsed_document in parsed_documents]
    chunks = [
        chunk
        for parsed_document in parsed_documents
        for chunk in parsed_document.chunks
    ]
    if report_dir is not None:
        _write_ingestion_reports(reports, report_dir)
    if not chunks:
        return 0

    pipeline.validate(parsed_documents)
    pipeline.embed(chunks)
    return pipeline.index(chunks, replace_existing=replace_existing)
