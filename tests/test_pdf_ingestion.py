from pathlib import Path

from backend.app.services.pdf_ingestion_service import _split_loaded_document


def test_pdf_ingestion_extracts_section_and_page_metadata() -> None:
    file_path = Path("data/raw/RAG.pdf")

    chunks = _split_loaded_document(file_path)

    assert chunks
    first_chunk = chunks[0]
    metadata = first_chunk["metadata"]
    assert first_chunk["source_type"] == "pdf_chunk"
    assert metadata["source_file"] == "RAG.pdf"
    assert metadata["section_title"]
    assert metadata["page_start"] is not None
    assert metadata["page_end"] is not None
