from pathlib import Path
from unittest.mock import patch

from backend.app.services.pdf_ingestion_service import _split_loaded_document, ingest_pdf_corpus


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


def test_pdf_ingestion_adds_contextual_embedding_text_and_parent_metadata() -> None:
    file_path = Path("data/raw/RAG.pdf")

    chunks = _split_loaded_document(file_path)

    assert chunks
    first_chunk = chunks[0]
    metadata = first_chunk["metadata"]
    assert first_chunk["contextualized_embedding_text"].startswith("Document: ")
    assert first_chunk["contextualized_embedding_text"] != first_chunk["content"]
    assert metadata["contextual_header"] in first_chunk["contextualized_embedding_text"]
    assert metadata["parent_chunk_id"]
    assert metadata["chunk_role"] == "child"
    assert metadata["parser_version"]


def test_pdf_ingestion_produces_stable_chunk_ids_for_same_file_and_settings() -> None:
    file_path = Path("data/raw/RAG.pdf")

    first_run = _split_loaded_document(file_path)
    second_run = _split_loaded_document(file_path)

    assert [chunk["document_id"] for chunk in first_run] == [
        chunk["document_id"] for chunk in second_run
    ]
    assert [chunk["metadata"]["parent_chunk_id"] for chunk in first_run] == [
        chunk["metadata"]["parent_chunk_id"] for chunk in second_run
    ]


def test_ingest_pdf_corpus_writes_ingestion_report(tmp_path: Path) -> None:
    with (
        patch(
            "backend.app.services.pdf_ingestion_service.generate_embeddings",
            side_effect=lambda texts: [[0.0, 1.0] for _ in texts],
        ) as mocked_embeddings,
        patch("backend.app.services.pdf_ingestion_service.upsert_documents", return_value=1) as mocked_upsert,
    ):
        inserted = ingest_pdf_corpus(
            Path("data/raw"),
            replace_existing=False,
            report_dir=tmp_path,
            file_pattern="RAG.pdf",
        )

    report_path = tmp_path / "RAG.ingestion.json"
    assert inserted == 1
    assert report_path.exists()
    report_text = report_path.read_text(encoding="utf-8")
    assert '"pages_processed"' in report_text
    assert '"chunks_created"' in report_text
    assert '"duplicate_percentage"' in report_text
    embedded_texts = mocked_embeddings.call_args.args[0]
    assert embedded_texts[0].startswith("Document: ")
    assert mocked_upsert.call_args.args[0][0]["embedding"] == [0.0, 1.0]
