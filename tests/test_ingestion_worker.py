from pathlib import Path
from unittest.mock import patch

from backend.app.workers.ingestion_worker import IngestionWorker, WorkerConfig


def test_ingestion_worker_processes_raw_pdf_directory_once(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    (raw_dir / "paper.pdf").write_bytes(b"%PDF-1.4")
    config = WorkerConfig(raw_dir=raw_dir, poll_interval_seconds=1, run_once=True)
    worker = IngestionWorker(config=config)

    with patch("backend.app.workers.ingestion_worker.ingest_raw_pdfs", return_value=1) as mocked_ingest:
        result = worker.run_once()

    mocked_ingest.assert_called_once_with(raw_dir)
    assert result.processed_count == 1
    assert result.status == "completed"


def test_ingestion_worker_skips_when_directory_has_no_pdfs(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    worker = IngestionWorker(config=WorkerConfig(raw_dir=raw_dir, run_once=True))

    with patch("backend.app.workers.ingestion_worker.ingest_raw_pdfs") as mocked_ingest:
        result = worker.run_once()

    mocked_ingest.assert_not_called()
    assert result.processed_count == 0
    assert result.status == "idle"
