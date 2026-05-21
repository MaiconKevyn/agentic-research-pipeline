from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

from backend.app.core.config import settings
from backend.app.core.logging import get_logger
from backend.app.services.pdf_ingestion_service import ingest_pdf_corpus


logger = get_logger(__name__)


@dataclass(frozen=True)
class WorkerConfig:
    raw_dir: Path = Path(settings.raw_pdf_dir)
    poll_interval_seconds: int = settings.worker_poll_interval_seconds
    run_once: bool = False


@dataclass(frozen=True)
class WorkerResult:
    status: str
    processed_count: int


def ingest_raw_pdfs(raw_dir: Path) -> int:
    return ingest_pdf_corpus(raw_dir, replace_existing=True)


class IngestionWorker:
    def __init__(self, config: WorkerConfig | None = None) -> None:
        self.config = config or WorkerConfig()

    def run_once(self) -> WorkerResult:
        raw_dir = self.config.raw_dir
        if not raw_dir.exists() or not any(raw_dir.glob("*.pdf")):
            logger.info("No PDFs found for ingestion in %s", raw_dir)
            return WorkerResult(status="idle", processed_count=0)
        processed_count = ingest_raw_pdfs(raw_dir)
        logger.info("Ingestion worker processed %s chunks from %s", processed_count, raw_dir)
        return WorkerResult(status="completed", processed_count=processed_count)

    def run_forever(self) -> None:
        while True:
            self.run_once()
            if self.config.run_once:
                return
            time.sleep(self.config.poll_interval_seconds)


def main() -> None:
    IngestionWorker().run_forever()


if __name__ == "__main__":
    main()
