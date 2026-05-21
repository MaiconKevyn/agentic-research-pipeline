from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.app.services.pdf_ingestion_service import ingest_pdf_corpus


RAW_DIR = ROOT_DIR / "data" / "raw"


def main() -> None:
    inserted = ingest_pdf_corpus(RAW_DIR, replace_existing=True)
    print(
        f"Ingested {inserted} PDF chunks into PostgreSQL. "
        "Per-document reports were written to ingestion_reports/."
    )


if __name__ == "__main__":
    main()
