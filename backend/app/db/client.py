from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import psycopg
from psycopg.rows import dict_row

from backend.app.core.config import settings


@contextmanager
def get_connection() -> Iterator[psycopg.Connection]:
    connection = psycopg.connect(settings.postgres_dsn, row_factory=dict_row)
    try:
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()
