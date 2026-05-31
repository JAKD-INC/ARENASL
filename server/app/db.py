"""Database engine, SQLite pragmas, session factory, and the get_db dependency.

Sync SQLAlchemy 2.0 on purpose: SQLite is a single-writer store, so an async
driver buys nothing here. DB-touching routes are defined with `def` (not
`async def`) so FastAPI runs them in its threadpool and they never block the
event loop.
"""

from __future__ import annotations

from collections.abc import Iterator

from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from app.config import get_settings

_settings = get_settings()
_is_sqlite = _settings.db_url.startswith("sqlite")

engine: Engine = create_engine(
    _settings.db_url,
    # SQLite connections are reused across FastAPI's threadpool threads.
    connect_args={"check_same_thread": False} if _is_sqlite else {},
)


if _is_sqlite:

    @event.listens_for(engine, "connect")
    def _set_sqlite_pragmas(dbapi_conn, _record) -> None:  # noqa: ANN001
        cur = dbapi_conn.cursor()
        cur.execute("PRAGMA journal_mode=WAL")     # readers + 1 writer concurrently
        cur.execute("PRAGMA foreign_keys=ON")      # per-connection; off by default in SQLite
        cur.execute("PRAGMA synchronous=NORMAL")   # safe under WAL, fewer fsyncs
        cur.execute("PRAGMA busy_timeout=5000")    # wait, don't error, on brief write locks
        cur.close()


SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


def get_db() -> Iterator[Session]:
    """One session per request: commit on success, rollback on error, always close."""
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()
