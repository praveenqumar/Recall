"""
recall.store — SQLite index of generated meetings, keyed by audio content hash.

So a repeat run on the same recording returns the existing transcript/notes paths
instead of regenerating (saving ASR time + Claude tokens). stdlib sqlite3, one
table, no new dependency.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Optional

from .identity import slugify

_SCHEMA = """CREATE TABLE IF NOT EXISTS meetings(
  audio_sha256  TEXT PRIMARY KEY,
  audio_path    TEXT,
  title         TEXT,
  duration_s    REAL,
  created_at    TEXT,
  transcript_md TEXT,
  notes_md      TEXT,
  coverage      REAL
)"""


def _conn(db: Path) -> sqlite3.Connection:
    db.parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(db)
    c.row_factory = sqlite3.Row
    c.execute(_SCHEMA)
    return c


def lookup(db: Path, sha: str) -> Optional[dict]:
    with _conn(db) as c:
        row = c.execute("SELECT * FROM meetings WHERE audio_sha256=?",
                        (sha,)).fetchone()
    return dict(row) if row else None


def record(db: Path, **row) -> None:
    cols = ", ".join(row)
    ph = ", ".join("?" * len(row))
    with _conn(db) as c:
        c.execute(f"INSERT OR REPLACE INTO meetings ({cols}) VALUES ({ph})",
                  tuple(row.values()))
        c.commit()


def dated_stem(audio_path: Path, title: Optional[str], today: str) -> str:
    """<DD-MM-YYYY>_<title>_<filename-without-space> (title segment dropped if none)."""
    parts = [today]
    if title:
        parts.append(slugify(title))
    parts.append(slugify(audio_path.stem))
    return "_".join(parts)
