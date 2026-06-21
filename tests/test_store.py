"""store + content-hash + dated-naming checks (no ML deps, no network)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from recall import store  # noqa: E402
from recall.common import audio_sha256  # noqa: E402


def test_audio_sha256_is_content_based(tmp_path):
    a, b, c = tmp_path / "a", tmp_path / "b", tmp_path / "c"
    a.write_bytes(b"xyz")
    b.write_bytes(b"xyz")          # same content, different name
    c.write_bytes(b"zzz")
    assert audio_sha256(a) == audio_sha256(b)
    assert audio_sha256(a) != audio_sha256(c)


def test_store_roundtrip_and_replace(tmp_path):
    db = tmp_path / "recall.db"
    assert store.lookup(db, "h1") is None
    store.record(db, audio_sha256="h1", audio_path="/x.m4a", title="T",
                 duration_s=10.0, created_at="2026", transcript_md="/t.md",
                 notes_md="/n.md", coverage=0.9)
    r = store.lookup(db, "h1")
    assert r["notes_md"] == "/n.md" and r["title"] == "T"
    store.record(db, audio_sha256="h1", audio_path="/x.m4a", title="T2",
                 duration_s=10.0, created_at="2026", transcript_md="/t.md",
                 notes_md="/n.md", coverage=0.9)
    assert store.lookup(db, "h1")["title"] == "T2"   # PK replace, no dup row


def test_dated_stem():
    p = Path("/x/My Meeting.m4a")
    assert store.dated_stem(p, "Q3 Review", "21-06-2026") == \
        "21-06-2026_q3-review_my-meeting"
    assert store.dated_stem(p, None, "21-06-2026") == "21-06-2026_my-meeting"
