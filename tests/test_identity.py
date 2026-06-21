"""Unit tests for the identity slice (pure-Python; no ML deps)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from recall import identity  # noqa: E402
from recall.common import Segment  # noqa: E402


def test_cosine_and_norm():
    assert abs(identity.cosine([1, 0], [1, 0]) - 1.0) < 1e-9
    assert abs(identity.cosine([1, 0], [0, 1])) < 1e-9
    assert identity.cosine([1, 0], []) == -1.0
    n = identity.l2norm([3.0, 4.0])
    assert abs((n[0] ** 2 + n[1] ** 2) - 1.0) < 1e-9


def test_is_finite_vec():
    assert identity.is_finite_vec([0.1, 0.2])
    assert not identity.is_finite_vec([])
    assert not identity.is_finite_vec([float("nan"), 1.0])
    assert not identity.is_finite_vec(None)


def test_slugify():
    assert identity.slugify("Priya Sharma") == "priya-sharma"
    assert identity.slugify("  A_B-c ") == "a-b-c"
    assert identity.slugify("!!!") == "person"


def test_voicestore_match_enroll_roundtrip(tmp_path):
    vs = identity.VoiceStore(tmp_path / "vp.json")
    # empty store -> unknown
    _, _, status = vs.match([1.0, 0.0, 0.0], 0.70, 0.45)
    assert status == "unknown"
    vs.enroll("Priya", [1.0, 0.0, 0.0])
    # near-identical -> match
    name, score, status = vs.match([0.99, 0.01, 0.0], 0.70, 0.45)
    assert name == "Priya" and status == "match" and score >= 0.70
    # mid -> ambiguous
    name, score, status = vs.match([0.6, 0.8, 0.0], 0.70, 0.45)
    assert status in ("ambiguous", "match")
    # orthogonal -> unknown
    _, _, status = vs.match([0.0, 0.0, 1.0], 0.70, 0.45)
    assert status == "unknown"
    # centroid averaging increments n_samples
    vs.enroll("Priya", [0.9, 0.1, 0.0])
    assert vs.data["people"]["Priya"]["n_samples"] == 2
    vs.save()
    vs2 = identity.VoiceStore(tmp_path / "vp.json")
    assert vs2.names() == ["Priya"]


def test_resolve_and_apply(tmp_path):
    vs = identity.VoiceStore(tmp_path / "vp.json")
    vs.enroll("Rahul", [0.0, 1.0, 0.0])
    segs = [Segment(0, 1, "hi", speaker="SPEAKER_00"),
            Segment(1, 2, "yes", speaker="SPEAKER_00")]
    emb_map = {"SPEAKER_00": [0.01, 0.99, 0.0]}
    mapping = identity.resolve_identities(segs, emb_map, vs, 0.70, 0.45,
                                          enroll=False)
    assert mapping == {"SPEAKER_00": "Rahul"}
    identity.apply_identities(segs, mapping)
    assert all(s.speaker == "Rahul" for s in segs)
    assert identity.raw_utterances_for(segs, "Rahul")[0]["text"] == "hi"


def test_personastore(tmp_path):
    from recall.personas import PersonaStore
    ps = PersonaStore(tmp_path / "people")
    ps.add_utterances("Priya", "2026-06-21",
                      [{"start": 1.0, "end": 2.0, "text": "ship it"}])
    assert (tmp_path / "people" / "priya" / "utterances.jsonl").exists()
    ps.write_profile("Priya", "## Snapshot\nLeads frontend.\n")
    assert "frontend" in ps.read_profile("Priya")
    captured = {}

    def fake_gen(instr, content):
        captured["called"] = True
        return "## Snapshot\nUpdated profile.\n"

    ok = ps.update_profile("Priya", "2026-06-21",
                           [{"start": 1.0, "end": 2.0, "text": "ship it"}],
                           "instructions", fake_gen)
    assert ok and captured.get("called")
    assert "Updated profile" in ps.read_profile("Priya")
