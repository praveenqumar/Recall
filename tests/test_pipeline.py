"""
End-to-end pipeline test with the ASR + diarization backends mocked out, so it
runs anywhere (only ffmpeg is required, for ingest). Also covers the new
coverage-diagnostic and the notes/reports writers.
"""
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from recall import asr, diarize, notes as notes_mod, transcript as tx  # noqa: E402
from recall.cli import build_parser  # noqa: E402
from recall.common import Segment  # noqa: E402
from recall.personas import PersonaStore  # noqa: E402
from recall.pipeline import run  # noqa: E402


def _make_wav(path: Path, seconds: int = 4) -> None:
    subprocess.run(
        ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-f", "lavfi",
         "-i", f"sine=frequency=440:duration={seconds}", "-ar", "16000",
         "-ac", "1", str(path)], check=True)


def test_coverage_flags_gap():
    segs = [Segment(0, 4, "hello"), Segment(600, 605, "after a big gap")]
    cov = tx.coverage(segs, duration=1200)
    assert cov["n_segments"] == 2
    assert cov["coverage"] < 0.6           # mostly silence -> low coverage
    assert cov["top_gaps"][0][0] >= 590    # ~596s gap detected


def test_coverage_flags_hallucination_loop():
    # 1 real segment + 9 identical loop segments: coverage looks full, but the
    # repetition signal must catch the runaway loop that coverage alone misses.
    segs = [Segment(0, 1, "real opening line")]
    segs += [Segment(i, i + 1, "झाल झाल") for i in range(1, 10)]
    cov = tx.coverage(segs, duration=10)
    assert cov["repetition"] >= 0.3        # dominant phrase dominates
    assert cov["coverage"] >= 0.9          # ...yet raw coverage looks fine


def test_coverage_clean_transcript_no_false_loop():
    segs = [Segment(i, i + 1, f"distinct sentence number {i}") for i in range(10)]
    cov = tx.coverage(segs, duration=10)
    assert cov["repetition"] < 0.3         # diverse text -> no loop flag


def test_build_transcript_with_speakers_and_coverage():
    segs = [Segment(0, 1, "hi", speaker="Priya"),
            Segment(1, 2, "yes", speaker="Priya"),
            Segment(2, 3, "ok", speaker="Rahul")]
    cov = tx.coverage(segs, duration=3)
    md, payload = tx.build_transcript(segs, "demo", cov)
    assert "**Priya**" in md and "**Rahul**" in md
    assert "Coverage:" in md
    assert payload["segments"][0]["speaker"] == "Priya"
    assert payload["coverage"]["coverage"] == cov["coverage"]


def test_pipeline_end_to_end(tmp_path, monkeypatch):
    audio = tmp_path / "demo.wav"
    _make_wav(audio)
    out = tmp_path / "out"
    data = tmp_path / "data"

    # mock ASR so no model/network is needed
    fake_segs = [Segment(0.0, 1.5, "नमस्ते, let's start the standup"),
                 Segment(1.5, 3.0, "frontend is on track")]
    monkeypatch.setattr(asr, "transcribe",
                        lambda *a, **k: [Segment(s.start, s.end, s.text)
                                         for s in fake_segs])

    cfg = build_parser().parse_args([
        str(audio), "-o", str(out), "--data-dir", str(data),
        "--asr", "faster", "--enhance", "none", "--no-diarize",
        "--notes-engine", "none", "--no-progress",
    ])
    run(cfg)

    md = out / "demo.transcript.md"
    js = out / "demo.transcript.json"
    assert md.exists() and js.exists()
    payload = json.loads(js.read_text())
    assert len(payload["segments"]) == 2
    assert "coverage" in payload and payload["coverage"]["n_segments"] == 2
    assert "नमस्ते" in md.read_text()


def test_pipeline_with_mocked_diarization(tmp_path, monkeypatch):
    audio = tmp_path / "demo2.wav"
    _make_wav(audio)
    out = tmp_path / "out2"
    data = tmp_path / "data2"

    fake_segs = [Segment(0.0, 2.0, "I'll own the API work"),
                 Segment(2.0, 4.0, "sounds good")]
    monkeypatch.setattr(asr, "transcribe",
                        lambda *a, **k: [Segment(s.start, s.end, s.text)
                                         for s in fake_segs])
    # mock diarization: two turns, one embedding per speaker
    turns = [(0.0, 2.0, "SPEAKER_00"), (2.0, 4.0, "SPEAKER_01")]
    emb = {"SPEAKER_00": [1.0, 0.0], "SPEAKER_01": [0.0, 1.0]}
    monkeypatch.setattr(diarize, "diarize", lambda *a, **k: (turns, emb))

    cfg = build_parser().parse_args([
        str(audio), "-o", str(out), "--data-dir", str(data),
        "--asr", "faster", "--no-enroll", "--no-personas",
        "--notes-engine", "none", "--no-progress", "--hf-token", "x",
    ])
    run(cfg)

    payload = json.loads((out / "demo2.transcript.json").read_text())
    speakers = {s["speaker"] for s in payload["segments"]}
    # unknown voiceprints with --no-enroll stay as diarization labels
    assert speakers == {"SPEAKER_00", "SPEAKER_01"}


def test_notes_and_reports_writers(tmp_path):
    out = tmp_path / "o"
    out.mkdir()
    ps = PersonaStore(tmp_path / "people")
    ps.write_profile("Priya", "Prefers concise bullets.")

    def fake_gen(instr, content, label="notes"):
        return f"# {label}\nok"

    np = notes_mod.write_notes("transcript md", "instr", out, "m", fake_gen)
    assert np and np.exists()
    reports = notes_mod.write_reports(["Priya"], "transcript md", "rinstr",
                                      ps, out, "m", fake_gen)
    assert reports and reports[0][1].exists()
    assert (out / "m.report.priya.md").exists()
