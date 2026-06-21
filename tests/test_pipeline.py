"""
End-to-end pipeline test with the ASR + diarization backends mocked out, so it
runs anywhere (only ffmpeg is required, for ingest). Also covers the new
coverage-diagnostic and the notes/reports writers.
"""
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

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


def test_claude_token_accounting(monkeypatch):
    from recall import generate as gen
    from recall.metrics import Metrics
    payload = {"is_error": False, "result": "NOTES TEXT",
               "usage": {"input_tokens": 10, "cache_read_input_tokens": 90,
                         "cache_creation_input_tokens": 0, "output_tokens": 5},
               "total_cost_usd": 0.01}

    class R:
        returncode, stdout, stderr = 0, json.dumps(payload), ""

    monkeypatch.setattr(gen, "have", lambda c: True)
    monkeypatch.setattr(gen.subprocess, "run", lambda *a, **k: R())
    gen.TOKENS.update(input=0, output=0, cost=0.0)
    out = gen._engine_claude("instr", "content", "notes", Metrics(), False)
    assert out == "NOTES TEXT"
    assert gen.TOKENS["input"] == 100 and gen.TOKENS["output"] == 5
    assert "100 in + 5 out" in gen.token_summary()


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

    md = next(out.glob("*demo.transcript.md"))
    js = next(out.glob("*demo.transcript.json"))
    assert md.exists() and js.exists()
    payload = json.loads(js.read_text())
    assert len(payload["segments"]) == 2
    assert "coverage" in payload and payload["coverage"]["n_segments"] == 2
    assert "नमस्ते" in md.read_text()


def test_compress_repeats():
    segs = [Segment(0, 1, "hello"),
            Segment(1, 2, "झाल झाल"), Segment(2, 3, "झाल झाल"),
            Segment(3, 4, "झाल झाल"), Segment(4, 5, "bye")]
    out = tx.compress_repeats(segs)
    assert [s.text for s in out] == ["hello", "झाल झाल", "bye"]  # looped segs -> 1
    assert out[1].end == 4.0                                      # end extended over run
    # within-segment single-word run
    assert tx.compress_repeats([Segment(0, 1, "that that that is good")])[0].text \
        == "that is good"
    # within-segment PHRASE loop (period 2)
    assert tx.compress_repeats(
        [Segment(0, 1, "so we go and go and go and go and stop")])[0].text \
        == "so we go and stop"


def test_pmap_preserves_order():
    from recall.generate import pmap
    assert pmap([lambda i=i: i * 10 for i in range(5)]) == [0, 10, 20, 30, 40]


def test_personas_parallel_updates_all(tmp_path):
    from recall import personas as pm
    store = pm.PersonaStore(tmp_path / "people")
    segs = [Segment(0, 1, "hi", speaker="A"), Segment(1, 2, "yo", speaker="B")]
    gen = lambda instr, content, label: f"profile {label}"   # noqa: E731
    pm.build_personas(segs, ["A", "B"], "m1", store, "P", gen, parallel=True)
    assert store.read_profile("A").strip() == "profile persona:A"
    assert store.read_profile("B").strip() == "profile persona:B"


def test_enhance_chain_order(tmp_path, monkeypatch):
    from recall import enhance as enh
    seen = []
    monkeypatch.setattr(enh, "enhance",
                        lambda name, wav, *a, **k: (seen.append(name) or wav))
    out = enh.enhance_chain("demucs, ffmpeg, none, deepfilternet",
                            tmp_path / "x.wav", tmp_path, None, False)
    assert seen == ["demucs", "ffmpeg", "deepfilternet"]   # in order, 'none' skipped
    assert out == tmp_path / "x.wav"


def test_second_run_dedups(tmp_path, monkeypatch):
    audio = tmp_path / "demo.wav"
    _make_wav(audio)
    out = tmp_path / "out"
    data = tmp_path / "data"
    calls = {"n": 0}

    def fake_asr(*a, **k):
        calls["n"] += 1
        return [Segment(0.0, 1.0, "hello")]

    monkeypatch.setattr(asr, "transcribe", fake_asr)
    args = [str(audio), "-o", str(out), "--data-dir", str(data),
            "--asr", "faster", "--enhance", "none", "--no-diarize",
            "--notes-engine", "none", "--no-progress"]
    run(build_parser().parse_args(args))
    run(build_parser().parse_args(args))          # same audio → short-circuits
    assert calls["n"] == 1
    run(build_parser().parse_args(args + ["--force"]))   # --force regenerates
    assert calls["n"] == 2


def test_regenerates_when_output_deleted(tmp_path, monkeypatch):
    audio = tmp_path / "demo.wav"
    _make_wav(audio)
    out = tmp_path / "out"
    data = tmp_path / "data"
    calls = {"n": 0}

    def fake_asr(*a, **k):
        calls["n"] += 1
        return [Segment(0.0, 1.0, "hello")]

    monkeypatch.setattr(asr, "transcribe", fake_asr)
    args = [str(audio), "-o", str(out), "--data-dir", str(data),
            "--asr", "faster", "--enhance", "none", "--no-diarize",
            "--notes-engine", "none", "--no-progress"]
    run(build_parser().parse_args(args))
    assert calls["n"] == 1
    for f in out.glob("*demo.transcript.*"):     # delete the recorded outputs
        f.unlink()
    run(build_parser().parse_args(args))          # stale row → regenerate, not skip
    assert calls["n"] == 2


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

    payload = json.loads(next(out.glob("*demo2.transcript.json")).read_text())
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
