#!/usr/bin/env python
"""
Tier-0 transcript-coverage eval (issue #9), mlx-only.

The pipeline's own coverage() is computed from the ASR's own segments, so speech
the ASR never emits is invisible. This compares the mlx transcript against an
INDEPENDENT speech detector (faster-whisper's Silero VAD) — the mlx path uses no
VAD, so it is a genuine second opinion.

Reports:
  vad_voiced_s   = seconds an independent VAD says someone is talking
  asr_speech_s   = seconds the mlx transcript actually covers
  dropped_s      = vad_voiced_s - asr_speech_s  (speech the pipeline lost)
  wpm_voiced     = words / minute over voiced time (capture density)
  silent_holes   = voiced spans with ~0 transcript words underneath

Run with the recall tool's interpreter (has mlx_whisper + faster_whisper.vad):
  /Users/praveen/.local/share/uv/tools/recall/bin/python eval/vad_coverage.py AUDIO
"""
from __future__ import annotations

import json
import sys
import tempfile
import wave
from pathlib import Path

import numpy as np

from recall.asr import transcribe
from recall.common import fmt_ts, ingest, log, wav_duration
from recall.metrics import Metrics
from recall.transcript import coverage

SR = 16000


def read_pcm16(wav: Path) -> np.ndarray:
    """Load a 16 kHz mono PCM-s16le wav (what ingest() writes) as float32 [-1,1]."""
    with wave.open(str(wav), "rb") as w:
        assert w.getframerate() == SR, f"expected {SR} Hz, got {w.getframerate()}"
        assert w.getnchannels() == 1, "expected mono"
        raw = w.readframes(w.getnframes())
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


def voiced_spans(audio: np.ndarray, threshold: float):
    """Independent VAD → list of (start_s, end_s) voiced spans + total voiced sec."""
    from faster_whisper.vad import VadOptions, get_speech_timestamps
    # threshold is the one calibration knob (doc §5): too high drops soft speech,
    # too low calls noise speech. Default 0.5 is the VAD's own; we expose it.
    opts = VadOptions(threshold=threshold, min_silence_duration_ms=500,
                      speech_pad_ms=200)
    stamps = get_speech_timestamps(audio, opts, sampling_rate=SR)
    spans = [(s["start"] / SR, s["end"] / SR) for s in stamps]
    total = sum(e - s for s, e in spans)
    return spans, total


def words_per_span(segs, spans):
    """For each voiced span, sum transcript words whose midpoint falls inside it.
    Returns list of (start_s, end_s, words) — a span with ~0 words is a hole."""
    mids = [((s.start + s.end) / 2.0, len(s.text.split())) for s in segs]
    out = []
    for a, b in spans:
        w = sum(n for mid, n in mids if a <= mid < b)
        out.append((a, b, w))
    return out


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    audio_in = Path(sys.argv[1]).expanduser()
    threshold = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
    chunk_s = float(sys.argv[3]) if len(sys.argv) > 3 else 240.0
    if not audio_in.exists():
        log(f"no such file: {audio_in}")
        return 1

    with tempfile.TemporaryDirectory() as td:
        work = Path(td)
        log(f"ingest {audio_in.name} -> 16 kHz mono wav")
        wav = ingest(audio_in, work)
        duration = wav_duration(wav)

        log(f"independent VAD (Silero, threshold={threshold}) over {fmt_ts(duration)}")
        audio = read_pcm16(wav)
        spans, vad_voiced = voiced_spans(audio, threshold)

        log("mlx transcription (real pipeline path)")
        segs = transcribe("mlx", wav, "en", chunk_s, None, Metrics(),
                          progress=True, work=work)
        cov = coverage(segs, duration)

    asr_speech = cov["speech_s"]
    dropped = max(0.0, vad_voiced - asr_speech)
    n_words = sum(len(s.text.split()) for s in segs)
    wpm_voiced = (n_words / (vad_voiced / 60.0)) if vad_voiced else 0.0

    per = words_per_span(segs, spans)
    holes = [(a, b) for a, b, w in per if (b - a) >= 3.0 and w == 0]
    holes_s = sum(b - a for a, b in holes)

    report = {
        "audio": str(audio_in),
        "duration_s": round(duration, 1),
        "vad_threshold": threshold,
        "vad_voiced_s": round(vad_voiced, 1),
        "asr_speech_s": asr_speech,
        "asr_coverage_self": cov["coverage"],          # the old, self-graded number
        "dropped_s": round(dropped, 1),
        "dropped_pct_of_voiced": round(dropped / vad_voiced, 3) if vad_voiced else 0.0,
        "n_words": n_words,
        "wpm_voiced": round(wpm_voiced, 1),
        "n_voiced_spans": len(spans),
        "silent_holes_count": len(holes),
        "silent_holes_s": round(holes_s, 1),
        "top_holes": [[fmt_ts(a), round(b - a, 1)] for a, b in
                      sorted(holes, key=lambda h: h[1] - h[0], reverse=True)[:5]],
        "n_segments": cov["n_segments"],
        "repetition": cov["repetition"],
        "mean_logprob": cov["mean_logprob"],
    }
    out = audio_in.with_suffix(".eval.json")
    out.write_text(json.dumps(report, indent=2))

    log("=" * 60)
    log(f"VAD says people talked : {fmt_ts(vad_voiced)}  ({len(spans)} spans)")
    log(f"transcript covers      : {fmt_ts(asr_speech)}")
    log(f"DROPPED speech         : {fmt_ts(dropped)}  "
        f"({report['dropped_pct_of_voiced']*100:.0f}% of voiced)")
    log(f"words / voiced-minute  : {wpm_voiced:.0f}")
    log(f"silent holes (>=3s)    : {len(holes)} spans, {fmt_ts(holes_s)} total")
    if report["top_holes"]:
        log(f"biggest hole           : {report['top_holes'][0][1]:.0f}s at "
            f"{report['top_holes'][0][0]}")
    log(f"report -> {out}")
    return 0


def _selftest() -> None:
    # words_per_span: midpoint-in-span attribution + hole detection, no model.
    class Seg:
        def __init__(self, start, end, text):
            self.start, self.end, self.text = start, end, text
    segs = [Seg(0, 2, "one two three"), Seg(20, 22, "")]   # span2 has no words
    per = words_per_span(segs, [(0.0, 5.0), (18.0, 25.0)])
    assert per == [(0.0, 5.0, 3), (18.0, 25.0, 0)], per
    print("selftest ok")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--selftest":
        _selftest()
    else:
        sys.exit(main())
