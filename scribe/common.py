"""
scribe.common — shared primitives used across every slice.

Kept dependency-free (stdlib only) so it imports cleanly anywhere, including in
tests that never touch the ML backends.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


# --------------------------------------------------------------------------- #
# logging — everything to stderr so stdout stays clean for piping
# --------------------------------------------------------------------------- #
def log(msg: str) -> None:
    print(f"[scribe] {msg}", file=sys.stderr, flush=True)


def die(msg: str, code: int = 1) -> None:
    log(f"ERROR: {msg}")
    sys.exit(code)


def have(cmd: str) -> bool:
    return shutil.which(cmd) is not None


def fmt_ts(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


# --------------------------------------------------------------------------- #
# the unit every slice downstream of ASR consumes
# --------------------------------------------------------------------------- #
@dataclass
class Segment:
    start: float
    end: float
    text: str
    speaker: Optional[str] = None
    avg_logprob: Optional[float] = None


def segs_from(raw_segments, offset: float = 0.0) -> list[Segment]:
    """Normalize a backend's raw segment dicts into Segment objects."""
    out: list[Segment] = []
    for s in raw_segments:
        txt = (s.get("text") or "").strip()
        if not txt:
            continue
        out.append(Segment(
            start=float(s["start"]) + offset,
            end=float(s["end"]) + offset,
            text=txt,
            avg_logprob=(float(s["avg_logprob"]) if s.get("avg_logprob") is not None
                         else None),
        ))
    return out


# --------------------------------------------------------------------------- #
# ffmpeg / wav helpers (shared by ingest + every enhancer + ASR chunking)
# --------------------------------------------------------------------------- #
def run_ffmpeg(src: Path, dst: Path, *, sr: int = 16000, af: Optional[str] = None,
               extra_in: Optional[list[str]] = None) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error"]
    cmd += extra_in or []
    cmd += ["-i", str(src)]
    if af:
        cmd += ["-af", af]
    cmd += ["-ar", str(sr), "-ac", "1", "-c:a", "pcm_s16le", str(dst)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        die(f"ffmpeg failed:\n{r.stderr.strip()}")


def ingest(audio_in: Path, work: Path) -> Path:
    """Decode any input to 16 kHz mono PCM wav."""
    if not have("ffmpeg"):
        die("ffmpeg not found. Install with: brew install ffmpeg")
    out = work / "ingest_16k_mono.wav"
    run_ffmpeg(audio_in, out, sr=16000)
    return out


def wav_duration(wav: Path) -> float:
    with wave.open(str(wav), "rb") as w:
        return w.getnframes() / float(w.getframerate())


def slice_wav(src: Path, start_s: float, dur_s: float, dst: Path) -> None:
    """Write [start_s, start_s+dur_s) of a 16-bit mono wav to dst."""
    with wave.open(str(src), "rb") as w:
        fr = w.getframerate()
        nframes = w.getnframes()
        start_f = min(int(start_s * fr), nframes)
        want_f = int(dur_s * fr)
        w.setpos(start_f)
        frames = w.readframes(min(want_f, nframes - start_f))
        params = w.getparams()
    with wave.open(str(dst), "wb") as o:
        o.setparams(params)
        o.writeframes(frames)
