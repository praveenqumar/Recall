"""
recall.enhance — C2 slice: pluggable audio enhancement.

One interface, four implementations, selected by name:

    none           passthrough (the ingested 16 kHz mono wav as-is)
    ffmpeg         DSP speech cleanup (high/low-pass, afftdn, loudnorm)
    deepfilternet  AI speech enhancement (deepFilter CLI), then re-normalize
    demucs         vocal isolation (--two-stems vocals)

The A/B harness compares these; whatever wins per file becomes the default. All
implementations take the ingested wav and return a 16 kHz mono wav, so the rest of
the pipeline is identical regardless of choice. An unavailable backend logs and
falls back to the input rather than crashing.
"""
from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Callable

from .common import have, log, run_ffmpeg
from .metrics import LiveStatus, Metrics

# filters carried over from the proven Track A pipeline
FFMPEG_DSP = "highpass=f=80,lowpass=f=8000,afftdn=nf=-25,loudnorm=I=-18:TP=-1.5:LRA=11"
FFMPEG_POST_AI = "highpass=f=80,lowpass=f=8000,loudnorm=I=-18:TP=-1.5:LRA=11"

NAMES = ("none", "ffmpeg", "deepfilternet", "demucs")


def _none(wav: Path, work: Path, metrics: Metrics, progress: bool,
          df_cmd: str) -> Path:
    return wav


def _ffmpeg(wav: Path, work: Path, metrics: Metrics, progress: bool,
            df_cmd: str) -> Path:
    out = work / "enh_ffmpeg.wav"
    with LiveStatus("enhance (ffmpeg DSP)", metrics, progress):
        run_ffmpeg(wav, out, sr=16000, af=FFMPEG_DSP)
    return out


def _deepfilternet(wav: Path, work: Path, metrics: Metrics, progress: bool,
                   df_cmd: str) -> Path:
    if not have(df_cmd):
        log(f"enhance: {df_cmd} not found (pip install deepfilternet); "
            "using original audio")
        return wav
    # DeepFilterNet works best at 48 kHz; upsample, enhance, then back to 16 kHz.
    df_in = work / "df_in_48k.wav"
    run_ffmpeg(wav, df_in, sr=48000, af="highpass=f=60,loudnorm=I=-18:TP=-1.5:LRA=11")
    df_out = work / "df_out"
    df_out.mkdir(exist_ok=True)
    with LiveStatus("enhance (DeepFilterNet)", metrics, progress):
        r = subprocess.run([df_cmd, "--output-dir", str(df_out), str(df_in)],
                           capture_output=True, text=True, cwd=work)
    if r.returncode != 0:
        log("enhance: DeepFilterNet failed; using original audio")
        return wav
    cands = sorted(df_out.rglob("*.wav"), key=lambda p: p.stat().st_mtime,
                   reverse=True)
    if not cands:
        log("enhance: DeepFilterNet produced no output; using original audio")
        return wav
    out = work / "enh_deepfilternet.wav"
    run_ffmpeg(cands[0], out, sr=16000, af=FFMPEG_POST_AI)
    return out


def _demucs(wav: Path, work: Path, metrics: Metrics, progress: bool,
            df_cmd: str) -> Path:
    if not have("demucs"):
        log("enhance: demucs not installed (pip install demucs); "
            "using original audio")
        return wav
    out_root = work / "demucs"
    out_root.mkdir(exist_ok=True)
    with LiveStatus("enhance (Demucs vocals)", metrics, progress):
        r = subprocess.run(
            ["demucs", "--two-stems", "vocals", "-o", str(out_root), str(wav)],
            capture_output=True, text=True)
    if r.returncode != 0:
        log("enhance: demucs failed; using original audio")
        return wav
    hits = list(out_root.rglob("vocals.wav"))
    if not hits:
        return wav
    out = work / "enh_demucs.wav"
    run_ffmpeg(hits[0], out, sr=16000)
    return out


_REGISTRY: dict[str, Callable] = {
    "none": _none,
    "ffmpeg": _ffmpeg,
    "deepfilternet": _deepfilternet,
    "demucs": _demucs,
}


def enhance(name: str, wav: Path, work: Path, metrics: Metrics,
            progress: bool, df_cmd: str = "deepFilter") -> Path:
    fn = _REGISTRY.get(name)
    if fn is None:
        log(f"enhance: unknown enhancer '{name}'; using original audio")
        return wav
    return fn(wav, work, metrics, progress, df_cmd)


def enhance_chain(spec: str, wav: Path, work: Path, metrics: Metrics,
                  progress: bool, df_cmd: str = "deepFilter") -> Path:
    """Apply a comma-separated chain of enhancers in order, e.g. 'demucs,ffmpeg'
    (isolate vocals, then DSP-clean them). Each step's output feeds the next;
    'none'/blank steps are skipped."""
    for name in [n.strip() for n in spec.split(",")]:
        if name and name != "none":
            wav = enhance(name, wav, work, metrics, progress, df_cmd)
    return wav
