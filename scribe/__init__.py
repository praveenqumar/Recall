"""
meeting-scribe — offline, on-device Hinglish meeting transcription + notes.

Organized as vertical slices, one capability per module:

    common      Segment + ffmpeg/wav primitives (stdlib only)
    metrics     resource readout + live progress UX
    enhance     C2 — pluggable audio enhancers (none/ffmpeg/deepfilternet/demucs)
    asr         C1 — pluggable ASR backends (faster-whisper / mlx-whisper)
    diarize     pyannote diarization + speaker assignment
    identity    persistent voiceprints + cross-session resolution
    personas    per-person living collaboration profiles
    generate    Claude (`claude -p`) → local-MLX text engine (shared)
    transcript  assemble .md/.json + romanize + coverage diagnostics
    notes       meeting notes + per-person tailored reports
    pipeline    orchestration (wires the slices together)
    cli         argument parsing / config

Each slice owns its own optional heavy dependency and degrades gracefully when it
is missing, so the package imports cleanly even without the ML stack installed.
"""
from __future__ import annotations

PROMPTS_DIR = __import__("pathlib").Path(__file__).resolve().parent / "prompts"

__all__ = ["PROMPTS_DIR"]
