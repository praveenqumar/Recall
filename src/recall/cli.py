"""
recall.cli — argument parsing and entry point.

Run as:  python -m recall AUDIO [options]
"""
from __future__ import annotations

import argparse
import sys

from . import PROMPTS_DIR
from .common import log
from .pipeline import run


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="recall",
        description="Offline Hinglish meeting transcription + AI notes.")
    p.add_argument("audio", help="input audio/video file (m4a, mp3, wav, mp4 ...)")
    p.add_argument("-o", "--output-dir", default="~/.recall",
                   help="output directory (default: ~/.recall)")
    p.add_argument("--title", default=None,
                   help="meeting title used in output filenames and the store")
    p.add_argument("--force", action="store_true",
                   help="regenerate even if this audio was already processed")

    # --- ASR (C1) ---
    p.add_argument("--asr", choices=["auto", "mlx", "faster"], default="auto",
                   help="ASR backend: auto = mlx on Apple Silicon, faster-whisper "
                        "elsewhere; or force mlx / faster (default: auto)")
    p.add_argument("--model", default=None,
                   help="ASR model id (default: backend-specific turbo model)")
    p.add_argument("--language", default="en",
                   help="ASR language hint: 'en' Roman/English (default, best for "
                        "Hinglish notes), 'hi' native Devanagari, 'auto' to detect")
    p.add_argument("--chunk-seconds", type=float, default=240.0,
                   help="ASR chunk size for the progress bar; 0 = single "
                        "max-accuracy pass (default: 240)")

    # --- enhancement (C2) ---
    p.add_argument("--enhance", default="ffmpeg",
                   help="audio enhancer(s): none | ffmpeg | deepfilternet | demucs. "
                        "Default 'ffmpeg' = conservative DSP (highpass/lowpass/"
                        "loudnorm), safe for Whisper. Comma-chain in order, e.g. "
                        "'demucs,ffmpeg'. Use 'none' for raw audio")
    p.add_argument("--denoise", dest="enhance", action="store_const",
                   const="demucs",
                   help="deprecated alias for --enhance demucs")
    p.add_argument("--deepfilter-command", default="deepFilter",
                   help="DeepFilterNet CLI command (default: deepFilter)")

    # --- diarization ---
    diar = p.add_mutually_exclusive_group()
    diar.add_argument("--diarize", dest="diarize", action="store_true",
                      help="speaker labels via pyannote (default: on)")
    diar.add_argument("--no-diarize", dest="diarize", action="store_false",
                      help="disable speaker labels")
    p.set_defaults(diarize=True)
    p.add_argument("--hf-token", default=None,
                   help="HF token for pyannote (or set HF_TOKEN env)")

    # --- output / notes ---
    p.add_argument("--romanize", action="store_true",
                   help="transliterate Devanagari to Roman (default: off)")
    p.add_argument("--keep-repeats", action="store_true",
                   help="keep Whisper's repeated/looped text (default: collapse it "
                        "to save tokens + clean the transcript)")
    p.add_argument("--notes-engine", choices=["auto", "claude", "local", "none"],
                   default="auto", help="auto = Claude then local (default)")
    p.add_argument("--local-model",
                   default="mlx-community/Qwen2.5-7B-Instruct-4bit",
                   help="mlx-lm model for the offline notes fallback")
    p.add_argument("--notes-prompt", default=str(PROMPTS_DIR / "notes.md"),
                   help="path to the notes instruction file")
    p.add_argument("--no-progress", action="store_true",
                   help="disable progress bars / live resource readout")

    # --- identity & personas ---
    ident = p.add_argument_group("speaker identity & personas")
    ident.add_argument("--data-dir", default="~/.recall/data",
                       help="persistent store for voiceprints + personas "
                            "(default: ~/.recall/data)")
    ident.add_argument("--no-enroll", action="store_true",
                       help="don't prompt to name unknown speakers; only "
                            "auto-assign confident voiceprint matches")
    ident.add_argument("--id-high", type=float, default=0.70,
                       help="cosine >= this auto-assigns a known speaker "
                            "(default: 0.70)")
    ident.add_argument("--id-low", type=float, default=0.45,
                       help="cosine in [id-low, id-high) is ambiguous and asks "
                            "you (default: 0.45)")
    ident.add_argument("--no-personas", dest="personas", action="store_false",
                       help="don't build/update per-person profiles")
    p.set_defaults(personas=True)
    ident.add_argument("--persona-prompt", default=str(PROMPTS_DIR / "persona.md"),
                       help="path to the persona-update instruction file")
    ident.add_argument("--report-for", action="append", metavar="NAME",
                       help="also write a report tailored to NAME's profile "
                            "(repeatable)")
    ident.add_argument("--report-prompt", default=str(PROMPTS_DIR / "report.md"),
                       help="path to the tailored-report instruction file")
    return p


def main(argv: list[str] | None = None) -> None:
    if sys.platform != "darwin":
        log("warning: this pipeline targets Apple Silicon (MLX). The mlx backend "
            "will be unavailable here; use --asr faster.")
    cfg = build_parser().parse_args(argv)
    if str(cfg.language).lower() == "auto":
        cfg.language = None
    run(cfg)


if __name__ == "__main__":
    main()
