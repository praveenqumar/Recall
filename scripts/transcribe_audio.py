#!/usr/bin/env python3
"""Transcribe an audio file with Whisper Large-v3-Turbo.

Example:
    python scripts/transcribe_audio.py hinglish_audio.m4a --language en

Outputs, by default, are written next to the input file:
    <input-stem>.transcript.txt
    <input-stem>.transcript.md
    <input-stem>.transcript.json
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

from faster_whisper import WhisperModel


FFMPEG_SPEECH_FILTER = "highpass=f=80,lowpass=f=8000,afftdn=nf=-25,loudnorm=I=-18:TP=-1.5:LRA=11"
FFMPEG_POST_AI_FILTER = "highpass=f=80,lowpass=f=8000,loudnorm=I=-18:TP=-1.5:LRA=11"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Transcribe audio with Whisper Large-v3-Turbo")
    parser.add_argument("audio", type=Path, help="Input audio file, e.g. .m4a/.wav/.mp3")
    parser.add_argument("--model", default="large-v3-turbo", help="faster-whisper model name")
    parser.add_argument("--language", default="en", help="Language hint. Use 'en' for Hinglish in Roman script")
    parser.add_argument("--device", default="cpu", help="Inference device: cpu/cuda/auto")
    parser.add_argument("--compute-type", default="int8", help="Compute type, e.g. int8/float16/float32")
    parser.add_argument("--output-dir", type=Path, default=None, help="Directory for transcript outputs")
    parser.add_argument("--keep-enhanced", action="store_true", help="Keep normalized/enhanced WAV used for transcription")
    parser.add_argument(
        "--enhancer",
        choices=("deepfilternet", "ffmpeg", "none"),
        default="deepfilternet",
        help="Audio enhancement backend. deepfilternet is AI speech enhancement; ffmpeg is DSP-only cleanup.",
    )
    parser.add_argument(
        "--deepfilter-command",
        default="deepFilter",
        help="DeepFilterNet CLI command. Usually `deepFilter` when deepfilternet is installed.",
    )
    parser.add_argument(
        "--deepfilter-atten-lim",
        type=int,
        default=None,
        help="Optional DeepFilterNet attenuation limit in dB. Lower values preserve more background ambience.",
    )
    parser.add_argument(
        "--no-enhance",
        action="store_true",
        help="Deprecated alias for `--enhancer none`.",
    )
    return parser.parse_args()


def require_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None:
        raise SystemExit("ffmpeg is required. Install it first, e.g. `brew install ffmpeg`.")


def require_command(command: str, install_hint: str) -> None:
    if shutil.which(command) is None:
        raise SystemExit(f"Required command not found: {command}\n{install_hint}")


def run_ffmpeg(input_audio: Path, output_wav: Path, *, sample_rate: int, audio_filter: str | None = None) -> None:
    output_wav.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(input_audio),
    ]
    if audio_filter:
        command += ["-af", audio_filter]
    command += ["-ar", str(sample_rate), "-ac", "1", str(output_wav)]
    subprocess.run(command, check=True)


def enhance_with_ffmpeg(input_audio: Path, output_wav: Path) -> None:
    """Convert to mono 16 kHz WAV and apply DSP voice-memo cleanup."""
    run_ffmpeg(input_audio, output_wav, sample_rate=16000, audio_filter=FFMPEG_SPEECH_FILTER)


def enhance_with_deepfilternet(
    input_audio: Path,
    output_wav: Path,
    *,
    work_dir: Path,
    deepfilter_command: str,
    atten_lim: int | None,
) -> None:
    """Use DeepFilterNet AI speech enhancement, then convert to Whisper-friendly WAV.

    DeepFilterNet expects speech audio and works best at 48 kHz. After AI enhancement,
    we normalize and resample to 16 kHz mono for Whisper.
    """
    require_command(
        deepfilter_command,
        "Install the AI enhancer with `pip install -r requirements.txt` or `pip install deepfilternet`.",
    )

    df_input = work_dir / f"{input_audio.stem}.deepfilter-input.wav"
    df_output_dir = work_dir / "deepfilternet-output"
    df_output_dir.mkdir(parents=True, exist_ok=True)

    # Keep this conversion light: do not denoise before the neural enhancer.
    run_ffmpeg(input_audio, df_input, sample_rate=48000, audio_filter="highpass=f=60,loudnorm=I=-18:TP=-1.5:LRA=11")

    command = [deepfilter_command, "--output-dir", str(df_output_dir)]
    if atten_lim is not None:
        command += ["--atten-lim", str(atten_lim)]
    command.append(str(df_input))

    subprocess.run(command, check=True, cwd=work_dir)

    candidates = sorted(df_output_dir.rglob("*.wav"), key=lambda path: path.stat().st_mtime, reverse=True)
    if not candidates:
        raise RuntimeError(f"DeepFilterNet did not write a WAV file in {df_output_dir}")

    ai_enhanced = candidates[0]
    run_ffmpeg(ai_enhanced, output_wav, sample_rate=16000, audio_filter=FFMPEG_POST_AI_FILTER)


def fmt_time(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"


def transcript_paths(audio: Path, output_dir: Path | None) -> tuple[Path, Path, Path]:
    out_dir = output_dir or audio.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = audio.stem
    return (
        out_dir / f"{stem}.transcript.txt",
        out_dir / f"{stem}.transcript.md",
        out_dir / f"{stem}.transcript.json",
    )


def transcribe(audio_for_model: Path, args: argparse.Namespace) -> dict:
    model = WhisperModel(args.model, device=args.device, compute_type=args.compute_type)
    segments_iter, info = model.transcribe(
        str(audio_for_model),
        language=args.language,
        task="transcribe",
        beam_size=5,
        condition_on_previous_text=False,
        vad_filter=True,
        vad_parameters={
            "threshold": 0.2,
            "min_speech_duration_ms": 100,
            "min_silence_duration_ms": 500,
            "speech_pad_ms": 400,
        },
        hallucination_silence_threshold=2,
    )

    segments = []
    for s in segments_iter:
        text = s.text.strip()
        if not text:
            continue
        segment = {
            "start": float(s.start),
            "end": float(s.end),
            "text": text,
            "avg_logprob": float(s.avg_logprob),
            "no_speech_prob": float(s.no_speech_prob),
            "compression_ratio": float(s.compression_ratio),
        }
        segments.append(segment)
        print(f"[{fmt_time(s.start)} -> {fmt_time(s.end)}] {text}")

    return {
        "model": args.model,
        "language": info.language,
        "language_probability": info.language_probability,
        "duration": info.duration,
        "enhancer": args.enhancer,
        "segments": segments,
        "text": " ".join(s["text"] for s in segments).strip(),
    }


def write_outputs(result: dict, audio: Path, output_dir: Path | None) -> tuple[Path, Path, Path]:
    txt_path, md_path, json_path = transcript_paths(audio, output_dir)

    txt_path.write_text(result["text"] + "\n", encoding="utf-8")
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        f"# Transcript: {audio.name}",
        "",
        f"Model: `{result['model']}`",
        f"Detected language: `{result['language']}`",
        f"Audio enhancer: `{result['enhancer']}`",
        "",
    ]
    for segment in result["segments"]:
        lines.append(f"[{fmt_time(segment['start'])} – {fmt_time(segment['end'])}] {segment['text']}")

    md_path.write_text("\n\n".join(lines) + "\n", encoding="utf-8")
    return txt_path, md_path, json_path


def prepare_audio_for_model(args: argparse.Namespace, input_audio: Path) -> tuple[Path, Path | None, tempfile.TemporaryDirectory | None]:
    if args.no_enhance:
        args.enhancer = "none"

    if args.enhancer == "none":
        return input_audio, None, None

    if args.keep_enhanced:
        enhanced_path = (args.output_dir or input_audio.parent) / f"{input_audio.stem}.enhanced.wav"
        temp_dir = tempfile.TemporaryDirectory(prefix="transcribe-audio-work-")
    else:
        temp_dir = tempfile.TemporaryDirectory(prefix="transcribe-audio-")
        enhanced_path = Path(temp_dir.name) / f"{input_audio.stem}.enhanced.wav"

    work_dir = Path(temp_dir.name)

    try:
        if args.enhancer == "deepfilternet":
            enhance_with_deepfilternet(
                input_audio,
                enhanced_path,
                work_dir=work_dir,
                deepfilter_command=args.deepfilter_command,
                atten_lim=args.deepfilter_atten_lim,
            )
        elif args.enhancer == "ffmpeg":
            enhance_with_ffmpeg(input_audio, enhanced_path)
        else:
            raise ValueError(f"Unsupported enhancer: {args.enhancer}")
    except Exception:
        temp_dir.cleanup()
        raise

    return enhanced_path, enhanced_path, temp_dir


def main() -> None:
    args = parse_args()
    input_audio = args.audio.expanduser().resolve()
    if not input_audio.exists():
        raise SystemExit(f"Input audio not found: {input_audio}")

    require_ffmpeg()
    audio_for_model, enhanced_path, temp_dir = prepare_audio_for_model(args, input_audio)

    try:
        result = transcribe(audio_for_model, args)
        txt_path, md_path, json_path = write_outputs(result, input_audio, args.output_dir)
    finally:
        if temp_dir is not None:
            # If --keep-enhanced was used, the final enhanced WAV lives outside this temp work dir.
            temp_dir.cleanup()

    print("\nWrote:")
    print(f"- {txt_path}")
    print(f"- {md_path}")
    print(f"- {json_path}")
    if enhanced_path is not None and args.keep_enhanced:
        print(f"- {enhanced_path}")


if __name__ == "__main__":
    main()
