"""
recall.asr — C1 slice: pluggable ASR backends behind one interface.

    transcribe(backend, wav, language, chunk_s, model, metrics, progress, work)
        -> list[Segment]

Two backends:
  * mlx     mlx-whisper on Apple-Silicon Metal (Track B premise; unproven —
            see design doc §6). Chunked with a duration-weighted tqdm bar.
  * faster  faster-whisper (the proven Track A path); streams segments so the
            progress bar advances by segment end-time, with built-in VAD.

backend="auto" tries mlx first, falls back to faster. Each backend resolves its
own model id default so the caller can stay backend-agnostic.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from .common import Segment, die, fmt_ts, log, segs_from, slice_wav, wav_duration
from .metrics import LiveStatus, Metrics

# silence HuggingFace's "Fetching N files / Download complete" spam (our own bar
# already shows ASR progress). Set before mlx_whisper/faster_whisper import hf_hub.
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

HINGLISH_PRIMER = (
    "The following is a casual business meeting spoken in Hinglish, a natural "
    "mix of Hindi and English. Speakers switch between Hindi and English "
    "mid-sentence. Transcribe Hindi in Devanagari and keep English as English."
)

# full large-v3 (more accurate than the distilled turbo; slower but fine for
# meeting notes). Override with --model …-turbo for speed.
DEFAULT_MODEL = {
    "mlx": "mlx-community/whisper-large-v3-mlx",
    "faster": "large-v3",
}


# --------------------------------------------------------------------------- #
# mlx-whisper backend (chunked for an honest progress bar)
# --------------------------------------------------------------------------- #
def _mlx(wav: Path, language: Optional[str], chunk_s: float, model: str,
         metrics: Metrics, progress: bool, work: Path) -> list[Segment]:
    import mlx_whisper  # raises ImportError if unavailable → handled by caller

    duration = wav_duration(wav)
    common = dict(path_or_hf_repo=model, language=language,
                  initial_prompt=HINGLISH_PRIMER, word_timestamps=False,
                  condition_on_previous_text=False, verbose=None)

    if chunk_s <= 0 or duration <= chunk_s:
        log(f"ASR(mlx): {model} single pass, {fmt_ts(duration)}")
        with LiveStatus("ASR (mlx-whisper)", metrics, progress):
            result = mlx_whisper.transcribe(str(wav), **common)
        return segs_from(result.get("segments", []), 0.0)

    log(f"ASR(mlx): {model} chunked @ {int(chunk_s)}s, total {fmt_ts(duration)}")
    try:
        from tqdm import tqdm
        bar = tqdm(total=round(duration), unit="s", disable=not progress,
                   bar_format="  {desc} {percentage:3.0f}%|{bar}| "
                              "{n:.0f}/{total:.0f}s [{elapsed}<{remaining}] {postfix}",
                   desc="ASR")
    except ImportError:
        bar = None

    segs: list[Segment] = []
    offset = 0.0
    while offset < duration:
        dur = min(chunk_s, duration - offset)
        chunk = work / f"chunk_{int(offset)}.wav"
        slice_wav(wav, offset, dur, chunk)
        result = mlx_whisper.transcribe(str(chunk), **common)
        segs.extend(segs_from(result.get("segments", []), offset))
        chunk.unlink(missing_ok=True)
        if bar:
            bar.update(round(dur))
            bar.set_postfix_str(metrics.snapshot())
        else:
            pct = min(100, round((offset + dur) / duration * 100))
            log(f"ASR {pct:3d}%  {fmt_ts(offset + dur)}/{fmt_ts(duration)}")
        offset += dur
    if bar:
        bar.close()
    return segs


# --------------------------------------------------------------------------- #
# faster-whisper backend (streaming; progress by segment end-time)
# --------------------------------------------------------------------------- #
def _faster(wav: Path, language: Optional[str], chunk_s: float, model: str,
            metrics: Metrics, progress: bool, work: Path) -> list[Segment]:
    from faster_whisper import WhisperModel  # ImportError handled by caller

    duration = wav_duration(wav)
    log(f"ASR(faster): {model} streaming, {fmt_ts(duration)}")
    m = WhisperModel(model, device="cpu", compute_type="int8")
    seg_iter, _info = m.transcribe(
        str(wav), language=language, task="transcribe", beam_size=5,
        condition_on_previous_text=False, vad_filter=True,
        # anti-hallucination VAD: stricter speech gate + cap segment length so a
        # runaway decode can't grow into a 500-word loop; split on natural pauses.
        vad_parameters={"threshold": 0.45, "min_speech_duration_ms": 250,
                        "min_silence_duration_ms": 2000, "max_speech_duration_s": 15,
                        "speech_pad_ms": 400},
        hallucination_silence_threshold=2,
        compression_ratio_threshold=2.2,   # reject high-compression (repetitive) text
    )
    try:
        from tqdm import tqdm
        bar = tqdm(total=round(duration), unit="s", disable=not progress,
                   bar_format="  {desc} {percentage:3.0f}%|{bar}| "
                              "{n:.0f}/{total:.0f}s [{elapsed}<{remaining}] {postfix}",
                   desc="ASR")
    except ImportError:
        bar = None

    segs: list[Segment] = []
    last_end = 0.0
    for s in seg_iter:
        txt = (s.text or "").strip()
        if txt:
            segs.append(Segment(start=float(s.start), end=float(s.end), text=txt,
                                avg_logprob=float(s.avg_logprob)))
        if bar:
            bar.update(max(0, round(float(s.end) - last_end)))
            bar.set_postfix_str(metrics.snapshot())
            last_end = float(s.end)
    if bar:
        bar.close()
    return segs


_REGISTRY = {"mlx": _mlx, "faster": _faster}


def transcribe(backend: str, wav: Path, language: Optional[str], chunk_s: float,
               model: Optional[str], metrics: Metrics, progress: bool,
               work: Path) -> list[Segment]:
    """Dispatch to a backend. 'auto' = mlx then faster-whisper."""
    order = ["mlx", "faster"] if backend == "auto" else [backend]
    last_err: Optional[Exception] = None
    for name in order:
        fn = _REGISTRY.get(name)
        if fn is None:
            die(f"unknown ASR backend: {name}")
        mdl = model or DEFAULT_MODEL[name]
        try:
            segs = fn(wav, language, chunk_s, mdl, metrics, progress, work)
            log(f"ASR({name}): {len(segs)} segments")
            return segs
        except ImportError as e:
            last_err = e
            log(f"ASR: backend '{name}' unavailable ({e}); "
                + ("trying next" if name != order[-1] else "no fallback left"))
        except Exception as e:  # API drift etc. — try the next backend
            last_err = e
            log(f"ASR: backend '{name}' failed ({e}); "
                + ("trying next" if name != order[-1] else "no fallback left"))
    die(f"all ASR backends failed; last error: {last_err}")
    return []  # unreachable
