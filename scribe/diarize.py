"""
scribe.diarize — speaker diarization slice (pyannote 3.1).

diarize()        -> (turns, emb_map): who-spoke-when + one voiceprint per speaker.
assign_speakers() -> stamps each Segment with its best-overlap speaker label.

Degrades gracefully: no HF token, missing pyannote, or a load failure all just
disable speaker labels rather than crashing the run.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from .common import Segment, log
from .metrics import LiveStatus, Metrics


def diarize(wav: Path, hf_token: Optional[str], metrics: Metrics,
            progress: bool) -> tuple[Optional[list[tuple]], dict]:
    if not hf_token:
        log("diarization: no HF token (set HF_TOKEN); skipping speaker labels")
        return None, {}
    try:
        from pyannote.audio import Pipeline
    except ImportError:
        log("diarization: pyannote.audio not installed; skipping "
            "(pip install pyannote.audio)")
        return None, {}
    try:
        try:
            pipe = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1", token=hf_token)
        except TypeError:
            # older pyannote.audio used the `use_auth_token` kwarg
            pipe = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1", use_auth_token=hf_token)
    except Exception as e:
        log(f"diarization: could not load pipeline ({e}); skipping. "
            "Did you accept the model conditions on huggingface.co?")
        return None, {}
    try:
        import torch
        if torch.backends.mps.is_available():
            pipe.to(torch.device("mps"))
    except Exception:
        pass

    with LiveStatus("diarization (pyannote)", metrics, progress):
        emb_map: dict[str, list] = {}
        out = pipe(str(wav))
        # pyannote >= 4 returns DiarizeOutput(speaker_diarization, ...,
        # speaker_embeddings); 3.x returned an Annotation, or an
        # (annotation, embeddings) tuple when called with return_embeddings.
        # One embedding per speaker, aligned to annotation.labels() — this is
        # the bridge into the identity layer.
        embeddings = None
        if hasattr(out, "speaker_diarization"):
            annotation = out.speaker_diarization
            embeddings = out.speaker_embeddings
        elif isinstance(out, tuple):
            annotation, embeddings = out
        else:
            annotation = out
        if embeddings is not None:
            for i, lab in enumerate(annotation.labels()):
                try:
                    emb_map[lab] = [float(x) for x in embeddings[i]]
                except Exception:
                    pass

    turns = [(t.start, t.end, spk)
             for t, _, spk in annotation.itertracks(yield_label=True)]
    log(f"diarization: {len(turns)} turns, "
        f"{len({t[2] for t in turns})} speakers, {len(emb_map)} voiceprints")
    return turns, emb_map


def assign_speakers(segs: list[Segment], turns: Optional[list[tuple]]) -> None:
    if not turns:
        return
    for seg in segs:
        best, best_ov = None, 0.0
        for t_start, t_end, spk in turns:
            ov = min(seg.end, t_end) - max(seg.start, t_start)
            if ov > best_ov:
                best_ov, best = ov, spk
        seg.speaker = best or "SPEAKER_?"
