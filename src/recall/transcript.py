"""
recall.transcript — assemble outputs + coverage diagnostics.

build_transcript()  -> (markdown, json_dict): timestamped, speaker-labelled.
romanize()           -> optional Devanagari -> ITRANS transliteration.
coverage()           -> the silent-gap diagnostic (design doc L6 / §4): both
                        example recordings silently dropped ~19 minutes, so every
                        run now reports speech-coverage and the largest gaps and
                        warns when coverage is low.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import asdict

from .common import Segment, fmt_ts, log

# immediate same-word run: "that that that" -> "that", "है है है" -> "है"
_WORD_RUN = re.compile(r"\b(\w+)(?:\s+\1\b)+", re.IGNORECASE | re.UNICODE)


def compress_repeats(segs: list[Segment]) -> list[Segment]:
    """Strip Whisper hallucination repetition before the LLM sees it: collapse
    immediate same-word runs inside each segment, then merge consecutive segments
    with identical text into one (extending its end). Cuts tokens fed to Claude
    without losing real content; near-no-op on clean transcripts."""
    out: list[Segment] = []
    for s in segs:
        s.text = _WORD_RUN.sub(r"\1", s.text)
        if out and out[-1].text.strip().lower() == s.text.strip().lower():
            out[-1].end = s.end          # absorb the duplicate run
        else:
            out.append(s)
    return out


def romanize(segs: list[Segment]) -> None:
    try:
        from indic_transliteration import sanscript
        from indic_transliteration.sanscript import transliterate
    except ImportError:
        log("romanize requested but indic-transliteration not installed; "
            "skipping (pip install indic-transliteration)")
        return
    for seg in segs:
        seg.text = transliterate(seg.text, sanscript.DEVANAGARI, sanscript.ITRANS)


def repetition_ratio(segs: list[Segment]) -> tuple[float, str]:
    """
    Fraction of segments whose (normalized) text equals the single most common
    text. A healthy transcript is lexically diverse; a Whisper hallucination loop
    repeats one short phrase (e.g. the 'झाल झाल' loop that fooled raw coverage at
    101% while 95% of the transcript was garbage). Returns (ratio, dominant_text).
    """
    if not segs:
        return 0.0, ""
    norm = [s.text.strip().lower() for s in segs if s.text.strip()]
    if not norm:
        return 0.0, ""
    text, count = Counter(norm).most_common(1)[0]
    return count / len(norm), text


def coverage(segs: list[Segment], duration: float, warn_below: float = 0.6,
             repeat_warn: float = 0.3, logprob_warn: float = -1.0) -> dict:
    """
    Speech-coverage ratio, largest silent gaps, and hallucination-loop detection.
    Warns on (a) low coverage — audio dropped/silent (the original L6 finding) —
    and (b) high repetition or very low mean avg-logprob — a runaway ASR loop that
    inflates coverage with garbage. Coverage alone is fooled by loops; these two
    extra signals catch the failure mode coverage cannot see.
    """
    speech = sum(max(0.0, s.end - s.start) for s in segs)
    ratio = (speech / duration) if duration else 0.0
    ordered = sorted(segs, key=lambda s: s.start)
    gaps = []
    for i in range(len(ordered) - 1):
        g = ordered[i + 1].start - ordered[i].end
        if g > 0:
            gaps.append((round(g, 1), fmt_ts(ordered[i].end)))
    gaps.sort(reverse=True)
    rep, dominant = repetition_ratio(segs)
    logprobs = [s.avg_logprob for s in segs if s.avg_logprob is not None]
    mean_lp = (sum(logprobs) / len(logprobs)) if logprobs else None
    report = {
        "duration_s": round(duration, 1),
        "speech_s": round(speech, 1),
        "coverage": round(ratio, 3),
        "n_segments": len(segs),
        "top_gaps": gaps[:3],
        "repetition": round(rep, 3),
        "mean_logprob": round(mean_lp, 3) if mean_lp is not None else None,
    }
    msg = (f"coverage: {ratio*100:.0f}% speech "
           f"({fmt_ts(speech)} of {fmt_ts(duration)}), {len(segs)} segments")
    if gaps:
        g, at = gaps[0]
        msg += f"; largest silent gap {g:.0f}s at {at}"
    log(msg)
    if ratio < warn_below:
        log(f"⚠ low coverage ({ratio*100:.0f}%) — significant audio may be "
            "silent/inaudible or dropped by VAD; review before trusting the "
            "transcript.")
    if rep >= repeat_warn:
        log(f"⚠ likely ASR hallucination loop: {rep*100:.0f}% of segments are the "
            f"same phrase (\"{dominant[:40]}\"). Coverage is unreliable here — try "
            "a different --asr backend, --language, or --enhance.")
    if mean_lp is not None and mean_lp < logprob_warn:
        log(f"⚠ low mean confidence (avg_logprob {mean_lp:.2f}) — transcript may be "
            "largely hallucinated; review before trusting it.")
    return report


def build_transcript(segs: list[Segment], title: str,
                     cov: dict | None = None) -> tuple[str, dict]:
    lines = [f"# Transcript — {title}", ""]
    if cov:
        gap = (f", largest gap {cov['top_gaps'][0][0]:.0f}s at "
               f"{cov['top_gaps'][0][1]}" if cov.get("top_gaps") else "")
        lines.append(f"_Coverage: {cov['coverage']*100:.0f}% speech over "
                     f"{fmt_ts(cov['duration_s'])}, {cov['n_segments']} "
                     f"segments{gap}._")
        lines.append("")
    last_spk = object()
    for seg in segs:
        ts = fmt_ts(seg.start)
        if seg.speaker is not None:
            if seg.speaker != last_spk:
                lines.append(f"\n**{seg.speaker}**")
                last_spk = seg.speaker
            lines.append(f"- `[{ts}]` {seg.text}")
        else:
            lines.append(f"`[{ts}]` {seg.text}")
    md = "\n".join(lines) + "\n"
    payload = {"title": title, "coverage": cov or {},
               "segments": [asdict(s) for s in segs]}
    return md, payload
