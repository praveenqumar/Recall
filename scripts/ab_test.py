#!/usr/bin/env python3
"""
ab_test.py — settle the three contested pipeline choices on YOUR audio.

Sweeps the contested axes from docs/reconciliation-and-merged-design.md:

    enhancer  : none | ffmpeg | deepfilternet        (C2)
    asr       : faster | mlx                          (C1)
    language  : en | hi                               (C3)

For every combination it transcribes the SAME clip, records objective metrics
(speed, segment count, avg-logprob, speech-coverage ratio, largest silent gaps),
and — with --judge — asks Claude to rank transcript fidelity AND the notes each
transcript produces. Writes per-variant transcripts + a comparison report.

Run this on the Apple Silicon Mac (mlx-whisper needs Metal). A backend or
enhancer that is not installed is marked "unavailable" and skipped, not fatal.

Examples
--------
    # fast first pass: first 6 minutes, all axes, with Claude judging
    python scripts/ab_test.py example_1on1.m4a --judge

    # only the language question, faster-whisper, ffmpeg enhancer, full file
    python scripts/ab_test.py meeting.m4a --enhancers ffmpeg --backends faster \\
        --languages en hi --minutes 0

    # just compare enhancers on the backend you already have working
    python scripts/ab_test.py meeting.m4a --backends faster --languages en
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import time
import wave
from pathlib import Path

# ffmpeg filter strings reused from scripts/transcribe_audio.py
FFMPEG_DSP = "highpass=f=80,lowpass=f=8000,afftdn=nf=-25,loudnorm=I=-18:TP=-1.5:LRA=11"
FFMPEG_POST_AI = "highpass=f=80,lowpass=f=8000,loudnorm=I=-18:TP=-1.5:LRA=11"


def log(msg: str) -> None:
    print(f"[ab] {msg}", file=sys.stderr, flush=True)


def have(cmd: str) -> bool:
    return shutil.which(cmd) is not None


def run_ffmpeg(src: Path, dst: Path, *, sr: int, af: str | None = None,
               extra_in: list[str] | None = None) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error"]
    cmd += extra_in or []
    cmd += ["-i", str(src)]
    if af:
        cmd += ["-af", af]
    cmd += ["-ar", str(sr), "-ac", "1", str(dst)]
    subprocess.run(cmd, check=True)


def wav_duration(wav: Path) -> float:
    with wave.open(str(wav), "rb") as w:
        return w.getnframes() / float(w.getframerate())


# --------------------------------------------------------------------------- #
# stage: clip + base 16k mono wav (one per run, shared across all variants)
# --------------------------------------------------------------------------- #
def make_base_wav(audio_in: Path, minutes: float, work: Path) -> Path:
    """16 kHz mono WAV, optionally clipped to the first `minutes` (0 = full)."""
    base = work / "base_16k_mono.wav"
    extra = ["-t", str(minutes * 60)] if minutes and minutes > 0 else None
    run_ffmpeg(audio_in, base, sr=16000, extra_in=extra)
    return base


# --------------------------------------------------------------------------- #
# enhancers (each returns a 16 kHz mono wav). Cached per enhancer name.
# --------------------------------------------------------------------------- #
def enhance(name: str, base_wav: Path, audio_in: Path, minutes: float,
            work: Path, df_cmd: str) -> Path:
    out = work / f"enh_{name}.wav"
    if out.exists():
        return out
    if name == "none":
        shutil.copy(base_wav, out)
    elif name == "ffmpeg":
        run_ffmpeg(base_wav, out, sr=16000, af=FFMPEG_DSP)
    elif name == "deepfilternet":
        if not have(df_cmd):
            raise RuntimeError(f"{df_cmd} not found (pip install deepfilternet)")
        df_in = work / "df_in_48k.wav"
        extra = ["-t", str(minutes * 60)] if minutes and minutes > 0 else None
        run_ffmpeg(audio_in, df_in, sr=48000,
                   af="highpass=f=60,loudnorm=I=-18:TP=-1.5:LRA=11", extra_in=extra)
        df_out = work / "df_out"
        df_out.mkdir(exist_ok=True)
        subprocess.run([df_cmd, "--output-dir", str(df_out), str(df_in)],
                       check=True, cwd=work)
        cands = sorted(df_out.rglob("*.wav"), key=lambda p: p.stat().st_mtime,
                       reverse=True)
        if not cands:
            raise RuntimeError("DeepFilterNet produced no output")
        run_ffmpeg(cands[0], out, sr=16000, af=FFMPEG_POST_AI)
    else:
        raise ValueError(f"unknown enhancer: {name}")
    return out


# --------------------------------------------------------------------------- #
# ASR backends. Each returns (segments, info) where segments are dicts with
# start/end/text and optional avg_logprob.
# --------------------------------------------------------------------------- #
def asr_faster(wav: Path, language: str, model: str):
    from faster_whisper import WhisperModel
    m = WhisperModel(model, device="cpu", compute_type="int8")
    seg_iter, info = m.transcribe(
        str(wav), language=language, task="transcribe", beam_size=5,
        condition_on_previous_text=False, vad_filter=True,
        vad_parameters={"threshold": 0.2, "min_speech_duration_ms": 100,
                        "min_silence_duration_ms": 500, "speech_pad_ms": 400},
        hallucination_silence_threshold=2,
    )
    segs = []
    for s in seg_iter:
        t = (s.text or "").strip()
        if t:
            segs.append({"start": float(s.start), "end": float(s.end),
                         "text": t, "avg_logprob": float(s.avg_logprob)})
    return segs, {"detected_language": info.language}


# faster-whisper model id differs from the mlx repo id
MLX_MODEL = "mlx-community/whisper-large-v3-turbo"
HINGLISH_PRIMER = ("The following is a casual business meeting spoken in Hinglish, "
                   "a natural mix of Hindi and English. Speakers switch between "
                   "Hindi and English mid-sentence.")


def asr_mlx(wav: Path, language: str, model: str):
    import mlx_whisper
    res = mlx_whisper.transcribe(
        str(wav), path_or_hf_repo=model, language=language,
        initial_prompt=HINGLISH_PRIMER, condition_on_previous_text=False,
        word_timestamps=False, verbose=False)
    segs = []
    for s in res.get("segments", []):
        t = (s.get("text") or "").strip()
        if t:
            d = {"start": float(s["start"]), "end": float(s["end"]), "text": t}
            if "avg_logprob" in s:
                d["avg_logprob"] = float(s["avg_logprob"])
            segs.append(d)
    return segs, {"detected_language": res.get("language", language)}


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #
def metrics_for(segs: list[dict], duration: float, elapsed: float) -> dict:
    speech = sum(s["end"] - s["start"] for s in segs)
    lps = [s["avg_logprob"] for s in segs if "avg_logprob" in s]
    gaps = sorted(
        ((round(segs[i + 1]["start"] - segs[i]["end"], 1), round(segs[i]["end"], 1))
         for i in range(len(segs) - 1)), reverse=True)[:3]
    return {
        "n_segments": len(segs),
        "wall_seconds": round(elapsed, 1),
        "rtf": round(elapsed / duration, 2) if duration else None,  # <1 is faster than realtime
        "coverage": round(speech / duration, 2) if duration else None,
        "avg_logprob_mean": round(sum(lps) / len(lps), 2) if lps else None,
        "avg_logprob_min": round(min(lps), 2) if lps else None,
        "top_gaps_sec": gaps,
        "chars": sum(len(s["text"]) for s in segs),
    }


def transcript_text(segs: list[dict]) -> str:
    return "\n".join(s["text"] for s in segs)


# --------------------------------------------------------------------------- #
# optional Claude judging
# --------------------------------------------------------------------------- #
JUDGE_PROMPT = """You are evaluating competing transcriptions of the SAME Hinglish
(Hindi+English) business meeting audio. You do not have the audio, so judge on
internal evidence: coherence, plausible code-switching, absence of obvious
hallucination/looping/repetition, and how usable each is as a basis for English
meeting notes.

For EACH labelled variant return a score. Then state which single variant you would
build the production pipeline on and why. Respond ONLY with JSON of the form:
{"scores":[{"variant":"<label>","fidelity":1-10,"notes_usability":1-10,
"issues":"<short>"}],"winner":"<label>","reasoning":"<2-3 sentences>"}"""


def judge_with_claude(variants: list[dict]) -> dict | None:
    if not have("claude"):
        log("--judge requested but `claude` CLI not found; skipping judging")
        return None
    blob = []
    for v in variants:
        if v.get("error"):
            continue
        excerpt = transcript_text(v["segments"])[:4000]
        blob.append(f"=== VARIANT: {v['label']} ===\n{excerpt}\n")
    if not blob:
        return None
    try:
        r = subprocess.run(["claude", "-p", JUDGE_PROMPT],
                           input="\n".join(blob), capture_output=True,
                           text=True, timeout=600)
    except subprocess.TimeoutExpired:
        log("claude judging timed out")
        return None
    if r.returncode != 0 or not r.stdout.strip():
        log(f"claude judging failed (exit {r.returncode})")
        return None
    out = r.stdout.strip()
    try:
        start, end = out.find("{"), out.rfind("}")
        return json.loads(out[start:end + 1])
    except Exception:
        log("could not parse Claude's JSON; saving raw output")
        return {"raw": out}


# --------------------------------------------------------------------------- #
# report
# --------------------------------------------------------------------------- #
def write_report(audio_in: Path, minutes: float, duration: float,
                 variants: list[dict], verdict: dict | None, outdir: Path) -> Path:
    rp = outdir / "ab_report.md"
    clip = f"first {minutes:g} min" if minutes else "full file"
    lines = [
        f"# A/B report — {audio_in.name}",
        "",
        f"Clip: **{clip}** ({duration:.0f}s analysed). "
        f"Generated {time.strftime('%Y-%m-%d %H:%M')}.",
        "",
        "RTF = wall ÷ audio (lower is faster; <1 beats realtime). "
        "Coverage = transcribed speech ÷ duration (low ⇒ audio dropped — see §4 "
        "of the design doc). avg_logprob nearer 0 is more confident.",
        "",
        "| variant | status | RTF | coverage | segs | logprob μ | min | chars | top gap (s) |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for v in variants:
        if v.get("error"):
            lines.append(f"| `{v['label']}` | ❌ {v['error']} | | | | | | | |")
            continue
        m = v["metrics"]
        top = m["top_gaps_sec"][0][0] if m["top_gaps_sec"] else 0
        lines.append(
            f"| `{v['label']}` | ✓ | {m['rtf']} | {m['coverage']} | "
            f"{m['n_segments']} | {m['avg_logprob_mean']} | {m['avg_logprob_min']} | "
            f"{m['chars']} | {top} |")
    lines += ["", "## Claude verdict", ""]
    if verdict and "scores" in verdict:
        for s in verdict["scores"]:
            lines.append(f"- `{s.get('variant')}` — fidelity {s.get('fidelity')}/10, "
                         f"notes {s.get('notes_usability')}/10 — {s.get('issues','')}")
        lines += ["", f"**Winner: `{verdict.get('winner')}`** — "
                  f"{verdict.get('reasoning','')}"]
    elif verdict and "raw" in verdict:
        lines += ["```", verdict["raw"], "```"]
    else:
        lines.append("_(run with `--judge` and the `claude` CLI for an automatic "
                     "ranking; otherwise compare the metrics above and skim the "
                     "per-variant transcripts in this folder)_")
    lines += [
        "", "## How to decide", "",
        "Per the design doc decision rule: pick the variant with the best "
        "Claude-judged notes usability; break ties by coverage, then RTF. If "
        "enhancers split by file (likely), keep the flag and note which file types "
        "favour which. Record the chosen per-axis defaults back into the merged "
        "`recall.py` config.", "",
        "## Per-variant transcripts", "",
    ]
    for v in variants:
        if not v.get("error"):
            lines.append(f"- `{v['label']}.txt`")
    rp.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return rp


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description="A/B the contested pipeline axes.")
    ap.add_argument("audio", type=Path)
    ap.add_argument("--minutes", type=float, default=6.0,
                    help="analyse only the first N minutes (0 = full file; "
                         "default 6 for a fast first pass)")
    ap.add_argument("--enhancers", nargs="+",
                    default=["none", "ffmpeg", "deepfilternet"],
                    choices=["none", "ffmpeg", "deepfilternet"])
    ap.add_argument("--backends", nargs="+", default=["faster", "mlx"],
                    choices=["faster", "mlx"])
    ap.add_argument("--languages", nargs="+", default=["en", "hi"],
                    choices=["en", "hi"])
    ap.add_argument("--faster-model", default="large-v3-turbo")
    ap.add_argument("--mlx-model", default=MLX_MODEL)
    ap.add_argument("--deepfilter-command", default="deepFilter")
    ap.add_argument("--judge", action="store_true",
                    help="have the `claude` CLI rank the variants")
    ap.add_argument("-o", "--outdir", type=Path, default=Path("./ab-out"))
    args = ap.parse_args()

    audio_in = args.audio.expanduser().resolve()
    if not audio_in.exists():
        sys.exit(f"audio not found: {audio_in}")
    if not have("ffmpeg"):
        sys.exit("ffmpeg required (brew install ffmpeg)")
    args.outdir.mkdir(parents=True, exist_ok=True)

    backends = {"faster": asr_faster, "mlx": asr_mlx}
    models = {"faster": args.faster_model, "mlx": args.mlx_model}

    variants: list[dict] = []
    with tempfile.TemporaryDirectory(prefix="ab_") as tmp:
        work = Path(tmp)
        base = make_base_wav(audio_in, args.minutes, work)
        duration = wav_duration(base)
        log(f"analysing {duration:.0f}s; matrix = "
            f"{len(args.enhancers)}×{len(args.backends)}×{len(args.languages)} "
            f"= {len(args.enhancers)*len(args.backends)*len(args.languages)} runs")

        for enh in args.enhancers:
            try:
                wav = enhance(enh, base, audio_in, args.minutes, work,
                              args.deepfilter_command)
            except Exception as e:  # enhancer unavailable/failed → skip its row
                for be in args.backends:
                    for lang in args.languages:
                        variants.append({"label": f"{enh}+{be}+{lang}",
                                         "error": f"enhancer: {e}"})
                log(f"enhancer {enh} unavailable: {e}")
                continue
            for be in args.backends:
                for lang in args.languages:
                    label = f"{enh}+{be}+{lang}"
                    log(f"running {label} …")
                    t0 = time.time()
                    try:
                        segs, info = backends[be](wav, lang, models[be])
                    except Exception as e:
                        variants.append({"label": label, "error": f"{be}: {e}"})
                        log(f"  {label} failed: {e}")
                        continue
                    el = time.time() - t0
                    m = metrics_for(segs, duration, el)
                    (args.outdir / f"{label}.txt").write_text(
                        transcript_text(segs) + "\n", encoding="utf-8")
                    variants.append({"label": label, "segments": segs,
                                     "metrics": m, "info": info})
                    log(f"  {label}: RTF {m['rtf']}  coverage {m['coverage']}  "
                        f"segs {m['n_segments']}")

    verdict = judge_with_claude(variants) if args.judge else None
    (args.outdir / "ab_results.json").write_text(
        json.dumps({"audio": str(audio_in), "minutes": args.minutes,
                    "duration": duration,
                    "variants": [{k: v for k, v in d.items() if k != "segments"}
                                 for d in variants],
                    "verdict": verdict}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    rp = write_report(audio_in, args.minutes, duration, variants, verdict,
                      args.outdir)
    print(f"\n✅ A/B done. Report: {rp}")
    print(f"   Transcripts + results.json in: {args.outdir}")


if __name__ == "__main__":
    main()
