"""
recall.pipeline — orchestration. Wires the vertical slices together, sequentially,
to respect the ~18 GB unified-memory budget (Whisper, then pyannote, then the
local LLM only if needed — never co-resident).

    ingest -> enhance -> ASR -> diarize+identify -> personas
           -> assemble (+coverage) -> notes (+reports)
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from . import asr, diarize as diar, enhance as enh, identity, notes as notes_mod
from . import personas as personas_mod, transcript as tx
from .common import die, fmt_ts, ingest, log, wav_duration
from .generate import make_generator
from .metrics import Metrics, stage


def run(cfg) -> None:
    audio_in = Path(cfg.audio).expanduser().resolve()
    if not audio_in.exists():
        die(f"audio file not found: {audio_in}")
    out_dir = Path(cfg.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = audio_in.stem
    meeting_id = stem

    metrics = Metrics()
    progress = not cfg.no_progress
    generate = make_generator(cfg.notes_engine, cfg.local_model, metrics, progress)

    data_dir = Path(cfg.data_dir).expanduser().resolve()
    vstore = identity.VoiceStore(data_dir / "voiceprints.json")
    pstore = personas_mod.PersonaStore(data_dir / "people")

    n_stages = 7
    identified: list[str] = []

    with tempfile.TemporaryDirectory(prefix="recall_") as tmp:
        work = Path(tmp)

        stage(1, n_stages, "Ingest")
        wav = ingest(audio_in, work)
        duration = wav_duration(wav)
        log(f"duration: {fmt_ts(duration)}")

        stage(2, n_stages, f"Enhance ({cfg.enhance})")
        wav = enh.enhance(cfg.enhance, wav, work, metrics, progress,
                          cfg.deepfilter_command)

        stage(3, n_stages, f"ASR ({cfg.asr})")
        segs = asr.transcribe(cfg.asr, wav, cfg.language, cfg.chunk_seconds,
                              cfg.model, metrics, progress, work)
        if not segs:
            die("ASR produced no segments — is the audio silent or corrupt?")

        stage(4, n_stages,
              "Diarize + identify" + ("" if cfg.diarize else " (skipped)"))
        if cfg.diarize:
            turns, emb_map = diar.diarize(
                wav, cfg.hf_token or os.environ.get("HF_TOKEN"), metrics, progress)
            diar.assign_speakers(segs, turns)
            label_to_name = identity.resolve_identities(
                segs, emb_map, vstore, cfg.id_high, cfg.id_low,
                enroll=not cfg.no_enroll)
            identity.apply_identities(segs, label_to_name)
            vstore.save()
            identified = sorted(set(label_to_name.values()))

        # personas must use RAW utterances, before any romanization
        do_personas = cfg.personas and identified
        stage(5, n_stages, "Personas" + ("" if do_personas else " (skipped)"))
        if do_personas:
            persona_prompt = Path(cfg.persona_prompt).read_text()
            personas_mod.build_personas(segs, identified, meeting_id, pstore,
                                        persona_prompt, generate)

        stage(6, n_stages, "Assemble")
        cov = tx.coverage(segs, duration)
        if cfg.romanize:
            tx.romanize(segs)
        transcript_md, transcript_json = tx.build_transcript(segs, stem, cov)

    md_path = out_dir / f"{stem}.transcript.md"
    json_path = out_dir / f"{stem}.transcript.json"
    md_path.write_text(transcript_md)
    json_path.write_text(json.dumps(transcript_json, ensure_ascii=False, indent=2))
    log(f"wrote {md_path.name} and {json_path.name}")

    stage(7, n_stages, "Notes" + (" (skipped)" if cfg.notes_engine == "none" else ""))
    outputs = [("Transcript", md_path)]
    notes_path = None
    if cfg.notes_engine != "none":
        notes_path = notes_mod.write_notes(transcript_md, Path(cfg.notes_prompt).read_text(),
                                           out_dir, stem, generate)
        if notes_path:
            outputs.append(("Notes", notes_path))

    if cfg.report_for:
        outputs += notes_mod.write_reports(
            cfg.report_for, transcript_md, Path(cfg.report_prompt).read_text(),
            pstore, out_dir, stem, generate)

    print("\n✅ Done.")
    for label, path in outputs:
        print(f"  {label}: {path}")
    if identified:
        print(f"  Known speakers this meeting: {', '.join(identified)}")
    if not notes_path and cfg.notes_engine != "none":
        log("notes engine produced nothing; pipe manually:  "
            f"cat '{md_path}' | claude -p \"$(cat '{cfg.notes_prompt}')\"")
