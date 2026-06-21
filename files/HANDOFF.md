# Recall — Project Handoff & Build Spec

> **Purpose of this document.** A complete, self-contained handoff so Claude
> Cowork (or any agent/developer) can resume this project without re-reading the
> original chat. It captures the goal, every finalized decision and its
> rationale, the current implementation, what's tested vs. unverified, and the
> exact next actions.
>
> **Status:** Working first version built and partially unit-tested in a Linux
> sandbox. The MLX / pyannote / mlx-lm calls have **not** been run on real
> hardware yet — see [§9 Validation checklist](#9-validation-checklist-run-these-first-on-the-mac).
>
> **Target machine:** Apple Silicon **MacBook M3 Pro, 18 GB unified memory**.

---

## 1. Goal

A **CLI tool** that turns a recorded meeting (phone voice memo or laptop audio,
conversations in **Hinglish** — mixed Hindi + English) into:

1. A clean, **timestamped, speaker-labelled transcript** (offline, on-device).
2. **English meeting notes** (summary, decisions, action items).
3. **Persistent speaker identities** — tag a speaker once, auto-recognize them in
   future meetings, bootstrapped from meeting audio (no clean voice samples
   needed).
4. **Per-person collaboration profiles** ("personas") accumulated over meetings,
   plus optional **reports tailored to how a specific person likes to receive
   information**.

Everything runs locally except the final note/persona/report generation, which
uses the user's **Claude subscription** via `claude -p` (with a fully-local MLX
fallback).

### Environment / assets available
- Apple Silicon M3 Pro, 18 GB RAM.
- Claude **Pro + Max** subscription (so `claude -p` headless is free to use — no
  paid API needed).
- **Codex** and **Claude Code** available for editing/debugging on the machine.
- Offline-first requirement: must be able to run end-to-end without internet
  (using the local LLM fallback) after one-time model downloads.

---

## 2. Finalized decisions (the contract)

| # | Decision | Choice | Why |
|---|----------|--------|-----|
| D1 | **No local Hindi→English translation step** | Removed. Local pipeline produces only a faithful raw transcript; Claude/local-LLM does translation **and** notes in one shot. | A small local model translating code-switched Hindi injects errors Claude can't undo. Claude reads Devanagari + Hinglish natively. Fewer stages, higher quality. |
| D2 | **Audio denoising** | **OFF by default**; optional `--denoise` via Demucs vocal isolation (not spectral gating). | Research shows generic denoisers often **raise** Whisper WER (~20%) by distorting spectral cues. Demucs-style isolation helps only on genuinely noisy audio. Must be A/B tested per recording. |
| D3 | **Transcript script format** | **Native Devanagari + English** (Whisper's natural output). Optional `--romanize` (ITRANS) for human readability. | Forcing romanization adds a lossy transliteration pass. Claude and Qwen both read Devanagari fine. |
| D4 | **Diarization (speaker labels)** | **ON by default**; one-time Hugging Face token + model-accept; auto-skips gracefully if no token. | Speaker attribution massively improves meeting notes (action items by person). One-time setup, then fully local. |
| D5 | **Notes generation** | `claude -p` **primary**, automatic **local MLX fallback** (`Qwen2.5-7B-Instruct-4bit`). Engine = `auto`. | `claude -p` runs on the Max subscription (no API billing, same weekly limit pool), reads stdin, fully scriptable. Local fallback keeps it offline-capable. |
| D6 | **Progress display** | Chunked ASR (default 240 s chunks) with a real `tqdm` % bar tied to audio duration; `--chunk-seconds 0` = single max-accuracy pass (spinner only). | mlx-whisper has no progress callback; chunking is the only honest way to show real % and also bounds memory. |
| D7 | **Live resource readout** | Per-stage spinner + RAM / system-mem% / CPU% / **GPU mem** (MLX Metal active allocation), via `psutil` + `mlx.core`. No `sudo`. | User asked to see resource usage. GPU **utilization %** needs `sudo powermetrics`; documented `macmon`/`asitop` as the alternative. |
| D8 | **Cross-session speaker identity** | Persistent **voiceprints** (pyannote embeddings) in a local store; cosine match; auto-assign ≥ `--id-high` (0.70), ask in [0.45, 0.70), treat < 0.45 as new. Interactive naming with `--no-enroll` for unattended runs. | Bootstraps identity from meetings; improves as samples accumulate; never silently guesses on weak matches. |
| D9 | **Personas** | **ON by default** when identified people exist. Built from **raw** utterances (captured pre-romanization). Profile separates evidence-backed *observed patterns* from a hedged *tentative read* of tone. `--no-personas` to disable. | Inferring fixed traits from noisy ASR is unreliable; framed as a collaboration aid with provenance, not a character verdict. |
| D10 | **Tailored reports** | `--report-for NAME` (repeatable) writes a report whose **format** matches that person's profile while **facts** stay strictly from the transcript. | Directly serves the "prepare a report in the format the person understands" requirement. |
| D11 | **Persistence location** | `--data-dir` (default `./recall-data/`): `voiceprints.json` + `people/<slug>/{profile.md, utterances.jsonl}`. | Plain JSON/JSONL/Markdown — inspectable and hand-editable. |

---

## 3. Pipeline architecture

All stages run **sequentially** to respect the 18 GB unified-memory budget
(Whisper, then pyannote, then the local LLM only if Claude is unavailable —
never co-resident).

```
audio ─► (1) Ingest ─► (2) Denoise ─► (3) ASR ─► (4) Diarize+Identify ─►
         ffmpeg         Demucs        whisper     pyannote + voiceprints
         16k mono       (off)         (chunked)   match/enroll

      ─► (5) Personas ─► (6) Assemble ─► (7) Notes  ─► [optional reports]
         raw utterances    transcript      claude -p     --report-for
         → profile.md      .md + .json     → local
```

1. **Ingest** — `ffmpeg` decodes any input (m4a/mp3/wav/mp4) → 16 kHz mono PCM wav.
2. **Denoise** *(optional, off)* — Demucs `--two-stems vocals`, re-normalized to 16 kHz.
3. **ASR** — `mlx-whisper` (`whisper-large-v3-turbo`), primed with a Hinglish
   `initial_prompt`, `language="hi"` by default, chunked for the progress bar with
   per-chunk timestamp offsetting.
4. **Diarize + Identify** — pyannote `speaker-diarization-3.1` returns turns **and**
   per-speaker embeddings; embeddings matched against the voiceprint store;
   confident matches auto-applied, others confirmed interactively.
5. **Personas** — raw utterances per identified person appended to their log; LLM
   updates `profile.md`.
6. **Assemble** — optional romanization, then build `*.transcript.md` + `*.transcript.json`.
7. **Notes** — transcript → `claude -p` (or local) → `*.notes.md`; optional
   `*.report.<slug>.md` per `--report-for`.

---

## 4. Tech stack / models

| Role | Choice | Notes |
|------|--------|-------|
| Audio decode | `ffmpeg` (system, `brew install ffmpeg`) | |
| ASR | `mlx-whisper`, model `mlx-community/whisper-large-v3-turbo` | `whisper-large-v3` for higher accuracy (slower). Consider A/B vs **Qwen3-ASR** (newer, claimed to beat large-v3 on-device). |
| Diarization + embeddings | `pyannote.audio`, `pyannote/speaker-diarization-3.1` | Gated model, free; needs HF token + condition accept. |
| Notes / personas / reports | `claude -p` (Claude Code headless) → fallback `mlx-lm`, `mlx-community/Qwen2.5-7B-Instruct-4bit` | `claude -p` uses Max subscription, reads stdin. |
| Denoise *(optional)* | `demucs` | Off by default. |
| Romanize *(optional)* | `indic-transliteration` (Devanagari → ITRANS) | Off by default. |
| Progress / metrics | `tqdm`, `psutil`, `mlx.core` (Metal memory) | |
| Identity / persona logic | pure-Python (`identity.py`), no numpy dependency | |

**Memory budget (18 GB):** turbo whisper ~1.5–2 GB, pyannote a few hundred MB,
Qwen2.5-7B-4bit ~4.5 GB — all fine sequentially. Do **not** hold whisper + local
LLM resident simultaneously.

---

## 5. Repository layout

```
Recall/
├── recall.py            # main CLI orchestrator (~750 lines)
├── identity.py          # VoiceStore + PersonaStore + cosine/match (~176 lines, pure python)
├── requirements.txt     # core: mlx-whisper, mlx-lm, tqdm, psutil, pyannote.audio
├── README.md            # user-facing setup + usage
├── HANDOFF.md           # this document
└── prompts/
    ├── notes.md         # meeting-notes instruction (TL;DR, decisions, action items…)
    ├── persona.md       # living profile updater (observed patterns vs tentative read)
    └── report.md        # per-person tailored-report instruction
```

### Key functions in `recall.py`
- `ingest`, `denoise`, `wav_duration`, `_slice_wav` — audio prep + chunking.
- `transcribe` — chunked ASR with tqdm bar + timestamp stitching (`_segs_from`).
- `diarize` — returns `(turns, emb_map)`; `assign_speakers` — overlap-based labelling.
- `resolve_identities` / `apply_identities` / `raw_utterances_for` — identity layer.
- `romanize`, `build_transcript`.
- `_engine_claude`, `_engine_local`, `make_generator` — shared text engine
  (Claude→local) reused by notes, personas, reports.
- `Metrics`, `LiveStatus`, `stage` — resource readout + progress UX.
- `run` — orchestration; `build_parser` — CLI.

### Key classes in `identity.py`
- `VoiceStore(path)` — `match(emb, high, low) → (name, score, status)`,
  `enroll(name, emb)` (running-average centroid), `save()`, `names()`.
- `PersonaStore(root)` — `add_utterances`, `read_profile`, `write_profile`,
  `update_profile(name, meeting, utts, instructions, generate_fn)`.
- Helpers: `cosine`, `l2norm`, `is_finite_vec`, `slugify`.

---

## 6. Data model

```
recall-data/
├── voiceprints.json
│     {
│       "version": 1,
│       "people": {
│         "Priya": { "centroid": [...256 floats...], "n_samples": 4, "dim": 256, "updated": <ts> },
│         "Rahul": { ... }
│       }
│     }
└── people/
    └── priya/
        ├── profile.md          # living collaboration profile (LLM-maintained, hand-editable)
        └── utterances.jsonl     # {date, start, end, text} per raw utterance, append-only
```

- Voiceprints are keyed by **real name**; persona folders by **slug** (note slug
  edge case in §8).
- Centroid is L2-normalized running average; each confirmed match calls
  `enroll()` to reinforce.

---

## 7. CLI reference

```
python recall.py AUDIO [options]
```

| Flag | Default | Purpose |
|------|---------|---------|
| `AUDIO` | — | input audio/video file |
| `-o, --output-dir` | `./recall-out` | where transcript/notes/reports go |
| `--model` | `mlx-community/whisper-large-v3-turbo` | ASR model |
| `--language` | `hi` | ASR language hint; `auto` to detect |
| `--chunk-seconds` | `240` | ASR chunk size for progress; `0` = single pass |
| `--denoise` | off | Demucs vocal isolation first |
| `--diarize` / `--no-diarize` | on | speaker labels |
| `--hf-token` | env `HF_TOKEN` | pyannote auth |
| `--romanize` | off | Devanagari → Roman (ITRANS) |
| `--notes-engine` | `auto` | `auto` \| `claude` \| `local` \| `none` |
| `--local-model` | `mlx-community/Qwen2.5-7B-Instruct-4bit` | offline notes model |
| `--notes-prompt` | `prompts/notes.md` | notes instruction file |
| `--no-progress` | off | disable bars/metrics (for scripts/cron) |
| `--data-dir` | `./recall-data` | voiceprint + persona store |
| `--no-enroll` | off | unattended: auto-assign only, no naming prompts |
| `--id-high` | `0.70` | cosine ≥ → auto-assign known speaker |
| `--id-low` | `0.45` | cosine in [low, high) → ask user |
| `--no-personas` | off (personas on) | skip profile building |
| `--persona-prompt` | `prompts/persona.md` | persona instruction file |
| `--report-for NAME` | — | tailored report for NAME (repeatable) |
| `--report-prompt` | `prompts/report.md` | report instruction file |

### Common invocations
```bash
# typical: labels on, Claude notes + local fallback
export HF_TOKEN=hf_xxx
python recall.py ~/VoiceMemos/standup.m4a

# later run: auto-recognize people + tailor two reports
python recall.py standup.m4a --report-for "Priya" --report-for "Rahul"

# fully offline notes
python recall.py call.m4a --notes-engine local

# unattended (no prompts), no personas
python recall.py call.m4a --no-enroll --no-personas
```

---

## 8. Design rationale & caveats (important context)

- **Code-switching is inherently hard.** Hinglish causes a 30–50% relative WER
  increase and unreliable per-word language ID in Whisper. Expect ASR errors on
  rapid switches — this is mitigated downstream (the notes prompt tells Claude to
  repair obvious garbles and mark the rest `[unclear]`), not eliminated.
- **Why no VAD trimming.** Silence-trimming was intentionally deferred: cutting
  audio breaks the timestamp alignment that diarization depends on. Could be
  re-added later as a non-destructive segment filter.
- **Denoising can hurt — keep it off by default.** See D2.
- **Identity accuracy ramps up.** Early matches on noisy phone/laptop audio are
  shaky; thresholds (0.70/0.45) need calibration once a few real people are
  enrolled. The system asks rather than guesses in the ambiguous band.
- **Personas are patterns, not verdicts.** The prompt enforces evidence-backed
  observations + a clearly hedged tone read. Profiles are editable Markdown.
- **Privacy.** Voiceprints + per-person logs are biometric/personal data about
  colleagues. Keep local; note that voiceprints are regulated in some
  jurisdictions (e.g. BIPA/GDPR special-category).
- **Slug edge case (minor).** `slugify()` strips Devanagari combining marks
  (matras), so a fully-Devanagari name yields an imperfect folder slug, and two
  names could in theory collide. Real names are preserved in `voiceprints.json`
  and inside `profile.md`. Fix later by storing a name↔slug map if needed.

---

## 9. Validation checklist (run these FIRST on the Mac)

These calls could not be executed in the build sandbox (no network, not Apple
Silicon). Verify each on first real run; each is a small, localized fix if the
library API has drifted:

- [ ] **`mlx_whisper.transcribe` kwargs** — confirm `path_or_hf_repo`,
  `initial_prompt`, `language`, `condition_on_previous_text`, `verbose`,
  `word_timestamps` are accepted and that `result["segments"]` has `start/end/text`.
- [ ] **`pyannote` `return_embeddings=True`** — confirm
  `annotation, embeddings = pipe(wav, return_embeddings=True)` returns embeddings
  aligned to `annotation.labels()` order, shape `(n_speakers, dim)`. This is the
  bridge feeding the **entire identity layer**; if shape/order differs, fix the
  `emb_map` build in `diarize()`. (Some speakers may yield NaN rows — already
  guarded by `is_finite_vec`.)
- [ ] **`mlx_lm.load` / `generate`** — confirm
  `generate(model, tokenizer, prompt=..., max_tokens=..., verbose=False)` matches
  the installed version (the sampler/args API has changed across releases).
- [ ] **MLX Metal memory API** — `Metrics._mlx_active_gb` tries
  `mx.metal.get_active_memory()` then `mx.get_active_memory()`; confirm one works.
- [ ] **pyannote on MPS** — `pipe.to(torch.device("mps"))` is wrapped in
  try/except; confirm it helps or harmlessly no-ops.
- [ ] **`claude -p`** — confirm logged in and that
  `cat transcript.md | claude -p "$(cat prompts/notes.md)"` returns notes.
- [ ] **Threshold calibration** — after enrolling ~3–5 real people, tune
  `--id-high`/`--id-low` (too high → repeated asking; too low → conflated voices).

### Already unit-tested in sandbox (passing)
`wav_duration`, `_slice_wav`, chunk **timestamp-offset stitching**, transcript
assembly + speaker grouping, `Metrics` graceful fallback, and all of
`identity.py`: `cosine`/`match` thresholds, `enroll` centroid averaging,
save/reload round-trip, `is_finite_vec` NaN guard, `slugify`, and `PersonaStore`
add/read/write/update.

---

## 10. One-time setup

```bash
# system
brew install ffmpeg
npm install -g @anthropic-ai/claude-code   # then run `claude` once to log in

# python
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install demucs                 # optional (--denoise)
pip install indic-transliteration  # optional (--romanize)

# diarization access (one-time, then local forever)
#  1. free account at huggingface.co
#  2. accept conditions on BOTH:
#       https://huggingface.co/pyannote/segmentation-3.0
#       https://huggingface.co/pyannote/speaker-diarization-3.1
#  3. token at https://huggingface.co/settings/tokens
export HF_TOKEN=hf_xxx
```

First run downloads model weights into the local HF cache; subsequent runs are
offline.

---

## 11. Roadmap / open items (not yet built)

Priority order for resuming:

1. **Validate §9** on a real recording (blocking everything else).
2. **`recall people` management command** — list / show / rename / merge / delete
   stored identities and view profiles. (First thing needed once the store has
   real people; merging matters because the same person may get enrolled twice
   before recognition stabilizes.)
3. **`setup.sh`** — automate venv + pip + checks that `ffmpeg`, `claude` login,
   and `HF_TOKEN` are present before first run.
4. **`--batch` mode** — process a folder of recordings (shared store), with a
   summary index.
5. **Threshold auto-calibration** — suggest `--id-high/--id-low` from observed
   match-score distributions.
6. **ASR A/B harness** — compare `whisper-large-v3-turbo` vs `large-v3` vs
   **Qwen3-ASR** on real Hinglish; wire the winner behind `--model`.
7. **Optional non-destructive VAD** — drop long silences without breaking
   diarization timestamps.
8. **Cross-meeting rollups** — "what did Priya commit to across the last month"
   from the per-person utterance logs.

---

## 12. How to resume in Claude Cowork

1. Open the `Recall/` folder in Cowork.
2. Read this `HANDOFF.md`, then `README.md`, then skim `recall.py` + `identity.py`.
3. Run the **§9 validation checklist** against one real meeting recording; patch
   any drifted library call (paste tracebacks to Claude Code for one-line fixes).
4. Calibrate identity thresholds after enrolling a few real speakers.
5. Then build **`recall people`** (#2) and **`setup.sh`** (#3) from §11.

**Guardrails to preserve when extending:**
- Keep the Claude-primary / local-fallback engine split (D5) and never require
  the paid API.
- Keep personas as evidence + hedged-read (D9); don't let profiles become fixed
  trait scores.
- Keep all speaker/persona data local (§8 privacy).
- Keep stages sequential to respect 18 GB (§3).
- Default denoise OFF and script native, not romanized (D2, D3).
