# Reconciliation & Merged Design

> **Purpose.** Resolve the split between the two designs that currently coexist in
> this repo and define **one** pipeline going forward. Decisions that both tracks
> agree on (or that only one track has) are **locked** here. The three decisions
> where the tracks genuinely contradict each other are **deferred to an A/B test**
> on real recordings (`scripts/ab_test.py`), because only evidence can settle them.
>
> This is a point-in-time design record. References below to `HANDOFF.md` and the
> early `files/*.py` prototypes are historical — those artifacts are no longer kept
> in the repo; the merged design they led to is the current `recall/` package.
>
> Written after studying: `HANDOFF.md`, the early prototype modules,
> `docs/transcription-pipeline-design.md`, `docs/requirements-solution-decisions.md`,
> `scripts/transcribe_audio.py`, the committed transcripts, and `git log`.

---

## 1. The two tracks, side by side

| Aspect | Track A — *built, run, committed* (`scripts/transcribe_audio.py`) | Track B — *spec + unrun code* (`files/recall.py`, `files/identity.py`) |
|---|---|---|
| ASR backend | `faster-whisper` (large-v3-turbo, CPU, int8) | `mlx-whisper` (large-v3-turbo, Metal/GPU) |
| Enhancement | DeepFilterNet (default) + ffmpeg DSP fallback; VAD on | Demucs vocal isolation, **off by default**; no VAD |
| Language | `en` — **tested**: `hi` gave worse output + Devanagari hallucination | `hi` + native Devanagari — Claude translates at the end |
| Diarization | none | pyannote 3.1, on by default |
| Identity (voiceprints) | none | persistent cross-session store, interactive enrollment |
| Personas | none | living per-person `profile.md` |
| Notes generation | none (separate manual summary) | `claude -p` primary, local MLX Qwen fallback |
| Tailored reports | none | `--report-for NAME` |
| Progress / metrics | nohup + log files | chunked ASR + tqdm bar + live RAM/CPU/GPU readout |
| Persistent store | none (manual `meetings/` folders) | `recall-data/` (voiceprints + people) |
| Validation status | **proven** — generated every transcript here | **unrun** — HANDOFF §9 is an explicit "verify these calls" list |

The tracks **only contradict each other on the first three rows**. Everything
below "Language" is a capability that exists in Track B and simply does not exist
in Track A — there is nothing to reconcile there, only to adopt.

---

## 2. Locked decisions (no A/B needed)

These are settled now so the build can proceed in parallel with the A/B test.

**L1 — Adopt Track B's superstructure wholesale.** Diarization, the
voiceprint/identity layer, personas, `claude -p`→local-MLX notes, tailored
reports, chunked progress with the live resource readout, and the `recall-data/`
store are all net-new and uncontested. Track A has none of them; the requirements
doc lists all of them as the intended roadmap. Keep `files/identity.py` essentially
as-is (it is already unit-tested in the sandbox per HANDOFF §9).

**L2 — Keep the Claude-primary / local-fallback split for all text generation**
(notes, personas, reports). Never require the paid API. This is a hard guardrail.

**L3 — Keep personas as evidence + hedged-read**, not fixed trait scores; keep all
speaker/persona data local. (HANDOFF D9 + privacy note.)

**L4 — Keep stages sequential** to respect the 18 GB unified-memory budget; never
hold Whisper and the local LLM resident at once.

**L5 — The pipeline is built backend-agnostic.** The ASR backend, the enhancer,
and the language hint become **configuration behind clean interfaces**, not
hard-coded choices. This is the key architectural move that lets the A/B winners
(§3) drop in as defaults without a rewrite — and lets us keep both backends
available if one wins on some files and the other on others (which the evidence
below suggests is likely).

**L6 — Add coverage/silence diagnostics to every run** (see §4 — the existing
recordings expose a serious silent-gap problem that neither track currently
surfaces).

---

## 3. Contested decisions — deferred to the A/B test

Each of these has real evidence pointing in *both* directions, so each is wired
into `scripts/ab_test.py` rather than decided by argument.

### C1 — ASR backend: `faster-whisper` vs `mlx-whisper`

- *For faster-whisper:* it is the only path that has actually produced output here;
  it runs everywhere; mature VAD + hallucination controls.
- *For mlx-whisper:* native Metal on the M3 Pro should be substantially faster and
  is the HANDOFF's whole premise — but **every mlx call is unverified** (HANDOFF
  §9). API drift is likely.
- **A/B measures:** wall-clock speed, segment coverage, avg-logprob distribution,
  and Claude-judged transcript fidelity on the *same* enhanced WAV.

### C2 — Enhancement: DeepFilterNet vs ffmpeg-DSP vs none vs Demucs

The repo already contains a natural experiment. The same example 1:1 call exists in
two variants:

- `example_1on1.deepfilter.transcript` (DeepFilterNet): correct
  opening segmentation, but **runaway empty `...` hallucination loops** and triple
  repetitions later (50 segments).
- `example_1on1.transcript` (ffmpeg): **collapses 19 minutes into a
  single hallucinated segment** at the start (31 segments).

The existing summary records the operator's verdict: *"DeepFilterNet AI
enhancement initially ran, but the resulting transcript was poor for this file… I
corrected course and regenerated with the ffmpeg speech-cleanup enhancer."* So the
right answer is **file-dependent** — which is exactly why L5 keeps the enhancer
selectable and the A/B compares them per recording rather than picking a global
winner.

### C3 — Language: `en` (Roman) vs `hi` (Devanagari)

This is the decision the user explicitly chose to settle by A/B.

- *For `en` (Track A, tested):* the committed Hinglish transcript is mostly
  coherent English and easy to summarize; the docs report `hi` produced worse
  output and Devanagari hallucinations in earlier tests.
- *For `hi`+Devanagari (Track B):* a faithful native-script transcript preserves
  Hindi that `en` flattens or mangles (e.g. the garbled `"Our teacher, the
  schedule, CPO phase, the women, digital oncology…"` line in the `en` transcript
  is the failure mode Track B claims to avoid), letting Claude translate once at
  the end.
- **A/B measures:** run both on the same WAV, then have Claude (a) rate transcript
  fidelity and (b) generate notes from each and rate the *notes* — because notes
  quality, not transcript aesthetics, is the real objective.

> **Decision rule.** For each contested axis, pick the variant with the better
> Claude-judged **notes** quality, breaking ties by coverage then speed. Record the
> winner per axis in the A/B report; it becomes the new default in the merged
> `recall.py` config. If results split by file type (likely for C2), keep the flag
> and document when to use which.

---

## 4. New finding that overrides both tracks: silent-gap coverage

Both example-1on1 transcripts independently jump from `00:04` to `19:14` — **~19 minutes
unaccounted for** — and the existing summary confirms the audio "appears to contain
a lot of silence/very-low-volume sections." Neither track surfaces this; a user
could easily trust a transcript that silently dropped a third of the meeting.

**Merged-design requirement (L6):** every run computes and prints a *coverage
ratio* (transcribed speech seconds ÷ audio duration) and the largest silent gaps,
and warns when coverage is low. The A/B harness reports this per variant so a
config that *appears* cleaner but actually drops more audio is caught. This also
informs whether a non-destructive VAD/segment-filter (HANDOFF roadmap #7) is worth
adding.

---

## 5. Target merged architecture

```
audio
  └► (1) ingest            ffmpeg → 16 kHz mono wav
  └► (2) enhance [pluggable] none | ffmpeg-dsp | deepfilternet | demucs   ← C2 config
  └► (3) ASR     [pluggable] faster-whisper | mlx-whisper                 ← C1 config
                            language = en | hi                            ← C3 config
                            chunked + tqdm bar + live RAM/CPU/GPU readout
  └► (4) diarize+identify   pyannote 3.1 → voiceprint match/enroll        (Track B)
  └► (5) personas           raw utterances → per-person profile.md        (Track B)
  └► (6) assemble           timestamped, speaker-labelled .md + .json
                            + coverage/gap diagnostics                    ← L6
  └► (7) notes              claude -p → local MLX fallback                 (Track B)
                            + optional --report-for NAME
```

Two abstractions make L5 concrete:

- `ASRBackend` — one method `transcribe(wav, language, chunk_s) -> list[Segment]`,
  with `FasterWhisperBackend` and `MLXWhisperBackend` implementations. `recall.py`
  picks one via `--asr {faster,mlx,auto}`.
- `Enhancer` — one method `process(in_wav) -> out_wav`, with `none`, `ffmpeg`,
  `deepfilternet`, `demucs` implementations, selected via `--enhance`.

Everything downstream (diarization, identity, personas, notes) consumes
`list[Segment]` and is therefore identical regardless of which backend/enhancer/
language won the A/B. The A/B test literally exercises these same two abstractions,
so its results map 1:1 onto the production defaults.

---

## 6. What still must be validated on the Mac (carried over from HANDOFF §9)

Unchanged and still blocking for the mlx path specifically: `mlx_whisper.transcribe`
kwargs, `pyannote` `return_embeddings=True` shape/order, `mlx_lm.load/generate`
signature, MLX Metal memory API, pyannote-on-MPS, and `claude -p` round-trip. The
A/B harness deliberately exercises the first of these as a side effect (if the mlx
backend throws, that axis is simply marked unavailable and the run continues).

---

## 7. Build order

1. **(this doc)** lock decisions, isolate contested axes. ✅
2. **A/B harness** (`scripts/ab_test.py`) — runs the contested matrix on a real
   recording, emits metrics + Claude-scored comparison. → user runs on the Mac.
3. **Read the A/B report**, set the per-axis defaults.
4. **Build merged `recall.py`** around the two abstractions (§5), reusing
   `files/identity.py` and the Track B prompts, wiring in coverage diagnostics (L6).
5. Validate the remaining §6 items end-to-end on one meeting; calibrate identity
   thresholds after enrolling a few real speakers.
6. Then the roadmap items (`recall people` management, `setup.sh`, `--batch`).

**Guardrails to preserve throughout:** L2 (never require paid API), L3 (personas =
evidence + hedge, data local), L4 (sequential stages / 18 GB), L5 (backends are
config, not forks).
