# AGENTS.md — working brief for Recall

Read this first. It's the short brief for getting productive fast. For depth
(pipeline diagram, ML models, full CLI, API, design decisions) see
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md). The code's per-module docstrings
are the source of truth.

## Current state & open work (read before resuming)
- **Branch:** all work is on `init-recall` (origin `praveenqumar/Recall`), **not
  merged to `main`** — `main` holds only the seed README. PR/merge when ready.
- **Status:** src-layout, pip-installable (src kept *because* it's installable),
  tests green. Token-usage display shipped (issue #2, closed).
- **Done — [issue #3](https://github.com/praveenqumar/Recall/issues/3):** dedup by
  audio sha256 + SQLite index (`store.py`). Default output + db now under
  `~/.recall/` (db = `<data-dir>/../recall.db`). Re-run on same audio prints existing
  paths; `--force` regenerates; `--title` sets the dated filename. `<title>` resolved
  as: `--title` if given, else filename slug.
- **ASR reality (this decides C1 per file):** `faster` (CPU) is most reliable;
  `mlx` (GPU) is faster but can hallucinate junk in silent gaps (e.g. `唐唐` / `drift`
  loops) even with `--language en`. **The deciding metric is notes quality, not
  transcript prettiness** — mlx notes can still win because Claude repairs the gaps.
  `--enhance ffmpeg` cuts that gap-hallucination.
- **Known L6 gap:** the repetition warning catches one dominant repeated phrase but
  NOT multi-phrase / CJK-burst hallucination (the `唐唐` case slipped through at
  "101% coverage"). Candidate fix if it bites.
- **Env (this machine):** `HF_TOKEN` in `~/.zshrc`; the 3 gated pyannote models
  already accepted; HF weights cached in `~/.cache/huggingface` (~8 GB, shared).
  Recall has its own `.venv-transcribe`; run tests from the repo root.

## What Recall is
An offline, on-device CLI that turns Hinglish (Hindi+English) meeting audio into a
speaker-labelled transcript + English notes, **remembers speakers across meetings**
(voiceprints), and builds per-person collaboration profiles. Everything runs
locally except notes/personas/reports, which use `claude -p` with a local MLX
fallback. Apple-Silicon target; `faster-whisper` path also runs portably.

## Layout (src layout, package = `recall`)
```
src/recall/        one capability per module (vertical slices):
  common      Segment + ffmpeg/wav primitives (stdlib only) — the data unit
  metrics     resource readout + progress UX
  enhance     C2 pluggable enhancers (none/ffmpeg/deepfilternet/demucs)
  asr         C1 pluggable ASR backends (faster/mlx)
  diarize     pyannote 4.x → turns + per-speaker embeddings
  identity    persistent voiceprints, cosine match, cross-session resolution
  personas    per-person living profiles
  generate    shared Claude→local-MLX text engine
  transcript  assemble md/json + coverage & hallucination diagnostics
  notes       meeting notes + tailored reports
  store       SQLite dedup index (audio sha256 → existing files) + dated naming
  pipeline    orchestration — start here to read control flow (run())
  cli         argparse / entry point
  prompts/    notes.md, persona.md, report.md (editable, not code)
scripts/     ab_test.py (A/B harness), seed_voiceprints.py, transcribe_audio.py (legacy)
tests/       run.py (no pytest needed) + test_identity.py + test_pipeline.py
docs/        ARCHITECTURE.md (the one deep doc)
```

## Build / test / run
```bash
python3 -m venv .venv-transcribe && source .venv-transcribe/bin/activate
pip install -e '.[all]'                 # extras: mlx|faster|diarize|enhance|romanize|all
python tests/run.py                      # mocks ASR+diarize, only ffmpeg required
python -m recall meeting.m4a --asr faster --language en   # proven config
```
Diarization needs `HF_TOKEN` + one-time accept of three gated pyannote models
(`segmentation-3.0`, `speaker-diarization-3.1`, `speaker-diarization-community-1`).

### Common commands (copy-paste)
```bash
# GPU (mlx) + ffmpeg cleanup, no prompts, persistent identity  ← good default
recall meeting.m4a --asr mlx --language en --enhance ffmpeg --no-enroll --data-dir ~/.recall/data
# GPU (mlx), raw audio
recall meeting.m4a --asr mlx --language en --no-enroll --data-dir ~/.recall/data
# CPU (faster), portable, strongest anti-hallucination
recall meeting.m4a --asr faster --language en --no-enroll --data-dir ~/.recall/data
# name speakers interactively (omit --no-enroll) → voiceprints persist + auto-match
recall meeting.m4a --asr mlx --language en --data-dir ~/.recall/data
# tailored report / transcript only / offline notes
recall meeting.m4a --asr mlx --language en --report-for "Priya" --data-dir ~/.recall/data
recall meeting.m4a --asr faster --language en --notes-engine none
recall meeting.m4a --asr mlx    --language en --notes-engine local
```
`--asr mlx` GPU / `faster` CPU · `--enhance ffmpeg` cleans gap-hallucination ·
`--no-enroll` skips naming prompts · `--data-dir ~/.recall/data` persists identity.

## Updating the installed CLI
A globally-installed `recall` (via `uv tool` / `pipx`) is a frozen copy — it does
**not** pick up new commits automatically. After pushing changes, refresh it:
```bash
uv tool upgrade recall
# or pin to a branch/ref:
uv tool install --force "recall @ git+https://github.com/<owner>/<repo>.git@<branch>"
```
An **editable** install (`pip install -e .` into a venv) reflects edits/pulls live —
no reinstall needed. Use editable for development, `uv tool`/`pipx` for daily use.

## How to extend (the designed path)
- **New ASR backend / enhancer:** add a function to the `_REGISTRY` in `asr.py` /
  `enhance.py` and a `--asr` / `--enhance` choice in `cli.py`. Everything downstream
  consumes `list[Segment]`, so nothing else changes. An enhancer must return a
  16 kHz mono wav and **fall back to its input on failure** (never crash the run).
- **New text engine:** extend `generate.py` — keep Claude-primary / local-fallback.
- **New output:** writer in `notes.py` (or a new slice), called from `pipeline.py`.

## Conventions
- Degrade gracefully: a missing optional dep **logs and no-ops**, never raises.
  Every heavy import is lazy (inside the function that needs it).
- Logs go to **stderr** (`common.log`, `[recall] …`); stdout stays clean for piping.
- Name things for their domain purpose; minimal diffs; don't reformat unrelated code.
- Add/adjust a test for any behavior change (`tests/`).

## Guardrails — do not regress (full rationale: ARCHITECTURE §9)
- **L2** never require the paid API (Claude-primary / local-fallback).
- **L3** personas = evidence + hedge; biometric/persona data stays local.
- **L4** stages sequential; never co-resident Whisper + local LLM (18 GB).
- **L5** ASR / enhancer / language are config behind interfaces, not forks.
- **L6** every run emits coverage + hallucination diagnostics.

## Debugging
- Reproduce with `--no-progress` first.
- Backends fall back instead of crashing — **read the `[recall]` warnings**, not
  just the exit code.
- Most breakage is upstream **library API drift** (pyannote/mlx move fast): suspect
  a renamed kwarg or changed return shape; the fix is usually a one-liner.
- Coverage diagnostic is a correctness signal: low coverage ⇒ dropped audio; high
  repetition / very low logprob ⇒ ASR hallucination loop.

## Privacy / security
Voiceprints, personas, transcripts, notes, audio, and `meetings/` are git-ignored —
keep them out of commits. The transcript is fed to an LLM (`claude -p`); treat
generated text as influenced by recording content (prompt-injection surface).
