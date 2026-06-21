# Recall

A professional-but-personal **meeting synthesizer**: offline, on-device
transcription for **Hinglish** (Hindi + English) meeting audio, with AI-generated
English notes, cross-session speaker identity, and per-person collaboration
profiles. Recall *remembers people across meetings* — name a voice once and it is
recognized in every later recording. Everything runs locally except the final
notes/personas/reports, which use your Claude subscription via `claude -p` with a
local MLX fallback.

> The CLI ships as the `recall` package — run it with `python -m recall`.

```
audio ─► ingest ─► [enhance] ─► ASR ─► [diarize+identify] ─► [personas]
        ffmpeg     pluggable    pluggable  pyannote+voiceprints  profiles
                  ─► assemble (+coverage) ─► notes (+tailored reports)
                     .md / .json              claude → local MLX
```

The two historically contested choices are now **configuration, not forks** — the
ASR backend (`--asr`) and the enhancer (`--enhance`) are pluggable, so the per-file
winners drop in as defaults (the design decisions are in ARCHITECTURE §9; the A/B
harness is `scripts/ab_test.py`).

> **Full reference:** [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — pipeline
> diagram, ML models, install + hardware requirements, every CLI flag, the package
> API, design decisions, and an agent/developer guide. Contributing agents: start
> with [`AGENTS.md`](AGENTS.md).

## Install

Recall is a pip-installable package (`src/` layout). Create the project venv and
install it editable with the backend extras you want:

```bash
brew install ffmpeg
python3 -m venv .venv-transcribe && source .venv-transcribe/bin/activate
pip install -e '.[all]'                    # or '.[mlx,faster,diarize]' — pick backends
npm install -g @anthropic-ai/claude-code   # for `claude -p`; log in once
```

Extras: `mlx` (Apple-Silicon ASR + offline notes), `faster` (portable CPU ASR),
`diarize` (speaker labels), `demucs`/`deepfilternet` (enhancers), `romanize`,
`all`. `all` excludes `deepfilternet` — its native lib (`deepfilterlib`) ships no
prebuilt wheel and **compiles from source via Rust/cargo**. To use it: install a
Rust toolchain first (`brew install rust`), then `recall[deepfilternet]`. The core
install needs
only stdlib + tqdm/psutil. (`requirements.txt` mirrors `all` if you prefer
`pip install -r requirements.txt`.) On `uv`, use
`uv pip install -e '.[all]'` into `.venv-transcribe`.

Diarization (optional, one-time): make a free huggingface.co account, accept the
conditions on `pyannote/segmentation-3.0`, `pyannote/speaker-diarization-3.1`, and
`pyannote/speaker-diarization-community-1`, create a token, and
`export HF_TOKEN=hf_xxx`.

## Editor / LSP setup

The editable install above already makes `recall` resolvable to any language
server. The repo also ships `pyrightconfig.json` (server pointed at
`.venv-transcribe`, `src/` on the path, optional lazy-imported ML backends
downgraded to warnings) and `.vscode/settings.json` (interpreter + pytest). In your
editor, select the `.venv-transcribe` interpreter and you get go-to-def,
find-references, and type checks across the package.

## Usage

```bash
# typical: speaker labels on, Claude notes with local fallback
python -m recall ~/VoiceMemos/standup.m4a

# pick the portable backend (works off Apple Silicon)
python -m recall meeting.m4a --asr faster

# try an enhancer (A/B it first — see scripts/ab_test.py)
python -m recall meeting.m4a --enhance ffmpeg

# fully offline notes / transcript only / no speaker labels
python -m recall meeting.m4a --notes-engine local
python -m recall meeting.m4a --notes-engine none
python -m recall meeting.m4a --no-diarize

# later runs auto-recognise people; tailor reports to two of them
python -m recall team-sync.m4a --report-for "Priya" --report-for "Rahul"
```

`python -m recall --help` lists every flag. After `pip install`, the `recall`
console command works too: `recall standup.m4a`.

### Recipes (copy-paste)
```bash
# GPU (mlx) + ffmpeg cleanup, no speaker prompts, persistent identity   ← good default
recall meeting.m4a --asr mlx --language en --enhance ffmpeg --no-enroll --data-dir ~/.recall/data

# GPU (mlx), raw audio (faster, transcript noisier in silent gaps)
recall meeting.m4a --asr mlx --language en --no-enroll --data-dir ~/.recall/data

# CPU (faster-whisper) — portable, no GPU, strongest anti-hallucination
recall meeting.m4a --asr faster --language en --no-enroll --data-dir ~/.recall/data

# name speakers interactively (omit --no-enroll); voiceprints persist + auto-match next time
recall meeting.m4a --asr mlx --language en --data-dir ~/.recall/data

# tailored report for a known person
recall meeting.m4a --asr mlx --language en --report-for "Priya" --data-dir ~/.recall/data

# transcript only (no notes) / fully offline notes (no Claude)
recall meeting.m4a --asr faster --language en --notes-engine none
recall meeting.m4a --asr mlx    --language en --notes-engine local
```
Flag cheatsheet: `--asr mlx` GPU / `faster` CPU · `--enhance ffmpeg` cleans silent-gap
hallucination · `--no-enroll` skips speaker-naming prompts · `--data-dir ~/.recall/data`
makes identity persist across runs from any folder. Full list: `recall --help`.

### Re-running the same audio (dedup)
Outputs and a small SQLite index live under `~/.recall/`. Each run is keyed by the
audio's content hash, so **re-running the same recording prints the existing
transcript/notes paths instead of regenerating** (saves ASR time + Claude tokens).
Pass `--force` to regenerate, `--title "My Meeting"` to set the output filename
(`<DD-MM-YYYY>_<title>_<file>.notes.md`).

## Package layout (vertical slices)

One capability per module under `src/recall/`:

| module | responsibility |
|---|---|
| `common` | `Segment` + ffmpeg/wav primitives (stdlib only) |
| `metrics` | resource readout + live progress UX |
| `enhance` | **C2** — pluggable enhancers: `none` / `ffmpeg` / `deepfilternet` / `demucs` |
| `asr` | **C1** — pluggable backends: `faster-whisper` / `mlx-whisper` (`auto` tries mlx then faster) |
| `diarize` | pyannote diarization + speaker assignment |
| `identity` | persistent voiceprints + cross-session resolution |
| `personas` | per-person living collaboration profiles |
| `generate` | Claude (`claude -p`) → local-MLX text engine (shared) |
| `transcript` | assemble `.md`/`.json` + romanize + **coverage diagnostics** |
| `notes` | meeting notes + per-person tailored reports |
| `store` | SQLite dedup index (audio-hash → existing transcript/notes) |
| `pipeline` | orchestration (wires the slices together, sequentially) |
| `cli` | argument parsing / entry point (`python -m recall`) |

Prompts live in `src/recall/prompts/{notes,persona,report}.md`. Each slice owns its
optional heavy dependency and degrades gracefully when it's missing, so the package
imports cleanly even without the ML stack.

## Data

Voiceprints + personas persist under `--data-dir` (default `~/.recall/data/`):
`voiceprints.json` and `people/<slug>/{profile.md, utterances.jsonl}`. This is
biometric/personal data about colleagues — keep it local.

## Coverage diagnostic

Every run reports a **speech-coverage ratio** and the largest silent gaps, and
warns when coverage is low — both example recordings here silently dropped ~19
minutes, which this catches. (Design doc §4 / L6.)

## Tests

No pytest required:

```bash
python tests/run.py        # identity logic + mocked end-to-end pipeline + store/dedup
# or, if you have pytest:
pytest tests/
```

## Repo map

- `src/recall/` — the package (this README)
- `scripts/ab_test.py` — A/B harness for the contested axes (run on the Mac)
- `scripts/transcribe_audio.py` — original standalone faster-whisper script (legacy)
- `AGENTS.md` — working brief for contributing agents
- `docs/ARCHITECTURE.md` — the deep reference (architecture, models, CLI, API, decisions)
- `meetings/` — per-meeting outputs
```
