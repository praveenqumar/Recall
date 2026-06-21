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
ASR backend (`--asr`) and the enhancer (`--enhance`) are pluggable, so the A/B
winners drop in as defaults. See `docs/reconciliation-and-merged-design.md` for the
decisions and `docs/ab-runbook.md` for how to settle them on your own audio.

> **Full reference:** [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — pipeline
> diagram, ML models, install + hardware requirements, every CLI flag, the package
> API, and an agent/developer guide for extending and debugging the code.

## Install

```bash
brew install ffmpeg
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # install the ASR backend(s) you want
npm install -g @anthropic-ai/claude-code  # for `claude -p`; log in once
```

Diarization (optional, one-time): make a free huggingface.co account, accept the
conditions on `pyannote/segmentation-3.0` and `pyannote/speaker-diarization-3.1`,
create a token, and `export HF_TOKEN=hf_xxx`.

## Editor / LSP setup

For full code intelligence (go-to-def, references, type checks) install the
package editable so the language server resolves `recall` and its backends:

```bash
VIRTUAL_ENV=.venv-transcribe uv pip install -e '.[all]'   # or .[mlx,faster,diarize]
```

`pyrightconfig.json` points the language server at `.venv-transcribe` and treats
the optional ML backends (mlx/faster/pyannote/torch) as warnings — they are
lazy-imported, so a partial install still type-checks cleanly. VS Code picks up
`.vscode/settings.json` automatically.

## Usage

```bash
# typical: speaker labels on, Claude notes with local fallback
python -m recall ~/VoiceMemos/standup.m4a

# pick the portable backend (works off Apple Silicon)
python -m recall meeting.m4a --asr faster

# try an enhancer (A/B it first — see docs/ab-runbook.md)
python -m recall meeting.m4a --enhance ffmpeg

# fully offline notes / transcript only / no speaker labels
python -m recall meeting.m4a --notes-engine local
python -m recall meeting.m4a --notes-engine none
python -m recall meeting.m4a --no-diarize

# later runs auto-recognise people; tailor reports to two of them
python -m recall team-sync.m4a --report-for "Priya" --report-for "Rahul"
```

`python -m recall --help` lists every flag.

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
| `pipeline` | orchestration (wires the slices together, sequentially) |
| `cli` | argument parsing / entry point (`python -m recall`) |

Prompts live in `src/recall/prompts/{notes,persona,report}.md`. Each slice owns its
optional heavy dependency and degrades gracefully when it's missing, so the package
imports cleanly even without the ML stack.

## Data

Voiceprints + personas persist under `--data-dir` (default `./recall-data/`):
`voiceprints.json` and `people/<slug>/{profile.md, utterances.jsonl}`. This is
biometric/personal data about colleagues — keep it local.

## Coverage diagnostic

Every run reports a **speech-coverage ratio** and the largest silent gaps, and
warns when coverage is low — both example recordings here silently dropped ~19
minutes, which this catches. (Design doc §4 / L6.)

## Tests

No pytest required:

```bash
python tests/run.py        # 13 tests: identity logic + mocked end-to-end pipeline
# or, if you have pytest:
pytest tests/
```

## Repo map

- `src/recall/` — the package (this README)
- `scripts/ab_test.py` — A/B harness for the contested axes (run on the Mac)
- `scripts/transcribe_audio.py` — original standalone faster-whisper script (legacy)
- `docs/` — design, reconciliation, A/B runbook
- `files/HANDOFF.md` — original project handoff (historical reference)
- `meetings/` — per-meeting outputs
```
