# Recall

A professional-but-personal **meeting synthesizer**: offline, on-device
transcription for **Hinglish** (Hindi + English) meeting audio, with AI-generated
English notes, cross-session speaker identity, and per-person collaboration
profiles. Recall *remembers people across meetings* — name a voice once and it is
auto-recognized in every later recording.

Everything runs **locally on your Mac** except the final notes/personas/reports,
which use your Claude subscription via `claude -p` (with a fully-local MLX
fallback). Your audio never leaves the machine.

```
audio ─► ingest ─► enhance ─► ASR ─► diarize+identify ─► personas
        ffmpeg     ffmpeg DSP  Whisper  pyannote+voiceprints  profiles
                  ─► assemble (+coverage) ─► notes (+tailored reports)
                     .md / .json              claude → local MLX
```

> **Deeper docs:** [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) (full architecture,
> package API, design decisions) · [`AGENTS.md`](AGENTS.md) (contributor brief).

---

## How it works (the 7 stages)

| # | stage | what it does |
|---|---|---|
| 1 | **ingest** | `ffmpeg` decodes any input (m4a/mp3/wav/mp4) → 16 kHz mono PCM wav (Whisper's native format) |
| 2 | **enhance** | light DSP cleanup (highpass/lowpass/denoise/loudnorm) so levels are consistent |
| 3 | **ASR** | Whisper turns speech → timestamped text segments |
| 4 | **diarize + identify** | pyannote finds *who spoke when*; voiceprints match speakers to known people across meetings |
| 5 | **personas** | per-person collaboration profiles updated from their utterances |
| 6 | **assemble** | build the `.md`/`.json` transcript + coverage/hallucination diagnostics |
| 7 | **notes** | Claude writes meeting notes (+ optional per-person tailored reports) |

Stages run **sequentially** to respect the 18 GB unified-memory budget (never holds
Whisper and the local LLM at once).

---

## Models — what runs, and why

| job | model | engine | why |
|---|---|---|---|
| **speech → text** | `whisper-large-v3` | faster-whisper (CPU) **or** mlx-whisper (Apple GPU) | most accurate Whisper; handles Hindi + English |
| **who spoke when** | `pyannote/speaker-diarization-3.1` (+ `segmentation-3.0`, `speaker-diarization-community-1`) | pyannote.audio | speaker turns + a ~256-d voiceprint per speaker |
| **notes / personas / reports** | Claude | `claude -p` (your subscription) | translates Hinglish + writes notes, repairs ASR garbles |
| **notes (offline fallback)** | `Qwen2.5-7B-Instruct-4bit` | mlx-lm (Apple GPU) | fully-local notes when Claude is unavailable |
| **enhance (optional)** | DeepFilterNet / Demucs | CLI | AI denoise / vocal isolation — only for genuinely noisy audio |

> **model vs engine:** `whisper-large-v3` is the *model* (the trained weights).
> `--asr faster` / `--asr mlx` picks the *engine* that runs it (CPU vs Apple GPU).
> Same model either way. Model weights auto-download from HuggingFace on first use
> (cached in `~/.cache/huggingface`, ~8 GB).

### What downloads from HuggingFace (and when)

All weights auto-download on **first use** of the relevant feature and cache to
`~/.cache/huggingface` (override with `HF_HOME`). Nothing is bundled; first run of
each stage is slower while it fetches.

| repo | pulled by | when | size approx | gated? |
|---|---|---|---|---|
| `mlx-community/whisper-large-v3-mlx` | `--asr mlx` | first GPU transcribe | ~3 GB | no |
| `Systran/faster-whisper-large-v3` | `--asr faster` | first CPU transcribe | ~3 GB | no |
| `pyannote/speaker-diarization-3.1` | `--diarize` | first run with diarization | small | **yes** |
| `pyannote/segmentation-3.0` | `--diarize` | "" | small | **yes** |
| `pyannote/speaker-diarization-community-1` | `--diarize` | "" | small | **yes** |
| `mlx-community/Qwen2.5-7B-Instruct-4bit` | offline notes (`--notes-engine local`, or Claude unavailable) | first local-notes run | ~4 GB | no |

**Gated models** (the 3 `pyannote` repos) need a free HuggingFace account, accepting
each repo's conditions, and `export HF_TOKEN=hf_xxx` — see [Diarization setup](#install-uv--recommended)
below. Skip them entirely with `--no-diarize`. Total cache once everything is pulled: ~8–10 GB.

---

## Install (uv — recommended)

**Requirements:** macOS (Apple Silicon for `mlx` GPU ASR), Python ≥ 3.10.

```bash
# 1. install uv (the installer/runner) — skip if you already have it
curl -LsSf https://astral.sh/uv/install.sh | sh    # then restart your shell

# 2. system tools
brew install ffmpeg
npm install -g @anthropic-ai/claude-code      # for notes; run `claude` once to log in

# 3. install Recall as a global command (Apple Silicon)
uv tool install "recall[mlx,faster,diarize] @ git+https://github.com/praveenqumar/Recall.git"
```

> This installs the **Python libraries only** (~hundreds MB). The ML model weights
> (whisper, pyannote, Qwen — ~8–10 GB total) are **not** downloaded here; they fetch
> lazily on first use of each feature — see
> [What downloads from HuggingFace](#what-downloads-from-huggingface-and-when).

Now `recall` works from any folder. Verify the build:
```bash
recall --version          # e.g. recall 0.1.1.dev42+g395a63d
```

**Diarization (optional, one-time):** make a free huggingface.co account, accept the
conditions on all three gated models —
[`segmentation-3.0`](https://huggingface.co/pyannote/segmentation-3.0),
[`speaker-diarization-3.1`](https://huggingface.co/pyannote/speaker-diarization-3.1),
[`speaker-diarization-community-1`](https://huggingface.co/pyannote/speaker-diarization-community-1)
— create a token, and `export HF_TOKEN=hf_xxx` (put it in `~/.zshrc`).

**Update later** (after new commits): `uv tool install --force "recall[mlx,faster,diarize] @ git+https://github.com/praveenqumar/Recall.git"`. Confirm with `recall --version`.

### Extras (pick the backends you want)
`mlx` (Apple-Silicon ASR + offline notes) · `faster` (portable CPU ASR) ·
`diarize` (speaker labels) · `demucs` / `deepfilternet` (enhancers) · `romanize` · `all`.

> `[all]` excludes `deepfilternet` — its native lib needs a Rust toolchain. To use
> it: grab the prebuilt `deep-filter` binary from
> [DeepFilterNet releases](https://github.com/Rikorose/DeepFilterNet/releases) and
> pass `--deepfilter-command ./deep-filter` (no Rust), or `brew install rust` then
> add `recall[deepfilternet]`.

### Dev install (edit code, test live — no reinstall)
```bash
git clone https://github.com/praveenqumar/Recall.git && cd Recall
uv venv .venv-transcribe && source .venv-transcribe/bin/activate
uv pip install -e '.[mlx,faster,diarize]'
python -m recall meeting.m4a            # every src/ edit applies instantly
python tests/run.py                     # 24 tests
```

---

## Quickstart

```bash
recall meeting.m4a                      # transcribe + speaker labels + Claude notes
recall meeting.mp4                      # video works too (audio is extracted)
cat ~/.recall/*_meeting.notes.md        # read the notes
```
Defaults are sensible: **Apple GPU ASR**, **English output**, **ffmpeg cleanup**,
speaker labels on, Claude notes. Outputs land in `~/.recall/`.

### Where outputs go

Everything lands under `~/.recall/` (change with `-o, --output-dir`):

| path | what |
|---|---|
| `~/.recall/<DD-MM-YYYY_HHMMSS>_<title>_<file>.transcript.md` / `.json` | transcript (+ coverage diagnostics) |
| `~/.recall/<…>.notes.md` | Claude/local meeting notes |
| `~/.recall/<…>.report.<name>.md` | per-person tailored report (`--report-for NAME`) |
| `~/.recall/data/voiceprints.json` | speaker voiceprints (biometric, git-ignored) |
| `~/.recall/data/people/<slug>/` | per-person `profile.md` + `utterances.jsonl` |
| `~/.recall/recall.db` | SQLite dedup index (audio content-hash → outputs) |

Re-running the same audio reuses cached outputs; pass `--force` to regenerate (see
[dedup](#dedup)).

---

## Speed vs reliability — `--asr mlx` vs `--asr faster`

| | `--asr mlx` (default on Apple Silicon) | `--asr faster` |
|---|---|---|
| engine | mlx-whisper (Apple **GPU/Metal**) | faster-whisper (**CPU**) |
| speed | **fast** ⚡ — uses the GPU | slower (CPU-bound) |
| robustness | no built-in silence trimming → can hallucinate on quiet/silent gaps | **built-in VAD** trims silence → fewer hallucinations |
| best for | clean audio where you want speed | rough/quiet recordings, or when mlx output looks repetitive |

`--asr auto` (the default) picks **mlx on Apple Silicon, faster elsewhere**. If an
mlx transcript comes out looping/garbled on a rough recording, switch to
`--asr faster` — same model, more reliable engine.

---

## CLI flags — full reference

`recall AUDIO [options]` · run `recall --help` for the live list.

### Core
| flag | default | purpose |
|---|---|---|
| `AUDIO` | — | input audio/video (m4a/mp3/wav/mp4…) |
| `-o, --output-dir` | `~/.recall` | where transcript/notes/reports are written |
| `--title NAME` | filename | meeting title used in the output filename + dedup store |
| `--force` | off | **regenerate even if this audio was already processed** (see dedup below) |
| `--version` | — | print the installed build (with git commit) |
| `--no-progress` | off | disable progress bars / live resource readout (for scripts/cron) |

### ASR (speech → text)
| flag | default | purpose |
|---|---|---|
| `--asr {auto,mlx,faster}` | `auto` | engine: `auto` = mlx on Apple Silicon, faster elsewhere. `mlx` = GPU/fast, `faster` = CPU/robust |
| `--model ID` | `large-v3` | Whisper model id. Pass a `*-turbo` id to trade accuracy for speed |
| `--language {en,hi,auto}` | `en` | hint. `en` = Roman/English (best for Hinglish notes), `hi` = native Devanagari, `auto` = detect |
| `--chunk-seconds N` | `240` | mlx chunk size for the progress bar; `0` = single max-accuracy pass |

### Enhance (audio cleanup)
| flag | default | purpose |
|---|---|---|
| `--enhance SPEC` | `ffmpeg` | `none` \| `ffmpeg` \| `deepfilternet` \| `demucs`. Comma-chain in order, e.g. `demucs,ffmpeg`. `none` = raw audio |
| `--deepfilter-command CMD` | `deepFilter` | path/name of the DeepFilterNet binary |

### Diarization & identity
| flag | default | purpose |
|---|---|---|
| `--diarize` / `--no-diarize` | on | speaker labels via pyannote |
| `--hf-token TOK` | env `HF_TOKEN` | HuggingFace token for the gated pyannote models |
| `--data-dir DIR` | `~/.recall/data` | persistent voiceprints + personas (biometric — kept local) |
| `--no-enroll` | off | don't prompt to name unknown speakers; only auto-assign confident matches |
| `--id-high F` | `0.70` | cosine ≥ this → auto-assign a known speaker |
| `--id-low F` | `0.45` | cosine in `[id-low, id-high)` → ask you |

### Notes, personas, reports
| flag | default | purpose |
|---|---|---|
| `--notes-engine {auto,claude,local,none}` | `auto` | `auto` = Claude then local MLX; `none` = transcript only |
| `--local-model ID` | `Qwen2.5-7B-Instruct-4bit` | offline notes model |
| `--report-for NAME` | — | also write a report tailored to NAME's profile (repeatable) |
| `--no-personas` | personas on | skip building/updating per-person profiles |
| `--romanize` | off | transliterate Devanagari → Roman (ITRANS) |
| `--keep-repeats` | off | keep Whisper's looped/repeated text (default: collapse it to save tokens) |
| `--notes-prompt PATH` | bundled `notes.md` | override the notes instruction prompt file |
| `--persona-prompt PATH` | bundled `persona.md` | override the persona-update instruction prompt file |
| `--report-prompt PATH` | bundled `report.md` | override the tailored-report instruction prompt file |

---

<a id="dedup"></a>
## Re-running the same audio (dedup) & `--force`

Every run is keyed by the **audio's content hash** in a small SQLite index
(`~/.recall/recall.db`). So:

- **Run the same recording again** → Recall prints the **existing** transcript/notes
  paths and skips the work — no wasted ASR time or Claude tokens.
- **`--force`** → ignore the cache and **regenerate** (e.g. after changing `--asr`,
  `--enhance`, or `--model`). Each forced run writes a fresh timestamped file, so
  old outputs aren't overwritten.

Output filenames: `<DD-MM-YYYY_HHMMSS>_<title>_<file>.transcript.md` / `.notes.md`.

---

## Recipes

```bash
# fastest (Apple GPU), default cleanup + Claude notes
recall meeting.m4a

# rough/quiet recording → CPU engine with VAD (more reliable)
recall meeting.m4a --asr faster

# name speakers interactively → voiceprints persist + auto-match next time
recall team-sync.m4a            # (omit --no-enroll to be prompted)

# tailored report for a known person
recall team-sync.m4a --report-for "Priya"

# fully offline notes (no Claude) / transcript only / raw audio
recall call.m4a --notes-engine local
recall call.m4a --notes-engine none
recall call.m4a --enhance none

# unattended (no prompts), regenerate from scratch
recall call.m4a --no-enroll --force
```

---

## Data & privacy

Voiceprints + personas live under `--data-dir` (`~/.recall/data/`):
`voiceprints.json` and `people/<slug>/{profile.md, utterances.jsonl}`. This is
**biometric/personal data** about colleagues — it stays on your disk and is
git-ignored. The transcript is sent to Claude for notes; treat generated text as
influenced by the recording's content.

## Coverage & hallucination diagnostics

Every run reports a **speech-coverage ratio** + largest silent gaps, and warns on
likely **hallucination loops** (repeated phrases / low confidence) — so a transcript
that silently dropped audio or looped is caught, not trusted blindly.

## Tests

```bash
python tests/run.py        # 24 tests, no pytest needed (only ffmpeg required)
```

## Repo map
- `src/recall/` — the package (one capability per module; see ARCHITECTURE)
- `scripts/ab_test.py` — A/B harness for ASR backend / enhancer / language
- `scripts/seed_voiceprints.py` — enroll a meeting's speakers without the tty prompt
- `docs/ARCHITECTURE.md` — deep reference · `AGENTS.md` — contributor brief
