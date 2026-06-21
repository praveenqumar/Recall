# CLAUDE.md

See [`AGENTS.md`](AGENTS.md) — the working brief (layout, build/test/run,
conventions, guardrails, debugging). For depth, [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

This file is intentionally a pointer so guidance lives in one place and can't drift.

## Common commands (copy-paste)
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
