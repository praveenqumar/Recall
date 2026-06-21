# Meeting Synthesis: Requirements, Implemented Solution, and Decisions

## Requirements Captured

### 1. Local audio transcription

- Accept local meeting audio files such as `.m4a` voice memos.
- Generate transcript files from the input audio.
- Support Hinglish conversations with Hindi + English + technical vocabulary.
- Produce readable transcript output that can later be summarized.

### 2. Long-running processing

- Transcription can take time, so the process should be runnable in the background.
- Logs should be observed without continuously polluting the main working context.
- The process should not depend on a fixed timeout.

### 3. Audio enhancement

- Improve transcription quality by cleaning noise and enhancing speech.
- Support an AI-based speech enhancement option.
- Keep a safer fallback enhancement path if AI enhancement performs poorly.

### 4. Meeting organization

- Each meeting should live in its own folder.
- Folder names should be based on:
  - recording date from metadata
  - short 3–4 word description of the meeting content
- Store transcript, enhanced audio, summary, and metadata together.

### 5. Summary generation

- Generate a meeting summary from the transcript.
- Summary should capture key themes, feedback, decisions, concerns, and action items.
- For one-on-one/manager feedback conversations, preserve the intent and caveats around transcription quality.

### 6. Speaker intelligence requirement

- Future requirement: identify speakers as `SPEAKER_00`, `SPEAKER_01`, etc.
- Allow manual tagging of detected speaker labels to real names such as `Priya` or `Rahul`.
- Persist that mapping for future meetings.
- Eventually learn reusable voice signatures from already-tagged meeting segments instead of requiring separate clean voice samples.

### 7. Persona / collaboration intelligence requirement

- Future requirement: derive communication-style observations from raw speaker-attributed transcript.
- Capture patterns such as:
  - feedback style
  - seriousness/directness
  - collaboration preference
  - expectation around status updates
  - preferred report structure
- Use this to prepare future updates in a format each person understands better.

---

## Solution Implemented

### 1. Reusable transcription script

Implemented:

```text
scripts/transcribe_audio.py
```

The script accepts an audio file and generates:

```text
<input>.transcript.txt
<input>.transcript.md
<input>.transcript.json
```

The Markdown transcript includes timestamps. The JSON transcript stores structured segment metadata.

### 2. Whisper Large-v3-Turbo transcription

Implemented local transcription using:

```text
faster-whisper
large-v3-turbo
```

The script uses `--language en` for Hinglish/Roman-script output because the meetings contain a mix of Hindi, English, and technical terms.

### 3. Audio enhancement paths

Implemented two enhancement options:

```bash
--enhancer deepfilternet
--enhancer ffmpeg
```

- `deepfilternet`: AI speech enhancement / denoising path.
- `ffmpeg`: reliable DSP-based cleanup path.

Also available:

```bash
--enhancer none
```

for direct transcription without enhancement.

### 4. Background execution pattern

Used `nohup` + log files + PID files for long-running jobs.

Example pattern:

```bash
nohup bash -lc 'source .venv-transcribe/bin/activate && python scripts/transcribe_audio.py ...' \
  > logs/<meeting>_transcription.log 2>&1 &
```

Observed only small log tails and process status when needed.

### 5. Meeting folder organization

Implemented meeting-level folder structure manually for the latest meeting:

```text
meetings/2026-05-15-ontology-ai-feedback/
```

The folder contains:

```text
another_meeting.m4a
another_meeting.enhanced.wav
another_meeting.transcript.txt
another_meeting.transcript.md
another_meeting.transcript.json
meeting_summary.md
metadata.json
```

### 6. Meeting summary and metadata

Created:

```text
meeting_summary.md
metadata.json
```

Metadata stores source file, creation time, duration, short description, and transcription settings.

---

## Decisions That Drove the Solution

### Decision 1: Use local transcription instead of LM Studio audio API

LM Studio showed the Whisper model as installed, but the local server exposed text/chat/embedding endpoints, not a reliable audio transcription endpoint in this setup.

Decision:

```text
Use local faster-whisper inference for transcription.
```

Reason:

- Reliable local control.
- Works directly with audio files.
- Produces timestamped segments.

### Decision 2: Use `language=en` for Hinglish transcription

Although the audio contains Hindi, using `hi` caused poorer output and more Devanagari/Hindi hallucination in tests.

Decision:

```text
Use --language en for Hinglish meetings when the desired output is Roman/English text.
```

Reason:

- Better handling of English technical terms.
- More readable output for mixed Hindi-English meetings.
- Easier to summarize later.

### Decision 3: Keep ffmpeg fallback even after adding AI enhancement

DeepFilterNet was added for AI vocal enhancement, but on one file it produced a poor/sparse transcript.

Decision:

```text
Keep both deepfilternet and ffmpeg enhancer paths.
```

Reason:

- AI enhancement can help, but may fail depending on audio characteristics.
- ffmpeg cleanup was more reliable for the tested meeting files.
- The pipeline needs operational fallback, not only best-case quality.

### Decision 4: Store enhanced audio

Decision:

```text
Support --keep-enhanced.
```

Reason:

- Enables quality review.
- Allows retranscription without repeating preprocessing.
- Helps compare enhancement strategies.

### Decision 5: Organize by meeting folders

Decision:

```text
Use meetings/YYYY-MM-DD-short-description/.
```

Reason:

- Keeps source audio, transcript, metadata, and summary together.
- Makes later search and review easier.
- Supports multiple meetings without filename confusion.

### Decision 6: Treat speaker recognition as a staged feature

Speaker naming requires diarization and identity mapping. Clean per-person samples may be hard to collect.

Decision:

```text
Start with diarized labels, allow manual tagging, then learn voice signatures from tagged segments over time.
```

Reason:

- Avoids needing separate speaker sample collection upfront.
- Improves incrementally as more meetings are processed.
- Allows human correction when the model is uncertain.

### Decision 7: Treat persona as communication-style assistance, not psychological labeling

Decision:

```text
Store observed communication patterns, not fixed personality judgments.
```

Reason:

- Safer and more useful.
- Helps prepare better reports and updates.
- Avoids overclaiming from imperfect transcripts.

---

## Current Implementation Status

Implemented:

- Local transcription script.
- Whisper Large-v3-Turbo via `faster-whisper`.
- ffmpeg speech enhancement.
- DeepFilterNet AI enhancement option.
- Background execution with logs.
- Meeting folder organization.
- Metadata and summary files for processed meeting.

Not yet implemented:

- Automatic diarization.
- Manual speaker-name mapping file support.
- Persistent speaker registry.
- Voice-signature learning from tagged segments.
- Persona profile generation from speaker-attributed transcripts.
