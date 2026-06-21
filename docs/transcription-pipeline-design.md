# Audio Transcription Pipeline Design

## Goal

Take an input meeting audio file, such as:

```text
hinglish_audio.m4a
```

and generate readable transcript artifacts:

```text
hinglish_audio.transcript.txt   # plain transcript
hinglish_audio.transcript.md    # timestamped transcript for review
hinglish_audio.transcript.json  # structured segments and metadata
```

The pipeline was built for a Hinglish voice memo where Hindi and English are mixed, but the expected transcript is mostly English/Roman script.

## Tooling Used

- `ffmpeg` for audio conversion, normalization, and resampling.
- `DeepFilterNet` (`deepfilternet`) for AI speech enhancement/noise suppression.
- `faster-whisper` for local Whisper inference.
- Whisper model: `large-v3-turbo`.
- Python virtual environment for repeatable local execution.

LM Studio had `whisper-large-v3-turbo` installed and visible in `/v1/models`, but its local server only exposed text/chat/embeddings endpoints in this setup. For the actual audio transcription step, the reliable path was to run Whisper locally through `faster-whisper`.

## Pipeline

### 1. Validate input

The script checks that the input audio exists before doing any work.

### 2. AI audio preprocessing

The default enhancer is now `deepfilternet`, an AI speech enhancement model. The flow is:

1. Convert the input voice memo to mono 48 kHz WAV for DeepFilterNet.
2. Run `deepFilter` to suppress background noise and enhance speech/vocals.
3. Convert the AI-enhanced result to mono 16 kHz WAV for Whisper.
4. Apply light post-processing only: high-pass, low-pass, and loudness normalization.

Post-AI ffmpeg filter:

```text
highpass=f=80,lowpass=f=8000,loudnorm=I=-18:TP=-1.5:LRA=11
```

The script also supports a DSP-only fallback:

```bash
python scripts/transcribe_audio.py hinglish_audio.m4a --enhancer ffmpeg
```

DSP fallback filter:

```text
highpass=f=80,lowpass=f=8000,afftdn=nf=-25,loudnorm=I=-18:TP=-1.5:LRA=11
```

Use `--enhancer none` when you want Whisper to consume the original audio directly.

### 3. Whisper inference

The script runs `large-v3-turbo` with these important settings:

- `language="en"`: best practical hint for Hinglish when the desired output is Roman/English text.
- `task="transcribe"`: transcribe instead of translate.
- `beam_size=5`: improves decoding quality versus greedy-only decoding.
- `condition_on_previous_text=False`: reduces repeated phrases and runaway loops.
- `vad_filter=True`: removes long non-speech regions.
- Low VAD threshold (`0.2`) to avoid dropping soft speech.
- `hallucination_silence_threshold=2`: reduces repeated hallucinations over silence.

### 4. Segment collection

Each Whisper segment is saved with:

- start timestamp
- end timestamp
- text
- average log probability
- no-speech probability
- compression ratio

These fields are useful for debugging low-confidence or noisy segments later.

### 5. Output generation

The same result object is written in three formats:

1. TXT: clean full transcript.
2. MD: timestamped transcript for reading/review.
3. JSON: structured machine-readable output for later summarization, QA, or search.

## How to Run

From this directory:

```bash
python3.11 -m venv .venv-transcribe
source .venv-transcribe/bin/activate
pip install -r requirements.txt
python scripts/transcribe_audio.py hinglish_audio.m4a --language en --keep-enhanced
```

By default this uses AI speech enhancement:

```text
--enhancer deepfilternet
```

If DeepFilterNet is not installed or you want the previous non-AI cleanup path:

```bash
python scripts/transcribe_audio.py hinglish_audio.m4a --language en --enhancer ffmpeg --keep-enhanced
```

Output files will be written next to the input audio.

## Regenerating the Transcript

The transcript input file is:

```text
hinglish_audio.m4a
```

To regenerate using the current AI-enhanced pipeline:

```bash
source .venv-transcribe/bin/activate
python scripts/transcribe_audio.py hinglish_audio.m4a --language en --enhancer deepfilternet --keep-enhanced
```

Generated files:

```text
hinglish_audio.transcript.txt
hinglish_audio.transcript.md
hinglish_audio.transcript.json
hinglish_audio.enhanced.wav
```

## Notes and Limitations

- Hinglish transcription quality depends strongly on audio quality, speaker distance, background noise, and code-switching.
- Some Hindi words may be incorrectly converted into nearby English-sounding words.
- `large-v3-turbo` is faster than full `large-v3`, but full `large-v3` may sometimes produce better accuracy if speed is less important.
- The generated transcript should be treated as a draft and reviewed for important decisions/action items.
