# A/B Runbook — settle the 3 contested choices

This runs on your **Apple Silicon Mac** (mlx-whisper needs Metal; this can't run
in the Cowork sandbox). It produces the evidence that turns the three contested
choices in `docs/reconciliation-and-merged-design.md` into defaults.

## 0. One-time setup

```bash
cd meeting-scribe
source .venv-transcribe/bin/activate          # the env that already runs faster-whisper

# to also test the mlx backend (Track B's premise — currently unproven):
pip install mlx-whisper
# optional, only if you want the deepfilternet enhancer in the sweep:
pip install deepfilternet
# optional, for automatic ranking: ensure the `claude` CLI is installed + logged in
```

## 1. Fast first pass (recommended)

Six minutes of one file, all axes, Claude ranks the results:

```bash
python scripts/ab_test.py example_1on1.m4a --judge
```

This runs up to 3 enhancers × 2 backends × 2 languages = 12 transcriptions of the
same 6-minute clip and writes to `./ab-out/`:

- `ab_report.md` — the comparison table + Claude's verdict (**read this first**)
- `<variant>.txt` — each transcript, e.g. `ffmpeg+faster+en.txt`
- `ab_results.json` — machine-readable metrics

If `mlx-whisper` or `deepfilternet` isn't installed, those rows show ❌ and the run
continues — so you get a useful report even with only what you have today.

> ⚠️ Pick a clip window that actually contains speech. The example 1:1 recording is
> mostly silent until ~19:14, so for that file use a window over the speech:
> ```bash
> # transcode the speech-bearing slice first, then point the harness at it
> ffmpeg -y -ss 00:19:00 -t 360 -i example_1on1.m4a example_speech.m4a
> python scripts/ab_test.py example_speech.m4a --judge --minutes 0
> ```
> `hinglish_audio.m4a` has speech from the start, so it needs no pre-slice and is the
> better file for the language (en vs hi) question.

## 2. Read the report

In `ab-out/ab_report.md`, for each variant you get:

- **RTF** (wall ÷ audio): how fast — expect mlx ≪ faster on your M3 Pro if mlx works.
- **coverage** (speech ÷ duration): how much audio survived. **Low coverage = dropped
  speech** (this is the silent-gap problem from §4 of the design doc — distrust a
  "clean-looking" variant with low coverage).
- **segs / logprob μ / min / chars**: confidence and verbosity sanity checks.
- **top gap (s)**: the largest silent jump — a big number flags lost minutes.
- **Claude verdict**: per-variant fidelity + notes-usability scores and a winner.

## 3. Decide (the rule)

For each axis, choose the variant with the best **notes usability** (notes quality is
the real objective, not transcript prettiness); break ties by coverage, then RTF.

- **C3 language (en vs hi):** the headline question. Compare on `hinglish_audio.m4a`.
- **C1 backend (faster vs mlx):** if mlx works and wins on speed without losing
  fidelity, it becomes default; if mlx errors (likely first try — see design doc §6),
  note the traceback and keep faster-whisper.
- **C2 enhancer:** expect this to split by file. Whatever wins, the enhancer stays a
  flag; just record the recommended default + when to override.

## 4. Confirm on a second file

Re-run on a contrasting recording (e.g. `another_meeting.m4a`) to make sure the
winners hold across content, not just one clip:

```bash
python scripts/ab_test.py another_meeting.m4a --judge
```

## 5. Hand the results back

Drop `ab-out/ab_report.md` (and any mlx tracebacks) back into Cowork. With the
per-axis winners decided, the next step is building the merged `scribe.py` around the
`ASRBackend` / `Enhancer` abstractions in design-doc §5, with the winners as defaults.
