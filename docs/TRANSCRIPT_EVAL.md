# Transcript Coverage Evaluation — Strategy

> Issue [#9](https://github.com/praveenqumar/Recall/issues/9). Goal: turn the
> observation *"long (~1h) conversations produce very little transcript"* into a
> **repeatable measurement** that (a) quantifies how much speech is captured vs
> lost, (b) localizes **where** loss happens, and (c) attributes it to a specific
> pipeline knob — before changing the pipeline.
>
> This doc is the working reference. **Decided** vs **Pending** is split
> explicitly so a fresh context can resume with no re-explanation.

---

## 1. Why the current signal is not enough

`transcript.coverage()` already emits `coverage = sum(segment durations) / audio
duration`, top silent gaps, repetition ratio, and mean `avg_logprob`. That made
the problem *visible* but cannot explain it, because:

- **Coverage is a time-span proxy, not a content measure.** A 15 s segment with
  2 words scores identically to 15 s of dense speech. Under-capture *inside*
  voiced regions is invisible to it.
- **It is fooled both directions.** A hallucination loop already inflated it to
  101 % on garbage; conversely, real speech dropped before it becomes a segment
  never appears in the numerator *or* shows only as a silent gap.
- **It has no independent reference.** Coverage is computed from the ASR's own
  output, so any speech the ASR/VAD silently discards is invisible — the metric
  can't see what the pipeline never emitted.

We need a measurement that compares the transcript against the **audio**, not
against the ASR's own segments.

---

## 2. Where loss most likely originates (hypotheses, ranked)

Model-internal reasoning, to be confirmed/denied by measurement — not assumed.

| # | Hypothesis | Mechanism | Knob |
|---|------------|-----------|------|
| H1 | **VAD drops speech** (most likely) | `faster` runs Silero `vad_filter=True`; only voiced chunks are concatenated before decode, so dropped spans vanish silently. `threshold=0.2`, `min_silence_duration_ms=500`, `speech_pad_ms=400`. Soft/accented Hinglish below threshold is gone. | `vad_parameters` |
| H2 | **Anti-hallucination filters discard real speech** | `hallucination_silence_threshold=2`, `compression_ratio_threshold=2.2`. Repetitive-but-real Hinglish trips the compression gate → whole segment dropped. | both thresholds |
| H3 | **mlx chunk-boundary loss** | each `chunk_s` slice is decoded independently with no overlap/stitch; words straddling a boundary are cut. Error scales with chunk count → worse on long files. | `chunk_s`, stitching |
| H4 | **Length-dependent degradation** | decode stops early / accumulates error on long input. Confirmed only if capture declines with duration in the controlled sweep (§4). | n/a |

H1 and H2 are silent (no error, no warning). H3 is partial. H4 is the one the
"size vs capture" experiment exists to break.

---

## 3. The measurement (three tiers, cheap → gold)

### Tier 0 — structural, zero ground truth
Automated, repeatable, CI-able. Build first.

- **`vad_ref`** — run an **independent** VAD (`webrtcvad`, or a standalone Silero
  pass) on the raw 16 kHz wav → total voiced seconds + voiced spans. Compare to
  the pipeline's emitted `speech_s`. **`vad_ref − speech_s` = speech the pipeline
  dropped.** This is the direct test of H1, and the single highest-value metric.
- **`wpm_voiced`** — words per minute computed *only over voiced minutes* (per
  `vad_ref`). Flag any voiced minute yielding ~0 words → a capture hole coverage
  cannot see. Expected conversational range is a calibration target (§6).
- **truncation check** — assert last segment end ≈ `wav_duration`. An early stop
  is H4.

### Tier 1 — reference without manual labels
- **engine A/B** — `mlx` vs `faster` on the same file, time-aligned; divergence
  localizes loss per region (one engine emits where the other is empty).
- **loudness vs words** — per-minute RMS vs `wpm_voiced`; loud-but-wordless
  minutes = failure region.

### Tier 2 — gold (the only true correctness measure)
- **hand-correct one 10–15 min slice** of a real 1 h Hinglish meeting → compute
  **WER and, separately, deletion rate (`del`)**. WER averages substitutions /
  insertions / deletions together; the "missing transcript" complaint *is* the
  deletion rate, so it must be reported on its own.
- **forced alignment** of the gold slice (WhisperX / aeneas) → word-level recall
  by time region; correlate missed words with `vad_ref` drops to prove the
  mechanism (H1 vs H2 vs genuine silence).

---

## 4. The "audio size vs capture" experiment (issue #9's literal ask)

A controlled sweep to break H4:

- Run the pipeline on the **same content** at **5 / 15 / 30 / 60 / 90 min** and
  plot `coverage`, `wpm_voiced`, and (on the gold slice) `del` vs duration.
  - **Flat** → no length bug; loss is uniform (points at H1/H2).
  - **Declining with length** → confirms a length-dependent bug (H4) → bisect.
- **Ablation grid** on the gold slice — toggle one knob at a time and measure the
  three metrics:
  - VAD `threshold` (0.2 / 0.35 / 0.5) and `min_silence_duration_ms`
  - `hallucination_silence_threshold` (on / off)
  - `compression_ratio_threshold` (2.2 / 2.6 / off)
  - `--enhance` (none / ffmpeg / deepfilternet)
  - `chunk_s` for mlx (and with/without boundary overlap)
  → attributes each lost word to a specific knob.

---

## 5. Calibration knobs (not just less code — the physical signal needs tuning)

- `wpm_voiced` "too low" threshold is **content-dependent** (Hinglish code-switch
  is slower than monolingual English). Leave it a configurable floor, calibrated
  from the gold slice, not a hardcoded constant.
- `vad_ref` aggressiveness (`webrtcvad` mode 0–3) trades missed-speech vs
  false-voiced; calibrate against the gold slice so `vad_ref` itself is trusted
  before we trust `vad_ref − speech_s`.

---

## 6. Decided vs Pending

### Decided
- D1 — work proceeds **design-doc-first**; build nothing until this is reviewed.
- D2 — a **real 1 h Hinglish recording** will be supplied (user has one); no
  synthetic audio for primary measurement (evidence-first).
- D3 — tier order is fixed: **Tier 0 → Tier 1 → Tier 2**; Tier 2 / forced
  alignment deps (WhisperX) are **deferred** until Tier 0 proves loss is real.
- D4 — deletion rate is reported **separately** from WER.
- D5 — branch: `eval/transcript-coverage`.

### Pending (need user decision)
- P1 — **independent VAD choice**: `webrtcvad` (tiny dep, fast) vs a second
  Silero pass (no new dep, but correlated with the pipeline's own VAD → weaker as
  an independent reference). *Default proposal: `webrtcvad`.*
- P2 — **harness shape**: standalone `recall-eval <wav> [<transcript.json>]`
  script vs a `--eval` flag on `recall`. *Default proposal: standalone script —
  keeps eval deps out of the main path.*
- P3 — **sweep content source**: tile one real clip to length vs five distinct
  real recordings of increasing length. *Default proposal: real clips if
  available; tiling only as fallback, clearly logged as synthetic.*
- P4 — `wpm_voiced` floor value (set after gold-slice calibration, §5).
- P5 — whether the size-vs-capture plot is a committed artifact (CSV + chart) or
  log-only.

---

## 7. Risks
- R1 — without the gold slice, Tier 0 can **mislocate** loss (genuine silence
  read as dropped speech). Gold slice is what converts "looks low" into a
  defensible root cause.
- R2 — `vad_ref` must itself be calibrated (§5) or `vad_ref − speech_s` is noise.
- R3 — A/B engine divergence shows *that* they disagree, not *which* is right;
  only the gold slice adjudicates.
