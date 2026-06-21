"""One-off: enroll this meeting's diarized speakers under given names, so the
next pipeline run auto-matches them and exercises identity + personas + reports.

    python scripts/seed_voiceprints.py AUDIO NAME0 NAME1 [NAME2 ...]

NAMEn maps to the diarization label SPEAKER_0n (by sorted label order).
"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scribe import diarize as diar, identity  # noqa: E402
from scribe.common import ingest  # noqa: E402
from scribe.metrics import Metrics  # noqa: E402
import os  # noqa: E402


def main() -> None:
    audio = Path(sys.argv[1]).expanduser().resolve()
    names = sys.argv[2:]
    with tempfile.TemporaryDirectory() as tmp:
        wav = ingest(audio, Path(tmp))
        turns, emb_map = diar.diarize(
            wav, os.environ.get("HF_TOKEN"), Metrics(), progress=False)
    if not emb_map:
        sys.exit("no embeddings — diarization failed")
    vstore = identity.VoiceStore(Path("./scribe-data/voiceprints.json"))
    for label, name in zip(sorted(emb_map), names):
        vstore.enroll(name, emb_map[label])
        print(f"enrolled {label} -> {name}")
    vstore.save()
    print("saved", vstore.path, "| people:", vstore.names())


if __name__ == "__main__":
    main()
