"""
scribe.identity — persistent cross-session speaker identity.

VoiceStore: name -> running-average voiceprint; match() classifies a new speaker
embedding as match / ambiguous / unknown by cosine. resolve_identities() turns
diarization labels into real names (auto-assigning confident matches, asking the
human on ambiguous/unknown ones), reinforcing the centroid on every confirmation.

Embeddings are only ~256 floats, so the vector math is pure-Python (no numpy).
This slice is the most heavily unit-tested part of the codebase.
"""
from __future__ import annotations

import json
import math
import re
import sys
import time
from pathlib import Path

from .common import Segment, log


# --------------------------------------------------------------------------- #
# vector + name helpers
# --------------------------------------------------------------------------- #
def l2norm(v: list[float]) -> list[float]:
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


def cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return -1.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(x * x for x in b)) or 1.0
    return dot / (na * nb)


def is_finite_vec(v) -> bool:
    try:
        return bool(v) and all(math.isfinite(float(x)) for x in v)
    except (TypeError, ValueError):
        return False


def slugify(name: str) -> str:
    s = re.sub(r"[^\w\s-]", "", name.strip().lower())
    s = re.sub(r"[\s_-]+", "-", s)
    return s or "person"


# --------------------------------------------------------------------------- #
# voiceprint store
# --------------------------------------------------------------------------- #
class VoiceStore:
    """name -> {centroid: [...], n_samples: int, dim: int, updated: ts}."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        if self.path.exists():
            self.data = json.loads(self.path.read_text())
        else:
            self.data = {"version": 1, "people": {}}

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data, indent=2))

    def names(self) -> list[str]:
        return sorted(self.data["people"].keys())

    def match(self, emb: list[float], high: float, low: float):
        """
        Returns (name, score, status):
          'match'      best cosine >= high          -> safe to auto-assign
          'ambiguous'  low <= best < high           -> ask the human
          'unknown'    best < low, or store empty    -> a new/unseen person
        """
        emb = l2norm([float(x) for x in emb])
        best_name, best_score = None, -1.0
        for name, rec in self.data["people"].items():
            score = cosine(emb, rec["centroid"])
            if score > best_score:
                best_name, best_score = name, score
        if best_name is None:
            return None, -1.0, "unknown"
        if best_score >= high:
            return best_name, best_score, "match"
        if best_score >= low:
            return best_name, best_score, "ambiguous"
        return best_name, best_score, "unknown"

    def enroll(self, name: str, emb: list[float]) -> None:
        """Fold a new embedding into the person's running-average centroid."""
        emb = l2norm([float(x) for x in emb])
        rec = self.data["people"].get(name)
        if rec is None:
            self.data["people"][name] = {
                "centroid": emb, "n_samples": 1,
                "dim": len(emb), "updated": int(time.time()),
            }
            return
        n = rec["n_samples"]
        merged = [(c * n + e) / (n + 1) for c, e in zip(rec["centroid"], emb)]
        rec["centroid"] = l2norm(merged)
        rec["n_samples"] = n + 1
        rec["updated"] = int(time.time())


# --------------------------------------------------------------------------- #
# resolution: diarization labels -> real names
# --------------------------------------------------------------------------- #
def sample_utterances(segs: list[Segment], label: str, k: int = 3) -> list[str]:
    texts = [s.text for s in segs if s.speaker == label]
    texts.sort(key=len, reverse=True)
    return texts[:k]


def resolve_identities(segs: list[Segment], emb_map: dict, vstore: VoiceStore,
                       id_high: float, id_low: float,
                       enroll: bool = True) -> dict:
    """
    Match each diarized speaker's voiceprint against the store. Confident matches
    are auto-applied and reinforced; ambiguous/unknown are confirmed interactively
    (unless enroll=False or non-tty). Returns {diarization_label: real_name}.
    """
    label_to_name: dict[str, str] = {}
    if not emb_map:
        return label_to_name

    interactive = sys.stdin.isatty() and enroll
    pending = []
    for label, emb in emb_map.items():
        if not is_finite_vec(emb):
            continue
        name, score, status = vstore.match(emb, id_high, id_low)
        if status == "match":
            label_to_name[label] = name
            vstore.enroll(name, emb)
            log(f"identity: {label} -> {name} (auto, cos {score:.2f})")
        else:
            pending.append((label, emb, name, score, status))

    if pending and interactive:
        sys.stderr.write("\n— name the speakers (Enter to skip) —\n")
        for label, emb, suggestion, score, status in pending:
            hint = (f"  [maybe {suggestion}? cos {score:.2f}]"
                    if status == "ambiguous" and suggestion else "")
            print(f"\n{label}{hint}", file=sys.stderr)
            for t in sample_utterances(segs, label):
                print(f"   “{t[:90]}”", file=sys.stderr)
            try:
                ans = input(f"   name for {label}: ").strip()
            except EOFError:
                ans = ""
            if ans:
                label_to_name[label] = ans
                vstore.enroll(ans, emb)
                log(f"identity: enrolled {label} -> {ans}")
    elif pending:
        for label, _emb, suggestion, score, status in pending:
            log(f"identity: {label} unresolved ({status}, best guess "
                f"{suggestion} cos {score:.2f}); left as {label}")
    return label_to_name


def apply_identities(segs: list[Segment], label_to_name: dict) -> None:
    for seg in segs:
        if seg.speaker in label_to_name:
            seg.speaker = label_to_name[seg.speaker]


def raw_utterances_for(segs: list[Segment], name: str) -> list[dict]:
    return [{"start": s.start, "end": s.end, "text": s.text}
            for s in segs if s.speaker == name]
