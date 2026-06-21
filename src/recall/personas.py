"""
recall.personas — per-person living collaboration profiles.

PersonaStore keeps, per person, an append-only raw-utterance log and a Markdown
profile.md that an LLM folds new meetings into over time. build_personas() runs
the per-identified-person update loop.

Profiles are framed as evidence-backed observed patterns + a hedged tentative
read (enforced by prompts/persona.md), never fixed trait scores. All data is
local biometric/personal data about colleagues — keep it on disk only.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Optional

from .common import Segment, log
from .identity import raw_utterances_for, slugify


class PersonaStore:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    def _dir(self, name: str) -> Path:
        d = self.root / slugify(name)
        d.mkdir(parents=True, exist_ok=True)
        return d

    def add_utterances(self, name: str, meeting: str,
                       utterances: list[dict]) -> None:
        """utterances: [{start,end,text}] — stored raw, pre-romanization."""
        if not utterances:
            return
        log_path = self._dir(name) / "utterances.jsonl"
        with log_path.open("a", encoding="utf-8") as f:
            for u in utterances:
                f.write(json.dumps({"date": meeting, **u},
                                   ensure_ascii=False) + "\n")

    def profile_path(self, name: str) -> Path:
        return self._dir(name) / "profile.md"

    def read_profile(self, name: str) -> str:
        p = self.profile_path(name)
        return p.read_text(encoding="utf-8") if p.exists() else ""

    def write_profile(self, name: str, text: str) -> None:
        self.profile_path(name).write_text(text, encoding="utf-8")

    def update_profile(self, name: str, meeting: str, utterances: list[dict],
                       instructions: str,
                       generate: Callable[[str, str], Optional[str]]) -> bool:
        """Fold this meeting's raw utterances into the person's living profile."""
        if not utterances:
            return False
        existing = self.read_profile(name) or "(no profile yet)"
        spoken = "\n".join(f"[{u.get('start', 0):.0f}s] {u['text']}"
                           for u in utterances)
        content = (
            f"PERSON: {name}\nMEETING DATE: {meeting}\n\n"
            f"=== EXISTING PROFILE ===\n{existing}\n\n"
            f"=== THIS MEETING'S RAW UTTERANCES (Hinglish, ASR errors likely) ===\n"
            f"{spoken}\n")
        updated = generate(instructions, content)
        if updated and updated.strip():
            self.write_profile(name, updated.strip() + "\n")
            return True
        return False


def build_personas(segs: list[Segment], names: list[str], meeting_id: str,
                   pstore: PersonaStore, persona_prompt: str,
                   generate: Callable[[str, str, str], Optional[str]]) -> None:
    """Capture raw utterances + update each identified person's profile."""
    for name in names:
        utts = raw_utterances_for(segs, name)
        pstore.add_utterances(name, meeting_id, utts)
        ok = pstore.update_profile(
            name, meeting_id, utts, persona_prompt,
            lambda instr, content, n=name: generate(instr, content, f"persona:{n}"))
        log(f"persona: {name} {'updated' if ok else 'logged (no LLM)'}")
