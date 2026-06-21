"""
scribe.notes — meeting notes + per-person tailored reports.

write_notes()   transcript -> notes.md (via the shared generate engine).
write_reports() for each --report-for NAME, a report whose FORMAT matches that
                person's persona profile while the FACTS stay strictly from the
                transcript.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

from .common import log
from .identity import slugify
from .personas import PersonaStore


def write_notes(transcript_md: str, instructions: str, out_dir: Path, stem: str,
                generate: Callable[..., Optional[str]]) -> Optional[Path]:
    notes = generate(instructions, transcript_md, "notes")
    if not notes:
        return None
    path = out_dir / f"{stem}.notes.md"
    path.write_text(notes + "\n")
    return path


def write_reports(report_for: list[str], transcript_md: str, report_prompt: str,
                  pstore: PersonaStore, out_dir: Path, stem: str,
                  generate: Callable[..., Optional[str]]) -> list[tuple[str, Path]]:
    out: list[tuple[str, Path]] = []
    for name in report_for:
        profile = pstore.read_profile(name)
        if not profile:
            log(f"report: no profile for '{name}' yet; using generic format")
        content = (f"=== RECIPIENT PROFILE: {name} ===\n"
                   f"{profile or '(no profile on file)'}\n\n"
                   f"=== MEETING TRANSCRIPT ===\n{transcript_md}")
        report = generate(report_prompt, content, f"report:{name}")
        if report:
            rp = out_dir / f"{stem}.report.{slugify(name)}.md"
            rp.write_text(report + "\n")
            out.append((f"Report for {name}", rp))
    return out
