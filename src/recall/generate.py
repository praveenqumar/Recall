"""
recall.generate — shared text-generation engine.

Claude (`claude -p`, on the Max subscription) is primary; a local MLX model
(Qwen2.5-7B-4bit) is the offline fallback. make_generator() returns a single
generate(instructions, content, label) used by notes, personas, and reports —
honouring --notes-engine (auto | claude | local | none).

Guardrail: the paid API is never required. 'auto' uses Claude when present and
silently falls back to local when it isn't.
"""
from __future__ import annotations

import json
import subprocess
from typing import Callable, Optional

from .common import have, log
from .metrics import LiveStatus, Metrics

# the local model is loaded once and cached across calls (notes + N personas)
_LOCAL_LM: dict = {}

# running Claude token total across this run's notes/personas/reports
TOKENS = {"input": 0, "output": 0, "cost": 0.0}


def token_summary() -> Optional[str]:
    if not (TOKENS["input"] or TOKENS["output"]):
        return None
    cost = f", ${TOKENS['cost']:.4f}" if TOKENS["cost"] else ""
    return (f"Claude tokens: {TOKENS['input']:,} in + "
            f"{TOKENS['output']:,} out{cost}")


def _engine_claude(instructions: str, content: str, label: str,
                   metrics: Metrics, progress: bool) -> Optional[str]:
    if not have("claude"):
        return None
    with LiveStatus(f"{label} (claude -p)", metrics, progress):
        try:
            r = subprocess.run(
                ["claude", "-p", instructions, "--output-format", "json"],
                input=content, capture_output=True, text=True, timeout=600)
        except subprocess.TimeoutExpired:
            log(f"{label}: claude timed out")
            return None
    if r.returncode != 0 or not r.stdout.strip():
        log(f"{label}: claude failed (exit {r.returncode}); trying local")
        if r.stderr.strip():
            log(r.stderr.strip()[:400])
        return None
    try:
        d = json.loads(r.stdout)
    except json.JSONDecodeError:
        return r.stdout.strip()  # back-compat if --output-format ignored
    if d.get("is_error") or not (d.get("result") or "").strip():
        log(f"{label}: claude returned an error; trying local")
        log((d.get("result") or "")[:400])
        return None
    u = d.get("usage", {})
    inp = (u.get("input_tokens", 0) + u.get("cache_read_input_tokens", 0)
           + u.get("cache_creation_input_tokens", 0))
    out = u.get("output_tokens", 0)
    TOKENS["input"] += inp
    TOKENS["output"] += out
    TOKENS["cost"] += d.get("total_cost_usd") or 0.0
    log(f"{label}: {inp:,} in + {out:,} out tokens")
    return d["result"].strip()


def _engine_local(instructions: str, content: str, model: str, label: str,
                  metrics: Metrics, progress: bool) -> Optional[str]:
    try:
        from mlx_lm import load, generate as mlx_generate
    except ImportError:
        log(f"{label}: mlx-lm not installed (pip install mlx-lm)")
        return None
    with LiveStatus(f"{label} (local {model.split('/')[-1]})", metrics, progress):
        if model not in _LOCAL_LM:
            _LOCAL_LM[model] = load(model)
        model_obj, tokenizer = _LOCAL_LM[model]
        messages = [{"role": "system", "content": instructions},
                    {"role": "user", "content": content}]
        try:
            prompt = tokenizer.apply_chat_template(messages,
                                                   add_generation_prompt=True)
        except Exception:
            prompt = instructions + "\n\n" + content
        text = mlx_generate(model_obj, tokenizer, prompt=prompt,
                            max_tokens=2048, verbose=False)
    return text.strip() if text else None


def make_generator(engine: str, local_model: str, metrics: Metrics,
                   progress: bool) -> Callable[..., Optional[str]]:
    """Returns generate(instructions, content, label) honouring `engine`."""
    def generate(instructions: str, content: str, label: str = "notes"):
        out = None
        if engine in ("auto", "claude"):
            out = _engine_claude(instructions, content, label, metrics, progress)
        if out is None and engine in ("auto", "local"):
            out = _engine_local(instructions, content, local_model, label,
                                metrics, progress)
        return out

    return generate
