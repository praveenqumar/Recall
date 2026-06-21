"""
recall.metrics — sudo-free resource readout + live progress UX.

Metrics: a cheap snapshot of RAM / system-mem% / CPU% / MLX-Metal GPU memory.
LiveStatus: a spinner + readout for stages with no natural progress %.
stage(): the per-stage header line.
"""
from __future__ import annotations

import itertools
import sys
import threading
import time
from typing import Optional


def _gb(n: float) -> float:
    return n / (1024 ** 3)


class Metrics:
    """Cheap, sudo-free snapshot of how hard the machine is working."""

    def __init__(self) -> None:
        try:
            import psutil
            self.psutil = psutil
            self.proc = psutil.Process()
            self.proc.cpu_percent(None)      # prime the counters
            psutil.cpu_percent(None)
        except Exception:
            self.psutil = None
            self.proc = None
        try:
            import mlx.core as mx
            self.mx = mx
        except Exception:
            self.mx = None

    def _mlx_active_gb(self) -> Optional[float]:
        if not self.mx:
            return None
        # API location shifted across MLX versions; prefer the current top-level
        # mx.get_active_memory, fall back to the deprecated mx.metal.* on old MLX.
        candidates = [
            (self.mx, "get_active_memory"),
            (getattr(self.mx, "metal", None), "get_active_memory"),
        ]
        for mod, attr in candidates:
            if mod is not None and hasattr(mod, attr):
                try:
                    return _gb(getattr(mod, attr)())
                except Exception:
                    pass
        return None

    def snapshot(self) -> str:
        parts: list[str] = []
        if self.psutil:
            try:
                rss = _gb(self.proc.memory_info().rss)
                parts.append(f"RAM {rss:.1f}G")
            except Exception:
                pass
            try:
                parts.append(f"sys {self.psutil.virtual_memory().percent:.0f}%")
            except Exception:
                pass
            try:
                parts.append(f"CPU {self.psutil.cpu_percent(None):.0f}%")
            except Exception:
                pass
        mlx_mem = self._mlx_active_gb()
        if mlx_mem is not None:
            parts.append(f"GPU {mlx_mem:.1f}G")
        return "  ".join(parts) if parts else "metrics n/a"


_SPIN = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"


class LiveStatus:
    def __init__(self, label: str, metrics: Metrics, enabled: bool = True) -> None:
        self.label = label
        self.metrics = metrics
        self.enabled = enabled and sys.stderr.isatty()
        self._stop = threading.Event()
        self._t: Optional[threading.Thread] = None
        self._start = 0.0

    def __enter__(self) -> "LiveStatus":
        self._start = time.time()
        if self.enabled:
            self._t = threading.Thread(target=self._run, daemon=True)
            self._t.start()
        else:
            print(f"[recall] {self.label} …", file=sys.stderr, flush=True)
        return self

    def _run(self) -> None:
        for ch in itertools.cycle(_SPIN):
            if self._stop.is_set():
                break
            el = time.time() - self._start
            line = f"\r  {ch} {self.label}  {el:5.1f}s  {self.metrics.snapshot()}"
            sys.stderr.write(line.ljust(80))
            sys.stderr.flush()
            time.sleep(0.4)

    def __exit__(self, *exc) -> None:
        self._stop.set()
        if self._t:
            self._t.join(timeout=1)
        el = time.time() - self._start
        if self.enabled:
            sys.stderr.write("\r" + " " * 90 + "\r")
        sys.stderr.write(f"  ✓ {self.label}  ({el:.1f}s)\n")
        sys.stderr.flush()


def stage(n: int, total: int, name: str) -> None:
    print(f"[recall] [{n}/{total}] {name}", file=sys.stderr, flush=True)
