"""Minimal pytest-free runner for environments without pytest.

Emulates the `tmp_path` and `monkeypatch` fixtures and runs every test_* function
in the test modules. Exit code is non-zero if anything fails.
"""
import inspect
import sys
import tempfile
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import test_identity  # noqa: E402
import test_pipeline  # noqa: E402
import test_store  # noqa: E402


class MonkeyPatch:
    def __init__(self):
        self._undo = []

    def setattr(self, target, name, value=None):
        # support setattr(obj, "name", value)
        self._undo.append((target, name, getattr(target, name)))
        setattr(target, name, value)

    def undo(self):
        for target, name, old in reversed(self._undo):
            setattr(target, name, old)
        self._undo.clear()


def run_module(mod):
    passed = failed = 0
    for name, fn in sorted(inspect.getmembers(mod, inspect.isfunction)):
        if not name.startswith("test_"):
            continue
        if fn.__module__ != mod.__name__:
            continue
        params = inspect.signature(fn).parameters
        kwargs = {}
        mp = None
        tmp = None
        if "tmp_path" in params:
            tmp = Path(tempfile.mkdtemp(prefix="recalltest_"))
            kwargs["tmp_path"] = tmp
        if "monkeypatch" in params:
            mp = MonkeyPatch()
            kwargs["monkeypatch"] = mp
        try:
            fn(**kwargs)
            print(f"  PASS {mod.__name__}.{name}")
            passed += 1
        except Exception:
            print(f"  FAIL {mod.__name__}.{name}")
            traceback.print_exc()
            failed += 1
        finally:
            if mp:
                mp.undo()
    return passed, failed


def main():
    total_p = total_f = 0
    for mod in (test_identity, test_pipeline, test_store):
        print(f"\n== {mod.__name__} ==")
        p, f = run_module(mod)
        total_p += p
        total_f += f
    print(f"\n{'='*40}\n{total_p} passed, {total_f} failed")
    sys.exit(1 if total_f else 0)


if __name__ == "__main__":
    main()
