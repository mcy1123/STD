"""Engine lifecycle guard: never leak an EngineCore process.

WHY THIS EXISTS
---------------
A previous run left `VLLM::EngineCore` (pid 1442533) alive holding 41 GiB on GPU2
for ~48 minutes. Cause: the shell driver was killed with SIGTERM while the python
child was mid-generation. vLLM's EngineCore runs in a separate process, so killing
the parent does NOT reap the engine -- and the parent never got to `shutdown()`.

Two independent defences, because either alone is insufficient:

  1. `try/finally: llm.shutdown()` around every engine's lifetime -- covers normal
     exits and exceptions raised in the script.
  2. SIGTERM/SIGINT handlers -- covers the driver being killed mid-run, which is
     exactly the case that leaked last time. A bare `finally` does NOT run on
     SIGTERM (the default disposition terminates the process immediately).

Usage
-----
    from std_vllm.engine_guard import engine_guard

    with engine_guard(LLM, dict(model=..., max_model_len=...)) as llm:
        ...                       # llm is shut down on every exit path
"""

from __future__ import annotations

import atexit
import signal
from contextlib import contextmanager
from typing import Any, Callable, Iterator

_ACTIVE: list[Any] = []
_INSTALLED = False


def _shutdown_all(*_args: Any) -> None:
    while _ACTIVE:
        eng = _ACTIVE.pop()
        try:
            eng.shutdown()
        except Exception:
            pass


def _install_signal_handlers() -> None:
    """Make SIGTERM/SIGINT shut engines down before dying."""
    global _INSTALLED
    if _INSTALLED:
        return
    _INSTALLED = True
    atexit.register(_shutdown_all)

    for sig in (signal.SIGTERM, signal.SIGINT):
        prev = signal.getsignal(sig)

        def handler(signum, frame, _prev=prev):  # noqa: ANN001
            _shutdown_all()
            # Restore default behaviour and re-raise so the exit status is honest.
            try:
                signal.signal(signum, signal.SIG_DFL)
            except Exception:
                pass
            if callable(_prev) and _prev not in (signal.SIG_DFL, signal.SIG_IGN):
                try:
                    _prev(signum, frame)
                    return
                except Exception:
                    pass
            signal.raise_signal(signum)

        try:
            signal.signal(sig, handler)
        except (ValueError, OSError):
            # Not on the main thread (e.g. inside a worker) -- atexit still covers us.
            pass


def guard_engine(llm: Any) -> Any:
    """Register an already-constructed engine for shutdown on every exit path."""
    _install_signal_handlers()
    _ACTIVE.append(llm)
    return llm


def shutdown_all() -> None:
    """Public alias: shut down every registered engine (idempotent)."""
    _shutdown_all()


@contextmanager
def engine_guard(ctor: Callable[..., Any], kwargs: dict) -> Iterator[Any]:
    """Construct an engine and guarantee `shutdown()` on every exit path."""
    _install_signal_handlers()
    llm = ctor(**kwargs)
    _ACTIVE.append(llm)
    try:
        yield llm
    finally:
        try:
            _ACTIVE.remove(llm)
        except ValueError:
            pass
        try:
            llm.shutdown()
        except Exception as exc:  # pragma: no cover - shutdown must never mask errors
            print(f"WARNING: engine shutdown raised {type(exc).__name__}: {exc}",
                  flush=True)
