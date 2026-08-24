"""CUDA-event based decode-loop profiler.

Wall-clock ``time.time()`` around async GPU kernels measures CPU launch overhead,
not GPU work. This module records per-component GPU time with ``torch.cuda.Event``
spans (deferred to a single sync) and accumulates decode statistics needed for the
verification-work analysis:

  * ``total_verified``   = sum(q_len) over rounds (dense verify forward length);
  * ``total_accepted``   = sum(accept_len) over rounds;
  * ``total_proposed``   = sum(propose_len) over rounds;
  * ``useful_verification_ratio`` = total_accepted / total_verified.

Components: ``draft``, ``verify_forward``, ``collect``, ``refresh``, ``accept``,
``misc``. ``misc`` is the wall-clock remainder (bonus append, cache-length adjust,
policy update, CPU/sync gaps).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch

COMPONENTS = ("draft", "verify_forward", "collect", "refresh", "accept")


@dataclass
class DecodeProfile:
    """Aggregated decode timing (ms) and verification-work statistics."""

    draft_ms: float = 0.0
    verify_forward_ms: float = 0.0
    collect_ms: float = 0.0
    refresh_ms: float = 0.0
    accept_ms: float = 0.0
    misc_ms: float = 0.0
    total_ms: float = 0.0

    decode_rounds: int = 0
    total_proposed: int = 0
    total_accepted: int = 0
    total_verified: int = 0
    proposed_lengths: List[int] = field(default_factory=list)
    accept_lengths: List[int] = field(default_factory=list)
    q_lengths: List[int] = field(default_factory=list)

    @property
    def acceptance_rate(self) -> float:
        return self.total_accepted / self.total_proposed if self.total_proposed else 0.0

    @property
    def mean_accept_length(self) -> float:
        return self.total_accepted / self.decode_rounds if self.decode_rounds else 0.0

    @property
    def useful_verification_ratio(self) -> float:
        return self.total_accepted / self.total_verified if self.total_verified else 0.0

    @property
    def wasted_verify_ratio(self) -> float:
        return 1.0 - self.useful_verification_ratio

    def as_dict(self, prefix: str = "") -> Dict[str, float]:
        p = prefix + "_" if prefix else ""
        return {
            f"{p}draft_ms": self.draft_ms,
            f"{p}verify_forward_ms": self.verify_forward_ms,
            f"{p}collect_ms": self.collect_ms,
            f"{p}refresh_ms": self.refresh_ms,
            f"{p}accept_ms": self.accept_ms,
            f"{p}misc_ms": self.misc_ms,
            f"{p}total_ms": self.total_ms,
            f"{p}decode_rounds": self.decode_rounds,
            f"{p}total_proposed": self.total_proposed,
            f"{p}total_accepted": self.total_accepted,
            f"{p}total_verified": self.total_verified,
            f"{p}acceptance_rate": self.acceptance_rate,
            f"{p}mean_accept_length": self.mean_accept_length,
            f"{p}useful_verification_ratio": self.useful_verification_ratio,
            f"{p}wasted_verify_ratio": self.wasted_verify_ratio,
        }


class DecodeProfiler:
    """Record per-component GPU-time spans and per-round statistics."""

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self._open: Dict[str, torch.cuda.Event] = {}
        self._spans: List[Tuple[str, torch.cuda.Event, torch.cuda.Event]] = []
        self.profile = DecodeProfile()

    def start(self, name: str) -> None:
        if not self.enabled:
            return
        if name in self._open:
            raise RuntimeError(f"Component {name!r} already started.")
        ev = torch.cuda.Event(enable_timing=True)
        ev.record()
        self._open[name] = ev

    def stop(self, name: str) -> None:
        if not self.enabled:
            return
        if name not in self._open:
            raise RuntimeError(f"Component {name!r} was not started.")
        ev = torch.cuda.Event(enable_timing=True)
        ev.record()
        self._spans.append((name, self._open.pop(name), ev))

    def record_round(self, propose_len: int, accept_len: int, q_len: int) -> None:
        if not self.enabled:
            return
        self.profile.decode_rounds += 1
        self.profile.total_proposed += propose_len
        self.profile.total_accepted += accept_len
        self.profile.total_verified += q_len
        self.profile.proposed_lengths.append(propose_len)
        self.profile.accept_lengths.append(accept_len)
        self.profile.q_lengths.append(q_len)

    def finalize(self, total_wall_ms: float) -> DecodeProfile:
        """Sync once, sum CUDA-event spans, and fill the misc remainder."""
        if not self.enabled:
            return self.profile
        torch.cuda.synchronize()
        acc: Dict[str, float] = {}
        for name, s, e in self._spans:
            acc[name] = acc.get(name, 0.0) + s.elapsed_time(e)
        self.profile.draft_ms = acc.get("draft", 0.0)
        self.profile.verify_forward_ms = acc.get("verify_forward", 0.0)
        self.profile.collect_ms = acc.get("collect", 0.0)
        self.profile.refresh_ms = acc.get("refresh", 0.0)
        self.profile.accept_ms = acc.get("accept", 0.0)
        self.profile.total_ms = total_wall_ms
        measured = (
            self.profile.draft_ms
            + self.profile.verify_forward_ms
            + self.profile.collect_ms
            + self.profile.refresh_ms
            + self.profile.accept_ms
        )
        self.profile.misc_ms = max(0.0, total_wall_ms - measured)
        return self.profile
