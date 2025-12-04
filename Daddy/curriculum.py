"""
Curriculum manager for savestate-based training.
"""

from __future__ import annotations

import random
from collections import defaultdict, deque
from pathlib import Path
from typing import Deque, Dict, Iterable, List, Optional


class CurriculumManager:
    """
    Tracks per-savestate success rates and samples states for each env.
    """

    def __init__(
        self,
        savestates: Optional[Iterable[Path]] = None,
        window: int = 50,
        promotion_threshold: float = 0.8,
        demotion_threshold: float = 0.2,
        max_total: int = 0,
        max_per_prefix: int = 0,
    ) -> None:
        self.savestates: List[Path] = list(savestates or [])
        self.history: Dict[str, Deque[bool]] = defaultdict(lambda: deque(maxlen=window))
        self.promotion_threshold = promotion_threshold
        self.demotion_threshold = demotion_threshold
        # safeguards; 0 disables
        self.max_total = max_total
        self.max_per_prefix = max_per_prefix

    def record(self, state_path: Path, success: bool) -> None:
        self.history[state_path.name].append(success)

    def success_rate(self, state_path: Path) -> float:
        hist = self.history[state_path.name]
        if not hist:
            return 0.5
        return sum(hist) / len(hist)

    def sample_state(self) -> Optional[Path]:
        if not self.savestates:
            return None
        weights = []
        for path in self.savestates:
            rate = self.success_rate(path)
            # favor states with mid success rate to encourage progression
            weights.append(max(0.05, 1.0 - abs(rate - 0.5)))
        total = sum(weights)
        probs = [w / total for w in weights]
        return random.choices(self.savestates, weights=probs, k=1)[0]

    def promote_or_demote(self) -> None:
        """
        Simple curriculum shaping: remove mastered states and keep challenging ones.
        """
        remaining: List[Path] = []
        for path in list(self.savestates):
            rate = self.success_rate(path)
            if rate > self.promotion_threshold:
                self._drop_path(path)
                continue
            remaining.append(path)
        self.savestates = remaining or self.savestates

    def add_state(self, path: Path) -> None:
        if path not in self.savestates:
            self.savestates.append(path)
            self._enforce_limits()

    def summary(self) -> str:
        """Human-friendly summary for terminal logs."""
        total = len(self.savestates)
        if total == 0:
            return "savestates=0"
        return f"savestates={total}"

    # ---------------- internal helpers ---------------- #
    def _prefix(self, path: Path) -> str:
        # use stem before first underscore as a lightweight category (mapXXX, badgeYY, epZZZ)
        stem = path.stem
        return stem.split("_")[0] if "_" in stem else stem

    def _drop_path(self, path: Path) -> None:
        if path in self.savestates:
            self.savestates.remove(path)
        self.history.pop(path.name, None)
        try:
            path.unlink(missing_ok=True)
        except Exception:
            pass

    def _enforce_limits(self) -> None:
        """Apply optional caps; removes oldest first to keep IO light."""
        if self.max_total and len(self.savestates) > self.max_total:
            # remove oldest overall
            sorted_paths = sorted(self.savestates, key=lambda p: p.stat().st_mtime if p.exists() else 0)
            for path in sorted_paths[: len(self.savestates) - self.max_total]:
                self._drop_path(path)
        if self.max_per_prefix and self.savestates:
            buckets: Dict[str, List[Path]] = defaultdict(list)
            for p in self.savestates:
                buckets[self._prefix(p)].append(p)
            for prefix, paths in buckets.items():
                if len(paths) > self.max_per_prefix:
                    to_drop = sorted(paths, key=lambda p: p.stat().st_mtime if p.exists() else 0)[
                        : len(paths) - self.max_per_prefix
                    ]
                    for p in to_drop:
                        self._drop_path(p)
