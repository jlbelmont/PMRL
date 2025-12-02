"""
Curriculum manager for savestate-based training.
"""

from __future__ import annotations

import random
from collections import defaultdict, deque
from typing import Deque, Dict, List, Optional


class CurriculumManager:
    """
    Tracks per-savestate success rates and samples states for each env.
    """

    def __init__(
        self,
        savestates: Optional[List[str]] = None,
        window: int = 50,
        promotion_threshold: float = 0.8,
        demotion_threshold: float = 0.2,
    ) -> None:
        self.savestates = savestates or []
        self.history: Dict[str, Deque[bool]] = defaultdict(lambda: deque(maxlen=window))
        self.promotion_threshold = promotion_threshold
        self.demotion_threshold = demotion_threshold

    def record(self, state_name: str, success: bool) -> None:
        self.history[state_name].append(success)

    def success_rate(self, state_name: str) -> float:
        hist = self.history[state_name]
        if not hist:
            return 0.5
        return sum(hist) / len(hist)

    def sample_state(self) -> Optional[str]:
        if not self.savestates:
            return None
        weights = []
        for name in self.savestates:
            rate = self.success_rate(name)
            # favor states with mid success rate to encourage progression
            weights.append(max(0.05, 1.0 - abs(rate - 0.5)))
        total = sum(weights)
        probs = [w / total for w in weights]
        return random.choices(self.savestates, weights=probs, k=1)[0]

    def promote_or_demote(self) -> None:
        """
        Simple curriculum shaping: remove mastered states and keep challenging ones.
        """
        remaining = []
        for name in self.savestates:
            rate = self.success_rate(name)
            if rate > self.promotion_threshold:
                continue
            remaining.append(name)
        self.savestates = remaining or self.savestates
