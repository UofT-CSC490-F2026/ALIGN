"""Monotonic trial deadline shared by execution stages; cleanup has its own budget."""
from math import isfinite
from threading import Event
from time import monotonic


class TrialDeadline:
    def __init__(self, seconds=600):
        if isinstance(seconds, bool) or not isinstance(seconds, (int, float)) or not isfinite(seconds) or seconds <= 0:
            raise ValueError("Trial timeout must be finite and positive")
        self.end = monotonic() + seconds
        self.cancelled = Event()

    def remaining(self, cap=None):
        remaining = self.end - monotonic()
        if self.cancelled.is_set() or remaining <= 0:
            raise TimeoutError("Trial stopped or deadline exceeded")
        return min(remaining, cap) if cap is not None else remaining

    def cancel(self):
        self.cancelled.set()
