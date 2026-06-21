"""A representative workload: a mix of easy/hard requests, with duplicates.

Duplicates make the cache matter; the easy/hard mix makes routing matter; both
together let the optimised pipeline beat the baseline on cost and latency while
keeping every answer correct.
"""

from __future__ import annotations

from .models import Request


def build_workload() -> list[Request]:
    base = [
        Request("q1", "What is 2 + 2?", "4", "easy"),
        Request("q2", "Capital of France?", "Paris", "easy"),
        Request("q3", "Summarize the geopolitical causes of World War I.",
                "A complex multi-factor summary.", "hard"),
        Request("q4", "Is the sky blue?", "Yes", "easy"),
        Request("q5", "Prove the infinitude of primes.",
                "Euclid's proof by contradiction.", "hard"),
        Request("q6", "Translate 'hello' to Spanish.", "hola", "easy"),
    ]
    # Add duplicates of popular easy queries (cache will absorb these).
    dupes = [
        Request("q7", "What is 2 + 2?", "4", "easy"),       # dup of q1
        Request("q8", "Capital of France?", "Paris", "easy"),  # dup of q2
        Request("q9", "what is 2 + 2?", "4", "easy"),        # dup of q1 (different case)
        Request("q10", "Is the sky blue?", "Yes", "easy"),   # dup of q4
    ]
    return base + dupes
