"""Response cache — the cheapest optimisation there is.

Many real workloads repeat prompts (same question, retries, popular queries). A
cache returns a stored answer for ~0 cost and negligible latency, so duplicate
work is never paid for twice. We key on the *normalised* prompt so trivial
differences (case, surrounding whitespace) still hit.
"""

from __future__ import annotations

from .models import Response

CACHE_HIT_MS = 2.0  # serving from cache is effectively instant


def _key(prompt: str) -> str:
    return " ".join(prompt.strip().lower().split())


class ResponseCache:
    def __init__(self) -> None:
        self._store: dict[str, Response] = {}
        self.hits = 0
        self.misses = 0

    def get(self, prompt: str) -> Response | None:
        hit = self._store.get(_key(prompt))
        if hit is None:
            self.misses += 1
            return None
        self.hits += 1
        # A cache hit costs nothing and is near-instant.
        return Response(hit.request_id, hit.text, hit.model, cost=0.0,
                        compute_ms=CACHE_HIT_MS, cached=True)

    def put(self, prompt: str, response: Response) -> None:
        self._store[_key(prompt)] = response
