"""Model router — send each request to the cheapest model that can handle it.

Using the expensive model for every request is the most common source of waste.
A router estimates how hard a request is and picks the smallest model likely to
get it right, escalating to the big model only when needed. Here the difficulty
estimate is explicit (the request carries it); in a real system you'd estimate it
from features like length, the presence of reasoning keywords, or a cheap
classifier.
"""

from __future__ import annotations

from collections.abc import Callable

from .models import BIG, SMALL, Model, Request

# A pluggable difficulty estimator. Default: trust the request's label, but also
# treat long prompts as hard (a realistic, content-based heuristic).
def default_estimator(req: Request) -> str:
    if req.difficulty == "hard" or len(req.prompt.split()) > 40:
        return "hard"
    return "easy"


class Router:
    def __init__(self, estimator: Callable[[Request], str] = default_estimator,
                 small: Model = SMALL, big: Model = BIG) -> None:
        self.estimator = estimator
        self.small = small
        self.big = big

    def route(self, req: Request) -> Model:
        return self.big if self.estimator(req) == "hard" else self.small
