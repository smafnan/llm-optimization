"""Simulated LLM models with deterministic cost and latency.

To make cost/latency optimisation *measurable and reproducible offline*, we model
an LLM as a deterministic function with:

  * a **cost per call** (proxy for token pricing),
  * a **compute latency** (time to generate),
  * a **competence** — which difficulty levels it answers correctly.

A real system has a strong/expensive model and a cheap/fast one. The cheap model
is right on *easy* items but wrong on *hard* ones — which is exactly why naive
"use the big model for everything" is wasteful, and why routing can cut cost
*without* losing quality. No real sleeping: latency is returned as a number so
the benchmark is instant and deterministic.
"""

from __future__ import annotations

from dataclasses import dataclass

# Fixed per-call network/overhead latency (ms), paid once per API call. Batching
# multiple prompts into one call pays this only once — that's the batching win.
CALL_OVERHEAD_MS = 120.0


@dataclass(frozen=True)
class Request:
    id: str
    prompt: str
    gold: str                 # the correct answer (for quality scoring)
    difficulty: str           # "easy" or "hard"


@dataclass
class Response:
    request_id: str
    text: str
    model: str
    cost: float
    compute_ms: float         # generation time, excluding call overhead
    cached: bool = False

    def correct(self, gold: str) -> bool:
        return self.text == gold


@dataclass(frozen=True)
class Model:
    name: str
    cost_per_call: float
    compute_ms: float
    solves: frozenset[str]    # difficulties it answers correctly
    first_token_ms: float = 20.0   # time to first token (for streaming TTFT)

    def run(self, req: Request) -> Response:
        text = req.gold if req.difficulty in self.solves else "I'm not sure."
        return Response(req.id, text, self.name, self.cost_per_call, self.compute_ms)


# A strong, slow, expensive model vs a cheap, fast one that only nails easy items.
BIG = Model(name="big", cost_per_call=0.010, compute_ms=700.0,
            solves=frozenset({"easy", "hard"}))
SMALL = Model(name="small", cost_per_call=0.001, compute_ms=120.0,
              solves=frozenset({"easy"}))
