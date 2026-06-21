"""Pipelines and metrics — measure the optimisation, don't assume it.

Two pipelines process the same workload:

  * **BaselinePipeline** — one call to the big model per request. No cache, no
    routing, no batching, no streaming. Correct, but slow and expensive.
  * **OptimizedPipeline** — applies the four levers:
      1. **cache**     — duplicate prompts are free and instant.
      2. **routing**   — easy requests go to the cheap/fast model.
      3. **batching**  — requests sharing a model are sent in one call, so the
                          per-call overhead is paid once instead of N times.
      4. **streaming** — reported via time-to-first-token (perceived latency),
                          which streaming slashes even when total work is equal.

Both return :class:`RunMetrics` so the win is a number, and both compute a
**quality** score against the gold answers — the optimised pipeline must not lose
quality to count as a win.
"""

from __future__ import annotations

from dataclasses import dataclass

from .cache import ResponseCache
from .models import BIG, CALL_OVERHEAD_MS, Model, Request, Response
from .router import Router


@dataclass
class RunMetrics:
    total_cost: float
    total_latency_ms: float       # wall-clock to finish the whole workload
    mean_ttft_ms: float           # mean perceived latency (time to first token)
    quality: float                # accuracy vs gold (0..1)
    model_calls: int              # number of API calls made
    cache_hits: int

    def as_dict(self) -> dict:
        return {
            "total_cost": round(self.total_cost, 4),
            "total_latency_ms": round(self.total_latency_ms, 1),
            "mean_ttft_ms": round(self.mean_ttft_ms, 1),
            "quality": round(self.quality, 4),
            "model_calls": self.model_calls,
            "cache_hits": self.cache_hits,
        }


def _quality(responses: dict[str, Response], requests: list[Request]) -> float:
    if not requests:
        return 0.0
    correct = sum(1 for r in requests if responses[r.id].correct(r.gold))
    return correct / len(requests)


class BaselinePipeline:
    """One big-model call per request — no optimisations."""

    def __init__(self, model: Model = BIG) -> None:
        self.model = model

    def run(self, requests: list[Request]) -> RunMetrics:
        responses: dict[str, Response] = {}
        total_cost = 0.0
        total_latency = 0.0
        ttfts: list[float] = []
        for req in requests:
            resp = self.model.run(req)
            responses[req.id] = resp
            total_cost += resp.cost
            # One call each: overhead + full generation, sequentially.
            total_latency += CALL_OVERHEAD_MS + resp.compute_ms
            # No streaming: you wait for the whole response before seeing anything.
            ttfts.append(CALL_OVERHEAD_MS + resp.compute_ms)
        return RunMetrics(total_cost, total_latency,
                          sum(ttfts) / len(ttfts), _quality(responses, requests),
                          model_calls=len(requests), cache_hits=0)


class OptimizedPipeline:
    """Cache + routing + batching + streaming."""

    def __init__(self, router: Router | None = None) -> None:
        self.router = router or Router()
        self.cache = ResponseCache()

    def run(self, requests: list[Request]) -> RunMetrics:
        from .cache import _key

        responses: dict[str, Response] = {}
        total_cost = 0.0
        ttfts: list[float] = []

        # 1) Cache + within-run dedup. A prompt's FIRST occurrence (and only it)
        #    becomes a model call; cross-run cache hits and within-run duplicates
        #    are served for free. assignment maps each request to a prompt key.
        unique: dict[str, Request] = {}     # prompt key -> the request to call
        assignment: dict[str, str] = {}     # request id -> prompt key (or "__cache__")
        for req in requests:
            cached = self.cache.get(req.prompt)     # persistent cross-run cache
            if cached is not None:
                cached.request_id = req.id
                responses[req.id] = cached
                ttfts.append(cached.compute_ms)
                assignment[req.id] = "__cache__"
                continue
            key = _key(req.prompt)
            if key in unique:
                # Duplicate within this run -> reuse the upcoming call's answer.
                self.cache.hits += 1
                ttfts.append(2.0)                    # served for free, instant
            else:
                unique[key] = req
            assignment[req.id] = key

        # 2) Route each unique miss and group by model so we can batch.
        by_model: dict[str, tuple[Model, list[Request]]] = {}
        for req in unique.values():
            model = self.router.route(req)
            by_model.setdefault(model.name, (model, []))[1].append(req)

        # 3) One call per model (batching); 4) streaming TTFT per item.
        model_calls = 0
        batch_latencies: list[float] = []
        key_to_response: dict[str, Response] = {}
        for model, batch in by_model.values():
            model_calls += 1
            for req in batch:
                resp = model.run(req)
                key_to_response[_key(req.prompt)] = resp
                total_cost += resp.cost
                self.cache.put(req.prompt, resp)
                ttfts.append(CALL_OVERHEAD_MS + model.first_token_ms)
            batch_latencies.append(
                CALL_OVERHEAD_MS + sum(model.compute_ms for _ in batch)
            )

        # Assign every not-yet-resolved request its prompt's response (dups free).
        for req in requests:
            if req.id in responses:
                continue
            src = key_to_response[assignment[req.id]]
            responses[req.id] = Response(req.id, src.text, src.model,
                                         cost=0.0, compute_ms=src.compute_ms)

        total_latency = sum(batch_latencies)
        return RunMetrics(
            total_cost=total_cost,
            total_latency_ms=total_latency,
            mean_ttft_ms=sum(ttfts) / len(ttfts) if ttfts else 0.0,
            quality=_quality(responses, requests),
            model_calls=model_calls,
            cache_hits=self.cache.hits,
        )
