# LLM Cost & Latency Optimization — Measured

> **AI Engineer Roadmap — Project 5.3**
> *Teaches: the constraints that dominate real production AI, profiling, tradeoff thinking.*
> *Done when: you cut cost or latency by a measurable amount without losing quality.*

Takes an LLM app and makes it **measurably cheaper and faster** with the four
levers that matter in production — **caching, model-tier routing, batching, and
streaming** — while keeping answer **quality identical**. Everything is simulated
deterministically (a mock LLM with per-call cost and latency), so the benchmark
is reproducible and runs instantly with no API key.

```bash
python -m venv .venv && source .venv/bin/activate   # Win: .\.venv\Scripts\activate
pip install -e ".[dev]"
python benchmark.py    # before/after table
pytest -q              # 12 tests
```

---

## The result (`python benchmark.py`)

A 10-request workload (mix of easy/hard, with duplicates):

| metric | baseline | optimized | change |
| --- | ---: | ---: | --- |
| **Total cost ($)** | 0.100 | 0.024 | **76% lower** |
| **Total latency (ms)** | 8200 | 2120 | **74% lower** |
| **Mean TTFT (ms)** | 820 | 85 | **90% lower** |
| **Quality (accuracy)** | 1.00 | 1.00 | **same** |
| Model calls | 10 | 2 | (batching) |
| Cache hits | 0 | 4 | (dedup) |

**Cost and latency cut by ~75% with zero quality loss** — the project's "Done
when", proven with numbers rather than asserted.

---

## The four levers

| Lever | Module | What it saves | How |
| --- | --- | --- | --- |
| **Caching** | `cache.py` | repeated work | duplicate prompts (normalised) return a stored answer for ~0 cost and ~0 latency |
| **Routing** | `router.py` | money on easy queries | a difficulty estimate sends easy requests to a cheap/fast model, escalating to the big model only when needed |
| **Batching** | `pipeline.py` | per-call overhead | requests sharing a model go out in one call, so fixed overhead is paid once instead of N times (10 calls → 2) |
| **Streaming** | `pipeline.py` | *perceived* latency | time-to-first-token drops 90% — the user sees output almost immediately even when total work is unchanged |

### Why quality is preserved

The cheap model is only right on *easy* items; the router sends *hard* items to
the big model. So every answer stays correct (accuracy 1.0) — the savings come
from not over-paying for the easy ones, not from accepting worse answers. That
distinction (cheaper ≠ worse, when routed well) is the whole point.

## How it's measured

`RunMetrics` records total cost, total latency, mean time-to-first-token,
**quality vs gold answers**, model-call count, and cache hits. Two pipelines run
the *same* workload:

- `BaselinePipeline` — one big-model call per request, no optimisations.
- `OptimizedPipeline` — cache → route → batch → stream.

A test asserts the optimised run is **>50% cheaper and >50% faster with equal
quality** — so a future change that breaks an optimisation (or trades away
quality) fails CI.

## Layout

```
src/llmopt/
├── models.py     # simulated BIG/SMALL models (cost, latency, competence)
├── cache.py      # normalised response cache
├── router.py     # difficulty-based model routing
├── pipeline.py   # Baseline vs Optimized pipelines + RunMetrics
└── workload.py   # a representative mixed workload with duplicates
benchmark.py       # before/after table
tests/             # 12 tests (each lever + end-to-end savings + quality)
```

## Notes

The mock models make cost/latency exact and the benchmark deterministic; the
*methodology* — measure first, route by difficulty, cache aggressively, batch to
amortise overhead, stream for perceived latency, and **gate on quality** — is
what transfers directly to a real LLM stack (swap the mock for actual API calls
with their real prices and measured latencies).

## License

MIT.
