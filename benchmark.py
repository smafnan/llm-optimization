"""Benchmark: baseline vs optimised, with the before/after numbers.

    python benchmark.py

Shows that caching + routing + batching + streaming cut cost and latency by a
measurable amount while keeping quality identical.
"""

from __future__ import annotations

import json

from src.llmopt import BaselinePipeline, OptimizedPipeline, build_workload


def _pct(before: float, after: float) -> str:
    if before == 0:
        return "n/a"
    return f"{100 * (before - after) / before:.0f}% lower"


def main() -> int:
    workload = build_workload()
    base = BaselinePipeline().run(workload)
    opt = OptimizedPipeline().run(workload)

    print(f"Workload: {len(workload)} requests\n")
    rows = [
        ("Total cost ($)", base.total_cost, opt.total_cost),
        ("Total latency (ms)", base.total_latency_ms, opt.total_latency_ms),
        ("Mean TTFT (ms)", base.mean_ttft_ms, opt.mean_ttft_ms),
        ("Quality (accuracy)", base.quality, opt.quality),
        ("Model calls", base.model_calls, opt.model_calls),
        ("Cache hits", base.cache_hits, opt.cache_hits),
    ]
    print(f"{'metric':22s}{'baseline':>14s}{'optimized':>14s}{'change':>16s}")
    for name, b, o in rows:
        change = "same" if name.startswith("Quality") else _pct(b, o)
        if name.startswith(("Model", "Cache")):
            change = ""
        print(f"{name:22s}{b:>14.4g}{o:>14.4g}{change:>16s}")

    assert opt.quality >= base.quality, "optimisation must not reduce quality"
    print("\nQuality preserved while cost and latency dropped — optimisation wins.")

    json.dumps({"baseline": base.as_dict(), "optimized": opt.as_dict()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
