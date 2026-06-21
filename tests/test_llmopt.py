"""Tests: each optimisation lever, and the end-to-end before/after win."""

from __future__ import annotations

from llmopt import (
    BIG,
    SMALL,
    BaselinePipeline,
    OptimizedPipeline,
    Request,
    ResponseCache,
    Router,
    build_workload,
)


# --- models ---------------------------------------------------------------- #

def test_small_model_fails_hard_items():
    easy = Request("e", "x", "ans", "easy")
    hard = Request("h", "y", "ans", "hard")
    assert SMALL.run(easy).correct("ans")
    assert not SMALL.run(hard).correct("ans")   # small can't do hard
    assert BIG.run(hard).correct("ans")         # big can


def test_small_is_cheaper_and_faster():
    assert SMALL.cost_per_call < BIG.cost_per_call
    assert SMALL.compute_ms < BIG.compute_ms


# --- cache ----------------------------------------------------------------- #

def test_cache_hit_is_free_and_marked():
    cache = ResponseCache()
    resp = BIG.run(Request("a", "Hello?", "hi", "easy"))
    cache.put("Hello?", resp)
    hit = cache.get("  hello?  ")              # normalised match
    assert hit is not None and hit.cached and hit.cost == 0.0
    assert cache.hits == 1


def test_cache_miss():
    cache = ResponseCache()
    assert cache.get("unseen") is None and cache.misses == 1


# --- router ---------------------------------------------------------------- #

def test_router_sends_easy_to_small_hard_to_big():
    r = Router()
    assert r.route(Request("a", "short", "x", "easy")).name == "small"
    assert r.route(Request("b", "x", "x", "hard")).name == "big"


def test_router_escalates_long_prompts():
    r = Router()
    long_prompt = " ".join(["word"] * 50)      # >40 words -> treated as hard
    assert r.route(Request("c", long_prompt, "x", "easy")).name == "big"


# --- end-to-end ------------------------------------------------------------ #

def test_optimized_is_cheaper():
    wl = build_workload()
    base = BaselinePipeline().run(wl)
    opt = OptimizedPipeline().run(wl)
    assert opt.total_cost < base.total_cost


def test_optimized_is_faster():
    wl = build_workload()
    base = BaselinePipeline().run(wl)
    opt = OptimizedPipeline().run(wl)
    assert opt.total_latency_ms < base.total_latency_ms
    assert opt.mean_ttft_ms < base.mean_ttft_ms          # streaming win


def test_quality_is_preserved():
    wl = build_workload()
    base = BaselinePipeline().run(wl)
    opt = OptimizedPipeline().run(wl)
    assert base.quality == 1.0
    assert opt.quality == base.quality                   # no quality lost


def test_cache_absorbs_duplicates():
    wl = build_workload()
    opt = OptimizedPipeline().run(wl)
    # The workload has 4 duplicate easy queries -> at least that many cache hits.
    assert opt.cache_hits >= 4


def test_batching_reduces_call_count():
    wl = build_workload()
    base = BaselinePipeline().run(wl)
    opt = OptimizedPipeline().run(wl)
    # Baseline makes one call per request; batching collapses misses to <= 2
    # calls (one per model).
    assert base.model_calls == len(wl)
    assert opt.model_calls <= 2


def test_meaningful_savings():
    wl = build_workload()
    base = BaselinePipeline().run(wl)
    opt = OptimizedPipeline().run(wl)
    cost_cut = (base.total_cost - opt.total_cost) / base.total_cost
    latency_cut = (base.total_latency_ms - opt.total_latency_ms) / base.total_latency_ms
    assert cost_cut > 0.5            # >50% cheaper
    assert latency_cut > 0.5         # >50% faster
