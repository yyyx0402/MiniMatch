"""Measure cold loading, uncached searches, and cache hits on this Windows host."""
import json
import time

import numpy as np
from threadpoolctl import threadpool_limits

from productsearch.catalog_search import CatalogSearchService
from productsearch.data import ROOT
from productsearch.release import MANIFEST, sha256


def statistics(values):
    return {"samples": len(values), "median_ms": float(np.median(values)),
            "p95_ms": float(np.quantile(values, .95)), "max_ms": max(values)}


def main():
    started = time.perf_counter()
    service = CatalogSearchService()
    loaded = time.perf_counter()
    service.search("wireless mouse")
    first_done = time.perf_counter()
    queries = ["stainless steel water bottle", "comfortable running shoes", "usb c cable",
               "laptop backpack", "ceramic coffee mug", "desk lamp"]
    timings = {}
    for model in ("hybrid", "minilm", "keyword"):
        uncached, cached = [], []
        for query in queries:
            service.cache.clear()
            tick = time.perf_counter()
            service.search(query, model)
            uncached.append((time.perf_counter() - tick) * 1000)
            tick = time.perf_counter()
            service.search(query, model)
            cached.append((time.perf_counter() - tick) * 1000)
        timings[model] = {"uncached": statistics(uncached), "cached": statistics(cached)}
    report = {"manifest_sha256": sha256(ROOT / MANIFEST), "catalog_load_seconds": loaded - started,
              "first_query_including_catalog_load_seconds": first_done - started,
              "queries": queries, "cpu_threads": 4, "timings": timings,
              "note": "Single-process local service timings, 6 queries per mode; includes product formatting, excludes interpreter imports, HTTP transport and browser rendering. Not a concurrency load test."}
    try:
        import psutil
        report["working_set_mb"] = psutil.Process().memory_info().rss / (1024 ** 2)
    except ImportError:
        report["working_set_mb"] = None
    output = ROOT / "reports/application"
    output.mkdir(parents=True, exist_ok=True)
    (output / "performance.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=True, indent=2), flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=4):
        main()
