#!/usr/bin/env python3
"""
Benchmark script using public ANN datasets.

Downloads and tests with standard vector search datasets:
- SIFT (128D, 1M vectors)
- GIST (960D, 1M vectors)
- GloVe (100D, 1.2M vectors)
- DEEP1B (96D, 1B vectors - optional)

Usage:
    python benchmark_datasets.py
"""

import os
import sys
import h5py
import numpy as np
import time
import urllib.request
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))

from zvec.gpu import search_faiss, search_numpy

DATASETS = {
    "sift-128-euclidean": {
        "url": "http://ann-benchmarks.com/sift-128-euclidean.h5",
        "dim": 128,
        "train_size": 100000,
        "test_size": 10000,
    },
    "glove-100-angular": {
        "url": "http://ann-benchmarks.com/glove-100-angular.h5",
        "dim": 100,
        "train_size": 100000,
        "test_size": 5000,
    },
    "nytimes-256-angular": {
        "url": "http://ann-benchmarks.com/nytimes-256-angular.h5",
        "dim": 256,
        "train_size": 100000,
        "test_size": 5000,
    },
}


def download_dataset(name: str, data_dir: Path) -> Path:
    """Download dataset if not exists."""
    path = data_dir / f"{name}.h5"
    if path.exists():
        print(f"  Using cached: {path.name}")
        return path

    info = DATASETS[name]
    url = info["url"]

    print(f"  Downloading {name}...")
    print(f"  URL: {url}")

    try:
        urllib.request.urlretrieve(url, path)
        print(f"  Downloaded: {path.stat().st_size / 1024 / 1024:.1f} MB")
        return path
    except Exception as e:
        print(f"  Error: {e}")
        return None


def load_dataset(path: Path, name: str):
    """Load dataset from HDF5 file."""
    info = DATASETS[name]

    with h5py.File(path, "r") as f:
        print(f"  Keys: {list(f.keys())}")

        # Try different possible key names
        for key in ["train", "test", "base", "neighbors"]:
            if key in f:
                data = f[key]
                print(f"  {key}: {data.shape}, {data.dtype}")

        # Get test data
        if "test" in f:
            queries = f["test"][: info["test_size"]]
        elif "queries" in f:
            queries = f["queries"][: info["test_size"]]
        else:
            queries = None

        # Get train/base data
        if "train" in f:
            database = f["train"][: info["train_size"]]
        elif "base" in f:
            database = f["base"][: info["train_size"]]
        else:
            database = None

        # Get ground truth if available
        neighbors = None
        if "neighbors" in f:
            neighbors = f["neighbors"][: info["test_size"], :10]

        return queries, database, neighbors


def run_benchmark(name: str, queries, database, k: int = 10):
    """Run benchmark on dataset."""
    print(f"\n{'=' * 60}")
    print(f"Benchmark: {name}")
    print(f"  Database: {database.shape}")
    print(f"  Queries: {queries.shape}")
    print(f"  k: {k}")
    print(f"{'=' * 60}")

    # NumPy benchmark
    print(f"\n--- NumPy (Accelerate) ---")
    start = time.perf_counter()
    distances, indices = search_numpy(queries, database, k=k)
    numpy_time = time.perf_counter() - start
    print(f"  Time: {numpy_time:.3f}s ({numpy_time * 1000 / len(queries):.2f}ms/query)")

    # FAISS benchmark
    print(f"\n--- FAISS ---")
    start = time.perf_counter()
    distances_faiss, indices_faiss = search_faiss(queries, database, k=k)
    faiss_time = time.perf_counter() - start
    print(f"  Time: {faiss_time:.3f}s ({faiss_time * 1000 / len(queries):.2f}ms/query)")

    # Compare results
    match_rate = np.mean(indices == indices_faiss)
    print(f"\n--- Comparison ---")
    print(f"  NumPy: {numpy_time * 1000:.1f}ms")
    print(f"  FAISS: {faiss_time * 1000:.1f}ms")
    print(f"  Speedup: {numpy_time / faiss_time:.1f}x")
    print(f"  Match: {match_rate * 100:.1f}%")

    return {
        "numpy_ms": numpy_time * 1000 / len(queries),
        "faiss_ms": faiss_time * 1000 / len(queries),
        "speedup": numpy_time / faiss_time,
    }


def main():
    data_dir = Path.home() / ".cache" / "zvec_benchmarks"
    data_dir.mkdir(parents=True, exist_ok=True)

    results = []

    for name in DATASETS.keys():
        print(f"\n{'#' * 60}")
        print(f"# Dataset: {name}")
        print(f"{'#' * 60}")

        # Download
        path = download_dataset(name, data_dir)
        if not path:
            print(f"  Skipping {name}")
            continue

        # Load
        queries, database, neighbors = load_dataset(path, name)
        if queries is None or database is None:
            print(f"  Could not load data from {name}")
            continue

        # Run benchmark
        result = run_benchmark(name, queries, database, k=10)
        results.append((name, result))

    # Summary
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    print(f"{'Dataset':<30} {'NumPy (ms/q)':<15} {'FAISS (ms/q)':<15} {'Speedup':<10}")
    print("-" * 70)

    for name, result in results:
        print(
            f"{name:<30} {result['numpy_ms']:<15.2f} {result['faiss_ms']:<15.2f} {result['speedup']:<10.1f}x"
        )


if __name__ == "__main__":
    main()
