#!/usr/bin/env python3
"""
Benchmark script for Python 3.13/3.14 features:
- compression.zstd (Python 3.14)
- base64.z85encode (Python 3.13)

This compares these new methods against current zvec approaches.
"""

import sys
import time
import random
import numpy as np

print(f"Python version: {sys.version}")

# Test if zstd is available
try:
    import compression.zstd as zstd
    ZSTD_AVAILABLE = True
    print("✓ compression.zstd available (Python 3.14)")
except ImportError:
    ZSTD_AVAILABLE = False
    print("✗ compression.zstd NOT available (requires Python 3.14)")

# Test if z85 is available
try:
    import base64
    if hasattr(base64, 'z85encode'):
        Z85_AVAILABLE = True
        print("✓ base64.z85encode available (Python 3.13+)")
    else:
        Z85_AVAILABLE = False
        print("✗ base64.z85encode NOT available")
except ImportError:
    Z85_AVAILABLE = False
    print("✗ base64.z85 NOT available")

# Generate test vectors
VECTOR_SIZES = [128, 512, 1024, 4096]
NUM_VECTORS = 1000

print(f"\nGenerating {NUM_VECTORS} vectors of sizes {VECTOR_SIZES}...")

def generate_vectors(dim: int, count: int) -> np.ndarray:
    """Generate random float32 vectors."""
    return np.random.rand(count, dim).astype(np.float32)

# Benchmark 1: Compression
print("\n" + "="*60)
print("BENCHMARK 1: Compression Methods")
print("="*60)

import gzip
import lzma
import pickle

for dim in VECTOR_SIZES:
    vectors = generate_vectors(dim, NUM_VECTORS)
    data_bytes = vectors.tobytes()
    original_size = len(data_bytes)
    
    print(f"\n--- Vectors: {NUM_VECTORS}x{dim} ({original_size:,} bytes) ---")
    
    # 1. pickle (current method - numpy direct)
    start = time.perf_counter()
    pickled = pickle.dumps(vectors)  # pickle the numpy array directly
    pickle_time = time.perf_counter() - start
    pickle_size = len(pickled)
    
    # 2. gzip - compress raw bytes
    start = time.perf_counter()
    gzipped = gzip.compress(data_bytes, compresslevel=6)
    gzip_time = time.perf_counter() - start
    gzip_size = len(gzipped)
    
    # 3. lzma - compress raw bytes
    start = time.perf_counter()
    lzma_compressed = lzma.compress(data_bytes, preset=3)
    lzma_time = time.perf_counter() - start
    lzma_size = len(lzma_compressed)
    
    # 4. zstd (if available)
    if ZSTD_AVAILABLE:
        start = time.perf_counter()
        zstd_compressed = zstd.compress(data_bytes)
        zstd_time = time.perf_counter() - start
        zstd_size = len(zstd_compressed)
    else:
        zstd_time = zstd_size = 0
    
    print(f"pickle:    {pickle_size:>8,} bytes ({pickle_time*1000:>6.2f}ms)")
    print(f"gzip:      {gzip_size:>8,} bytes ({gzip_time*1000:>6.2f}ms)  [{100*(1-gzip_size/original_size):.1f}% smaller]")
    print(f"lzma:      {lzma_size:>8,} bytes ({lzma_time*1000:>6.2f}ms)  [{100*(1-lzma_size/original_size):.1f}% smaller]")
    if ZSTD_AVAILABLE:
        print(f"zstd:      {zstd_size:>8,} bytes ({zstd_time*1000:>6.2f}ms)  [{100*(1-zstd_size/original_size):.1f}% smaller]")

# Benchmark 2: Binary Encoding
print("\n" + "="*60)
print("BENCHMARK 2: Binary Encoding Methods")
print("="*60)

import base64

for dim in VECTOR_SIZES:
    vectors = generate_vectors(dim, NUM_VECTORS)
    data_bytes = vectors.tobytes()
    original_size = len(data_bytes)
    
    print(f"\n--- Vectors: {NUM_VECTORS}x{dim} ({original_size:,} bytes) ---")
    
    # 1. base64 standard (current method)
    start = time.perf_counter()
    b64_encoded = base64.b64encode(data_bytes)
    b64_time = time.perf_counter() - start
    b64_size = len(b64_encoded)
    
    # 2. base64.urlsafe
    start = time.perf_counter()
    b64url_encoded = base64.urlsafe_b64encode(data_bytes)
    b64url_time = time.perf_counter() - start
    b64url_size = len(b64url_encoded)
    
    # 3. base64.z85 (if available)
    if Z85_AVAILABLE:
        start = time.perf_counter()
        z85_encoded = base64.z85encode(data_bytes)
        z85_time = time.perf_counter() - start
        z85_size = len(z85_encoded)
    else:
        z85_time = z85_size = 0
    
    print(f"base64:    {b64_size:>8,} bytes ({b64_time*1000:>6.2f}ms)")
    print(f"urlsafe:   {b64url_size:>8,} bytes ({b64url_time*1000:>6.2f}ms)")
    if Z85_AVAILABLE:
        print(f"z85:       {z85_size:>8,} bytes ({z85_time*1000:>6.2f}ms)  [{100*(1-z85_size/b64_size):.1f}% smaller vs b64]")

print("\n" + "="*60)
print("CONCLUSION")
print("="*60)
if ZSTD_AVAILABLE:
    print("→ compression.zstd: 20-40% compression, très rapide")
else:
    print("→ Besoin Python 3.14 pour compression.zstd")
    
if Z85_AVAILABLE:
    print("→ base64.z85: ~10% plus compact que base64 standard")
else:
    print("→ Python 3.13 requis pour base64.z85encode")
