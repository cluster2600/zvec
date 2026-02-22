# Metal MPS (Apple Silicon) Guide

This guide explains how to use Metal Performance Shaders (MPS) for GPU acceleration on Apple Silicon (M1/M2/M1 Max/M4) chips.

## Overview

Metal Performance Shaders is Apple's GPU framework that provides high-performance compute kernels for M-series chips. zvec includes native MPS support for vector operations.

## Quick Start

```python
from zvec.mps import MPSBackend, is_mps_available

# Check if MPS is available
print(f"MPS available: {is_mps_available()}")

# Create MPS backend
mps = MPSBackend()
```

## Requirements

- Apple Silicon (M1, M2, M1 Max, or M4)
- macOS 12.3+
- PyTorch with MPS support

Install PyTorch:
```bash
pip install torch
```

## Usage

### Vector Search

```python
import numpy as np
from zvec.mps import MPSBackend

# Create backend
mps = MPSBackend()

# Your data
database = np.random.rand(10000, 128).astype(np.float32)
queries = np.random.rand(100, 128).astype(np.float32)

# GPU-accelerated search
distances, indices = mps.vector_search(
    queries, 
    database, 
    k=10, 
    metric="L2"  # or "cosine"
)
```

### Batch Distance

```python
# Compute pairwise distances
a = np.random.rand(1000, 256).astype(np.float32)
b = np.random.rand(500, 256).astype(np.float32)

distances = mps.batch_distance(a, b, metric="L2")
# Result: (1000, 500) distance matrix
```

### Matrix Multiplication

```python
# GPU-accelerated matrix multiply
a = np.random.rand(100, 500).astype(np.float32)
b = np.random.rand(500, 200).astype(np.float32)

result = mps.batch_matrix_multiply(a, b)
# Result: (100, 200)
```

## Performance

### Benchmark Results (M1 Max)

| Operation | Data Size | Time |
|-----------|-----------|------|
| Search | 1K × 128D | ~10ms |
| Search | 10K × 128D | ~15ms |
| Search | 100K × 128D | ~100ms |
| Search | 1K × 512D | ~15ms |
| Search | 10K × 512D | ~20ms |

### Tips for Better Performance

1. **Use float32**: MPS works best with float32
2. **Batch queries**: Search multiple queries at once
3. **Dimension**: Smaller dimensions are faster
4. **Warmup**: First call is slower (kernel compilation)

## API Reference

### MPSBackend

```python
from zvec.mps import MPSBackend

mps = MPSBackend(device=0)  # device is for future CUDA compatibility
```

#### Methods

- `vector_search(queries, database, k, metric)` - Search vectors
- `batch_distance(a, b, metric)` - Compute distance matrix
- `batch_matrix_multiply(a, b)` - Matrix multiplication
- `to_mps(array)` - Convert numpy to MPS tensor
- `to_numpy(tensor)` - Convert MPS tensor to numpy

### Functions

- `is_mps_available()` - Check MPS availability
- `get_mps_info()` - Get device information

## Integration with zvec

```python
# Future: Use MPS with zvec collections
import zvec

schema = zvec.CollectionSchema(
    name="vectors",
    vectors=zvec.VectorSchema("emb", dimension=128),
    backend="mps"  # Use MPS backend
)
```

## Troubleshooting

### "Metal Performance Shaders not available"

1. Ensure you're on Apple Silicon (M1/M2/M1 Max/M4)
2. Update macOS to 12.3+
3. Reinstall PyTorch: `pip install torch`

### Slow Performance

1. Use float32, not float64
2. Warm up with a small query first
3. Use batch operations

### Memory Issues

MPS uses unified memory. If you get memory errors:
1. Reduce batch size
2. Process in chunks
