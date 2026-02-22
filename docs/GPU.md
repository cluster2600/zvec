# GPU Acceleration Guide

This guide explains how to use GPU acceleration with zvec on Apple Silicon (M-series) and other platforms.

## Overview

zvec supports GPU acceleration through multiple backends:
- **Apple Silicon (M1/M2/M3/M4)**: FAISS CPU (optimized), Metal MPS (future)
- **NVIDIA GPU**: CUDA via FAISS
- **AMD GPU**: ROCm via FAISS
- **CPU Fallback**: FAISS CPU (always available)

## Quick Start

```python
from zvec.gpu import GPUBackend, get_optimal_backend, get_gpu_info

# Check what's available
info = get_gpu_info()
print(f"Platform: {info['platform']}")
print(f"Backend: {info['selected']}")

# Get optimal backend
backend = get_optimal_backend()  # "faiss-cpu", "mps", "cuda", or "none"

# Create GPU-accelerated backend
gpu = GPUBackend(backend="auto")  # or specify "cuda", "faiss-cpu"
```

## GPU Information

```python
from zvec.gpu import get_gpu_info

info = get_gpu_info()
print(info)
```

Example output on Apple Silicon:
```python
{
    'platform': 'Darwin',
    'machine': 'arm64',
    'is_apple_silicon': True,
    'backends': {
        'faiss': True,
        'torch': False,
        'torch_mps': False,
        'cuda': False
    },
    'selected': 'faiss-cpu',
    'available': True
}
```

## Creating GPU Index

```python
import numpy as np
from zvec.gpu import GPUBackend

# Create backend
gpu = GPUBackend()

# Create GPU-accelerated index
index = gpu.create_index(
    dim=128,           # Vector dimension
    metric="L2",       # Distance metric: "L2", "IP", "cosine"
    nlist=100          # Number of clusters
)

# Prepare data
vectors = np.random.rand(10000, 128).astype('float32')

# Train index
index.train(vectors)

# Add vectors
index.add(vectors)

# Search
query = np.random.rand(5, 128).astype('float32')
distances, indices = gpu.search(index, query, k=10)

print(f"Found {len(indices)} results")
```

## Performance

### Expected Performance (Apple Silicon M3)

| Operation | CPU Time |
|-----------|----------|
| Index build (10K vectors) | ~2-5s |
| Index build (1M vectors) | ~5-10min |
| Search (10K vectors) | ~5ms |
| Search (1M vectors) | ~50ms |

### Tips for Better Performance

1. **Use appropriate nlist**: For N vectors, use nlist = 4*sqrt(N)
2. **Train with enough data**: Minimum 100x nlist vectors
3. **Batch queries**: Search multiple queries at once
4. **Use IP for cosine**: For cosine similarity, use IP metric

## GPU Memory

On Apple Silicon, GPU and CPU share unified memory. FAISS will automatically manage memory.

```python
# For very large datasets, consider:
# 1. Reducing nprobe for faster search
# 2. Using smaller batch sizes
# 3. Using quantization (PQ)
```

## Future: Metal Performance Shaders

Future versions will support Apple Metal Performance Shaders (MPS) for even better performance on M-series chips.

```python
# This is coming soon!
from zvec.gpu import GPUBackend

gpu = GPUBackend(backend="mps")  # Not yet available
```

## Troubleshooting

### "FAISS not available"
Install FAISS:
```bash
pip install faiss-cpu
# or for GPU support:
pip install faiss-gpu
```

### Slow performance
- Ensure vectors are float32
- Train with representative data
- Increase nlist for larger datasets
- Use batch queries

### Memory issues
- Reduce batch size
- Use smaller nlist
- Consider quantization
