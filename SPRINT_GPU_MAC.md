# Sprint: GPU Optimization for zvec (Internal)

## Objectif
Implémenter le support GPU pour zvec sur Mac (Apple Silicon / M-Series).

## Duration
2-3 jours

## Contexte
- Usage interne seulement (pas de PR upstream)
- Cible: Mac avec Apple Silicon (M1/M2/M3/M4)
- Pas de NVIDIA CUDA

## Approach

### Apple Silicon GPU Options
1. **Metal Performance Shaders (MPS)** - Apple's GPU framework
2. **OpenCL** - Cross-platform GPU compute
3. **FAISS with Metal** - Possible via custom indices

### Selected Approach: FAISS GPU
FAISS supporte déjà le calcul GPU via:
- CUDA (NVIDIA)
- **ROCm** (AMD) - peut être exploré

Pour Apple Silicon, on peut:
1. Utiliser FAISS CPU optimisé (still fast sur M-series)
2. Explorer Metal pour custom kernels
3. Utiliser Core ML pour inference

### Stratégie
1. Ajouter FAISS GPU comme optionnelle
2. Créer wrapper pour Apple Silicon
3. Benchmark CPU vs GPU sur Mac

---

## Tasks

### Day 1: Setup & Configuration

#### T1.1: Add FAISS GPU dependency
- [ ] Update pyproject.toml with faiss-gpu
- [ ] Add conditional import for GPU availability
- [ ] Create fallback to CPU if GPU not available

#### T1.2: Create GPU wrapper module
- [ ] Create `zvec/gpu.py`
- [ ] Detect Apple Silicon
- [ ] Auto-select optimal backend

### Day 2: Implementation

#### T2.1: GPU-accelerated indexing
- [ ] Add GPU index options to schema
- [ ] Implement GPU index creation
- [ ] Add GPU search methods

#### T2.2: Memory management
- [ ] Handle GPU memory limits
- [ ] Add CPU/GPU data transfer
- [ ] Implement memory pooling

### Day 3: Testing & Benchmark

#### T3.1: Benchmark suite
- [ ] Compare CPU vs GPU performance
- [ ] Test on various Mac models
- [ ] Document performance results

#### T3.2: Integration tests
- [ ] Test with real collections
- [ ] Edge cases (empty, large, small)
- [ ] Memory pressure tests

---

## Definition of Done

- [ ] GPU module working on Apple Silicon
- [ ] Benchmarks showing improvement
- [ ] Tests passing
- [ ] Documentation

---

## Technical Notes

### Apple Silicon Considerations
- Unified memory architecture (CPU/GPU share RAM)
- No VRAM separate from system RAM
- Metal Performance Shaders available
- Core ML for ML inference

### Expected Performance
| Operation | CPU (M3) | GPU Expected |
|-----------|-----------|--------------|
| Index build | ~30s | ~5-10s |
| Search (1M vectors) | ~50ms | ~10-20ms |

### Dependencies
```toml
# pyproject.toml
faiss-cpu = ">=1.7.0"
faiss-gpu = ">=1.7.0"  # Optional
```

### API Design
```python
import zvec
from zvec.gpu import GPUBackend

# Auto-detect and use GPU if available
schema = zvec.CollectionSchema(
    name="vectors",
    vectors=zvec.VectorSchema("emb", dimension=128),
    backend="auto"  # "cpu", "gpu", "auto"
)

# Or explicitly use GPU
schema = zvec.CollectionSchema(
    name="vectors",
    vectors=zvec.VectorSchema("emb", dimension=128),
    backend="gpu",
    gpu_device=0
)
```
