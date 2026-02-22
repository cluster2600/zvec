# Sprint: zvec Feature Opportunities

## Objectif
Identifier et planifier les nouvelles fonctionnalités basées sur les dernières versions des libraries utilisées par zvec.

## Durée
1-2 semaines

## Dependencies Analysis

### RocksDB (v10.10.1 - Feb 2026)
**GPU Acceleration**: ❌ Pas de support GPU natif dans RocksDB officiel

**Features interessantes**:
- Parallel Compression (v10.7.0): 65% reduction CPU
- MultiScan Optimizations (v10.5.0+)
- Manifest Auto-Tuning
- IO Activity Tagging
- Unified Memory Tracking

**H-Rocks**: Research extension CPU-GPU (pas production-ready)

### Faiss (v1.13.2 - Dec 2025)
**GPU Acceleration**: ✅ Oui - NVIDIA cuVS integration

**Features GPU**:
- GpuIndexCagra (CUDA-ANN Graph)
- GpuIndexIVFPQ optimisé
- Up to 12x index build, 90% lower latency
- BinaryCagra, FP16, int8 support

### zvec Current Features
- In-process vector DB
- SIMD-accelerated
- Dense + Sparse vectors
- Hybrid search with filters
- Full CRUD + RAG

---

## Proposed Features for zvec

### Priority 1: Performance

#### F1: GPU Acceleration (FAISS cuVS)
- **Description**: Add optional GPU support via FAISS cuVS
- **Impact**: 10-100x speedup for index build and search
- **Effort**: High (new bindings, CUDA integration)
- **Dependencies**: FAISS with cuVS, CUDA

#### F2: Parallel Compression
- **Description**: Enable RocksDB parallel compression
- **Impact**: 65% lower CPU overhead
- **Effort**: Low (config change in RocksDB options)
- **Status**: Can implement in current PR

#### F3: MultiScan Optimization
- **Description**: Enable async I/O and prefetch
- **Impact**: Faster range scans
- **Effort**: Low (RocksDB config)
- **Status**: Can implement now

### Priority 2: Storage

#### F4: Compression Level Control
- **Description**: Expose compression level as runtime parameter
- **Impact**: User control over speed/ratio tradeoff
- **Effort**: Medium
- **Status**: Add to CollectionSchema

#### F5: Tiered Storage
- **Description**: Hot/warm/cold data tiers
- **Impact**: Cost optimization
- **Effort**: High

### Priority 3: Search

#### F6: Cagra Index Support
- **Description**: GPU-optimized graph-based index
- **Impact**: Fastest ANN search
- **Effort**: High (FAISS integration)

#### F7: Advanced Filters
- **Description**: More complex filter expressions
- **Impact**: Better hybrid search
- **Effort**: Medium

---

## Sprint Recommendations

### Sprint 1: Quick Wins (1-2 days)
| Feature | Effort | Impact |
|---------|--------|--------|
| Parallel Compression | Low | High |
| MultiScan config | Low | Medium |
| Compression level param | Medium | Medium |

### Sprint 2: GPU Foundation (1 week)
| Feature | Effort | Impact |
|---------|--------|--------|
| FAISS GPU bindings | High | Very High |
| Cagra index | High | Very High |

### Sprint 3: Advanced (1-2 weeks)
| Feature | Effort | Impact |
|---------|--------|--------|
| Tiered storage | High | Medium |
| Advanced filters | Medium | Medium |

---

## GPU Status for zvec

### Currently
- **SIMD acceleration**: ✅ Yes (CPU)
- **GPU support**: ❌ Not yet

### Roadmap
1. **Short term**: RocksDB optimizations (parallel compression)
2. **Medium term**: FAISS GPU integration
3. **Long term**: Custom GPU kernels

### Alternative: H-Rocks
Research project (not production-ready):
- https://github.com/csl-iisc/H-Rocks-SIGMOD25
- CPU-GPU heterogeneous RocksDB
- Would require significant porting work
