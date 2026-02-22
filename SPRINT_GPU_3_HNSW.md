# Sprint 3: Graph-Based Indexes (HNSW)

## Objective
Implement Hierarchical Navigable Small World (HNSW) graphs for fast approximate nearest neighbor search.

## Background

HNSW is currently the best single-thread ANN algorithm:
- Logarithmic search complexity: O(log N)
- Excellent recall (95%+)
- Memory proportional to graph size

## Tasks

### Day 1: HNSW Basics
- [ ] Study FAISS HNSW implementation
- [ ] Create wrapper/interface
- [ ] Basic search

### Day 2: Index Construction
- [ ] Implement build process
- [ ] Parameter tuning (M, efConstruction)
- [ ] Memory estimation

### Day 3: Query Optimization
- [ ] Implement efSearch parameter
- [ ] Parallel query handling
- [ ] Result ranking

### Day 4: Persistence
- [ ] Save/load index
- [ ] Incremental add
- [ ] Delete support

### Day 5: Benchmark & Tune
- [ ] Recall vs speed curves
- [ ] Memory profiling
- [ ] Comparison with IVF-PQ

## Research Papers

### Key Papers

1. **"Efficient and Robust Approximate Nearest Neighbor Search"** (Malkov & Yashunin)
   - Original HNSW paper
   - Comprehensive evaluation

2. **"HNSW On GPU: Accelerating Hierarchical Navigable Small World Graphs"**
   - GPU-accelerated HNSW

3. **"Fast Approximate Nearest Neighbor Search Through Hashing"**
   - Comparison with LSH

4. **"DiskANN: Fast Accurate Billion-scale Nearest Neighbor Search"**
   - Billion-scale ANN
   - Disk-based solution

## Technical Details

### Key Parameters
- `M`: Number of connections (16-64)
- `efConstruction`: Search width during build (100-500)
- `efSearch`: Search width during query (50-200)

### Trade-offs
| M | Memory | Search Speed | Recall |
|---|--------|--------------|--------|
| 16 | Low | Fast | Good |
| 32 | Medium | Medium | Better |
| 64 | High | Slow | Best |

## Success Metrics
- >95% recall@10
- <10ms search for 1M vectors
- <2GB memory for 1M vectors
