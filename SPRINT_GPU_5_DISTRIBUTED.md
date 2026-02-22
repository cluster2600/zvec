# Sprint 5: Distributed & Scale-Out

## Objective
Prepare zvec for distributed deployment and billion-scale datasets.

## Background

Single-machine solutions hit limits at ~100M vectors. Need distributed approach for larger.

## Tasks

### Day 1: Sharding
- [ ] Partition strategies (by bucket, by range)
- [ ] Consistent hashing
- [ ] Data rebalancing

### Day 2: Query Processing
- [ ] Scatter-gather pattern
- [ ] Result merging/ranking
- [ ] Query routing

### Day 3: Distributed Index
- [ ] Partitioned HNSW
- [ ] IVF index sharding
- [ ] Coordinator node

### Day 4: Replication
- [ ] Leader-follower replication
- [ ] Consistency models
- [ ] Failover handling

### Day 5: Benchmark
- [ ] Scale testing (10M+ vectors)
- [ ] Latency profiling
- [ ] Throughput testing

## Research Papers

### Key Papers

1. **"FAISS: A Library for Efficient Similarity Search"**
   - Distributed search techniques

2. **"DiskANN: Fast Accurate Billion-scale Nearest Neighbor Search on a Single Machine"**
   - Microsoft research
   - Single-machine billion-scale

3. **"PAnn: A Distributed System for Approximate Nearest Neighbor Search"**
   - Distributed ANN

4. **"SPANN: Efficiently Search Billionscale Vectors"**
   - Hierarchical clustering

## Architecture Options

### Option 1: Coordinator + Workers
- Central coordinator routes queries
- Workers handle local search
- Simple but coordinator is bottleneck

### Option 2: P2P
- No central node
- More complex but scalable

### Option 3: Hybrid (Recommended)
- Shard by vector bucket
- Local indexes
- Merge results

## Success Metrics
- Linear scaling to 1B vectors
- <100ms p99 latency
- 99.9% availability
