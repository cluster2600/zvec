# GPU-PIM Collaboration for Vector Search

## Based on
- USENIX ATC 2025: "Turbocharge ANNS on Real Processing-in-Memory by Enabling Fine-Grained PIM-GPU Collaboration"
- arXiv:2410.15621 - DRIM-ANN
- arXiv:2410.23805 - UpANNS

## Key Concepts

### Processing-in-Memory (PIM)
- Memory chips with compute capability
- Reduces data movement between CPU/GPU and memory
- Key for memory-bound workloads like vector search

### GPU-PIM Collaboration Patterns

1. **Pre-filtering**: Use PIM to filter candidates before GPU search
2. **Hybrid Index**: Hot data on GPU, cold data in PIM
3. **Pipeline**: PIM does coarse search, GPU does refinement

## Implementation Ideas

```python
class HybridGPUPIMIndex:
    """Hybrid index using GPU + PIM collaboration."""
    
    def __init__(self, pim_threshold_mb=1000):
        self.gpu_index = None  # cuVS/FAISS
        self.pim_index = None  # UPMEM or similar
        self.threshold = pim_threshold_mb
    
    def search(self, query, k=10):
        # Phase 1: PIM coarse search
        candidates = self.pim_index.search(query, k * 10)
        
        # Phase 2: GPU refinement
        refined = self.gpu_index.refine(query, candidates, k)
        return refined
```

## Expected Benefits
- 40-60% reduction in data movement
- Better performance for large datasets that don't fit in GPU memory
- Cost efficiency for billion-scale search

## Future Work
- Benchmark on actual PIM hardware (UPMUM)
- Integrate with DiskANN for out-of-core
