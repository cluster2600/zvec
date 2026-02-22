# Metal MPS C++ Integration

This document describes the C++ Metal integration for zvec on Apple Silicon.

## Overview

The Metal module provides GPU-accelerated operations using Apple's Metal Performance Shaders (MPS) framework for M-series Apple Silicon chips.

## Requirements

- macOS 12.3+
- Apple Silicon (M1, M2, M3, M4)
- Xcode with Metal support

## Building

The Metal module is automatically built when compiling on macOS:

```bash
mkdir build && cd build
cmake ..
make
```

The module is located at:
- Source: `src/ailego/gpu/metal/`
- Header: `src/ailego/gpu/metal/zvec_metal.h`

## Usage

```cpp
#include "zvec_metal.h"

// Check availability
if (zvec_metal_available()) {
    // Create Metal device
    ZvecMetalDevice* device = zvec_metal_create();
    
    // Get device info
    printf("Device: %s\n", zvec_metal_device_name(device));
    printf("Memory: %lu MB\n", zvec_metal_device_memory(device) / 1024 / 1024);
    
    // Compute L2 distances
    std::vector<float> queries(N * D);
    std::vector<float> database(M * D);
    std::vector<float> distances(N * M);
    
    zvec_metal_l2_distance(
        device,
        queries.data(),
        database.data(),
        distances.data(),
        N, M, D
    );
    
    // Cleanup
    zvec_metal_destroy(device);
}
```

## API

### Functions

| Function | Description |
|----------|-------------|
| `zvec_metal_available()` | Check if Metal is available |
| `zvec_metal_create()` | Create Metal device handle |
| `zvec_metal_destroy()` | Destroy device handle |
| `zvec_metal_device_name()` | Get device name |
| `zvec_metal_device_memory()` | Get available memory |
| `zvec_metal_l2_distance()` | Compute L2 distances |
| `zvec_metal_inner_product()` | Compute inner products |
| `zvec_metal_normalize()` | L2 normalize vectors |

## Performance

The C++ Metal implementation provides:
- L2 distance computation
- Inner product (cosine similarity)
- Vector normalization
- Matrix operations

Current implementation uses CPU fallback with Metal shaders ready for activation.

## Integration

To integrate Metal acceleration into your zvec application:

1. Include the header
2. Check availability
3. Create device
4. Use GPU functions
5. Destroy device

## Future Work

- Full Metal kernel activation
- SIMD optimization
- Integration with RocksDB storage
