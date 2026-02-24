/**
 * SIMD Optimized Vector Distance Functions for CPU
 * 
 * Based on:
 * - Intel SIMD documentation
 * - NEON optimization for ARM (Apple Silicon)
 * - x86 AVX2/AVX-512 intrinsics
 * 
 * Expected speedup: 4-16x vs scalar
 */

#ifndef ZVEC_CPU_SIMD_DISTANCE_H_
#define ZVEC_CPU_SIMD_DISTANCE_H_

#include <cstdint>
#include <cmath>
#include <algorithm>

#ifdef __SSE2__
#include <emmintrin.h>
#endif

#ifdef __AVX2__
#include <immintrin.h>
#endif

#ifdef __ARM_NEON
#include <arm_neon.h>
#endif

namespace zvec {
namespace simd {

// =============================================================================
// SSE2 Implementation (x86)
// =============================================================================

#ifdef __SSE2__

inline float sse2_l2_distance(const float* a, const float* b, size_t dim) {
    __m128 sum = _mm_setzero_ps();
    
    size_t i = 0;
    for (; i + 4 <= dim; i += 4) {
        __m128 va = _mm_loadu_ps(a + i);
        __m128 vb = _mm_loadu_ps(b + i);
        __m128 diff = _mm_sub_ps(va, vb);
        sum = _mm_add_ps(sum, _mm_mul_ps(diff, diff));
    }
    
    // Horizontal sum
    __m128 temp = _mm_movehdup_ps(sum);
    __m128 sum2 = _mm_addsub_ps(sum, temp);
    temp = _mm_movehl_ps(temp, sum2);
    sum2 = _mm_add_ss(sum2, temp);
    float result = _mm_cvtss_si32(sum2);
    
    // Handle remainder
    for (; i < dim; i++) {
        float d = a[i] - b[i];
        result += d * d;
    }
    
    return result;
}

inline void sse2_l2_distance_batch(
    const float* queries,
    const float* database,
    float* distances,
    size_t dim,
    size_t n_queries,
    size_t n_database
) {
    for (size_t q = 0; q < n_queries; q++) {
        const float* query = queries + q * dim;
        for (size_t d = 0; d < n_database; d++) {
            distances[q * n_database + d] = sse2_l2_distance(
                query, database + d * dim, dim
            );
        }
    }
}

#endif // __SSE2__

// =============================================================================
// AVX2 Implementation (x86)
// =============================================================================

#ifdef __AVX2__

inline float avx2_l2_distance(const float* a, const float* b, size_t dim) {
    __m256 sum = _mm256_setzero_ps();
    
    size_t i = 0;
    for (; i + 8 <= dim; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        __m256 diff = _mm256_sub_ps(va, vb);
        sum = _mm256_add_ps(sum, _mm256_mul_ps(diff, diff));
    }
    
    // Horizontal sum of 256-bit
    __m128 sum128 = _mm256_castps256_ps128(sum);
    __m128 high = _mm256_extractf128_ps(sum, 1);
    sum128 = _mm_add_ps(sum128, high);
    
    // Sum of 128-bit
    __m128 temp = _mm_movehdup_ps(sum128);
    sum128 = _mm_addsub_ps(sum128, temp);
    temp = _mm_movehl_ps(temp, sum128);
    sum128 = _mm_add_ss(sum128, temp);
    float result = _mm_cvtss_si32(sum128);
    
    for (; i < dim; i++) {
        float d = a[i] - b[i];
        result += d * d;
    }
    
    return result;
}

/**
 * AVX2 batch L2 with unrolling
 */
inline void avx2_l2_distance_batch_unrolled(
    const float* queries,
    const float* database,
    float* distances,
    size_t dim,
    size_t n_queries,
    size_t n_database
) {
    constexpr size_t UNROLL = 4;
    
    for (size_t q = 0; q < n_queries; q++) {
        const float* query = queries + q * dim;
        
        size_t d = 0;
        for (; d + UNROLL <= n_database; d += UNROLL) {
            __m256 sum0 = _mm256_setzero_ps();
            __m256 sum1 = _mm256_setzero_ps();
            __m256 sum2 = _mm256_setzero_ps();
            __m256 sum3 = _mm256_setzero_ps();
            
            for (size_t i = 0; i < dim; i += 8) {
                __m256 vq = _mm256_set1_ps(query[i]);
                
                __m256 vd0 = _mm256_loadu_ps(database + (d + 0) * dim + i);
                __m256 vd1 = _mm256_loadu_ps(database + (d + 1) * dim + i);
                __m256 vd2 = _mm256_loadu_ps(database + (d + 2) * dim + i);
                __m256 vd3 = _mm256_loadu_ps(database + (d + 3) * dim + i);
                
                sum0 = _mm256_add_ps(sum0, _mm256_mul_ps(_mm256_sub_ps(vq, vd0), _mm256_sub_ps(vq, vd0)));
                sum1 = _mm256_add_ps(sum1, _mm256_mul_ps(_mm256_sub_ps(vq, vd1), _mm256_sub_ps(vq, vd1)));
                sum2 = _mm256_add_ps(sum2, _mm256_mul_ps(_mm256_sub_ps(vq, vd2), _mm256_sub_ps(vq, vd2)));
                sum3 = _mm256_add_ps(sum3, _mm256_mul_ps(_mm256_sub_ps(vq, vd3), _mm256_sub_ps(vq, vd3)));
            }
            
            // Reduce
            __m128 s0 = _mm256_castps256_ps128(sum0);
            __m128 s0h = _mm256_extractf128_ps(sum0, 1);
            distances[q * n_database + d + 0] = _mm_cvtss_f32(_mm_add_ss(s0, s0h));
            
            __m128 s1 = _mm256_castps256_ps128(sum1);
            __m128 s1h = _mm256_extractf128_ps(sum1, 1);
            distances[q * n_database + d + 1] = _mm_cvtss_f32(_mm_add_ss(s1, s1h));
            
            __m128 s2 = _mm256_castps256_ps128(sum2);
            __m128 s2h = _mm256_extractf128_ps(sum2, 1);
            distances[q * n_database + d + 2] = _mm_cvtss_f32(_mm_add_ss(s2, s2h));
            
            __m128 s3 = _mm256_castps256_ps128(sum3);
            __m128 s3h = _mm256_extractf128_ps(sum3, 1);
            distances[q * n_database + d + 3] = _mm_cvtss_f32(_mm_add_ss(s3, s3h));
        }
        
        // Handle remainder
        for (; d < n_database; d++) {
            distances[q * n_database + d] = avx2_l2_distance(
                query, database + d * dim, dim
            );
        }
    }
}

#endif // __AVX2__

// =============================================================================
// NEON Implementation (ARM/Apple Silicon)
// =============================================================================

#ifdef __ARM_NEON

inline float neon_l2_distance(const float* a, const float* b, size_t dim) {
    float32x4_t sum = vdupq_n_f32(0.0f);
    
    size_t i = 0;
    for (; i + 4 <= dim; i += 4) {
        float32x4_t va = vld1q_f32(a + i);
        float32x4_t vb = vld1q_f32(b + i);
        float32x4_t diff = vsubq_f32(va, vb);
        sum = vmlaq_f32(sum, diff, diff);
    }
    
    // Horizontal sum
    float32x2_t sum2 = vadd_f32(vget_low_f32(sum), vget_high_f32(sum));
    float result = vget_lane_f32(vpadd_f32(sum2, sum2), 0);
    
    for (; i < dim; i++) {
        float d = a[i] - b[i];
        result += d * d;
    }
    
    return result;
}

inline void neon_l2_distance_batch(
    const float* queries,
    const float* database,
    float* distances,
    size_t dim,
    size_t n_queries,
    size_t n_database
) {
    for (size_t q = 0; q < n_queries; q++) {
        const float* query = queries + q * dim;
        for (size_t d = 0; d < n_database; d++) {
            distances[q * n_database + d] = neon_l2_distance(
                query, database + d * dim, dim
            );
        }
    }
}

#endif // __ARM_NEON

// =============================================================================
// Portable Fallback
// =============================================================================

inline float scalar_l2_distance(const float* a, const float* b, size_t dim) {
    float sum = 0.0f;
    for (size_t i = 0; i < dim; i++) {
        float diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sum;
}

// =============================================================================
// Dispatcher
// =============================================================================

struct SimdCapabilities {
    bool sse2 = false;
    bool avx2 = false;
    bool avx512 = false;
    bool neon = false;
    bool neon_dotprod = false;
};

inline SimdCapabilities detect_simd() {
    SimdCapabilities caps;
    
#ifdef __SSE2__
    caps.sse2 = true;
#endif

#ifdef __AVX2__
    caps.avx2 = true;
#endif

#ifdef __AVX512F__
    caps.avx512 = true;
#endif

#ifdef __ARM_NEON
    caps.neon = true;
#ifdef __ARM_FEATURE_DOTPROD
    caps.neon_dotprod = true;
#endif
#endif
    
    return caps;
}

} // namespace simd
} // namespace zvec

#endif // ZVEC_CPU_SIMD_DISTANCE_H_
