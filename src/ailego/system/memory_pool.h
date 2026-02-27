/**
 * Memory Pool and Allocator Optimizations
 *
 * Based on:
 * - FAISS: mimalloc allocator, huge pages
 * - https://github.com/facebookresearch/faiss/wiki/How-to-make-Faiss-run-faster
 * - OptiTrust: Cache tiling, SoA layout
 *
 * Optimizations:
 * - Memory pooling (减少allocation overhead)
 * - Huge pages (TLB miss reduction)
 * - Cache-aligned allocations
 * - Object pooling
 */

#ifndef ZVEC_SYSTEM_MEMORY_POOL_H_
#define ZVEC_SYSTEM_MEMORY_POOL_H_

#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

// Try to include mimalloc
#ifdef ZVEC_USE_MIMALLOC
#include <mimalloc.h>
#endif

namespace zvec {
namespace memory {

/**
 * Aligned memory allocator (cache-line or huge page)
 */
class AlignedAllocator {
 public:
  static void *allocate(size_t size, size_t alignment = 64) {
    void *ptr = nullptr;

#ifdef ZVEC_USE_MIMALLOC
    ptr = mi_aligned_alloc(size, alignment);
#else
    if (posix_memalign(&ptr, alignment, size) != 0) {
      return nullptr;
    }
#endif
    return ptr;
  }

  static void deallocate(void *ptr) {
#ifdef ZVEC_USE_MIMALLOC
    mi_free(ptr);
#else
    free(ptr);
#endif
  }
};

/**
 * Memory pool for fixed-size objects
 *
 * Reduces allocation overhead by pre-allocating chunks
 */
template <typename T>
class ObjectPool {
 public:
  ObjectPool(size_t chunk_size = 1024) : chunk_size_(chunk_size) {}

  ~ObjectPool() {
    for (auto *chunk : chunks_) {
      delete[] chunk;
    }
  }

  /**
   * Get object from pool
   */
  T *allocate() {
    std::lock_guard<std::mutex> lock(mutex_);

    if (free_list_.empty()) {
      // Allocate new chunk
      auto *chunk = new T[chunk_size_];
      chunks_.push_back(chunk);

      // Add all to free list
      for (size_t i = 0; i < chunk_size_; i++) {
        free_list_.push_back(&chunk[i]);
      }
    }

    T *obj = free_list_.back();
    free_list_.pop_back();
    return obj;
  }

  /**
   * Return object to pool
   */
  void deallocate(T *obj) {
    std::lock_guard<std::mutex> lock(mutex_);
    free_list_.push_back(obj);
  }

  size_t allocated_size() const {
    return chunks_.size() * chunk_size_;
  }

  size_t available_size() const {
    return free_list_.size();
  }

 private:
  size_t chunk_size_;
  std::vector<T *> chunks_;
  std::vector<T *> free_list_;
  std::mutex mutex_;
};

/**
 * Huge page support
 */
class HugePageAllocator {
 public:
  static void *allocate_huge_page(size_t size) {
#ifdef __linux__
    // Use madvise with MADV_HUGEPAGE
    void *ptr = aligned_alloc(1024 * 1024 * 2, size);  // 2MB huge pages
    if (ptr) {
      madvise(ptr, size, MADV_HUGEPAGE);
    }
    return ptr;
#else
    return AlignedAllocator::allocate(size, 1024 * 1024 * 2);
#endif
  }
};

/**
 * Cache-aligned vector (SoA layout for SIMD)
 */
template <typename T>
class CacheAlignedVector {
 public:
  CacheAlignedVector(size_t size = 0) {
    resize(size);
  }

  ~CacheAlignedVector() {
    for (auto *data : data_) {
      AlignedAllocator::deallocate(data);
    }
  }

  void resize(size_t size) {
    // Free old
    for (auto *data : data_) {
      AlignedAllocator::deallocate(data);
    }
    data_.clear();

    // Allocate aligned
    size_ = size;
    data_.push_back(
        static_cast<T *>(AlignedAllocator::allocate(size * sizeof(T), 64)));
  }

  T &operator[](size_t idx) {
    return data_[0][idx];
  }

  const T &operator[](size_t idx) const {
    return data_[0][idx];
  }

  size_t size() const {
    return size_;
  }

 private:
  std::vector<T *> data_;
  size_t size_ = 0;
};

/**
 * Slab allocator for index structures
 */
class SlabAllocator {
 public:
  SlabAllocator(size_t object_size, size_t objects_per_slab = 1024)
      : object_size_(object_size), objects_per_slab_(objects_per_slab) {}

  void *allocate() {
    std::lock_guard<std::mutex> lock(mutex_);

    // Try current slab
    if (current_slab_ && current_pos_ < objects_per_slab_) {
      char *ptr = current_slab_ + current_pos_ * object_size_;
      current_pos_++;
      return ptr;
    }

    // Allocate new slab
    char *new_slab = static_cast<char *>(
        AlignedAllocator::allocate(object_size_ * objects_per_slab_, 4096));

    slabs_.push_back(new_slab);
    current_slab_ = new_slab;
    current_pos_ = 1;

    return new_slab;
  }

  ~SlabAllocator() {
    for (char *slab : slabs_) {
      AlignedAllocator::deallocate(slab);
    }
  }

 private:
  size_t object_size_;
  size_t objects_per_slab_;
  std::vector<char *> slabs_;
  char *current_slab_ = nullptr;
  size_t current_pos_ = 0;
  std::mutex mutex_;
};

}  // namespace memory
}  // namespace zvec

#endif  // ZVEC_SYSTEM_MEMORY_POOL_H_
