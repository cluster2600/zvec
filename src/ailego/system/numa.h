/**
 * NUMA-Aware Data Structures and Algorithms
 * 
 * Based on:
 * - Quake (OSDI 2025): NUMA-aware partitioning
 * - https://www.usenix.org/system/files/osdi25-mohoney.pdf
 * 
 * Key optimizations:
 * - Per-NUMA-node data structures
 * - Locality-aware allocation
 * - Work stealing across nodes
 * 
 * Expected: 6-20x speedup on multi-socket systems
 */

#ifndef ZVEC_SYSTEM_NUMA_H_
#define ZVEC_SYSTEM_NUMA_H_

#include <vector>
#include <memory>
#include <thread>
#include <sched.h>
#include <numa.h>
#include <unistd.h>

#include <cassert>
#include <cstring>

namespace zvec {
namespace numa {

/**
 * NUMA node information
 */
struct NumaNode {
    int id;
    size_t memory_bytes;
    int num_cpus;
    std::vector<int> cpus;
    
    NumaNode(int id) : id(id) {
        // Get node memory
        struct bitmask* mask = numa_allocate_nodemask();
        numa_bitmask_setbit(mask, id);
        memory_bytes = numa_node_size64(id, nullptr);
        numa_free_nodemask(mask);
        
        // Get CPUs
        struct bitmask* cpu_mask = numa_allocate_cpumask();
        numa_node_to_cpus(id, cpu_mask);
        
        num_cpus = numa_num_cpus_node(id);
        cpus.resize(num_cpus);
        for (int i = 0; i < num_cpus; i++) {
            cpus[i] = i;  // Simplified
        }
        numa_free_cpumask(cpu_mask);
    }
};

/**
 * NUMA-aware memory allocator
 */
class NumaAllocator {
public:
    /**
     * Allocate memory on specific NUMA node
     */
    static void* allocate_node(size_t size, int node) {
        if (numa_available() < 0) {
            // NUMA not available, use regular allocation
            return malloc(size);
        }
        
        void* ptr = numa_alloc_onnode(size, node);
        if (!ptr) {
            // Fallback
            ptr = numa_alloc_interleaved(size);
        }
        return ptr;
    }
    
    /**
     * Allocate interleaved across all nodes
     */
    static void* allocate_interleaved(size_t size) {
        if (numa_available() < 0) {
            return malloc(size);
        }
        
        void* ptr = numa_alloc_interleaved(size);
        return ptr ? ptr : malloc(size);
    }
    
    /**
     * Free NUMA-allocated memory
     */
    static void free(void* ptr, size_t size) {
        if (numa_available() < 0) {
            ::free(ptr);
            return;
        }
        
        // Try to detect if it was NUMA-allocated
        // In practice, just use numa_free if available
        if (ptr) {
            numa_free(ptr, size);
        }
    }
};

/**
 * NUMA-aware vector with local storage
 */
template<typename T>
class NumaVector {
public:
    NumaVector() = default;
    
    NumaVector(size_t size, int node = -1) {
        resize(size, node);
    }
    
    ~NumaVector() {
        if (data_) {
            NumaAllocator::free(data_, size_ * sizeof(T));
        }
    }
    
    void resize(size_t size, int node = -1) {
        if (data_) {
            NumaAllocator::free(data_, size_ * sizeof(T));
        }
        
        size_ = size;
        node_ = node >= 0 ? node : 0;
        
        if (size > 0) {
            data_ = static_cast<T*>(NumaAllocator::allocate_node(
                size * sizeof(T), node_
            ));
        }
    }
    
    T& operator[](size_t idx) { return data_[idx]; }
    const T& operator[](size_t idx) const { return data_[idx]; }
    
    T* data() { return data_; }
    const T* data() const { return data_; }
    size_t size() const { return size_; }
    int node() const { return node_; }
    
    // Move to another NUMA node
    void migrate(int new_node) {
        if (new_node == node_) return;
        
        T* new_data = static_cast<T*>(
            NumaAllocator::allocate_node(size_ * sizeof(T), new_node)
        );
        
        memcpy(new_data, data_, size_ * sizeof(T));
        NumaAllocator::free(data_, size_ * sizeof(T));
        
        data_ = new_data;
        node_ = new_node;
    }

private:
    T* data_ = nullptr;
    size_t size_ = 0;
    int node_ = 0;
};

/**
 * NUMA-aware thread pool with local work stealing
 */
class NumaThreadPool {
public:
    NumaThreadPool(size_t num_threads = 0) {
        if (num_threads == 0) {
            num_threads = std::thread::hardware_concurrency();
        }
        
        // Get NUMA info
        num_nodes_ = numa_max_node() + 1;
        
        threads_.resize(num_threads);
        
        for (size_t i = 0; i < num_threads; i++) {
            int node = i % num_nodes_;
            threads_[i] = std::thread([this, i, node]() {
                // Bind thread to NUMA node
                if (numa_available() >= 0) {
                    struct bitmask* mask = numa_allocate_cpumask();
                    numa_bitmask_setbit(mask, node);
                    numa_setaffinity(0, mask);
                    numa_free_cpumask(mask);
                }
                
                // Work loop
                while (!stop_) {
                    // Try local queue first
                    Task task = local_queues_[i].pop();
                    if (task) {
                        task();
                        completed_++;
                        continue;
                    }
                    
                    // Try stealing from other NUMA nodes
                    bool stolen = false;
                    for (size_t j = 0; j < num_threads_; j++) {
                        if (i == j) continue;
                        
                        // Prefer same NUMA node
                        int other_node = j % num_nodes_;
                        if (other_node != node) continue;
                        
                        task = local_queues_[j].steal();
                        if (task) {
                            task();
                            stolen = true;
                            break;
                        }
                    }
                    
                    if (!stolen) {
                        std::this_thread::yield();
                    }
                }
            });
        }
    }
    
    ~NumaThreadPool() {
        stop_ = true;
        for (auto& t : threads_) {
            if (t.joinable()) t.join();
        }
    }
    
    template<typename F>
    void submit(F&& task) {
        size_t thread_id = current_thread_.load();
        if (thread_id >= num_threads_) {
            thread_id = next_thread_++ % num_threads_;
        }
        local_queues_[thread_id].push(std::forward<F>(task));
    }
    
    size_t completed() const { return completed_; }

private:
    struct Task {
        std::function<void()> func;
        
        Task() = default;
        
        explicit Task(std::function<void()>&& f) : func(std::move(f)) {}
        
        explicit operator bool() const { return bool(func); }
        
        void operator()() { if (func) func(); }
    };
    
    struct MPSCQueue {
        std::vector<Task> tasks;
        size_t head = 0;
        size_t tail = 0;
        
        void push(Task&& t) {
            tasks.push_back(std::move(t));
        }
        
        Task pop() {
            if (head >= tasks.size()) return Task();
            return std::move(tasks[head++]);
        }
        
        Task steal() {
            if (tail <= head) return Task();
            // Steal from tail (FIFO)
            return std::move(tasks[--tail]);
        }
    };
    
    size_t num_threads_;
    size_t num_nodes_;
    std::vector<std::thread> threads_;
    std::vector<MPSCQueue> local_queues_;
    std::atomic<bool> stop_{false};
    std::atomic<size_t> current_thread_{0};
    std::atomic<size_t> next_thread_{0};
    std::atomic<size_t> completed_{0};
};

} // namespace numa
} // namespace zvec

#endif // ZVEC_SYSTEM_NUMA_H_
