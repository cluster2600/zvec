/**
 * Graph-Based ANN Implementation (CAGRA-like)
 * 
 * Based on:
 * - NVIDIA cuVS CAGRA algorithm
 * - https://developer.nvidia.com/blog/optimizing-vector-search-for-indexing-and-real-time-retrieval-with-nvidia-cuvs
 * 
 * Features:
 * - GPU-friendly graph structure
 * - Configurable graph degree
 * - Hierarchical search
 */

#ifndef ZVEC_GPU_GRAPH_ANN_H_
#define ZVEC_GPU_GRAPH_ANN_H_

#include <vector>
#include <queue>
#include <random>
#include <algorithm>
#include <cmath>

namespace zvec {
namespace ann {

/**
 * Graph node representation
 */
struct GraphNode {
    std::vector<uint32_t> neighbors;  // Indices of neighboring nodes
    
    void add_neighbor(uint32_t idx) {
        neighbors.push_back(idx);
    }
    
    void sort_neighbors() {
        std::sort(neighbors.begin(), neighbors.end());
    }
};

/**
 * Graph-based ANN index
 */
template<typename T>
class GraphANNIndex {
public:
    GraphANNIndex(
        size_t dim,
        uint32_t graph_degree = 32,
        uint32_t intermediate_degree = 64
    ) : dim_(dim), 
        graph_degree_(graph_degree),
        intermediate_degree_(intermediate_degree) {}
    
    /**
     * Build the graph index from vectors
     * 
     * Uses NN-Descent algorithm
     */
    void build(const T* vectors, size_t n_vectors) {
        vectors_ = vectors;
        n_vectors_ = n_vectors;
        
        // Initialize graph
        graph_.resize(n_vectors_);
        
        // Random initialization
        std::mt19937 rng(42);
        std::uniform_int_distribution<uint32_t> dist(0, n_vectors_ - 1);
        
        for (size_t i = 0; i < n_vectors_; i++) {
            for (uint32_t j = 0; j < graph_degree_; j++) {
                graph_[i].add_neighbor(dist(rng));
            }
        }
        
        // NN-Descent iterations
        nn_descent(3);  // 3 iterations
    }
    
    /**
     * Search for k nearest neighbors
     */
    std::vector<std::pair<float, uint32_t>> search(
        const T* query,
        uint32_t k,
        uint32_t ef = 32
    ) const {
        if (n_vectors_ == 0) return {};
        
        // Initial candidates from random nodes
        std::mt19937 rng(42);
        std::vector<uint32_t> candidates;
        std::vector<float> candidate_distances;
        
        uint32_t init_count = std::min(ef, static_cast<uint32_t>(n_vectors_));
        for (uint32_t i = 0; i < init_count; i++) {
            candidates.push_back(i);
            candidate_distances.push_back(distance(query, vectors_ + i * dim_));
        }
        
        // Greedy search
        std::vector<char> visited(n_vectors_, 0);
        std::priority_queue<std::pair<float, uint32_t>> top_queue;
        
        while (!candidates.empty()) {
            // Get best candidate
            uint32_t best_idx = candidates.back();
            candidates.pop_back();
            
            if (visited[best_idx]) continue;
            visited[best_idx] = 1;
            
            float best_dist = candidate_distances.back();
            candidate_distances.pop_back();
            
            // Add to results
            top_queue.emplace(-best_dist, best_idx);
            if (top_queue.size() > ef) {
                top_queue.pop();
            }
            
            // Expand to neighbors
            for (uint32_t neighbor : graph_[best_idx].neighbors) {
                if (visited[neighbor]) continue;
                
                float dist = distance(query, vectors_ + neighbor * dim_);
                
                // Check if should be in candidates
                if (top_queue.size() < ef || 
                    dist < -top_queue.top().first) {
                    
                    candidates.push_back(neighbor);
                    candidate_distances.push_back(dist);
                }
            }
        }
        
        // Extract top-k
        std::vector<std::pair<float, uint32_t>> results;
        while (!top_queue.empty() && results.size() < k) {
            results.emplace_back(-top_queue.top().first, top_queue.top().second);
            top_queue.pop();
        }
        
        std::reverse(results.begin(), results.end());
        return results;
    }
    
    size_t size() const { return n_vectors_; }
    size_t dim() const { return dim_; }

private:
    size_t dim_;
    uint32_t graph_degree_;
    uint32_t intermediate_degree_;
    
    const T* vectors_ = nullptr;
    size_t n_vectors_ = 0;
    std::vector<GraphNode> graph_;
    
    /**
     * Compute L2 distance between two vectors
     */
    float distance(const T* a, const T* b) const {
        float sum = 0.0f;
        for (size_t i = 0; i < dim_; i++) {
            float diff = static_cast<float>(a[i]) - static_cast<float>(b[i]);
            sum += diff * diff;
        }
        return sum;
    }
    
    /**
     * NN-Descent algorithm for graph construction
     */
    void nn_descent(uint32_t iterations) {
        std::mt19937 rng(42);
        
        for (uint32_t iter = 0; iter < iterations; iter++) {
            // For each node, try to improve neighbors
            for (size_t i = 0; i < n_vectors_; i++) {
                const T* vec_i = vectors_ + i * dim_;
                
                std::vector<std::pair<float, uint32_t>> all_candidates;
                
                // Current neighbors
                for (uint32_t n : graph_[i].neighbors) {
                    all_candidates.emplace_back(
                        distance(vec_i, vectors_ + n * dim_), n
                    );
                }
                
                // Try to find better neighbors
                for (uint32_t n : graph_[i].neighbors) {
                    for (uint32_t nn : graph_[n].neighbors) {
                        if (nn == i) continue;
                        all_candidates.emplace_back(
                            distance(vec_i, vectors_ + nn * dim_), nn
                        );
                    }
                }
                
                // Sort and keep best
                std::sort(all_candidates.begin(), all_candidates.end());
                
                graph_[i].neighbors.clear();
                for (size_t j = 0; j < graph_degree_ && j < all_candidates.size(); j++) {
                    graph_[i].neighbors.push_back(all_candidates[j].second);
                }
            }
        }
    }
};

} // namespace ann
} // namespace zvec

#endif // ZVEC_GPU_GRAPH_ANN_H_
