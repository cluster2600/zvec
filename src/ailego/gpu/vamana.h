/**
 * Vamana Graph Index Implementation
 *
 * Based on:
 * - DiskANN paper (Microsoft)
 * - https://arxiv.org/abs/1907.06146
 *
 * Key features:
 * - Robust to search parameters
 * - Supports dynamic updates
 * - Works well with PQ
 * - Used in Azure AI Search
 */

#ifndef ZVEC_ANN_VAMANA_H_
#define ZVEC_ANN_VAMANA_H_

#include <algorithm>
#include <cmath>
#include <limits>
#include <queue>
#include <random>
#include <vector>

namespace zvec {
namespace ann {

/**
 * Vamana graph parameters
 */
struct VamanaParams {
  float alpha = 1.2f;             // Graph construction parameter
  uint32_t R = 64;                // Max neighbors (degree)
  uint32_t L = 100;               // Search width during construction
  uint32_t L_search = 50;         // Search width during query
  uint32_t max_candidates = 500;  // Candidate pool size
};

/**
 * Vamana graph index
 */
template <typename T>
class VamanaIndex {
 public:
  VamanaIndex(size_t dim, const VamanaParams &params = VamanaParams())
      : dim_(dim), params_(params) {}

  /**
   * Build graph from vectors
   *
   * @param vectors Source vectors
   * @param n_vectors Number of vectors
   * @param pindex Prestored graph (optional, for pruning)
   */
  void build(const T *vectors, size_t n_vectors,
             const uint32_t *pindex = nullptr) {
    vectors_ = vectors;
    n_vectors_ = n_vectors;

    // Initialize graph
    graph_.resize(n_vectors_);

    // Random starting points
    std::mt19937 rng(42);
    std::vector<uint32_t> start_nodes(n_vectors_);
    for (size_t i = 0; i < n_vectors_; i++) start_nodes[i] = i;
    std::shuffle(start_nodes.begin(), start_nodes.end(), rng);

    // Build graph in iterations
    for (size_t iter = 0; iter < 3; iter++) {
      for (size_t i = 0; i < n_vectors_; i++) {
        // Random search to find candidates
        auto candidates = search_pruning(vectors_ + i * dim_, params_.L,
                                         params_.max_candidates);

        // Prune candidates
        graph_[i].neighbors = prune_candidates(candidates, vectors_ + i * dim_,
                                               params_.R, params_.alpha);
      }
    }

    // Ensure reciprocal edges
    make_reciprocal();
  }

  /**
   * Search for k nearest neighbors
   */
  std::vector<std::pair<float, uint32_t>> search(const T *query,
                                                 size_t k) const {
    if (n_vectors_ == 0) return {};

    // Initialize with random nodes
    std::mt19937 rng(42);
    std::vector<uint32_t> visited(n_vectors_, 0);
    std::priority_queue<std::pair<float, uint32_t>> queue;  // min-heap

    // Start from a few random nodes
    uint32_t start = rng() % n_vectors_;
    queue.emplace(0.0f, start);

    std::vector<std::pair<float, uint32_t>> results;

    while (!queue.empty() && results.size() < params_.L_search) {
      auto [dist, id] = queue.top();
      queue.pop();

      if (visited[id]) continue;
      visited[id] = 1;

      results.emplace_back(dist, id);

      // Expand to neighbors
      for (uint32_t neighbor : graph_[id].neighbors) {
        if (!visited[neighbor]) {
          float d = distance(query, vectors_ + neighbor * dim_);
          queue.emplace(d, neighbor);
        }
      }
    }

    // Sort and return top-k
    std::partial_sort(results.begin(),
                      results.begin() + std::min(k, results.size()),
                      results.end());

    results.resize(std::min(k, results.size()));
    return results;
  }

  size_t size() const {
    return n_vectors_;
  }
  size_t dim() const {
    return dim_;
  }

 private:
  size_t dim_;
  VamanaParams params_;

  const T *vectors_ = nullptr;
  size_t n_vectors_ = 0;

  struct Node {
    std::vector<uint32_t> neighbors;
  };
  std::vector<Node> graph_;

  /**
   * L2 distance
   */
  float distance(const T *a, const T *b) const {
    float sum = 0;
    for (size_t i = 0; i < dim_; i++) {
      float d = static_cast<float>(a[i]) - static_cast<float>(b[i]);
      sum += d * d;
    }
    return sum;
  }

  /**
   * Search with pruning to find candidates
   */
  std::vector<std::pair<float, uint32_t>> search_pruning(
      const T *query, uint32_t L, uint32_t max_candidates) const {
    std::mt19937 rng(42);
    std::vector<uint32_t> visited(n_vectors_, 0);

    // Start from random node
    uint32_t start = rng() % n_vectors_;

    std::priority_queue<std::pair<float, uint32_t>> frontier;
    frontier.emplace(0.0f, start);

    std::vector<std::pair<float, uint32_t>> candidates;

    while (!frontier.empty() && candidates.size() < max_candidates) {
      auto [dist, id] = frontier.top();
      frontier.pop();

      if (visited[id]) continue;
      visited[id] = 1;

      candidates.emplace_back(dist, id);

      for (uint32_t neighbor : graph_[id].neighbors) {
        if (!visited[neighbor]) {
          float d = distance(query, vectors_ + neighbor * dim_);
          frontier.emplace(d, neighbor);
        }
      }
    }

    return candidates;
  }

  /**
   * Prune candidates to R neighbors
   */
  std::vector<uint32_t> prune_candidates(
      std::vector<std::pair<float, uint32_t>> &candidates, const T *query,
      uint32_t R, float alpha) {
    if (candidates.empty()) return {};

    // Sort by distance
    std::sort(candidates.begin(), candidates.end());

    std::vector<uint32_t> pruned;
    float max_dist = candidates.empty() ? std::numeric_limits<float>::max()
                                        : candidates[0].first * alpha;

    for (auto &[dist, id] : candidates) {
      if (pruned.size() >= R) break;
      if (dist > max_dist) break;

      // Check against already selected
      bool dominated = false;
      for (uint32_t selected : pruned) {
        float d = distance(vectors_ + selected * dim_, vectors_ + id * dim_);
        if (d < max_dist) {
          dominated = true;
          break;
        }
      }

      if (!dominated) {
        pruned.push_back(id);
        max_dist = std::max(max_dist, dist * alpha);
      }
    }

    return pruned;
  }

  /**
   * Make graph reciprocal (both directions)
   */
  void make_reciprocal() {
    std::vector<std::vector<uint32_t>> new_graph(n_vectors_);

    for (size_t i = 0; i < n_vectors_; i++) {
      std::vector<uint32_t> all_neighbors = graph_[i].neighbors;

      for (uint32_t neighbor : graph_[i].neighbors) {
        if (neighbor < n_vectors_) {
          all_neighbors.push_back(neighbor);
          // Add reverse edge
          new_graph[neighbor].push_back(i);
        }
      }

      // Remove duplicates
      std::sort(all_neighbors.begin(), all_neighbors.end());
      all_neighbors.erase(
          std::unique(all_neighbors.begin(), all_neighbors.end()),
          all_neighbors.end());

      new_graph[i] = all_neighbors;
    }

    // Apply and prune to R
    for (size_t i = 0; i < n_vectors_; i++) {
      auto &neighbors = new_graph[i];
      if (neighbors.size() > params_.R) {
        neighbors.resize(params_.R);
      }
      graph_[i].neighbors = neighbors;
    }
  }
};

}  // namespace ann
}  // namespace zvec

#endif  // ZVEC_ANN_VAMANA_H_
