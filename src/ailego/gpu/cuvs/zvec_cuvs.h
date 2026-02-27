/**
 * cuVS C++ Bindings for zvec
 *
 * Based on cuVS C++ API:
 * https://docs.rapids.ai/api/cuvs/stable/
 *
 * Requires: cuVS, CUDA 12+
 */

#ifndef ZVEC_CUVS_H_
#define ZVEC_CUVS_H_

#include <cstdint>
#include <memory>
#include <vector>

namespace zvec {
namespace cuvs {

// Forward declarations
template <typename T>
class IVFPQIndex;

template <typename T>
class CAGRAIndex;

template <typename T>
class HNSWIndex;

/**
 * IVF-PQ Index Parameters
 */
struct IVFPQParams {
  uint32_t nlist = 1024;         // Number of inverted file lists
  uint32_t nprobe = 32;          // Number of lists to search
  uint32_t pq_bits = 8;          // Bits per subvector
  uint32_t pq_dim = 0;           // Subvector dimension (0 = auto)
  std::string metric = "sq_l2";  // Distance metric

  IVFPQParams() = default;

  IVFPQParams &set_nlist(uint32_t v) {
    nlist = v;
    return *this;
  }
  IVFPQParams &set_nprobe(uint32_t v) {
    nprobe = v;
    return *this;
  }
  IVFPQParams &set_pq_bits(uint32_t v) {
    pq_bits = v;
    return *this;
  }
};

/**
 * CAGRA Index Parameters
 */
struct CAGRAParams {
  uint32_t graph_degree = 32;               // Connections in final graph
  uint32_t intermediate_graph_degree = 64;  // Construction connections
  uint32_t nn_min_num = 128;                // Min search neighbors
  uint32_t nn_max_num = 256;                // Max search neighbors
  std::string metric = "sq_l2";

  CAGRAParams() = default;
};

/**
 * HNSW Index Parameters
 */
struct HNSWParams {
  uint32_t m = 32;                 // Connections per node
  uint32_t ef_construction = 200;  // Construction width
  uint32_t ef_search = 50;         // Search width

  HNSWParams() = default;
};

/**
 * Search Results
 */
struct SearchResult {
  std::vector<float> distances;
  std::vector<int64_t> indices;

  SearchResult() = default;

  SearchResult(size_t n_queries, size_t k) {
    distances.resize(n_queries * k);
    indices.resize(n_queries * k);
  }

  float *distances_ptr() {
    return distances.data();
  }
  int64_t *indices_ptr() {
    return indices.data();
  }
};

/**
 * IVFPQ Index Implementation
 */
template <typename T>
class IVFPQIndex {
 public:
  IVFPQIndex() = default;

  explicit IVFPQIndex(const IVFPQParams &params) : params_(params) {}

  /**
   * Train the index on training vectors
   *
   * @param vectors Training vectors (n_vectors x dim)
   * @param dim Vector dimensionality
   */
  void train(const T *vectors, size_t n_vectors, size_t dim);

  /**
   * Add vectors to the index
   *
   * @param vectors Vectors to add (n_vectors x dim)
   * @param n_vectors Number of vectors
   */
  void add(const T *vectors, size_t n_vectors);

  /**
   * Search for k nearest neighbors
   *
   * @param queries Query vectors (n_queries x dim)
   * @param n_queries Number of queries
   * @param k Number of neighbors to return
   * @return SearchResult with distances and indices
   */
  SearchResult search(const T *queries, size_t n_queries, size_t k);

  /**
   * Get number of vectors in index
   */
  size_t size() const {
    return size_;
  }

  /**
   * Get vector dimensionality
   */
  size_t dim() const {
    return dim_;
  }

 private:
  IVFPQParams params_;
  size_t dim_ = 0;
  size_t size_ = 0;

  // cuVS index would be held here
  // std::unique_ptr<cuvs::IVFPQIndex> index_;
};

// Explicit instantiations
extern template class IVFPQIndex<float>;
extern template class IVFPQIndex<uint8_t>;
extern template class IVFPQIndex<int8_t>;

/**
 * CAGRA Index - GPU-native graph ANN
 */
template <typename T>
class CAGRAIndex {
 public:
  CAGRAIndex() = default;

  explicit CAGRAIndex(const CAGRAParams &params) : params_(params) {}

  void build(const T *vectors, size_t n_vectors, size_t dim);
  SearchResult search(const T *queries, size_t n_queries, size_t k,
                      size_t num_iters = 10);

 private:
  CAGRAParams params_;
  size_t dim_ = 0;
  size_t size_ = 0;
};

extern template class CAGRAIndex<float>;

/**
 * HNSW Index - Hierarchical Navigable Small World
 */
template <typename T>
class HNSWIndex {
 public:
  HNSWIndex() = default;

  explicit HNSWIndex(const HNSWParams &params) : params_(params) {}

  void build(const T *vectors, size_t n_vectors, size_t dim);
  SearchResult search(const T *queries, size_t n_queries, size_t k);

 private:
  HNSWParams params_;
  size_t dim_ = 0;
  size_t size_ = 0;
};

extern template class HNSWIndex<float>;

/**
 * Factory functions for index creation
 */
std::unique_ptr<IVFPQIndex<float>> create_ivf_pq_float(
    const IVFPQParams &params = IVFPQParams());
std::unique_ptr<CAGRAIndex<float>> create_cagra_float(
    const CAGRAParams &params = CAGRAParams());
std::unique_ptr<HNSWIndex<float>> create_hnsw_float(
    const HNSWParams &params = HNSWParams());

}  // namespace cuvs
}  // namespace zvec

#endif  // ZVEC_CUVS_H_
