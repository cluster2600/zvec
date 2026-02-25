// Copyright 2025-present the zvec project
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>
#include <rocksdb/db.h>
#include <zvec/db/common/rocksdb_context.h>
#include <zvec/db/status.h>

namespace zvec {

/*! Persistent vector storage backed by RocksDB.
 *
 * Stores raw float vectors and optional PQ codes in separate column families.
 * Designed for integration with the Metal GPU backend: vectors are loaded from
 * RocksDB into contiguous buffers that can be transferred to GPU memory.
 *
 * Column families:
 *   - "vectors"  : raw float vectors keyed by uint64 ID
 *   - "pq_codes" : PQ-encoded codes keyed by uint64 ID
 *   - "metadata" : dimension, count, and quantizer parameters
 */
class VectorStorage {
 public:
  VectorStorage() = default;
  ~VectorStorage() { close(); }

  //! Create a new vector storage at the given path.
  //! @param path Directory for the RocksDB instance.
  //! @param dim Vector dimension.
  Status create(const std::string &path, size_t dim) {
    dim_ = dim;
    Status s = ctx_.create(path, {"default", "vectors", "pq_codes", "metadata"});
    if (!s.ok()) return s;
    return write_metadata();
  }

  //! Open an existing vector storage.
  //! @param path Directory of the RocksDB instance.
  //! @param read_only Open in read-only mode.
  Status open(const std::string &path, bool read_only = false) {
    Status s = ctx_.open(path, {"default", "vectors", "pq_codes", "metadata"},
                         read_only);
    if (!s.ok()) return s;
    return read_metadata();
  }

  //! Close the storage.
  Status close() { return ctx_.close(); }

  //! Retrieve the vector dimension.
  size_t dim() const { return dim_; }

  //! Retrieve the number of stored vectors.
  size_t count() const { return count_; }

  //! Insert a vector with the given ID.
  //! @param id Unique vector identifier.
  //! @param vec Float vector of size dim().
  Status put_vector(uint64_t id, const float *vec) {
    auto *cf = ctx_.get_cf("vectors");
    if (!cf) return Status::InternalError("vectors CF not found");

    rocksdb::Slice key(reinterpret_cast<const char *>(&id), sizeof(id));
    rocksdb::Slice value(reinterpret_cast<const char *>(vec),
                         dim_ * sizeof(float));
    auto s = ctx_.db_->Put(ctx_.write_opts_, cf, key, value);
    if (!s.ok()) return Status::InternalError(s.ToString());

    ++count_;
    return Status::OK();
  }

  //! Retrieve a vector by ID.
  //! @param id Vector identifier.
  //! @param vec Output buffer of size dim().
  //! @return OK if found, NotFound otherwise.
  Status get_vector(uint64_t id, float *vec) const {
    auto *cf = ctx_.get_cf("vectors");
    if (!cf) return Status::InternalError("vectors CF not found");

    rocksdb::Slice key(reinterpret_cast<const char *>(&id), sizeof(id));
    std::string value;
    auto s = ctx_.db_->Get(ctx_.read_opts_, cf, key, &value);
    if (!s.ok()) return Status::NotFound("vector not found");

    if (value.size() != dim_ * sizeof(float)) {
      return Status::InternalError("corrupted vector data");
    }
    std::memcpy(vec, value.data(), dim_ * sizeof(float));
    return Status::OK();
  }

  //! Store PQ codes for a vector.
  //! @param id Vector identifier.
  //! @param codes PQ code array of size m.
  //! @param m Number of sub-quantizers.
  Status put_pq_codes(uint64_t id, const uint8_t *codes, size_t m) {
    auto *cf = ctx_.get_cf("pq_codes");
    if (!cf) return Status::InternalError( "pq_codes CF not found");

    rocksdb::Slice key(reinterpret_cast<const char *>(&id), sizeof(id));
    rocksdb::Slice value(reinterpret_cast<const char *>(codes), m);
    auto s = ctx_.db_->Put(ctx_.write_opts_, cf, key, value);
    if (!s.ok()) return Status::InternalError( s.ToString());

    return Status::OK();
  }

  //! Retrieve PQ codes for a vector.
  //! @param id Vector identifier.
  //! @param codes Output buffer of size m.
  //! @param m Number of sub-quantizers.
  Status get_pq_codes(uint64_t id, uint8_t *codes, size_t m) const {
    auto *cf = ctx_.get_cf("pq_codes");
    if (!cf) return Status::InternalError( "pq_codes CF not found");

    rocksdb::Slice key(reinterpret_cast<const char *>(&id), sizeof(id));
    std::string value;
    auto s = ctx_.db_->Get(ctx_.read_opts_, cf, key, &value);
    if (!s.ok()) return Status::NotFound( "pq codes not found");

    size_t copy_len = std::min(m, value.size());
    std::memcpy(codes, value.data(), copy_len);
    return Status::OK();
  }

  //! Batch insert vectors.
  //! @param ids Array of n vector IDs.
  //! @param vecs Contiguous float array (n x dim), row-major.
  //! @param n Number of vectors.
  Status put_vectors_batch(const uint64_t *ids, const float *vecs, size_t n) {
    auto *cf = ctx_.get_cf("vectors");
    if (!cf) return Status::InternalError( "vectors CF not found");

    rocksdb::WriteBatch batch;
    for (size_t i = 0; i < n; ++i) {
      rocksdb::Slice key(reinterpret_cast<const char *>(&ids[i]),
                         sizeof(uint64_t));
      rocksdb::Slice value(reinterpret_cast<const char *>(vecs + i * dim_),
                           dim_ * sizeof(float));
      batch.Put(cf, key, value);
    }

    auto s = ctx_.db_->Write(ctx_.write_opts_, &batch);
    if (!s.ok()) return Status::InternalError( s.ToString());

    count_ += n;
    return Status::OK();
  }

  //! Load all vectors into a contiguous buffer for GPU transfer.
  //! Returns vectors in insertion order. The caller should allocate
  //! the output buffers.
  //! @param out_ids Output vector IDs (resized to count).
  //! @param out_vecs Output float buffer (count x dim), row-major.
  Status load_all(std::vector<uint64_t> &out_ids,
                  std::vector<float> &out_vecs) const {
    auto *cf = ctx_.get_cf("vectors");
    if (!cf) return Status::InternalError( "vectors CF not found");

    out_ids.clear();
    out_vecs.clear();
    out_ids.reserve(count_);
    out_vecs.reserve(count_ * dim_);

    std::unique_ptr<rocksdb::Iterator> it(
        ctx_.db_->NewIterator(ctx_.read_opts_, cf));
    for (it->SeekToFirst(); it->Valid(); it->Next()) {
      auto key = it->key();
      auto value = it->value();

      if (key.size() != sizeof(uint64_t) ||
          value.size() != dim_ * sizeof(float)) {
        continue;
      }

      uint64_t id;
      std::memcpy(&id, key.data(), sizeof(uint64_t));
      out_ids.push_back(id);

      size_t offset = out_vecs.size();
      out_vecs.resize(offset + dim_);
      std::memcpy(out_vecs.data() + offset, value.data(),
                   dim_ * sizeof(float));
    }

    return Status::OK();
  }

  //! Flush pending writes to disk.
  Status flush() { return ctx_.flush(); }

 private:
  VectorStorage(const VectorStorage &) = delete;
  VectorStorage &operator=(const VectorStorage &) = delete;

  Status write_metadata() {
    auto *cf = ctx_.get_cf("metadata");
    if (!cf) return Status::InternalError( "metadata CF not found");

    // Write dimension
    auto s = ctx_.db_->Put(ctx_.write_opts_, cf, "dim",
                            rocksdb::Slice(reinterpret_cast<const char *>(&dim_),
                                           sizeof(dim_)));
    if (!s.ok()) return Status::InternalError( s.ToString());

    return Status::OK();
  }

  Status read_metadata() {
    auto *cf = ctx_.get_cf("metadata");
    if (!cf) return Status::InternalError( "metadata CF not found");

    std::string value;
    auto s = ctx_.db_->Get(ctx_.read_opts_, cf, "dim", &value);
    if (!s.ok()) return Status::InternalError( "missing dim");
    if (value.size() != sizeof(size_t)) {
      return Status::InternalError( "corrupted dim");
    }
    std::memcpy(&dim_, value.data(), sizeof(size_t));

    // Count vectors
    auto *vec_cf = ctx_.get_cf("vectors");
    if (vec_cf) {
      count_ = 0;
      std::unique_ptr<rocksdb::Iterator> it(
          ctx_.db_->NewIterator(ctx_.read_opts_, vec_cf));
      for (it->SeekToFirst(); it->Valid(); it->Next()) {
        ++count_;
      }
    }

    return Status::OK();
  }

  mutable RocksdbContext ctx_;
  size_t dim_{0};
  size_t count_{0};
};

}  // namespace zvec
