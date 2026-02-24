//
//  test_metal.cc
//  Tests for Metal GPU acceleration
//
//  Created by cluster2600 on 2026-02-22.
//

#include <cmath>
#include <cstdlib>
#include <random>
#include <vector>
#include "gtest/gtest.h"
#include "zvec_metal.h"

class MetalTest : public ::testing::Test {
 protected:
  void SetUp() override {
    device_ = zvec_metal_create();
  }

  void TearDown() override {
    if (device_) {
      zvec_metal_destroy(device_);
    }
  }

  ZvecMetalDevice *device_ = nullptr;
};

TEST_F(MetalTest, Availability) {
  int available = zvec_metal_available();
  // Test passes regardless of Metal availability
  EXPECT_TRUE(available == 0 || available == 1);
}

TEST_F(MetalTest, DeviceInfo) {
  if (!device_) {
    GTEST_SKIP() << "Metal not available";
  }

  const char *name = zvec_metal_device_name(device_);
  EXPECT_NE(name, nullptr);
  EXPECT_GT(strlen(name), 0);

  uint64_t memory = zvec_metal_device_memory(device_);
  EXPECT_GT(memory, 0);
}

TEST_F(MetalTest, L2Distance) {
  if (!device_) {
    GTEST_SKIP() << "Metal not available";
  }

  const int N = 10;
  const int M = 100;
  const int D = 128;

  std::vector<float> queries(N * D);
  std::vector<float> database(M * D);
  std::vector<float> distances(N * M);

  // Fill with random data
  std::mt19937 rng(42);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

  for (auto &v : queries) v = dist(rng);
  for (auto &v : database) v = dist(rng);

  // Compute distances
  int result = zvec_metal_l2_distance(device_, queries.data(), database.data(),
                                      distances.data(), N, M, D);

  EXPECT_EQ(result, 0);

  // Verify first distance manually
  float expected = 0.0f;
  for (int i = 0; i < D; i++) {
    float diff = queries[i] - database[i];
    expected += diff * diff;
  }

  EXPECT_NEAR(distances[0], expected, 1e-3);
}

TEST_F(MetalTest, InnerProduct) {
  if (!device_) {
    GTEST_SKIP() << "Metal not available";
  }

  const int N = 5;
  const int M = 20;
  const int D = 64;

  std::vector<float> queries(N * D);
  std::vector<float> database(M * D);
  std::vector<float> results(N * M);

  std::mt19937 rng(42);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

  for (auto &v : queries) v = dist(rng);
  for (auto &v : database) v = dist(rng);

  int result = zvec_metal_inner_product(
      device_, queries.data(), database.data(), results.data(), N, M, D);

  EXPECT_EQ(result, 0);

  // Verify
  float expected = 0.0f;
  for (int i = 0; i < D; i++) {
    expected += queries[i] * database[i];
  }

  EXPECT_NEAR(results[0], expected, 1e-3);
}

TEST_F(MetalTest, Normalize) {
  if (!device_) {
    GTEST_SKIP() << "Metal not available";
  }

  const int N = 10;
  const int D = 32;

  std::vector<float> vectors(N * D);

  std::mt19937 rng(42);
  std::uniform_real_distribution<float> dist(-2.0f, 2.0f);

  for (auto &v : vectors) v = dist(rng);

  int result = zvec_metal_normalize(device_, vectors.data(), N, D);

  EXPECT_EQ(result, 0);

  // Check normalization
  for (int i = 0; i < N; i++) {
    float norm = 0.0f;
    for (int j = 0; j < D; j++) {
      norm += vectors[i * D + j] * vectors[i * D + j];
    }
    EXPECT_NEAR(sqrt(norm), 1.0f, 1e-3);
  }
}

TEST_F(MetalTest, NullDevice) {
  // Test with null device
  int result =
      zvec_metal_l2_distance(nullptr, nullptr, nullptr, nullptr, 1, 1, 1);
  EXPECT_NE(result, 0);
}
