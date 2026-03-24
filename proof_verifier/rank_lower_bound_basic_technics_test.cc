#include "proof_verifier/rank_lower_bound_basic_technics.h"

#include <random>
#include <string>

#include <gtest/gtest.h>

#include "proof_verifier/tensor.h"

template <int n0, int n1, int n2>
void TestRankLowerBoundForcedProductA(const std::string &tensor_str,
                                      int expected_rank) {
  Tensor<n0, n1, n2> tensor = SparseStringToTensor<n0, n1, n2>(tensor_str);
  int rank = RankLowerBoundForcedProductA<n0, n1, n2>(tensor);
  EXPECT_EQ(rank, expected_rank);
}

TEST(RankLowerBoundForcedProductATest, N3) {
  constexpr int n = 3;
  TestRankLowerBoundForcedProductA<n, n, n>(
      "a10*b02*c12 + a11*b12*c12 + a12*b22*c12 + a20*b01*c12 + a20*b02*c22 + "
      "a21*b11*c12 + a21*b12*c22 + a22*b21*c12 + a22*b22*c22",
      9);
  TestRankLowerBoundForcedProductA<n, n, n>(
      "a00*b02*c02 + a01*b12*c02 + a02*b22*c02 + a10*b01*c02 + a10*b02*c12 + "
      "a11*b11*c02 + a11*b12*c12 + a12*b21*c02 + a12*b22*c12 + a20*b00*c02 + "
      "a20*b01*c12 + a21*b10*c02 + a21*b11*c12 + a22*b20*c02 + a22*b21*c12",
      12);
  TestRankLowerBoundForcedProductA<n, n, n>(
      "a00*b02*c02 + a01*b12*c02 + a02*b22*c02 + a10*b01*c02 + a10*b02*c12 + "
      "a11*b11*c02 + a11*b12*c12 + a12*b21*c02 + a12*b22*c12 + a20*b00*c02 + "
      "a20*b01*c12 + a21*b10*c02 + a21*b11*c12 + a22*b20*c02 + a22*b21*c12",
      12);
  TestRankLowerBoundForcedProductA<n, n, n>(
      " a00*b02*c12 + a01*b12*c12 + a02*b22*c12 + a10*b01*c02 + a11*b11*c02 + "
      "a12*b21*c02 + a20*b00*c02 + a20*b01*c12 + a21*b10*c02 + a21*b11*c12 + "
      "a22*b20*c02 + a22*b21*c12",
      12);
  TestRankLowerBoundForcedProductA<n, n, n>(
      "a00*b02*c11 + a01*b12*c11 + a02*b22*c11 + a10*b01*c11 + a11*b11*c11 + "
      "a12*b21*c11 + a20*b01*c12 + a20*b02*c22 + a21*b11*c12 + a21*b12*c22 + "
      "a22*b21*c12 + a22*b22*c22",
      12);
  TestRankLowerBoundForcedProductA<n, n, n>(
      "a00*b02*c12 + a01*b12*c12 + a02*b22*c12 + a10*b01*c02 + a11*b11*c02 + "
      "a12*b21*c02 + a20*b00*c02 + a20*b01*c12 + a20*b02*c22 + a21*b10*c02 + "
      "a21*b11*c12 + a21*b12*c22 + a22*b20*c02 + a22*b21*c12 + a22*b22*c22",
      15);
  TestRankLowerBoundForcedProductA<n, n, n>(
      "a00*b02*c02 + a00*b02*c11 + a01*b12*c02 + a01*b12*c11 + a02*b22*c02 + "
      "a02*b22*c11 + a10*b01*c11 + a11*b11*c11 + a12*b21*c11 + a20*b00*c02 + "
      "a20*b02*c22 + a21*b10*c02 + a21*b12*c22 + a22*b20*c02 + a22*b22*c22",
      12);
  TestRankLowerBoundForcedProductA<n, n, n>(
      "a00*b02*c02 + a01*b12*c02 + a02*b22*c02 + a10*b02*c12 + a11*b12*c12 + "
      "a12*b22*c12 + a20*b00*c02 + a20*b01*c12 + a20*b02*c22 + a21*b10*c02 + "
      "a21*b11*c12 + a21*b12*c22 + a22*b20*c02 + a22*b21*c12 + a22*b22*c22",
      15);
  TestRankLowerBoundForcedProductA<n, n, n>(
      "a00*b02*c02 + a00*b02*c11 + a01*b12*c02 + a01*b12*c11 + a02*b22*c02 + "
      "a02*b22*c11 + a10*b01*c11 + a10*b02*c21 + a11*b11*c11 + a11*b12*c21 + "
      "a12*b21*c11 + a12*b22*c21 + a20*b00*c02 + a20*b02*c21 + a21*b10*c02 + "
      "a21*b12*c21 + a22*b20*c02 + a22*b22*c21",
      12);
  TestRankLowerBoundForcedProductA<n, n, n>(
      "a00*b02*c20 + a01*b11*c10 + a02*b02*c00 + a02*b11*c00 + a02*b21*c10 + "
      "a02*b22*c20 + a10*b02*c21 + a11*b11*c11 + a12*b02*c01 + a12*b11*c01 + "
      "a12*b21*c11 + a12*b22*c21 + a20*b02*c22 + a21*b11*c12 + a22*b02*c02 + "
      "a22*b11*c02 + a22*b21*c12 + a22*b22*c22",
      15);
}

TEST(RankLowerBoundForcedProductATest, N234) {
  TestRankLowerBoundForcedProductA<2, 4, 3>(
      "a00*b02*c11 + a01*b12*c11 + a02*b22*c11 + a03*b32*c11 + a10*b01*c11 + "
      "a10*b02*c21 + a11*b11*c11 + a11*b12*c21 + a12*b21*c11 + a12*b22*c21 + "
      "a13*b31*c11 + a13*b32*c21",
      12);
  TestRankLowerBoundForcedProductA<3, 4, 2>(
      "a10*b01*c02 + a11*b11*c02 + a12*b21*c02 + a13*b31*c02 + a20*b00*c02 + "
      "a20*b01*c12 + a21*b10*c02 + a21*b11*c12 + a22*b20*c02 + a22*b21*c12 + "
      "a23*b30*c02 + a23*b31*c12",
      12);
  TestRankLowerBoundForcedProductA<4, 2, 3>(
      "a00*b02*c20 + a01*b02*c00 + a01*b11*c10 + a01*b12*c20 + a10*b02*c21 + "
      "a11*b02*c01 + a11*b11*c11 + a11*b12*c21 + a20*b02*c22 + a21*b02*c02 + "
      "a21*b11*c12 + a21*b12*c22 + a30*b02*c23 + a31*b02*c03 + a31*b11*c13 + "
      "a31*b12*c23",
      16);
}
