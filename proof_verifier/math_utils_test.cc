#include "proof_verifier/math_utils.h"

#include <string>

#include <gtest/gtest.h>

// Test with uint16_t type
TEST(GaussJordanEliminationTest, EmptyMatrix) {
  std::vector<uint16_t> matrix;
  int rank = GaussJordanElimination(4, &matrix);
  EXPECT_EQ(rank, 0);
}

TEST(GaussJordanEliminationTest, ZeroMatrix) {
  std::vector<uint16_t> matrix = {0, 0, 0};
  int rank = GaussJordanElimination(4, &matrix);
  EXPECT_EQ(rank, 0);
  EXPECT_EQ(matrix[0], 0);
  EXPECT_EQ(matrix[1], 0);
  EXPECT_EQ(matrix[2], 0);
}

TEST(GaussJordanEliminationTest, IdentityMatrix) {
  // Identity matrix 3x3: rows are [100, 010, 001] in binary
  // bit 0 = col 0, bit 1 = col 1, bit 2 = col 2
  std::vector<uint16_t> matrix = {
      0b001, // row 0: [1, 0, 0]
      0b010, // row 1: [0, 1, 0]
      0b100, // row 2: [0, 0, 1]
  };
  std::vector<uint16_t> expected = matrix;
  int rank = GaussJordanElimination(3, &matrix);
  EXPECT_EQ(rank, 3);
  // Identity should remain identity
  EXPECT_EQ(matrix[0], 0b001);
  EXPECT_EQ(matrix[1], 0b010);
  EXPECT_EQ(matrix[2], 0b100);
}

TEST(GaussJordanEliminationTest, FullRankMatrix) {
  // Matrix:
  // 110
  // 101
  // 010
  std::vector<uint16_t> matrix = {
      0b011, // row 0: [1, 1, 0]
      0b101, // row 1: [1, 0, 1]
      0b010, // row 2: [0, 1, 0]
  };
  int rank = GaussJordanElimination(3, &matrix);
  EXPECT_EQ(rank, 3);
  // After RREF, should become identity matrix
  EXPECT_EQ(matrix[0], 0b001);
  EXPECT_EQ(matrix[1], 0b010);
  EXPECT_EQ(matrix[2], 0b100);
}

TEST(GaussJordanEliminationTest, Rank2Matrix) {
  // Matrix:
  // 110
  // 101
  // 011
  std::vector<uint16_t> matrix = {
      0b011, // row 0: [1, 1, 0]
      0b101, // row 1: [1, 0, 1]
      0b110, // row 2: [0, 1, 1]
  };
  int rank = GaussJordanElimination(3, &matrix);
  EXPECT_EQ(rank, 2);
  EXPECT_EQ(matrix[0], 0);
  EXPECT_EQ(matrix[1], 0b011);
  EXPECT_EQ(matrix[2], 0b101);
}

TEST(GaussJordanEliminationTest, Rank1Matrix) {
  // Matrix:
  // 111
  // 111
  // 111
  // All rows identical, rank should be 1
  std::vector<uint16_t> matrix = {
      0b111, // row 0: [1, 1, 1]
      0b111, // row 1: [1, 1, 1]
      0b111, // row 2: [1, 1, 1]
  };
  int rank = GaussJordanElimination(3, &matrix);
  EXPECT_EQ(rank, 1);
  EXPECT_EQ(matrix[0], 0);
  EXPECT_EQ(matrix[1], 0);
  EXPECT_EQ(matrix[2], 0b111);
}

TEST(GaussJordanEliminationTest, SingleRow) {
  std::vector<uint16_t> matrix = {0b011};
  int rank = GaussJordanElimination(3, &matrix);
  EXPECT_EQ(rank, 1);
  EXPECT_EQ(matrix[0], 0b011);
}

TEST(GaussJordanEliminationTest, SingleRowZero) {
  std::vector<uint16_t> matrix = {0b000};
  int rank = GaussJordanElimination(3, &matrix);
  EXPECT_EQ(rank, 0);
  EXPECT_EQ(matrix[0], 0);
}

TEST(GaussJordanEliminationTest, MoreRowsThanColumns) {
  // Matrix 4x3:
  // 110
  // 010
  // 001
  // 111
  std::vector<uint16_t> matrix = {
      0b011, // row 0: [1, 1, 0]
      0b010, // row 1: [0, 1, 0]
      0b100, // row 2: [0, 0, 1]
      0b111, // row 3: [1, 1, 1]
  };
  int rank = GaussJordanElimination(3, &matrix);
  EXPECT_EQ(rank, 3); // Max rank is 3 (number of columns)
  EXPECT_EQ(matrix[0], 0);
  EXPECT_EQ(matrix[1], 0b001);
  EXPECT_EQ(matrix[2], 0b010);
  EXPECT_EQ(matrix[3], 0b100);
}

TEST(GaussJordanEliminationTest, MoreColumnsThanRows) {
  // Matrix 2x4:
  // 0110
  // 1010
  std::vector<uint16_t> matrix = {
      0b0110, // row 0: [0, 1, 1, 0]
      0b0101, // row 1: [1, 0, 1, 0]
  };
  int rank = GaussJordanElimination(4, &matrix);
  EXPECT_EQ(rank, 2);
  EXPECT_EQ(matrix[0], 0b0011);
  EXPECT_EQ(matrix[1], 0b0101);
}

TEST(GaussJordanEliminationTest, PermutedIdentity) {
  // Matrix with rows swapped:
  // 010
  // 001
  // 100
  std::vector<uint16_t> matrix = {
      0b010, // row 0: [0, 1, 0]
      0b100, // row 1: [0, 0, 1]
      0b001, // row 2: [1, 0, 0]
  };
  int rank = GaussJordanElimination(3, &matrix);
  EXPECT_EQ(rank, 3);
  // Should be reordered to identity-like form
  EXPECT_EQ(matrix[0], 0b001);
  EXPECT_EQ(matrix[1], 0b010);
  EXPECT_EQ(matrix[2], 0b100);
}

TEST(GaussJordanEliminationTest, UpperTriangular) {
  // Matrix:
  // 111
  // 011
  // 001
  std::vector<uint16_t> matrix = {
      0b111, // row 0: [1, 1, 1]
      0b110, // row 1: [0, 1, 1]
      0b100, // row 2: [0, 0, 1]
  };
  int rank = GaussJordanElimination(3, &matrix);
  EXPECT_EQ(rank, 3);
  // Should become identity
  EXPECT_EQ(matrix[0], 0b001);
  EXPECT_EQ(matrix[1], 0b010);
  EXPECT_EQ(matrix[2], 0b100);
}

TEST(GaussJordanEliminationTest, LowerTriangular) {
  // Matrix:
  // 100
  // 110
  // 111
  std::vector<uint16_t> matrix = {
      0b001, // row 0: [1, 0, 0]
      0b011, // row 1: [1, 1, 0]
      0b111, // row 2: [1, 1, 1]
  };
  int rank = GaussJordanElimination(3, &matrix);
  EXPECT_EQ(rank, 3);
  // Should become identity
  EXPECT_EQ(matrix[0], 0b001);
  EXPECT_EQ(matrix[1], 0b010);
  EXPECT_EQ(matrix[2], 0b100);
}

TEST(GaussJordanEliminationTest, WithUint32) {
  // Test with uint32_t type
  std::vector<uint32_t> matrix = {
      0b0011, // row 0: [1, 1, 0, 0]
      0b0101, // row 1: [1, 0, 1, 0]
      0b1100, // row 2: [0, 0, 1, 1]
  };
  int rank = GaussJordanElimination(4, &matrix);
  EXPECT_EQ(rank, 3);
  EXPECT_EQ(matrix[0], 0b0011);
  EXPECT_EQ(matrix[1], 0b0101);
  EXPECT_EQ(matrix[2], 0b1001);
}

TEST(GaussJordanEliminationTest, LargeBitwidth) {
  // Test with larger bitwidth
  std::vector<uint16_t> matrix = {
      0b1111111100000000,
      0b0000000011111111,
      0b1010101010101010,
  };
  int rank = GaussJordanElimination(16, &matrix);
  EXPECT_EQ(rank, 3);
  EXPECT_EQ(matrix[0], 0b0000000011111111);
  EXPECT_EQ(matrix[1], 0b0101010101010101);
  EXPECT_EQ(matrix[2], 0b1010101001010101);
}
