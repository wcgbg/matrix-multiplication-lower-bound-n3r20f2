#include "proof_verifier/dynamic_matrix.h"

#include <gtest/gtest.h>

TEST(DynamicMatrixTest, Constructor_ZeroMatrix) {
  DynamicMatrix m(2, 3);
  EXPECT_EQ(m.rows(), 2);
  EXPECT_EQ(m.cols(), 3);
  EXPECT_TRUE(m.IsZero());
  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 3; ++j) {
      EXPECT_EQ(m(i, j), 0);
    }
  }
  EXPECT_EQ(m.Rank(), 0);
}

TEST(DynamicMatrixTest, Resize) {
  DynamicMatrix m(2, 3);
  m(0, 0) = 1;
  m(1, 2) = 1;
  EXPECT_EQ(m.ToString(), "[100,001]");
  m.ResizeRows(3);
  EXPECT_EQ(m.rows(), 3);
  EXPECT_EQ(m.cols(), 3);
  EXPECT_EQ(m.ToString(), "[100,001,000]");
}

TEST(DynamicMatrixTest, SetAndGet) {
  DynamicMatrix m(2, 2);
  m(0, 0) = 1;
  m(1, 1) = 1;
  EXPECT_EQ(m(0, 0), 1);
  EXPECT_EQ(m(0, 1), 0);
  EXPECT_EQ(m(1, 0), 0);
  EXPECT_EQ(m(1, 1), 1);
  EXPECT_FALSE(m.IsZero());
  EXPECT_EQ(m.Rank(), 2);
}

TEST(DynamicMatrixTest, ToString) {
  DynamicMatrix m(3, 4);
  m(0, 0) = 1;
  m(0, 3) = 1; // row 0: 1001
  m(1, 0) = 1;
  m(1, 1) = 1;
  m(1, 3) = 1; // row 1: 1101
  m(2, 3) = 1; // row 2: 0001
  EXPECT_EQ(m.ToString(), "[1001,1101,0001]");
}

TEST(DynamicMatrixTest, ToString_Zero) {
  DynamicMatrix m(2, 2);
  EXPECT_EQ(m.ToString(), "[00,00]");
}

TEST(DynamicMatrixTest, ToString_SingleElement) {
  DynamicMatrix m(1, 1);
  m(0, 0) = 1;
  EXPECT_EQ(m.ToString(), "[1]");
}

TEST(DynamicMatrixTest, Plus_SameSize) {
  DynamicMatrix a(2, 2);
  a(0, 0) = 1;
  a(1, 1) = 1;
  DynamicMatrix b(2, 2);
  b(0, 0) = 1;

  DynamicMatrix sum = a.Plus(b);
  EXPECT_EQ(sum(0, 0), 0); // 1 + 1 = 0 in F_2
  EXPECT_EQ(sum(1, 1), 1); // 0 + 1 = 1
  EXPECT_EQ(sum.rows(), 2);
  EXPECT_EQ(sum.cols(), 2);
}

TEST(DynamicMatrixTest, Plus_Zero) {
  DynamicMatrix a(2, 2);
  a(0, 1) = 1;
  DynamicMatrix b(2, 2);

  DynamicMatrix sum = a.Plus(b);
  EXPECT_EQ(sum(0, 1), 1);
  EXPECT_TRUE(sum.Plus(a).IsZero()); // sum + a = a + a = 0 in F_2
}

TEST(DynamicMatrixTest, IsZero) {
  DynamicMatrix m(2, 2);
  EXPECT_TRUE(m.IsZero());
  m(0, 0) = 1;
  EXPECT_FALSE(m.IsZero());
  m(0, 0) = 0;
  EXPECT_TRUE(m.IsZero());
}

TEST(DynamicMatrixTest, Rank_Zero) {
  DynamicMatrix m(3, 4);
  EXPECT_EQ(m.Rank(), 0);
}

TEST(DynamicMatrixTest, Rank_Identity2x2) {
  DynamicMatrix m(2, 2);
  m(0, 0) = 1;
  m(1, 1) = 1;
  EXPECT_EQ(m.Rank(), 2);
}

TEST(DynamicMatrixTest, Rank_Identity3x3) {
  DynamicMatrix m(3, 3);
  m(0, 0) = 1;
  m(1, 1) = 1;
  m(2, 2) = 1;
  EXPECT_EQ(m.Rank(), 3);
}

TEST(DynamicMatrixTest, Rank_One) {
  DynamicMatrix m(2, 2);
  m(0, 0) = 1;
  m(0, 1) = 1;
  // Row 0: [1, 1], Row 1: [0, 0]
  EXPECT_EQ(m.Rank(), 1);
}

TEST(DynamicMatrixTest, Rank_LinearlyDependentRows) {
  DynamicMatrix m(2, 2);
  m(0, 0) = 1;
  m(0, 1) = 1;
  m(1, 0) = 1;
  m(1, 1) = 1;
  // Both rows [1, 1]
  EXPECT_EQ(m.Rank(), 1);
}

TEST(DynamicMatrixTest, Rank_NonSquare_RowsMoreThanCols) {
  DynamicMatrix m(4, 2);
  m(0, 0) = 1;
  m(1, 1) = 1;
  // Two independent columns, rank = 2
  EXPECT_EQ(m.Rank(), 2);
}

TEST(DynamicMatrixTest, Rank_NonSquare_ColsMoreThanRows) {
  DynamicMatrix m(2, 4);
  m(0, 0) = 1;
  m(0, 1) = 1;
  m(1, 2) = 1;
  m(1, 3) = 1;
  // Two independent rows, rank = 2
  EXPECT_EQ(m.Rank(), 2);
}

TEST(DynamicMatrixTest, Rank_EmptyRows) {
  DynamicMatrix m(0, 3);
  EXPECT_EQ(m.Rank(), 0);
}

TEST(DynamicMatrixTest, Rank_EmptyCols) {
  DynamicMatrix m(3, 0);
  EXPECT_EQ(m.Rank(), 0);
}

TEST(DynamicMatrixTest, Plus_DimensionsMatch) {
  DynamicMatrix a(1, 1);
  DynamicMatrix b(1, 1);
  a(0, 0) = 1;
  DynamicMatrix sum = a.Plus(b);
  EXPECT_EQ(sum.ToString(), "[1]");
}
