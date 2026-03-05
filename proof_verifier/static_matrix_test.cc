#include "proof_verifier/static_matrix.h"

#include <gtest/gtest.h>

#include <random>

// Test StaticMatrix<1>
TEST(MatrixTest, Matrix1_DefaultConstructor) {
  StaticMatrix<1> m;
  EXPECT_TRUE(m.IsZero());
  EXPECT_EQ(m.Get(0, 0), 0);
  EXPECT_EQ(m.Rank(), 0);
}

TEST(MatrixTest, Matrix1_SetAndGet) {
  StaticMatrix<1> m;
  m.Set(0, 0, 1);
  EXPECT_EQ(m.Get(0, 0), 1);
  EXPECT_FALSE(m.IsZero());
  EXPECT_EQ(m.Rank(), 1);

  m.Set(0, 0, 0);
  EXPECT_EQ(m.Get(0, 0), 0);
  EXPECT_TRUE(m.IsZero());
  EXPECT_EQ(m.Rank(), 0);
}

TEST(MatrixTest, Matrix1_Plus) {
  StaticMatrix<1> m1;
  StaticMatrix<1> m2;
  m1.Set(0, 0, 1);
  m2.Set(0, 0, 1);

  StaticMatrix<1> sum = m1.Plus(m2);
  EXPECT_TRUE(sum.IsZero()); // 1 + 1 = 0 in F_2

  m2.Set(0, 0, 0);
  sum = m1.Plus(m2);
  EXPECT_FALSE(sum.IsZero());
  EXPECT_EQ(sum.Get(0, 0), 1);
}

TEST(MatrixTest, Matrix1_FromData) {
  StaticMatrix<1> m(1); // bit 0 set
  EXPECT_EQ(m.Get(0, 0), 1);
  EXPECT_EQ(m.Data(), 1);
}

// Test StaticMatrix<2>
TEST(MatrixTest, Matrix2_DefaultConstructor) {
  StaticMatrix<2> m;
  EXPECT_TRUE(m.IsZero());
  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 2; ++j) {
      EXPECT_EQ(m.Get(i, j), 0);
    }
  }
  EXPECT_EQ(m.Rank(), 0);
}

TEST(MatrixTest, Matrix2_SetAndGet) {
  StaticMatrix<2> m;
  m.Set(0, 0, 1);
  m.Set(1, 1, 1);
  EXPECT_EQ(m.Get(0, 0), 1);
  EXPECT_EQ(m.Get(0, 1), 0);
  EXPECT_EQ(m.Get(1, 0), 0);
  EXPECT_EQ(m.Get(1, 1), 1);
  EXPECT_EQ(m.Rank(), 2);
}

TEST(MatrixTest, Matrix2_Identity) {
  StaticMatrix<2> m;
  m.Set(0, 0, 1);
  m.Set(1, 1, 1);
  EXPECT_EQ(m.Rank(), 2);
}

TEST(MatrixTest, Matrix2_Rank1) {
  StaticMatrix<2> m;
  m.Set(0, 0, 1);
  m.Set(0, 1, 1);
  // Row 0: [1, 1], Row 1: [0, 0]
  EXPECT_EQ(m.Rank(), 1);
}

TEST(MatrixTest, Matrix2_Rank0) {
  StaticMatrix<2> m;
  EXPECT_EQ(m.Rank(), 0);
}

TEST(MatrixTest, Matrix2_Plus) {
  StaticMatrix<2> m1;
  StaticMatrix<2> m2;
  m1.Set(0, 0, 1);
  m1.Set(1, 1, 1);
  m2.Set(0, 0, 1);

  StaticMatrix<2> sum = m1.Plus(m2);
  EXPECT_EQ(sum.Get(0, 0), 0); // 1 + 1 = 0
  EXPECT_EQ(sum.Get(1, 1), 1); // 0 + 1 = 1
}

// Test StaticMatrix<3>
TEST(MatrixTest, Matrix3_DefaultConstructor) {
  StaticMatrix<3> m;
  EXPECT_TRUE(m.IsZero());
  EXPECT_EQ(m.Rank(), 0);
}

TEST(MatrixTest, Matrix3_Identity) {
  StaticMatrix<3> m;
  m.Set(0, 0, 1);
  m.Set(1, 1, 1);
  m.Set(2, 2, 1);
  EXPECT_EQ(m.Rank(), 3);
}

TEST(MatrixTest, Matrix3_Rank2) {
  StaticMatrix<3> m;
  m.Set(0, 0, 1);
  m.Set(1, 0, 1);
  m.Set(1, 1, 1);
  // Two linearly independent rows
  EXPECT_EQ(m.Rank(), 2);
}

TEST(MatrixTest, Matrix3_Rank1) {
  StaticMatrix<3> m;
  m.Set(0, 0, 1);
  m.Set(0, 1, 1);
  m.Set(0, 2, 1);
  // Only one non-zero row
  EXPECT_EQ(m.Rank(), 1);
}

TEST(MatrixTest, Matrix3_Plus) {
  StaticMatrix<3> m1;
  StaticMatrix<3> m2;
  m1.Set(0, 0, 1);
  m1.Set(1, 1, 1);
  m2.Set(0, 0, 1);
  m2.Set(2, 2, 1);

  StaticMatrix<3> sum = m1.Plus(m2);
  EXPECT_EQ(sum.Get(0, 0), 0); // 1 + 1 = 0
  EXPECT_EQ(sum.Get(1, 1), 1); // 0 + 1 = 1
  EXPECT_EQ(sum.Get(2, 2), 1); // 0 + 1 = 1
}

// Test StaticMatrix<4>
TEST(MatrixTest, Matrix4_DefaultConstructor) {
  StaticMatrix<4> m;
  EXPECT_TRUE(m.IsZero());
  EXPECT_EQ(m.Rank(), 0);
}

TEST(MatrixTest, Matrix4_Identity) {
  StaticMatrix<4> m;
  m.Set(0, 0, 1);
  m.Set(1, 1, 1);
  m.Set(2, 2, 1);
  m.Set(3, 3, 1);
  EXPECT_EQ(m.Rank(), 4);
}

TEST(MatrixTest, Matrix4_Rank3) {
  StaticMatrix<4> m;
  m.Set(0, 0, 1);
  m.Set(0, 1, 1);
  m.Set(1, 0, 1);
  m.Set(1, 2, 1);
  m.Set(3, 2, 1);
  m.Set(3, 3, 1);
  // Three linearly independent rows
  EXPECT_EQ(m.Rank(), 3);
}

TEST(MatrixTest, Matrix4_FromData) {
  // Create a matrix with specific bit pattern
  // Row 0: [1, 0, 0, 0] = bit 0
  // Row 1: [0, 1, 0, 0] = bit 5
  // Row 2: [0, 0, 1, 0] = bit 10
  // Row 3: [0, 0, 0, 1] = bit 15
  StaticMatrix<4>::DataType data =
      (1U << 0) | (1U << 5) | (1U << 10) | (1U << 15);
  StaticMatrix<4> m(data);
  EXPECT_EQ(m.Get(0, 0), 1);
  EXPECT_EQ(m.Get(1, 1), 1);
  EXPECT_EQ(m.Get(2, 2), 1);
  EXPECT_EQ(m.Get(3, 3), 1);
  EXPECT_EQ(m.Rank(), 4);
}

// Test edge cases
TEST(MatrixTest, Matrix2_LinearlyDependentRows) {
  StaticMatrix<2> m;
  m.Set(0, 0, 1);
  m.Set(0, 1, 1);
  m.Set(1, 0, 1);
  m.Set(1, 1, 1);
  // Both rows are [1, 1], so rank should be 1
  EXPECT_EQ(m.Rank(), 1);
}

TEST(MatrixTest, Matrix3_AllOnes) {
  StaticMatrix<3> m;
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      m.Set(i, j, 1);
    }
  }
  // All rows are [1, 1, 1], so rank should be 1
  EXPECT_EQ(m.Rank(), 1);
}

TEST(MatrixTest, Matrix3_UpperTriangular) {
  StaticMatrix<3> m;
  m.Set(0, 0, 1);
  m.Set(0, 1, 1);
  m.Set(0, 2, 1);
  m.Set(1, 1, 1);
  m.Set(1, 2, 1);
  m.Set(2, 2, 1);
  // Upper triangular matrix, should have rank 3
  EXPECT_EQ(m.Rank(), 3);
}

TEST(MatrixTest, Matrix3_ZeroAfterPlus) {
  StaticMatrix<3> m1;
  StaticMatrix<3> m2;
  m1.Set(0, 0, 1);
  m1.Set(1, 1, 1);
  m2.Set(0, 0, 1);
  m2.Set(1, 1, 1);

  StaticMatrix<3> sum = m1.Plus(m2);
  EXPECT_TRUE(sum.IsZero());
  EXPECT_EQ(sum.Rank(), 0);
}

TEST(MatrixTest, DataAccess) {
  StaticMatrix<2> m;
  m.Set(0, 0, 1);
  m.Set(1, 1, 1);
  StaticMatrix<2>::DataType data = m.Data();
  EXPECT_EQ(data, (1U << 0) | (1U << 3)); // bit 0 and bit 3 (1*2+1)
}

// Test FromString round-trip with random matrices
template <int n0, int n1 = n0>
void TestFromDenseStringRoundTripOne(std::mt19937 *gen) {
  std::uniform_int_distribution<StaticMatrixData<n0, n1>> dist(
      0, (1 << (n0 * n1)) - 1);
  StaticMatrix<n0, n1> m(dist(*gen));
  EXPECT_EQ(m, (StaticMatrix<n0, n1>::FromString(m.ToString())));
}

TEST(MatrixTest, FromDenseString_RoundTrip) {
  std::mt19937 gen;
  const int kNumTrials = 100;
  for (int trial = 0; trial < kNumTrials; ++trial) {
    TestFromDenseStringRoundTripOne<1>(&gen);
    TestFromDenseStringRoundTripOne<2>(&gen);
    TestFromDenseStringRoundTripOne<3>(&gen);
    TestFromDenseStringRoundTripOne<4>(&gen);
    TestFromDenseStringRoundTripOne<2, 3>(&gen);
  }
}

// Test ToString
TEST(MatrixTest, ToCompactString_Matrix1) {
  StaticMatrix<1> m;
  EXPECT_EQ(m.ToString(), "[0]");
  m.Set(0, 0, 1);
  EXPECT_EQ(m.ToString(), "[1]");
}
TEST(MatrixTest, ToCompactString_Matrix3) {
  // Matrix from comment: [101,110,001]
  StaticMatrix<3> m;
  m.Set(0, 0, 1); // row 0: 101
  m.Set(0, 2, 1);
  m.Set(1, 0, 1); // row 1: 110
  m.Set(1, 1, 1);
  m.Set(2, 2, 1); // row 2: 001
  EXPECT_EQ(m.ToString(), "[101,110,001]");
}
TEST(MatrixTest, ToCompactString_Matrix4_Identity) {
  StaticMatrix<4> m;
  m.Set(0, 0, 1);
  m.Set(1, 1, 1);
  m.Set(2, 2, 1);
  m.Set(3, 3, 1);
  EXPECT_EQ(m.ToString(), "[1000,0100,0010,0001]");
}

// Test operator* (matrix multiplication)
TEST(MatrixTest, Matrix_Multiply) {
  // A = [1, 1; 0, 1]
  StaticMatrix<2> a;
  a.Set(0, 0, 1);
  a.Set(0, 1, 1);
  a.Set(1, 1, 1);

  // B = [1, 0; 1, 1]
  StaticMatrix<2> b;
  b.Set(0, 0, 1);
  b.Set(1, 0, 1);
  b.Set(1, 1, 1);

  // A * B = [1, 1; 0, 1] * [1, 0; 1, 1]
  //        = [1*1+1*1, 1*0+1*1; 0*1+1*1, 0*0+1*1]
  //        = [0, 1; 1, 1] (in F_2)
  StaticMatrix<2> product = a * b;
  EXPECT_EQ(product.Get(0, 0), 0);
  EXPECT_EQ(product.Get(0, 1), 1);
  EXPECT_EQ(product.Get(1, 0), 1);
  EXPECT_EQ(product.Get(1, 1), 1);
}

// Test Inversed
TEST(MatrixTest, Matrix1_Inverse) {
  StaticMatrix<1> m;
  m.Set(0, 0, 1);

  StaticMatrix<1> inv = m.Inversed();
  EXPECT_EQ(inv.Get(0, 0), 1); // 1^-1 = 1 in F_2

  // Verify m * inv = identity
  StaticMatrix<1> product = m * inv;
  EXPECT_EQ(product.Get(0, 0), 1);

  // Test non-invertible (zero matrix)
  StaticMatrix<1> zero;
  StaticMatrix<1> zero_inv = zero.Inversed();
  EXPECT_TRUE(zero_inv.IsZero());
}

TEST(MatrixTest, Matrix3_Inverse_Identity) {
  StaticMatrix<3> identity = StaticMatrix<3>::Identity();
  StaticMatrix<3> inv = identity.Inversed();
  EXPECT_TRUE(inv.IsIdentity());
}

TEST(MatrixTest, Matrix3_Inverse) {
  constexpr int n = 3;
  for (StaticMatrixData<n> data = 0; data < (StaticMatrixData<n>(1) << (n * n));
       ++data) {
    StaticMatrix<n> a(data);
    StaticMatrix<n> inv = a.Inversed();
    if (a.Rank() == n) {
      EXPECT_TRUE((a * inv).IsIdentity());
      EXPECT_TRUE((inv * a).IsIdentity());
      StaticMatrix<n> inv_inv = inv.Inversed();
      EXPECT_EQ(inv_inv, a);
    } else {
      EXPECT_TRUE(inv.IsZero());
    }
  }
}

TEST(MatrixTest, Matrix3_Inverse_NonInvertible) {
  // Matrix with rank 2 (not invertible)
  StaticMatrix<3> m;
  m.Set(0, 0, 1);
  m.Set(0, 1, 1);
  m.Set(1, 0, 1);
  m.Set(1, 1, 0);
  // Row 2 is all zeros, so rank = 2 < 3
  EXPECT_EQ(m.Rank(), 2);
  StaticMatrix<3> inv = m.Inversed();
  EXPECT_TRUE(inv.IsZero()); // Should return zero matrix for non-invertible
}

TEST(MatrixTest, Matrix4_Inverse) {
  // A = [1, 0, 0, 0; 1, 1, 0, 0; 0, 1, 1, 0; 0, 0, 1, 1]
  StaticMatrix<4> a;
  a.Set(0, 0, 1);
  a.Set(1, 0, 1);
  a.Set(1, 1, 1);
  a.Set(2, 1, 1);
  a.Set(2, 2, 1);
  a.Set(3, 2, 1);
  a.Set(3, 3, 1);

  StaticMatrix<4> inv = a.Inversed();
  EXPECT_TRUE((a * inv).IsIdentity());
  EXPECT_TRUE((inv * a).IsIdentity());
  StaticMatrix<4> inv_inv = inv.Inversed();
  EXPECT_EQ(inv_inv, a);
}

// Test Transposed
TEST(MatrixTest, Matrix3_Transposed) {
  // Original matrix:
  // 1 0 1
  // 1 1 0
  // 0 0 1
  StaticMatrix<3> m;
  m.Set(0, 0, 1);
  m.Set(0, 2, 1);
  m.Set(1, 0, 1);
  m.Set(1, 1, 1);
  m.Set(2, 2, 1);
  EXPECT_EQ(m.ToString(), "[101,110,001]");

  // Transposed should be:
  // 1 1 0
  // 0 1 0
  // 1 0 1
  StaticMatrix<3> transposed = m.Transposed();
  EXPECT_EQ(transposed.ToString(), "[110,010,101]");
}

// Test rectangular matrices
TEST(MatrixTest, Rectangular_2x3_SetGetPlusRank) {
  StaticMatrix<2, 3> m;
  m.Set(0, 0, 1);
  m.Set(0, 2, 1);
  m.Set(1, 0, 1);
  m.Set(1, 1, 1);
  EXPECT_EQ(m.ToString(), "[101,110]");
  EXPECT_EQ(m.Rank(), 2);
  StaticMatrix<3, 2> mt = m.Transposed();
  EXPECT_EQ(mt.ToString(), "[11,01,10]");
  EXPECT_EQ(mt.Rank(), 2);

  StaticMatrix<2, 3> m2;
  m2.Set(0, 0, 1);
  StaticMatrix<2, 3> sum = m.Plus(m2);
  EXPECT_EQ(sum.ToString(), "[001,110]");
  EXPECT_EQ(sum.Rank(), 2);
  EXPECT_EQ(sum.Transposed().Rank(), 2);
}

TEST(MatrixTest, Rectangular_MultiplyAndTranspose) {
  // A: 2x3, B: 3x4 => A*B: 2x4
  StaticMatrix<2, 3> a = StaticMatrix<2, 3>::FromString("[110,011]");
  StaticMatrix<3, 4> b = StaticMatrix<3, 4>::FromString("[1001,1100,0111]");
  StaticMatrix<2, 4> product = a * b;
  EXPECT_EQ(product.ToString(), "[0101,1011]");
  EXPECT_EQ((b.Transposed() * a.Transposed()).Transposed(), product);
}
