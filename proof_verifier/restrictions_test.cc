#include "proof_verifier/restrictions.h"

#include <array>
#include <gtest/gtest.h>
#include <initializer_list>

#include "proof_verifier/tensor.h"

TEST(ApplyRestrictionsToTensorTest, N2SubstituteLastVariableDimension0) {
  constexpr int n = 2;
  // Restriction a01+a10+a11=0: pivot is a11 (index 3), others are a01 (1), a10
  // (2). Bit indices: a00=0, a01=1, a10=2, a11=3.
  StaticMatrixData<n> restriction = (1u << 1) | (1u << 2) | (1u << 3);
  EXPECT_EQ(StaticMatrix<n>(restriction).ToString(), "[01,11]");
  Restrictions<n, n> restrictions;
  restrictions.push_back(restriction);

  // Tensor a01*b01*c01 + a11*b11*c11
  SquareTensor<n> tensor = {};
  tensor[1][1][1] = 1;
  tensor[3][3][3] = 1;
  EXPECT_EQ((TensorToSparseString<n, n, n>(tensor)),
            "a01*b01*c01 + a11*b11*c11");

  // a11=a01+a10
  SquareTensor<n> result =
      ApplyRestrictionsToTensor<n, n, n>(restrictions, tensor);

  EXPECT_EQ((TensorToSparseString<n, n, n>(result)),
            "a01*b01*c01 + a01*b11*c11 + a10*b11*c11");
}

TEST(ApplyRestrictionsToTensorTest, N2ZeroRestrictionSkipped) {
  constexpr int n = 2;
  Restrictions<n, n> restrictions;
  restrictions.push_back(0); // zero restriction

  SquareTensor<n> tensor = {};
  tensor[0][0][0] = 1;
  EXPECT_EQ((TensorToSparseString<n, n, n>(tensor)), "a00*b00*c00");

  SquareTensor<n> result =
      ApplyRestrictionsToTensor<n, n, n>(restrictions, tensor);

  EXPECT_EQ((TensorToSparseString<n, n, n>(result)), "a00*b00*c00");
}

TEST(ApplyRestrictionsToTensorTest, N2SingleVariableRestrictionZerosPivot) {
  constexpr int n = 2;
  // Restriction a11=0: only pivot index 3, no others.
  StaticMatrixData<n> restriction = (1u << 3);
  Restrictions<n, n> restrictions;
  restrictions.push_back(restriction);
  EXPECT_EQ((RestrictionsToString<n, n>(restrictions)), "[00,01]");

  SquareTensor<n> tensor = {};
  tensor[3][0][0] = 1;

  SquareTensor<n> result =
      ApplyRestrictionsToTensor<n, n, n>(restrictions, tensor);

  EXPECT_EQ((TensorToSparseString<n, n, n>(result)), "0");
}

TEST(ApplyRestrictionsToTensorTest, N3_24_68571) {
  constexpr int n = 3;
  Restrictions<n, n> restrictions_24;
  for (const std::string &str :
       {"[100,000,000]", "[010,000,000]", "[001,000,000]", "[000,100,000]",
        "[000,010,100]", "[000,001,010]"}) {
    restrictions_24.push_back(StaticMatrix<n>::FromString(str).Data());
  }
  SquareTensor<n> expected_tensor_24 = SparseStringToTensor<n, n, n>(
      "a11*b00*c02 + a11*b01*c12 + a11*b02*c22 + a11*b10*c01 + a11*b11*c11 + "
      "a11*b12*c21 + a12*b10*c02 + a12*b11*c12 + a12*b12*c22 + a12*b20*c01 + "
      "a12*b21*c11 + a12*b22*c21 + a22*b20*c02 + a22*b21*c12 + a22*b22*c22");
  SquareTensor<n> tensor_24 = ApplyRestrictionsToTensor<n, n, n>(
      restrictions_24, MatrixMultiplicationTensor<n, n, n>());
  EXPECT_EQ(tensor_24, expected_tensor_24);

  Restrictions<n, n> restrictions_68571;
  for (const std::string &str :
       {"[001,100,000]", "[100,010,000]", "[110,001,000]", "[001,000,010]",
        "[100,000,101]"}) {
    restrictions_68571.push_back(StaticMatrix<n>::FromString(str).Data());
  }
  SquareTensor<n> tensor_24_68571 = ApplyRestrictionsToTensor<n, n, n>(
      restrictions_68571, CyclicTranspose<n, n, n>(tensor_24));
  SquareTensor<n> expected_tensor_24_68571 = SparseStringToTensor<n, n, n>(
      "a00*b02*c11 + a00*b11*c11 + a00*b12*c12 + a00*b21*c11 + a00*b21*c12 + "
      "a00*b22*c12 + a00*b22*c22 + a01*b12*c11 + a01*b21*c11 + a01*b22*c12 + "
      "a02*b01*c11 + a02*b02*c12 + a02*b11*c12 + a02*b12*c22 + a02*b22*c11 + "
      "a20*b01*c12 + a20*b02*c22 + a20*b21*c12 + a20*b22*c22");
  EXPECT_EQ(tensor_24_68571, expected_tensor_24_68571);
}
