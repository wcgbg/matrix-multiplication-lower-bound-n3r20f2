#include "proof_verifier/tensor.h"
#include "proof_verifier/tensor_utils.h"

#include <gtest/gtest.h>

TEST(MatrixMultiplicationTensorTest, N2SparseString) {
  const SquareTensor<2> tensor = MatrixMultiplicationTensor<2, 2, 2>();
  const std::string sparse_string = TensorToSparseString<2, 2, 2>(tensor);
  EXPECT_EQ(sparse_string,
            "a00*b00*c00 + a00*b01*c10 + a01*b10*c00 + a01*b11*c10 + "
            "a10*b00*c01 + a10*b01*c11 + a11*b10*c01 + a11*b11*c11");
  const SquareTensor<2> tensor_recovered =
      SparseStringToTensor<2, 2, 2>(sparse_string);
  EXPECT_EQ(tensor_recovered, tensor);
}

TEST(MatrixMultiplicationTensorTest, N2DenseString) {
  const SquareTensor<2> tensor = MatrixMultiplicationTensor<2, 2, 2>();
  const std::string dense_string = TensorToDenseString<2, 2, 2>(tensor);
  EXPECT_EQ(dense_string, "1000,0010,0000,0000,\n"
                          "0000,0000,1000,0010,\n"
                          "0100,0001,0000,0000,\n"
                          "0000,0000,0100,0001,\n");
  const SquareTensor<2> tensor_recovered =
      DenseStringToTensor<2, 2, 2>(dense_string);
  EXPECT_EQ(tensor_recovered, tensor);
}

TEST(MatrixMultiplicationTensorTest, N2CyclicTranspose) {
  const SquareTensor<2> tensor = MatrixMultiplicationTensor<2, 2, 2>();
  EXPECT_EQ((CyclicTranspose<2, 2, 2>(tensor)), tensor);
}
