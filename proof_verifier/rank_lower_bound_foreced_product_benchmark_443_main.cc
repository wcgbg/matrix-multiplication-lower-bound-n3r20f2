#include "proof_verifier/rank_lower_bound_basic_technics.h"
#include "proof_verifier/tensor.h"

int main(int argc, char **argv) {
  Tensor<4, 4, 3> tensor = SparseStringToTensor<4, 4, 3>(
      "a10*b02*c03 + a11*b12*c03 + a12*b22*c03 + a13*b32*c03 + a20*b01*c03 + "
      "a21*b11*c03 + a22*b21*c03 + a23*b31*c03 + a30*b00*c03 + a30*b02*c23 + "
      "a31*b10*c03 + a31*b12*c23 + a32*b20*c03 + a32*b22*c23 + a33*b30*c03 + "
      "a33*b32*c23");
  int rank = RankLowerBoundForcedProductA<4, 4, 3>(tensor);
  CHECK_EQ(rank, 16);
  return 0;
}
