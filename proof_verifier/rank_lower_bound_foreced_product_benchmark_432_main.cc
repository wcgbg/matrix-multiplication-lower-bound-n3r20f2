#include "proof_verifier/rank_lower_bound_basic_technics.h"
#include "proof_verifier/tensor.h"

int main(int argc, char **argv) {
  Tensor<4, 3, 2> tensor = SparseStringToTensor<4, 3, 2>(
      "a00*b01*c10 + a01*b01*c00 + a02*b20*c00 + a02*b21*c10 + a10*b01*c11 + "
      "a11*b01*c01 + a12*b20*c01 + a12*b21*c11 + a20*b01*c12 + a21*b01*c02 + "
      "a22*b20*c02 + a22*b21*c12 + a30*b01*c13 + a31*b01*c03 + a32*b20*c03 + "
      "a32*b21*c13");
  int rank = RankLowerBoundForcedProductA<4, 3, 2>(tensor);
  CHECK_EQ(rank, 16);
  return 0;
}
