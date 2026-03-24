#pragma once

#include <cstdint>
#include <optional>

#include "proof_verifier/tensor.h"

// Parameters for the rank lower bound loop (iteration + reduction).
// Used by both CPU (TBB) and GPU (CUDA) paths.
template <int n0, int n1, int n2>
struct RankLowerBoundForcedProductALoopParams {
  using TensorType = Tensor<n0, n1, n2>;

  TensorType r2p;
  const uint8_t *r1_bc_data = nullptr;
  int r1_bc_rows = 0;
  int r2p_size0 = 0;
  int bit_width = 0;
  int known_lower_bound = 0;
};

// Returns true if a CUDA device is available and the GPU path can be used.
bool IsCudaAvailable();

template <int n0, int n1, int n2>
std::optional<int> RankLowerBoundForcedProductALoopCuda(
    const RankLowerBoundForcedProductALoopParams<n0, n1, n2> &params);
