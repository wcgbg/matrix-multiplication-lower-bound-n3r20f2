#pragma once

#include <algorithm>
#include <cstdint>
#include <limits>
#include <vector>

#include <ng-log/logging.h>
#include <tbb/blocked_range.h>
#include <tbb/parallel_reduce.h>

#include "proof_verifier/dynamic_matrix.h"
#include "proof_verifier/tensor.h"

// Return a matrix of shape (n0*n1, (n1*n2)*(n2*n0)).
template <int n0, int n1, int n2>
DynamicMatrix FlattenTensorAxBC(const Tensor<n0, n1, n2> &tensor) {
  constexpr int n01 = n0 * n1;
  constexpr int n12 = n1 * n2;
  constexpr int n20 = n2 * n0;
  DynamicMatrix result(n01, n12 * n20);
  for (int i = 0; i < n01; ++i) {
    for (int j = 0; j < n12; ++j) {
      for (int k = 0; k < n20; ++k) {
        result(i, j * n20 + k) = tensor[i][j][k];
      }
    }
  }
  return result;
}

template <int n0, int n1, int n2>
int RankLowerBoundFlattenMatrix(const Tensor<n0, n1, n2> &tensor) {
  int rank0 = FlattenTensorAxBC<n0, n1, n2>(tensor).Rank();
  Tensor<n1, n2, n0> tensor1 = CyclicTranspose<n0, n1, n2>(tensor);
  int rank1 = FlattenTensorAxBC<n1, n2, n0>(tensor1).Rank();
  Tensor<n2, n0, n1> tensor2 = CyclicTranspose<n1, n2, n0>(tensor1);
  int rank2 = FlattenTensorAxBC<n2, n0, n1>(tensor2).Rank();
  return std::max({rank0, rank1, rank2});
}

/*
Lemma 2: Let F={f[0],f[1],...,f[n-1]} be a set of expressions, where
f[0],f[1],...,f[k-1] are independent and each can be expressed as a single
product. If F can be computed with p multiplications, then there exists an
algorithm for F with p multiplications in which k of the multiplications are
f[0],f[1],...,f[k-1].
*/
template <int n0, int n1, int n2>
int RankLowerBoundForcedProductA(const Tensor<n0, n1, n2> &tensor,
                                 int known_lower_bound = 0) {
  constexpr int n01 = n0 * n1;
  constexpr int n12 = n1 * n2;
  constexpr int n20 = n2 * n0;
  std::vector<int> a_to_rank_bc(n01, 0);
  int r1_count = 0;
  int r2p_count = 0;
  for (int i = 0; i < n01; ++i) {
    DynamicMatrix bc_matrix(tensor[i]);
    int rank = bc_matrix.Rank();
    a_to_rank_bc[i] = rank;
    if (rank == 0) {
    } else if (rank == 1) {
      r1_count++;
    } else {
      r2p_count++;
    }
  }
  if (r1_count == 0) {
    return 0;
  }
  Tensor<n0, n1, n2> tensor_r2p = {};
  int tensor_r2p_idx = 0;
  DynamicMatrix r1_bc_collection(0, n12 * n20);
  int r1_bc_idx = 0;
  for (size_t i = 0; i < n01; ++i) {
    if (a_to_rank_bc[i] == 0) {
    } else if (a_to_rank_bc[i] == 1) {
      r1_bc_collection.ResizeRows(r1_bc_idx + 1);
      for (int j = 0; j < n12; ++j) {
        for (int k = 0; k < n20; ++k) {
          r1_bc_collection(r1_bc_idx, j * n20 + k) = tensor[i][j][k];
        }
      }
      if (r1_bc_collection.Rank() == r1_bc_idx + 1) {
        r1_bc_idx++;
      } else {
        r1_bc_collection.ResizeRows(r1_bc_idx);
        tensor_r2p[tensor_r2p_idx] = tensor[i];
        tensor_r2p_idx++;
      }
    } else {
      tensor_r2p[tensor_r2p_idx] = tensor[i];
      tensor_r2p_idx++;
    }
  }
  CHECK_EQ(r1_bc_idx + tensor_r2p_idx, r1_count + r2p_count);
  CHECK_EQ(r1_bc_idx, r1_bc_collection.rows());
  CHECK_EQ(r1_bc_collection.Rank(), r1_bc_collection.rows());

  int bit_width = tensor_r2p_idx * r1_bc_idx;
  if (bit_width > 32) {
    LOG(WARNING) << "Bit width is too large: " << bit_width;
    return 0;
  }
  const uint64_t prime = 73074167;
  const uint64_t mask = (uint64_t(1) << bit_width) - 1;
  const uint64_t num_iterations = uint64_t(1) << bit_width;
  std::atomic<bool> early_break = false;
  int rank_lower_bound = tbb::parallel_reduce(
      tbb::blocked_range<uint64_t>(0, num_iterations),
      std::numeric_limits<int>::max(),
      [&](const tbb::blocked_range<uint64_t> &range, int init) {
        int local_min = init;
        for (uint64_t t = range.begin(); t != range.end(); ++t) {
          if (early_break) {
            break;
          }
          uint64_t binary = (t * prime) & mask;
          Tensor<n0, n1, n2> tensor_t = tensor_r2p;
          for (int bit_idx = 0; bit_idx < bit_width; ++bit_idx) {
            if (((binary >> bit_idx) & 1) == 0) {
              continue;
            }
            int r1_idx = bit_idx % r1_bc_idx; // r1_idx in [0, r1_bc_idx)
            int i = bit_idx / r1_bc_idx;      // i in [0, tensor_r2p_idx)
            for (int j = 0; j < n12; ++j) {
              for (int k = 0; k < n20; ++k) {
                tensor_t[i][j][k] ^= r1_bc_collection(r1_idx, j * n20 + k);
              }
            }
          }
          int remaining_rank_lower_bound =
              RankLowerBoundFlattenMatrix<n0, n1, n2>(tensor_t);
          local_min =
              std::min(local_min, r1_bc_idx + remaining_rank_lower_bound);
          if (local_min <= known_lower_bound) {
            early_break = true;
          }
        }
        return local_min;
      },
      [](int a, int b) { return std::min(a, b); });
  return rank_lower_bound;
}
