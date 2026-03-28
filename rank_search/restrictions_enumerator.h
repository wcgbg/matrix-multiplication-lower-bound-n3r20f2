#pragma once

#include <algorithm>
#include <bit>
#include <initializer_list>
#include <limits>
#include <random>
#include <vector>

#include <boost/unordered/unordered_flat_set.hpp>
#include <ng-log/logging.h>
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>

#include "proof_verifier/math_utils.h"
#include "proof_verifier/restricted_mm.pb.h"
#include "proof_verifier/restrictions.h"
#include "proof_verifier/static_matrix.h"
#include "proof_verifier/tensor.h"

template <int n0, int n1, int n2> class RestrictionEnumerator {
public:
  RestrictionEnumerator() {
    static_assert(n0 <= 4);
    static_assert(n1 <= 4);
    static_assert(n0 * n0 < 32);
    static_assert(n0 * n1 < 32);
    static_assert(n1 * n1 < 32);

    // Build full_rank_matrices_n0_ for n0×n0 matrices (for gl_left)
    for (uint32_t d00 = 0; d00 < (uint32_t(1) << (n0 * n0)); ++d00) {
      StaticMatrix<n0> mat00(d00);
      int rank = mat00.Rank();
      if (rank == n0) {
        full_rank_matrices_n0_.push_back(mat00);
      }
    }

    // Build full_rank_matrices_n1_ for n1×n1 matrices (for gl_right)
    for (uint32_t d11 = 0; d11 < (uint32_t(1) << (n1 * n1)); ++d11) {
      StaticMatrix<n1> mat11(d11);
      int rank = mat11.Rank();
      if (rank == n1) {
        full_rank_matrices_n1_.push_back(mat11);
      }
    }

    // Build transpose table: n0×n1 → n1×n0
    for (uint32_t d01 = 0; d01 < (uint32_t(1) << (n0 * n1)); ++d01) {
      StaticMatrix<n0, n1> mat01(d01);
      m_to_transpose_n01_.push_back(mat01.Transposed());
    }
  }

  pb::RestrictedMMCollection Search() {
    some_visited_restrictions_.clear();
    minimal_restrictions_.clear();

    Restrictions<n0, n1> restrictions;
    minimal_restrictions_.push_back(restrictions);
    Search(restrictions);

    Tensor<n0, n1, n2> matrix_multiplication_tensor =
        MatrixMultiplicationTensor<n0, n1, n2>();
    std::vector<std::vector<Restrictions<n0, n1>>> rank_to_restrictions(
        n0 * n1 + 1);
    for (int i = 0; i < static_cast<int>(minimal_restrictions_.size()); ++i) {
      const Restrictions<n0, n1> &restrictions = minimal_restrictions_[i];
      if (i == 0 || !restrictions.empty()) {
        rank_to_restrictions.at(restrictions.size()).push_back(restrictions);
      }
    }
    pb::RestrictedMMCollection collection;
    for (int rank = rank_to_restrictions.size() - 1; rank >= 0; --rank) {
      LOG(INFO) << "restriction_rank=" << rank
                << " count=" << rank_to_restrictions.at(rank).size();
      std::sort(rank_to_restrictions[rank].begin(),
                rank_to_restrictions[rank].end());
      for (const auto &restrictions : rank_to_restrictions[rank]) {
        int index = collection.restricted_mm_size();
        pb::RestrictedMM *rmm = collection.add_restricted_mm();
        rmm->set_index(index);
        rmm->set_n0(n0);
        rmm->set_n1(n1);
        rmm->set_n2(n2);
        rmm->set_p(2);
        for (const auto &restriction : restrictions) {
          auto restriction_pb = rmm->add_restriction();
          for (int i = 0; i < n0; ++i) {
            for (int j = 0; j < n1; ++j) {
              int value = (restriction >> (i * n1 + j)) & 1u;
              restriction_pb->add_a(value);
            }
          }
          restriction_pb->set_text(
              StaticMatrix<n0, n1>(restriction).ToString());
        }
        rmm->set_tensor(TensorToSparseString<n0, n1, n2>(
            ApplyRestrictionsToTensor<n0, n1, n2>(
                restrictions, matrix_multiplication_tensor)));
      }
    }
    LOG(INFO) << "total_count=" << collection.restricted_mm_size();
    return collection;
  }

private:
  bool TransformRestrictions(const Restrictions<n0, n1> &restrictions,
                             bool transpose, const StaticMatrix<n0> &gl_left,
                             const StaticMatrix<n1> &gl_right,
                             Restrictions<n0, n1> *new_restrictions) const {
    new_restrictions->clear();
    for (auto restriction : restrictions) {
      if (transpose) {
        DCHECK_EQ(n0, n1);
        DCHECK_EQ(n1, n2);
        if constexpr (n0 == n1 && n1 == n2) {
          restriction = m_to_transpose_n01_[restriction].Data();
        }
      }
      StaticMatrixData<n0, n1> new_restriction_data =
          (gl_left * StaticMatrix<n0, n1>(restriction) * gl_right).Data();
      new_restrictions->push_back(new_restriction_data);
    }
    int new_restrictions_rank =
        GaussJordanElimination(n0 * n1, new_restrictions);
    new_restrictions->shrink_to_fit();
    return new_restrictions_rank == static_cast<int>(restrictions.size());
  }

  static StaticMatrixData<n0, n1>
  MakeRestrictionFromHiLo(uint32_t hi, uint32_t lo,
                          StaticMatrixData<n0, n1> pivot_mask,
                          int pivot_mask_width, int pivot_mask_weight) {
    StaticMatrixData<n0, n1> restriction = 0;
    int j = 0;
    for (int i = 0; i < pivot_mask_width; ++i) {
      if (pivot_mask & (StaticMatrixData<n0, n1>(1) << i)) {
        continue;
      }
      restriction |= ((lo >> j) & uint32_t(1)) << i;
      ++j;
    }
    CHECK_EQ(j, pivot_mask_width - pivot_mask_weight);
    restriction |= (hi << pivot_mask_width);
    return restriction;
  }

  void Search(Restrictions<n0, n1> &restrictions) {
    if (restrictions.size() == static_cast<size_t>(n0 * n1)) {
      return;
    }

    // pivot_mask is the bit-wise-or of the highest bit of each restriction.
    StaticMatrixData<n0, n1> pivot_mask = 0;
    for (const auto &restriction : restrictions) {
      // For each restriction, get the highest 1-bit present (pivot column).
      CHECK_NE(restriction, 0);
      // n0 * n1 is the total number of bits in restriction.
      int highest_bit_position =
          std::numeric_limits<StaticMatrixData<n0, n1>>::digits - 1 -
          std::countl_zero(restriction);
      pivot_mask |= (StaticMatrixData<n0, n1>(1) << highest_bit_position);
    }
    int pivot_mask_width =
        std::numeric_limits<StaticMatrixData<n0, n1>>::digits -
        std::countl_zero(pivot_mask);
    int pivot_mask_weight = std::popcount(pivot_mask);
    CHECK_EQ(pivot_mask_weight, static_cast<int>(restrictions.size()));
    CHECK_LE(pivot_mask_weight, pivot_mask_width);

    static_assert(n0 * n1 <=
                  std::numeric_limits<StaticMatrixData<n0, n1>>::digits);
    static_assert(n0 * n1 < 32);
    uint32_t hi_end = (uint32_t(1) << (n0 * n1 - pivot_mask_width));
    uint32_t lo_end = (uint32_t(1) << (pivot_mask_width - pivot_mask_weight));

    for (uint32_t hi = 1; hi < hi_end; ++hi) {
      for (uint32_t lo = 0; lo < lo_end; ++lo) {
        StaticMatrixData<n0, n1> restriction = MakeRestrictionFromHiLo(
            hi, lo, pivot_mask, pivot_mask_width, pivot_mask_weight);
        restrictions.push_back(restriction);
        if (Visit(restrictions)) {
          Search(restrictions);
        }
        restrictions.pop_back();
      }
    }
  }

  // Returns false if it has been visited.
  bool Visit(const Restrictions<n0, n1> &restrictions) {
    Restrictions<n0, n1> new_restrictions;
    new_restrictions.reserve(restrictions.size());

    for (bool transpose : {false, true}) {
      if (n0 != n1 || n1 != n2 || n0 != n2) {
        if (transpose) {
          break;
        }
      }
      for (const StaticMatrix<n0> &gl_left : full_rank_matrices_n0_) {
        CHECK(TransformRestrictions(restrictions, transpose, gl_left,
                                    StaticMatrix<n1>::Identity(),
                                    &new_restrictions));
        if (some_visited_restrictions_.contains(new_restrictions)) {
          return false;
        }
      }
    }
    CHECK_LT(minimal_restrictions_.size(), (uint64_t(1) << 32));
    for (const StaticMatrix<n1> &gl_right : full_rank_matrices_n1_) {
      CHECK(TransformRestrictions(restrictions, false,
                                  StaticMatrix<n0>::Identity(), gl_right,
                                  &new_restrictions));
      some_visited_restrictions_.insert(new_restrictions);
    }

    minimal_restrictions_.push_back(restrictions);
    LOG(INFO) << "mr.size=" << minimal_restrictions_.size()
              << ", svr.size=" << some_visited_restrictions_.size();
    return true;
  }

  std::vector<StaticMatrix<n0>> full_rank_matrices_n0_;
  std::vector<StaticMatrix<n1>> full_rank_matrices_n1_;
  std::vector<StaticMatrix<n1, n0>> m_to_transpose_n01_; // n0*n1 -> n1*n0

  std::mt19937_64 gen_;

  boost::unordered_flat_set<Restrictions<n0, n1>> some_visited_restrictions_;
  std::vector<Restrictions<n0, n1>> minimal_restrictions_;
};
