#pragma once

#include <algorithm>
#include <bit>
#include <limits>
#include <vector>

#include <ng-log/logging.h>
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>

#include "proof_verifier/math_utils.h"
#include "proof_verifier/restricted_mm.pb.h"
#include "proof_verifier/restrictions.h"
#include "proof_verifier/static_matrix.h"
#include "proof_verifier/tensor_utils.h"
#include "restrictions_set.h"

template <int n0, int n1, int n2> class RestrictionEnumerator {
public:
  RestrictionEnumerator() {
    static_assert(n0 <= 4);
    static_assert(n1 <= 4);
    static_assert(n0 * n0 < 32);
    static_assert(n0 * n1 < 32);
    static_assert(n1 * n1 < 32);

    for (uint32_t d00 = 0; d00 < (uint32_t(1) << (n0 * n0)); ++d00) {
      StaticMatrix<n0> mat00(d00);
      if (mat00.Rank() == n0) {
        full_rank_matrices_n0_.push_back(mat00);
      }
    }

    for (uint32_t d11 = 0; d11 < (uint32_t(1) << (n1 * n1)); ++d11) {
      StaticMatrix<n1> mat11(d11);
      if (mat11.Rank() == n1) {
        full_rank_matrices_n1_.push_back(mat11);
      }
    }

    for (uint32_t d01 = 0; d01 < (uint32_t(1) << (n0 * n1)); ++d01) {
      StaticMatrix<n0, n1> mat01(d01);
      m_to_transpose_n01_.push_back(mat01.Transposed());
    }
  }

  pb::RestrictedMMCollection Search() {
    std::vector<std::vector<Restrictions<n0, n1>>> dim_to_restrictions(n0 * n1 +
                                                                       1);
    dim_to_restrictions[0].push_back(Restrictions<n0, n1>());
    LOG(INFO) << "dim=0 count=1";

    for (int dim = 1; dim <= n0 * n1; ++dim) {
      RestrictionsSet<n0, n1> some_visited_restrictions;
      std::vector<Restrictions<n0, n1>> restrictions;
      for (const auto &parent : dim_to_restrictions[dim - 1]) {
        ExpandNextLayer(parent, &some_visited_restrictions, &restrictions);
      }
      std::sort(restrictions.begin(), restrictions.end());
      dim_to_restrictions[dim] = std::move(restrictions);
      LOG(INFO) << "dim=" << dim
                << " count=" << dim_to_restrictions[dim].size();
    }

    Tensor<n0, n1, n2> matrix_multiplication_tensor =
        MatrixMultiplicationTensor<n0, n1, n2>();
    pb::RestrictedMMCollection collection;
    for (int dim = n0 * n1; dim >= 0; --dim) {
      for (const auto &restrictions : dim_to_restrictions[dim]) {
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

  void ExpandNextLayer(const Restrictions<n0, n1> &base_restrictions,
                       RestrictionsSet<n0, n1> *some_visited_restrictions,
                       std::vector<Restrictions<n0, n1>> *next_layer) {
    StaticMatrixData<n0, n1> pivot_mask = 0;
    for (const auto &restriction : base_restrictions) {
      CHECK_NE(restriction, 0);
      int highest_bit_position =
          std::numeric_limits<StaticMatrixData<n0, n1>>::digits - 1 -
          std::countl_zero(restriction);
      pivot_mask |= (StaticMatrixData<n0, n1>(1) << highest_bit_position);
    }
    int pivot_mask_width =
        std::numeric_limits<StaticMatrixData<n0, n1>>::digits -
        std::countl_zero(pivot_mask);
    int pivot_mask_weight = std::popcount(pivot_mask);
    CHECK_EQ(pivot_mask_weight, static_cast<int>(base_restrictions.size()));
    CHECK_LE(pivot_mask_weight, pivot_mask_width);

    static_assert(n0 * n1 <=
                  std::numeric_limits<StaticMatrixData<n0, n1>>::digits);
    static_assert(n0 * n1 < 32);
    uint32_t hi_end = (uint32_t(1) << (n0 * n1 - pivot_mask_width));
    uint32_t lo_end = (uint32_t(1) << (pivot_mask_width - pivot_mask_weight));

    Restrictions<n0, n1> restrictions = base_restrictions;
    for (uint32_t hi = 1; hi < hi_end; ++hi) {
      for (uint32_t lo = 0; lo < lo_end; ++lo) {
        StaticMatrixData<n0, n1> restriction = MakeRestrictionFromHiLo(
            hi, lo, pivot_mask, pivot_mask_width, pivot_mask_weight);
        restrictions.push_back(restriction);
        if (Visit(restrictions, some_visited_restrictions)) {
          next_layer->push_back(restrictions);
        }
        restrictions.pop_back();
      }
    }
  }

  bool Visit(const Restrictions<n0, n1> &restrictions,
             RestrictionsSet<n0, n1> *some_visited_restrictions) {
    constexpr int kTransposeRange = (n0 == n1 && n1 == n2) ? 2 : 1;
    std::atomic<bool> found_duplicate = false;
    tbb::parallel_for(
        tbb::blocked_range<size_t>(0, kTransposeRange *
                                          full_rank_matrices_n0_.size()),
        [&](const tbb::blocked_range<size_t> &range) {
          Restrictions<n0, n1> new_restrictions;
          new_restrictions.reserve(restrictions.size());
          for (size_t idx = range.begin(); idx != range.end(); ++idx) {
            if (found_duplicate) {
              return;
            }
            bool transpose = false;
            size_t gl_left_idx = idx;
            if constexpr (kTransposeRange == 2) {
              transpose = idx % 2 == 1;
              gl_left_idx = idx / 2;
            }
            const StaticMatrix<n0> &gl_left =
                full_rank_matrices_n0_[gl_left_idx];
            CHECK(TransformRestrictions(restrictions, transpose, gl_left,
                                        StaticMatrix<n1>::Identity(),
                                        &new_restrictions));
            if (some_visited_restrictions->ContainsUnsafe(new_restrictions)) {
              found_duplicate = true;
              return;
            }
          }
        });
    if (found_duplicate) {
      return false;
    }

    tbb::parallel_for(
        tbb::blocked_range<size_t>(0, full_rank_matrices_n1_.size()),
        [&](const tbb::blocked_range<size_t> &range) {
          for (size_t idx = range.begin(); idx != range.end(); ++idx) {
            const StaticMatrix<n1> &gl_right = full_rank_matrices_n1_[idx];
            Restrictions<n0, n1> transformed;
            CHECK(TransformRestrictions(restrictions, false,
                                        StaticMatrix<n0>::Identity(), gl_right,
                                        &transformed));
            some_visited_restrictions->Insert(transformed);
          }
        });
    return true;
  }

  std::vector<StaticMatrix<n0>> full_rank_matrices_n0_;
  std::vector<StaticMatrix<n1>> full_rank_matrices_n1_;
  std::vector<StaticMatrix<n1, n0>> m_to_transpose_n01_; // n0*n1 -> n1*n0
};
