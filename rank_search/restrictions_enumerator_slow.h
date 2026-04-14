#pragma once

#include <algorithm>
#include <random>
#include <vector>

#include <boost/unordered/unordered_flat_set.hpp>
#include <ng-log/logging.h>

#include "proof_verifier/restricted_mm.pb.h"
#include "proof_verifier/restrictions.h"
#include "proof_verifier/static_matrix.h"
#include "proof_verifier/tensor_utils.h"

template <int n0, int n1, int n2> class RestrictionEnumeratorSlow {
public:
  RestrictionEnumeratorSlow() {
    static_assert(n0 <= 4);
    static_assert(n1 <= 4);
    static_assert(n2 <= 4);
    static_assert(n0 * n0 < 32);
    static_assert(n0 * n1 < 32);
    static_assert(n1 * n1 < 32);

    constexpr int n00 = n0 * n0;
    constexpr int n11 = n1 * n1;

    std::vector<StaticMatrix<n0>> full_rank_matrices_n0;
    std::vector<StaticMatrix<n1>> full_rank_matrices_n1;

    // Full-rank n0×n0 matrices
    for (uint32_t k = 0; k < (uint32_t(1) << n00); ++k) {
      StaticMatrix<n0> mat(k);
      if (mat.Rank() == n0) {
        full_rank_matrices_n0.push_back(mat);
      }
    }

    // Full-rank n1×n1 matrices
    for (uint32_t k = 0; k < (uint32_t(1) << n11); ++k) {
      StaticMatrix<n1> mat(k);
      if (mat.Rank() == n1) {
        full_rank_matrices_n1.push_back(mat);
      }
    }

    // Stabilizers
    for (bool transpose : {false, true}) {
      if (n0 != n1 || n1 != n2 || n0 != n2) {
        if (transpose) {
          break;
        }
      }
      for (const StaticMatrix<n0> &gl_left : full_rank_matrices_n0) {
        for (const StaticMatrix<n1> &gl_right : full_rank_matrices_n1) {
          all_transformations_.push_back({transpose, gl_left, gl_right});
        }
      }
    }

    std::shuffle(all_transformations_.begin(), all_transformations_.end(),
                 gen_);
  }

  pb::RestrictedMMCollection Search() {
    Restrictions<n0, n1> restrictions;
    minimal_restrictions_.clear();
    minimal_restrictions_.insert(restrictions);
    Search(restrictions);

    Tensor<n0, n1, n2> matrix_multiplication_tensor =
        MatrixMultiplicationTensor<n0, n1, n2>();
    std::vector<std::vector<Restrictions<n0, n1>>> dim_to_restrictions(n0 * n1 +
                                                                       1);
    for (const auto &r : minimal_restrictions_) {
      dim_to_restrictions.at(r.size()).push_back(r);
    }
    pb::RestrictedMMCollection collection;
    for (int dim = static_cast<int>(dim_to_restrictions.size()) - 1; dim >= 0;
         --dim) {
      LOG(INFO) << "dim=" << dim
                << " count=" << dim_to_restrictions.at(dim).size();
      std::sort(dim_to_restrictions[dim].begin(),
                dim_to_restrictions[dim].end());
      for (const Restrictions<n0, n1> &restrictions :
           dim_to_restrictions[dim]) {
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
  struct Transformation {
    bool transpose = false;
    StaticMatrix<n0> gl_left;
    StaticMatrix<n1> gl_right;
  };

  void Search(Restrictions<n0, n1> restrictions) {
    constexpr int num_cols = n0 * n1;
    if (restrictions.size() == static_cast<size_t>(num_cols)) {
      return;
    }
    StaticMatrixData<n0, n1> last_restriction_data = 0;
    if (!restrictions.empty()) {
      last_restriction_data = restrictions.back();
    }
    for (uint32_t data = last_restriction_data + 1;
         data < (uint32_t(1) << num_cols); ++data) {
      restrictions.push_back(static_cast<StaticMatrixData<n0, n1>>(data));
      if (Visit(restrictions)) {
        Search(restrictions);
      }
      restrictions.pop_back();
    }
  }

  bool Visit(const Restrictions<n0, n1> &restrictions) {
    for (const Transformation &transformation : all_transformations_) {
      Restrictions<n0, n1> new_restrictions = TransformRestrictions<n0, n1, n2>(
          restrictions, transformation.transpose, transformation.gl_left,
          transformation.gl_right);
      if (new_restrictions.size() != restrictions.size()) {
        return false;
      }
      if (new_restrictions < restrictions) {
        return false;
      }
    }
    LOG(INFO) << restrictions.size() << " "
              << RestrictionsToString<n0, n1>(restrictions);
    CHECK(minimal_restrictions_.insert(restrictions).second);
    return true;
  }

  std::vector<Transformation> all_transformations_;
  std::mt19937_64 gen_;
  boost::unordered_flat_set<Restrictions<n0, n1>, RestrictionsHash<n0, n1>>
      minimal_restrictions_;
};
