#pragma once

#include <cstdint>
#include <limits>
#include <vector>

#include <boost/unordered/unordered_flat_map.hpp>
#include <ng-log/logging.h>

#include "proof_verifier/backtracking_proof.h"
#include "proof_verifier/restrictions.h"
#include "proof_verifier/static_matrix.h"

template <int n0, int n1, int n2> class RankLowerBoundBacktrackingVerifier {
public:
  static void
  Verify(const Restrictions<n0, n1> &restrictions, int rank_lower_bound,
         const BacktrackingProof &proof,
         const boost::unordered_flat_map<Restrictions<n0, n1>, uint32_t>
             &restrictions_to_rank_lower_bound) {
    return RankLowerBoundBacktrackingVerifier(restrictions, rank_lower_bound,
                                              proof,
                                              restrictions_to_rank_lower_bound)
        .Verify();
  }

private:
  RankLowerBoundBacktrackingVerifier(
      const Restrictions<n0, n1> &restrictions, int rank_lower_bound,
      const BacktrackingProof &proof,
      const boost::unordered_flat_map<Restrictions<n0, n1>, uint32_t>
          &restrictions_to_rank_lower_bound)
      : base_restrictions_(restrictions), rank_lower_bound_(rank_lower_bound),
        proof_(proof),
        restrictions_to_rank_lower_bound_(restrictions_to_rank_lower_bound) {
    static_assert(n0 * n1 <
                  std::numeric_limits<StaticMatrixData<n0, n1>>::digits);
    for (StaticMatrixData<n0, n1> restriction = 1;
         restriction < (StaticMatrixData<n0, n1>(1) << (n0 * n1));
         ++restriction) {
      bool is_minimal = true;
      for (const auto &base_restriction : base_restrictions_) {
        if ((restriction ^ base_restriction) < restriction) {
          is_minimal = false;
          break;
        }
      }
      if (is_minimal) {
        minimal_restrictions_.push_back(restriction);
      }
    }
    CHECK(!minimal_restrictions_.empty());
  }

  void Verify() const {
    size_t proof_index = 0;
    Restrictions<n0, n1> dfs_restrictions;
    Verify(minimal_restrictions_.size() - 1, &dfs_restrictions, &proof_index);
    CHECK_EQ(proof_index, proof_.Size());
  }

  void Verify(int max_restriction_idx, Restrictions<n0, n1> *dfs_restrictions,
              size_t *proof_index) const {
    CHECK_LT(*proof_index, proof_.Size());
    size_t proof_dfs_restrictions_size =
        proof_.dfs_restrictions_size_array[*proof_index];
    CHECK_LE(dfs_restrictions->size(), proof_dfs_restrictions_size);
    if (dfs_restrictions->size() == proof_dfs_restrictions_size) {
      uint32_t mask = proof_.mask_array[*proof_index];
      bool transpose = proof_.transpose_array[*proof_index];
      uint16_t gl_left = proof_.gl_left_array[*proof_index];
      uint16_t gl_right = proof_.gl_right_array[*proof_index];
      Restrictions<n0, n1> extended_restrictions = base_restrictions_;
      for (int i = 0; i < dfs_restrictions->size(); ++i) {
        if (mask & (uint32_t(1) << i)) {
          extended_restrictions.push_back((*dfs_restrictions)[i]);
        }
      }
      Restrictions<n0, n1> transformed_restrictions =
          TransformRestrictions<n0, n1, n2>(
              extended_restrictions, StaticMatrix<n0>(gl_left),
              StaticMatrix<n1>(gl_right), transpose);
      auto it =
          restrictions_to_rank_lower_bound_.find(transformed_restrictions);
      CHECK(it != restrictions_to_rank_lower_bound_.end())
          << "transformed restrictions not found in map. base_restrictions="
          << RestrictionsToString<n0, n1>(base_restrictions_)
          << ", dfs_restrictions="
          << RestrictionsToString<n0, n1>(*dfs_restrictions)
          << ", extended_restrictions="
          << RestrictionsToString<n0, n1>(extended_restrictions)
          << ", transformed_restrictions="
          << RestrictionsToString<n0, n1>(transformed_restrictions);
      uint32_t rank_lower_bound = std::popcount(mask) + it->second;
      CHECK_LE(rank_lower_bound_, rank_lower_bound);
      (*proof_index)++;
      return;
    }

    for (int i = 0; i <= max_restriction_idx; ++i) {
      dfs_restrictions->push_back(minimal_restrictions_[i]);
      Verify(i, dfs_restrictions, proof_index);
      dfs_restrictions->pop_back();
    }
  }

  const Restrictions<n0, n1> &base_restrictions_;
  int rank_lower_bound_ = 0;
  const BacktrackingProof &proof_;
  const boost::unordered_flat_map<Restrictions<n0, n1>, uint32_t>
      &restrictions_to_rank_lower_bound_;
  std::vector<StaticMatrixData<n0, n1>> minimal_restrictions_;
};
