#pragma once

#include <format>
#include <string>

#include <boost/unordered/unordered_flat_map.hpp>
#include <ng-log/logging.h>
#include <tbb/parallel_for.h>

#include "proof_verifier/backtracking_proof.h"
#include "proof_verifier/math_utils.h"
#include "proof_verifier/rank_lower_bound_backtracking_verifier.h"
#include "proof_verifier/rank_lower_bound_basic_technics.h"
#include "proof_verifier/restricted_mm.pb.h"
#include "proof_verifier/restrictions.h"
#include "proof_verifier/static_matrix.h"
#include "proof_verifier/tensor.h"

// Transforms restrictions using the proof's (gl_left, gl_right, transpose)
// so the result can be looked up in restrictions_to_rank_lower_bound.
template <int n0, int n1, int n2>
Restrictions<n0, n1>
TransformRestrictions(const Restrictions<n0, n1> &restrictions,
                      const pb::Transformation &t) {
  StaticMatrix<n0> gl_left(t.gl_left());
  StaticMatrix<n1> gl_right(t.gl_right());
  return TransformRestrictions<n0, n1, n2>(restrictions, gl_left, gl_right,
                                           t.transpose());
}

// Verifies FlattenMatrixProof: computes rank lower bound and checks it
// matches the expected value.
template <int n0, int n1, int n2>
void VerifyFlattenProof(const Tensor<n0, n1, n2> &tensor,
                        int rank_lower_bound) {
  int computed = RankLowerBoundFlattenMatrix<n0, n1, n2>(tensor);
  CHECK_EQ(computed, rank_lower_bound);
}

// Verifies ForcedProductProof: computes rank lower bound based on
// projection_type and checks it matches the expected value.
template <int n0, int n1, int n2>
void VerifyForcedProductProof(
    const Tensor<n0, n1, n2> &tensor, int rank_lower_bound,
    const pb::ForcedProductProof &forced_product_proof) {
  uint32_t proj = forced_product_proof.projection_type();
  int computed = 0;
  if (proj == 0) {
    computed = RankLowerBoundForcedProductA<n0, n1, n2>(tensor);
  } else if (proj == 1) {
    Tensor<n1, n2, n0> t1 = CyclicTranspose<n0, n1, n2>(tensor);
    computed = RankLowerBoundForcedProductA<n1, n2, n0>(t1);
  } else if (proj == 2) {
    Tensor<n1, n2, n0> t1 = CyclicTranspose<n0, n1, n2>(tensor);
    Tensor<n2, n0, n1> t2 = CyclicTranspose<n1, n2, n0>(t1);
    computed = RankLowerBoundForcedProductA<n2, n0, n1>(t2);
  } else {
    LOG(FATAL) << "Invalid projection_type: " << proj;
  }
  CHECK_EQ(computed, rank_lower_bound);
}

// Verifies DegenerateProof: extends restrictions, transforms by proof's
// transformation, runs GaussJordanElimination, and looks up in the map.
template <int n0, int n1, int n2>
void VerifyDegenerateProof(
    const Restrictions<n0, n1> &restrictions, int rank_lower_bound,
    const pb::DegenerateProof &degenerate_proof,
    const boost::unordered_flat_map<Restrictions<n0, n1>, uint32_t>
        &restrictions_to_rank_lower_bound) {
  Restrictions<n0, n1> extended = restrictions;
  extended.push_back(degenerate_proof.extra_restriction());
  auto transformed = TransformRestrictions<n0, n1, n2>(
      extended, degenerate_proof.transformation());
  int rank = GaussJordanElimination(n0 * n1, &transformed);
  CHECK_EQ(rank, transformed.size());
  auto it = restrictions_to_rank_lower_bound.find(transformed);
  CHECK(it != restrictions_to_rank_lower_bound.end())
      << "transformed extended restrictions not found in map. restrictions="
      << RestrictionsToString<n0, n1>(restrictions)
      << ", extended=" << RestrictionsToString<n0, n1>(extended)
      << ", transformed=" << RestrictionsToString<n0, n1>(transformed);
  CHECK_EQ(it->second, rank_lower_bound);
}

// Verifies BacktrackingProof: loads proof from disk, checks size, and runs
// backtracking verifier.
template <int n0, int n1, int n2>
void VerifyBacktrackingProof(
    const Restrictions<n0, n1> &restrictions, int rank_lower_bound,
    const pb::RankLowerBoundProof &proof_proto, int rmm_index,
    const boost::unordered_flat_map<Restrictions<n0, n1>, uint32_t>
        &restrictions_to_rank_lower_bound,
    const std::string &bt_proof_root_dir) {
  std::string proof_path =
      GetBacktrackingProofPath(bt_proof_root_dir, rmm_index);
  BacktrackingProof proof = BacktrackingProof::Load(proof_path);
  CHECK_EQ(proof_proto.backtracking_proof().proof_size(), proof.Size());
  RankLowerBoundBacktrackingVerifier<n0, n1, n2>::Verify(
      restrictions, rank_lower_bound, proof, restrictions_to_rank_lower_bound);
}

// Verifies a single RestrictedMM with FlattenMatrixProof or ForcedProductProof.
// Returns true if the entry passes (or is skipped), false if verification
// fails.
template <int n0, int n1, int n2>
void VerifyOne(const pb::RestrictedMM &rmm,
               const boost::unordered_flat_map<Restrictions<n0, n1>, uint32_t>
                   &restrictions_to_rank_lower_bound,
               const std::string &bt_proof_root_dir) {
  CHECK_EQ(rmm.p(), 2);
  CHECK_EQ(rmm.n0(), n0);
  CHECK_EQ(rmm.n1(), n1);
  CHECK_EQ(rmm.n2(), n2);

  const auto &proof = rmm.rank_lower_bound_proof();
  CHECK_NE(proof.proof_case(), pb::RankLowerBoundProof::PROOF_NOT_SET)
      << rmm.index();

  Restrictions<n0, n1> restrictions = RestrictionsFromProto<n0, n1>(rmm);
  {
    Restrictions<n0, n1> rref_restrictions = restrictions;
    GaussJordanElimination(n0 * n1, &rref_restrictions);
    CHECK(rref_restrictions == restrictions);
  }
  Tensor<n0, n1, n2> tensor = ApplyRestrictionsToTensor<n0, n1, n2>(
      restrictions, MatrixMultiplicationTensor<n0, n1, n2>());
  CHECK_EQ((TensorToSparseString<n0, n1, n2>(tensor)), rmm.tensor())
      << rmm.index();

  if (proof.has_flatten_matrix_proof()) {
    VerifyFlattenProof<n0, n1, n2>(tensor, rmm.rank_lower_bound());
  } else if (proof.has_forced_product_proof()) {
    VerifyForcedProductProof<n0, n1, n2>(tensor, rmm.rank_lower_bound(),
                                         proof.forced_product_proof());
  } else if (proof.has_degenerate_proof()) {
    VerifyDegenerateProof<n0, n1, n2>(restrictions, rmm.rank_lower_bound(),
                                      proof.degenerate_proof(),
                                      restrictions_to_rank_lower_bound);
  } else if (proof.has_backtracking_proof()) {
    VerifyBacktrackingProof<n0, n1, n2>(
        restrictions, rmm.rank_lower_bound(), proof, rmm.index(),
        restrictions_to_rank_lower_bound, bt_proof_root_dir);
  } else {
    LOG(FATAL) << rmm.index() << ": Invalid proof type";
  }
}

// Verifies rank lower bound proofs in a RestrictedMMCollection.
template <int n0, int n1, int n2>
void VerifyRankLowerBound(const pb::RestrictedMMCollection &collection,
                          const std::string &bt_proof_root_dir) {
  boost::unordered_flat_map<Restrictions<n0, n1>, uint32_t>
      restrictions_to_rank_lower_bound;
  size_t total_count = 0;
  for (int restriction_size = n0 * n1; restriction_size >= 0;
       restriction_size--) {
    LOG(INFO) << "restriction_size=" << restriction_size;
    std::vector<const pb::RestrictedMM *> rmms;
    for (int i = 0; i < collection.restricted_mm_size(); ++i) {
      const pb::RestrictedMM &rmm = collection.restricted_mm(i);
      if (rmm.restriction_size() == restriction_size) {
        rmms.push_back(&rmm);
      }
    }
    tbb::parallel_for(size_t(0), rmms.size(), [&](size_t i) {
      VerifyOne<n0, n1, n2>(*rmms[i], restrictions_to_rank_lower_bound,
                            bt_proof_root_dir);
    });
    for (const pb::RestrictedMM *rmm : rmms) {
      CHECK(restrictions_to_rank_lower_bound
                .emplace(RestrictionsFromProto<n0, n1>(*rmm),
                         rmm->rank_lower_bound())
                .second);
    }
    total_count += rmms.size();
  }
  CHECK_EQ(total_count, collection.restricted_mm_size());
  CHECK_GT(collection.restricted_mm_size(), 0);
  const auto &last_rmm =
      collection.restricted_mm(collection.restricted_mm_size() - 1);
  CHECK_EQ(last_rmm.restriction_size(), 0);
  LOG(INFO) << std::format("Verified. The rank lower bound for {}x{}x{} matrix "
                           "multiplication tensor is {} over F_2.",
                           n0, n1, n2, last_rmm.rank_lower_bound());
}
