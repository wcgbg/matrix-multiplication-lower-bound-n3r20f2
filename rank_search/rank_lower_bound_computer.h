#pragma once

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <format>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <ng-log/logging.h>
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>

#include "proof_verifier/backtracking_proof.h"
#include "proof_verifier/math_utils.h"
#include "proof_verifier/proto_io.h"
#include "proof_verifier/rank_lower_bound_basic_technics.h"
#include "proof_verifier/restricted_mm.pb.h"
#include "proof_verifier/restrictions.h"
#include "proof_verifier/static_matrix.h"
#include "proof_verifier/tensor.h"
#include "rank_lower_bound_backtracking.h"
#include "restrictions_map.h"

template <int n0, int n1, int n2>
std::pair<int, pb::ForcedProductProof>
RankLowerBoundForcedProduct(const Tensor<n0, n1, n2> &tensor,
                            int known_lower_bound) {
  int rank0 =
      RankLowerBoundForcedProductA<n0, n1, n2>(tensor, known_lower_bound);
  known_lower_bound = std::max(known_lower_bound, rank0);
  Tensor<n1, n2, n0> tensor1 = CyclicTranspose<n0, n1, n2>(tensor);
  int rank1 =
      RankLowerBoundForcedProductA<n1, n2, n0>(tensor1, known_lower_bound);
  known_lower_bound = std::max(known_lower_bound, rank1);
  Tensor<n2, n0, n1> tensor2 = CyclicTranspose<n1, n2, n0>(tensor1);
  int rank2 =
      RankLowerBoundForcedProductA<n2, n0, n1>(tensor2, known_lower_bound);
  known_lower_bound = std::max(known_lower_bound, rank2);
  pb::ForcedProductProof forced_product_proof;
  if (rank0 == known_lower_bound) {
    forced_product_proof.set_projection_type(0);
  } else if (rank1 == known_lower_bound) {
    forced_product_proof.set_projection_type(1);
  } else if (rank2 == known_lower_bound) {
    forced_product_proof.set_projection_type(2);
  } else {
    return {0, forced_product_proof};
  }
  return {known_lower_bound, forced_product_proof};
}

template <int n0, int n1, int n2>
std::pair<int, pb::DegenerateProof> RankLowerBoundDegenerate(
    const RestrictionsMap<n0, n1, n2> &restrictions_to_rank_lower_bound,
    Restrictions<n0, n1> restrictions) {
  int rank_lower_bound = 0;
  pb::DegenerateProof degenerate_proof;
  static_assert(n0 * n1 <
                std::numeric_limits<StaticMatrixData<n0, n1>>::digits);
  for (StaticMatrixData<n0, n1> restriction = 1;
       restriction < (StaticMatrixData<n0, n1>(1) << (n0 * n1));
       ++restriction) {
    restrictions.push_back(restriction);
    if (IsLinearIndependent(n0 * n1, restrictions)) {
      bool transpose = false;
      StaticMatrix<n0> gl_left;
      StaticMatrix<n1> gl_right;
      int r = restrictions_to_rank_lower_bound.Get(restrictions, &transpose,
                                                   &gl_left, &gl_right);
      if (r > rank_lower_bound) {
        rank_lower_bound = r;
        degenerate_proof.set_extra_restriction(
            static_cast<uint32_t>(restriction));
        degenerate_proof.mutable_transformation()->set_transpose(transpose);
        degenerate_proof.mutable_transformation()->set_gl_left(gl_left.Data());
        degenerate_proof.mutable_transformation()->set_gl_right(
            gl_right.Data());
      }
    }
    restrictions.pop_back();
  }
  return {rank_lower_bound, degenerate_proof};
}

template <int n0, int n1, int n2>
void BuildRestrictionsMap(
    const pb::RestrictedMMCollection &collection, int rank_lower_bound_max,
    RestrictionsMap<n0, n1, n2> *restrictions_to_rank_lower_bound) {
  int size = collection.restricted_mm_size();
  std::atomic<int> progress = 0;
  tbb::parallel_for(
      tbb::blocked_range<int>(0, size),
      [&](const tbb::blocked_range<int> &range) {
        for (int i = range.begin(); i != range.end(); ++i) {
          std::cerr << std::format("  {}/{}    \r", progress.fetch_add(1),
                                   size);
          const pb::RestrictedMM &rmm = collection.restricted_mm(i);
          CHECK_EQ(rmm.n0(), n0);
          CHECK_EQ(rmm.n1(), n1);
          CHECK_EQ(rmm.n2(), n2);
          if (rmm.rank_lower_bound() > rank_lower_bound_max) {
            continue;
          }
          restrictions_to_rank_lower_bound->Set(
              RestrictionsFromProto<n0, n1>(rmm), rmm.rank_lower_bound());
        }
      });
}

template <int n0, int n1, int n2>
std::tuple<int, pb::RankLowerBoundProof, std::string> ProcessRestrictedMM(
    const pb::RestrictedMM &rmm,
    const RestrictionsMap<n0, n1, n2> &restrictions_to_rank_lower_bound,
    bool basic_method, bool degenerate_method, uint64_t backtracking_step_limit,
    const std::string &bt_proof_root_dir) {
  CHECK_EQ(rmm.n0(), n0);
  CHECK_EQ(rmm.n1(), n1);
  CHECK_EQ(rmm.n2(), n2);
  CHECK_EQ(rmm.p(), 2);

  int rank_lower_bound =
      rmm.has_rank_lower_bound() ? rmm.rank_lower_bound() : -1;
  pb::RankLowerBoundProof rank_lower_bound_proof;
  std::string proof_case_name;

  Restrictions<n0, n1> restrictions = RestrictionsFromProto<n0, n1>(rmm);
  Tensor<n0, n1, n2> tensor = SparseStringToTensor<n0, n1, n2>(rmm.tensor());

  if (basic_method && !rmm.has_rank_lower_bound()) {
    // FlattenMatrix method
    int flatten_matrix_rank = RankLowerBoundFlattenMatrix<n0, n1, n2>(tensor);
    if (flatten_matrix_rank > rank_lower_bound) {
      rank_lower_bound = flatten_matrix_rank;
      *rank_lower_bound_proof.mutable_flatten_matrix_proof() = {};
      proof_case_name = "flatten_matrix";
    }
  }

  if (degenerate_method) {
    // Degenerate method
    auto [degenerate_rank, degenerate_proof] =
        RankLowerBoundDegenerate<n0, n1, n2>(restrictions_to_rank_lower_bound,
                                             restrictions);
    if (degenerate_rank > rank_lower_bound) {
      rank_lower_bound = degenerate_rank;
      *rank_lower_bound_proof.mutable_degenerate_proof() =
          std::move(degenerate_proof);
      proof_case_name = "degenerate";
    }
  }

  if (basic_method && !rmm.has_rank_lower_bound()) {
    // ForcedProduct method
    auto [forced_product_rank, forced_product_proof] =
        RankLowerBoundForcedProduct<n0, n1, n2>(tensor, rank_lower_bound);
    if (forced_product_rank > rank_lower_bound) {
      rank_lower_bound = forced_product_rank;
      *rank_lower_bound_proof.mutable_forced_product_proof() =
          std::move(forced_product_proof);
      proof_case_name = "forced_prod";
    }
  }

  if (backtracking_step_limit > 0) {
    // Backtracking method
    std::string proof_path =
        GetBacktrackingProofPath(bt_proof_root_dir, rmm.index(), true);
    while (true) {
      auto [backtracking_rank, backtracking_proof] =
          RankLowerBoundBacktracking<n0, n1, n2>::Search(
              restrictions, restrictions_to_rank_lower_bound, rank_lower_bound,
              backtracking_step_limit, proof_path);
      if (backtracking_rank <= rank_lower_bound) {
        break;
      }
      rank_lower_bound = backtracking_rank;
      *rank_lower_bound_proof.mutable_backtracking_proof() =
          std::move(backtracking_proof);
      proof_case_name = "backtracking";
    }
  }

  return {rank_lower_bound, rank_lower_bound_proof, proof_case_name};
}

// Options for ProcessRestrictions.
struct ProcessOptions {
  bool basic_method = true;
  bool degenerate_method = true;
  uint64_t backtracking_step_limit = std::numeric_limits<uint64_t>::max();
  int rank_lower_bound_min = 0;
  int rank_lower_bound_max = std::numeric_limits<int>::max();
  int restriction_size_min = 0;
  int restriction_size_max = std::numeric_limits<int>::max();
  std::string bt_proof_root_dir;
};

// Processes all restricted MMs in collection that have rank_lower_bound ==
// rlb. Updates their rank lower bounds and restrictions_to_rank_lower_bound.
// Returns true if any rank lower bound was updated.
template <int n0, int n1, int n2>
bool ProcessOneRankLowerBound(
    int restriction_size, const ProcessOptions &options,
    pb::RestrictedMMCollection *collection,
    RestrictionsMap<n0, n1, n2> *restrictions_to_rank_lower_bound) {
  auto iteration_start = std::chrono::steady_clock::now();

  std::vector<pb::RestrictedMM *> rmms;
  for (int i = 0; i < collection->restricted_mm_size(); ++i) {
    pb::RestrictedMM *rmm = collection->mutable_restricted_mm(i);
    if (rmm->restriction_size() == restriction_size) {
      if (rmm->rank_lower_bound() >= options.rank_lower_bound_min &&
          rmm->rank_lower_bound() <= options.rank_lower_bound_max) {
        rmms.push_back(rmm);
      }
    }
  }
  LOG(INFO) << "Processing restriction_size=" << restriction_size
            << ", count=" << rmms.size();

  // Parallel processing with progress tracking
  std::vector<std::pair<int, pb::RankLowerBoundProof>> rank_and_proof_list(
      rmms.size());
  std::atomic<int> progress = 0;
  tbb::parallel_for(0, static_cast<int>(rmms.size()), [&](int idx) {
    std::cerr << std::format("  {}/{}    \r", progress.fetch_add(1),
                             rmms.size());
    auto [rank, proof, proof_case_name] = ProcessRestrictedMM<n0, n1, n2>(
        *rmms[idx], *restrictions_to_rank_lower_bound, options.basic_method,
        options.degenerate_method, options.backtracking_step_limit,
        options.bt_proof_root_dir);
    if (proof.proof_case() != pb::RankLowerBoundProof::PROOF_NOT_SET) {
      LOG(INFO) << "Better LB for rmm_index=" << rmms[idx]->index() << ": "
                << rmms[idx]->rank_lower_bound() << "->" << rank << " ("
                << proof_case_name << ")";
    }
    rank_and_proof_list[idx] = {rank, proof};
  });

  progress = 0;
  tbb::parallel_for(
      tbb::blocked_range<int>(0, static_cast<int>(rmms.size())),
      [&](const tbb::blocked_range<int> &range) {
        for (int i = range.begin(); i != range.end(); ++i) {
          std::cerr << std::format("  {}/{}    \r", progress.fetch_add(1),
                                   rmms.size());
          pb::RestrictedMM *rmm = rmms[i];
          if (!rmm->has_rank_lower_bound() ||
              rmm->rank_lower_bound() < rank_and_proof_list[i].first) {
            restrictions_to_rank_lower_bound->Set(
                RestrictionsFromProto<n0, n1>(*rmm),
                rank_and_proof_list[i].first);
          }
        }
      });

  bool has_update = false;
  for (int i = 0; i < rmms.size(); ++i) {
    std::cerr << std::format("  i={}/{}    \r", i, rmms.size());
    if (rank_and_proof_list[i].second.proof_case() ==
        pb::RankLowerBoundProof::PROOF_NOT_SET) {
      continue;
    }
    pb::RestrictedMM *rmm = rmms[i];
    CHECK_LE(rmm->rank_lower_bound(), rank_and_proof_list[i].first);
    has_update = true;
    if (!rank_and_proof_list[i].second.has_backtracking_proof()) {
      std::string proof_path =
          GetBacktrackingProofPath(options.bt_proof_root_dir, rmm->index());
      if (std::filesystem::exists(proof_path)) {
        std::filesystem::remove(proof_path);
      }
    }
    rmm->set_rank_lower_bound(rank_and_proof_list[i].first);
    *rmm->mutable_rank_lower_bound_proof() =
        std::move(rank_and_proof_list[i].second);
  }

  auto iteration_end = std::chrono::steady_clock::now();
  auto iteration_duration =
      std::chrono::duration<double>(iteration_end - iteration_start);
  LOG(INFO) << "  duration=" << std::fixed << std::setprecision(1)
            << iteration_duration.count();
  return has_update;
}

template <int n0, int n1, int n2>
bool ProcessRestrictions(
    const ProcessOptions &options, const std::string &output_path,
    pb::RestrictedMMCollection *collection,
    RestrictionsMap<n0, n1, n2> *restrictions_to_rank_lower_bound) {
  bool has_update = false;
  for (int restriction_size = n0 * n1; restriction_size >= 0;
       restriction_size--) {
    if (restriction_size < options.restriction_size_min ||
        restriction_size > options.restriction_size_max) {
      continue;
    }
    if (ProcessOneRankLowerBound<n0, n1, n2>(
            restriction_size, options, collection,
            restrictions_to_rank_lower_bound)) {
      has_update = true;
    }
    if (!output_path.empty()) {
      WriteProtoToFile(*collection, output_path);
    }
  }
  return has_update;
}
