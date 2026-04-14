#pragma once

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <limits>
#include <numeric>
#include <vector>

#include <ng-log/logging.h>
#include <tbb/blocked_range.h>
#include <tbb/concurrent_hash_map.h>
#include <tbb/parallel_for.h>

#include "proof_verifier/backtracking_proof.h"
#include "proof_verifier/restricted_mm.pb.h"
#include "proof_verifier/restrictions.h"
#include "proof_verifier/static_matrix.h"
#include "restrictions_map.h"

template <int n0, int n1, int n2> class RankLowerBoundBacktracking {
public:
  static std::pair<int, pb::BacktrackingProof>
  Search(const Restrictions<n0, n1> &restrictions,
         const RestrictionsMap<n0, n1, n2> &restrictions_to_rank_lower_bound,
         int known_rank_lower_bound, uint64_t step_limit,
         const std::string &proof_path) {
    return RankLowerBoundBacktracking(
               restrictions, restrictions_to_rank_lower_bound,
               known_rank_lower_bound, step_limit, proof_path)
        .Search();
  }

private:
  struct LocalMapValue {
    uint8_t rank = 0;
    uint8_t transpose = 0;
    StaticMatrix<n0> gl_left;
    StaticMatrix<n1> gl_right;
  };
  static_assert(sizeof(LocalMapValue) == 6);
  using LocalMap =
      boost::unordered_flat_map<Restrictions<n0, n1>, LocalMapValue>;

  RankLowerBoundBacktracking(
      const Restrictions<n0, n1> &restrictions,
      const RestrictionsMap<n0, n1, n2> &restrictions_to_rank_lower_bound,
      int known_rank_lower_bound, uint64_t step_limit,
      const std::string &proof_path)
      : base_restrictions_(restrictions),
        restrictions_to_rank_lower_bound_(restrictions_to_rank_lower_bound),
        step_limit_(step_limit), proof_path_(proof_path) {
    known_rank_lower_bound_ =
        std::max(known_rank_lower_bound,
                 restrictions_to_rank_lower_bound_.Get(base_restrictions_));
    if (base_restrictions_.size() != n0 * n1 && known_rank_lower_bound_ == 0) {
      LOG(WARNING) << "Known rank lower bound is 0 for "
                   << RestrictionsToString<n0, n1>(base_restrictions_);
    }
    target_rank_lower_bound_ = known_rank_lower_bound_ + 1;
    max_depth_ = known_rank_lower_bound_;

    CHECK_LT(max_depth_, 32);
    for (int bitwidth = 0; bitwidth < max_depth_; bitwidth++) {
      std::vector<uint32_t> masks_i(uint32_t(1) << bitwidth);
      std::iota(masks_i.begin(), masks_i.end(), uint32_t(0));
      std::sort(masks_i.begin(), masks_i.end(), [&](uint32_t a, uint32_t b) {
        return std::popcount(a) < std::popcount(b);
      });
      masks_.push_back(std::move(masks_i));
    }

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
  }

  std::pair<int, pb::BacktrackingProof> Search() const {
    if (base_restrictions_.size() == n0 * n1) {
      return {0, {}};
    }
    if (max_depth_ == 0) {
      return {0, {}};
    }
    if (step_limit_ == 0) {
      return {0, {}};
    }

    // Parallel in a top level of the DFS.
    std::atomic<bool> early_break = false;
    std::atomic<uint64_t> step_count(0);
    std::vector<std::pair<int, BacktrackingProof>> rank_and_proof_list(
        minimal_restrictions_.size());
    tbb::parallel_for(
        tbb::blocked_range<size_t>(0, minimal_restrictions_.size()),
        [&](const tbb::blocked_range<size_t> &range) {
          Restrictions<n0, n1> dfs_restrictions;
          LocalMap local_map;

          for (size_t i = range.begin(); i < range.end(); ++i) {
            if (early_break) {
              break;
            }
            dfs_restrictions.push_back(minimal_restrictions_[i]);
            rank_and_proof_list[i].first =
                Search(i, &step_count, &dfs_restrictions, &local_map,
                       &rank_and_proof_list[i].second);
            dfs_restrictions.pop_back();
            if (rank_and_proof_list[i].first <= known_rank_lower_bound_) {
              early_break = true;
            }
          }
        });

    int rank_lower_bound_child = std::numeric_limits<int>::max();
    size_t proof_size = 0;
    for (const auto &rank_and_proof : rank_and_proof_list) {
      rank_lower_bound_child =
          std::min(rank_lower_bound_child, rank_and_proof.first);
      proof_size += rank_and_proof.second.Size();
    }
    if (rank_lower_bound_child <= known_rank_lower_bound_) {
      return {0, {}};
    }
    BacktrackingProof proof;
    proof.Reserve(proof_size);
    for (const auto &rank_and_proof : rank_and_proof_list) {
      proof.Append(rank_and_proof.second);
    }
    CHECK_EQ(proof.Size(), proof_size);
    rank_and_proof_list.clear();
    proof.Save(proof_path_);
    pb::BacktrackingProof proof_proto;
    proof_proto.set_proof_size(proof_size);
    return {rank_lower_bound_child, proof_proto};
  }

  int Search(int max_restriction_idx, std::atomic<uint64_t> *step_count,
             Restrictions<n0, n1> *dfs_restrictions, LocalMap *local_map,
             BacktrackingProof *proof) const {
    if (step_count->fetch_add(1, std::memory_order_relaxed) >= step_limit_) {
      return 0;
    }
    int rank_lower_bound_self = 0;
    CHECK(!dfs_restrictions->empty());
    constexpr size_t max_map_size = 10'000'000;
    if (local_map->size() >= max_map_size) {
      // delete half of the elements to avoid OOM
      unsigned int bit = local_map->size() & 1;
      for (auto it = local_map->begin(); it != local_map->end();) {
        if (bit == 0) {
          it = local_map->erase(it);
        } else {
          ++it;
        }
        bit ^= 1;
      }
    }
    Restrictions<n0, n1> restrictions;
    restrictions.reserve(dfs_restrictions->size());
    for (uint32_t mask : masks_[dfs_restrictions->size() - 1]) {
      restrictions.clear();
      for (int i = 0; i < static_cast<int>(dfs_restrictions->size()) - 1; ++i) {
        if (mask & (1 << i)) {
          restrictions.push_back(dfs_restrictions->at(i));
        }
      }
      restrictions.push_back(dfs_restrictions->back());
      auto [it, inserted] = local_map->try_emplace(restrictions);
      LocalMapValue local_map_value;
      if (inserted) {
        // not found in the hash map, compute and insert it.
        restrictions.insert(restrictions.end(), base_restrictions_.begin(),
                            base_restrictions_.end());
        bool transpose = false;
        uint8_t rank = restrictions_to_rank_lower_bound_.Get(
            restrictions, &transpose, &local_map_value.gl_left,
            &local_map_value.gl_right);
        local_map_value.rank = rank;
        local_map_value.transpose = transpose;
        it->second = local_map_value;
      } else {
        // in the hash map, get the value.
        local_map_value = it->second;
      }
      rank_lower_bound_self =
          std::max<int>(rank_lower_bound_self,
                        std::popcount(mask) + 1 + local_map_value.rank);
      if (rank_lower_bound_self >= target_rank_lower_bound_) {
        uint32_t full_mask = mask | (1 << (dfs_restrictions->size() - 1));
        proof->Append(dfs_restrictions->size(), full_mask,
                      local_map_value.transpose, local_map_value.gl_left.Data(),
                      local_map_value.gl_right.Data());
        return rank_lower_bound_self;
      }
    }

    if (static_cast<int>(dfs_restrictions->size()) == max_depth_) {
      return rank_lower_bound_self;
    }

    int rank_lower_bound_child = std::numeric_limits<int>::max();
    for (int i = 0; i <= max_restriction_idx; ++i) {
      dfs_restrictions->push_back(minimal_restrictions_[i]);
      rank_lower_bound_child =
          std::min(rank_lower_bound_child,
                   Search(i, step_count, dfs_restrictions, local_map, proof));
      dfs_restrictions->pop_back();
      if (rank_lower_bound_child <= rank_lower_bound_self) {
        break;
      }
    }

    return std::max(rank_lower_bound_self, rank_lower_bound_child);
  }

  const Restrictions<n0, n1> &base_restrictions_;
  const RestrictionsMap<n0, n1, n2> &restrictions_to_rank_lower_bound_;
  int known_rank_lower_bound_ = 0;
  uint64_t step_limit_ = 0;
  int max_depth_ = 0;
  int target_rank_lower_bound_ = 0;
  std::vector<std::vector<uint32_t>> masks_;
  std::vector<StaticMatrixData<n0, n1>> minimal_restrictions_;
  std::string proof_path_;
};
